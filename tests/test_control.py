"""The experiment controller on the compiled model: effort allocation, support cone and reference rates."""
import unittest

import numpy as np

import conftest  # noqa: F401
from ipd import se3
from ipd.control import Allocator, SupportCone, tracking_error
from ipd.integrate import MinimalModel
from systems import CASES


class AllocationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = CASES['dancer_double']()
        cls.q, cls.u, cls.f = cls.case.sample(np.random.default_rng(0))
        cls.ev = cls.case.comp.evaluate(cls.q, cls.u)

    def cones(self):
        out = {}
        for m in self.case.sys.members:
            for site in m.comp.cmg['sole_sites']:
                out[f"{m.name}/{site['id']}"] = SupportCone(*site['half_extent_m'])
        return out

    def test_allocation_realises_the_generalized_force(self):
        comp, ev = self.case.comp, self.ev
        alloc = Allocator(comp, self.cones())
        ev0 = comp.evaluate(self.case.q0, np.zeros(comp.d))
        f, info = alloc.solve(ev0, ev0.g, np.zeros(comp.D1))
        self.assertLess(np.max(abs(ev0.L.T @ f - ev0.g)), 1e-6 * np.max(abs(ev0.g)))
        fz = sum(w[5] for w in info['wrenches'].values())
        self.assertAlmostEqual(fz / (self.case.sys.mass.sum() * 9.81), 1., places=6)
        for w in info['wrenches'].values():
            self.assertGreater(w[5], 0.)

    def test_failed_drive_is_taken_out(self):
        comp = self.case.comp
        alloc = Allocator(comp, self.cones())
        lo, hi = alloc.lower.copy(), alloc.upper.copy()
        k = self.case.sys.port_ids.index('d0/l_shoulder_pitch_drive')
        lo[k] = hi[k] = 0.
        alloc.set_bounds(lo, hi)
        ev0 = comp.evaluate(self.case.q0, np.zeros(comp.d))
        f, info = alloc.solve(ev0, ev0.g, np.zeros(comp.D1))
        self.assertEqual(f[k], 0.)
        self.assertFalse(info['failed'])
        self.assertEqual(sum(alloc.counts.values()), 1)
        # one dancer cannot hold that arm without its drive: the allocation says so through the slack
        self.assertGreater(info['slack'], 0.1)
        defect = ev0.L.T @ f - ev0.g
        self.assertAlmostEqual(np.max(abs(defect)), info['slack'], places=5)

    def test_infeasible_programme_is_reported_and_holds_the_efforts(self):
        """Contradictory rows: the solver reports infeasibility and the previous efforts are held."""
        comp = self.case.comp
        alloc = Allocator(comp, self.cones())
        ev0 = comp.evaluate(self.case.q0, np.zeros(comp.d))
        f0, info0 = alloc.solve(ev0, ev0.g, np.zeros(comp.D1))
        self.assertFalse(info0['failed'])
        row = np.zeros((2, alloc.na)); row[:, 0] = 1.
        contradictory = [(row, np.array([1., -np.inf]), np.array([np.inf, 0.]))]          # f_0 >= 1 and f_0 <= 0
        f, info = alloc.solve(ev0, ev0.g, np.zeros(comp.D1), f_prev=f0, extra_rows=contradictory)
        self.assertTrue(info['failed'])
        self.assertIn('infeasible', info['status'])
        self.assertTrue(np.isnan(info['slack']))
        np.testing.assert_allclose(f, np.clip(f0, alloc.lower, alloc.upper))               # previous efforts, not a marker value
        self.assertLess(np.max(abs(f)), 1e6)

    def test_active_rows_are_met_exactly(self):
        """With active cone rows the programme is solved to round-off: rows and the realised force hold."""
        comp = self.case.comp
        alloc = Allocator(comp, self.cones())
        ev0 = comp.evaluate(self.case.q0, np.zeros(comp.d))
        maps = alloc.support_maps(ev0, np.zeros(comp.D1))
        f0, _ = alloc.solve(ev0, ev0.g, np.zeros(comp.D1))
        # a row that the unconstrained solution violates but a nearby solution of L^T f = g meets: move along the
        # null space of L^T (internal forces of the double support) by one per cent of the effort bounds
        _, sv, Vt = np.linalg.svd(ev0.L.T)
        N = Vt[len(sv):].T if Vt.shape[0] > len(sv) else Vt[np.sum(sv > 1e-9 * sv[0]):].T
        k = int(np.argmax(np.max(abs(N), axis=1)))
        j = int(np.argmax(abs(N[k])))
        step = 0.01 * alloc.upper[k] * np.sign(N[k, j]) * N[:, j] / abs(N[k, j])
        target = f0[k] + 0.01 * alloc.upper[k]
        self.assertAlmostEqual((f0 + step)[k], target)
        row = np.zeros((1, alloc.na)); row[0, k] = 1.
        f, info = alloc.solve(ev0, ev0.g, np.zeros(comp.D1), extra_rows=[(row, np.array([target]), np.array([np.inf]))])
        k0 = k
        self.assertTrue(info['qp'])
        self.assertEqual(info['status'], 'solved')
        self.assertGreaterEqual(f[k0], target - 1e-8)
        self.assertLess(info['slack'], 1e-6)
        self.assertLess(np.max(abs(ev0.L.T @ f - ev0.g)), 1e-6)
        for Af, lo, hi in alloc.cone_rows(maps):
            v = Af @ f
            self.assertTrue(np.all(v >= lo - 1e-6) and np.all(v <= hi + 1e-6))

    def test_contact_wrench_cone_is_the_set_of_corner_force_wrenches(self):
        """The closed-form rows against an explicit search for friction-pyramid forces at the four corners."""
        from scipy.optimize import linprog
        cone = SupportCone(0.11, 0.05, mu=0.9, margin=1.0, fz_min=0.)
        mu, X, Y = cone.mu, cone.X, cone.Y
        G = np.zeros((6, 12))
        rows = []
        for i, r in enumerate(np.array([[sx * X, sy * Y, 0.] for sx in (-1, 1) for sy in (-1, 1)])):
            G[:3, 3 * i:3 * i + 3] = se3.skew(r)                      # moment about the sole centre
            G[3:, 3 * i:3 * i + 3] = np.eye(3)
            for c in (0, 1):
                for sg in (1., -1.):
                    row = np.zeros(12)
                    row[3 * i + c], row[3 * i + 2] = sg, -mu
                    rows.append(row)
        A_ub = np.array(rows)
        rng = np.random.default_rng(0)
        inside = 0
        for _ in range(400):
            fz = rng.uniform(50., 500.)
            w = np.r_[rng.uniform(-1.3, 1.3) * Y * fz, rng.uniform(-1.3, 1.3) * X * fz, rng.uniform(-1.5, 1.5) * mu * (X + Y) * fz,
                      rng.uniform(-1.3, 1.3) * mu * fz, rng.uniform(-1.3, 1.3) * mu * fz, fz]
            viol = cone.violation(w)
            if abs(viol) < 1e-6 * fz:
                continue
            res = linprog(np.zeros(12), A_ub=A_ub, b_ub=np.zeros(len(A_ub)), A_eq=G, b_eq=w, bounds=[(None, None)] * 12,
                          method='highs')
            self.assertEqual(res.status == 0, viol <= 0.)
            inside += viol <= 0.
        self.assertGreater(inside, 10)

    def test_support_weight_reduces_tangential_forces(self):
        comp = self.case.comp
        ev0 = comp.evaluate(self.case.q0, np.zeros(comp.d))
        tang = []
        for w in (0., 1e-3):
            f, info = Allocator(comp, self.cones(), support_weight=w).solve(ev0, ev0.g, np.zeros(comp.D1))
            self.assertLess(np.max(abs(ev0.L.T @ f - ev0.g)), 1e-6 * np.max(abs(ev0.g)))
            tang.append(sum(np.linalg.norm(x[3:5]) for x in info['wrenches'].values()))
        self.assertLessEqual(tang[1], tang[0] + 1e-9)


class ReferenceTest(unittest.TestCase):
    def test_root_block_rates_are_exact(self):
        """``edot`` and ``d/dt u_d`` of a floating root against differences along two motions."""
        case = CASES['dancer_floating']()
        s, comp = case.sys, case.comp
        ind = comp.mode.independent
        rng = np.random.default_rng(11)
        q, u, _ = case.sample(rng)
        v = comp.lift(comp.evaluate(q)) @ u
        q_ref = s.integrate(q, rng.normal(size=s.n) * 0.2 * (np.arange(s.n) < 6))      # a root pose 0.2 away
        v_ref = v + rng.normal(size=s.n) * 0.3
        a_ref = rng.normal(size=s.n)
        root = s.root_slice[0]

        def at(t):
            """State and reference at time t: constant body twist, reference twist linear in time."""
            qa = s.integrate(q, t * v)
            qr = s.integrate(q_ref, t * v_ref + .5 * t * t * a_ref)
            return tracking_error(s, ind, qr, v_ref + t * a_ref, a_ref, qa, v)

        e, e_dot, u_d, ud_d = at(0.)
        h = 1e-4
        ep, _, up_, _ = at(h)
        em, _, um_, _ = at(-h)
        blk = np.arange(6)                                             # the root twist is independent
        self.assertTrue(np.all(ind[:6] == np.arange(root.start, root.stop)))
        self.assertLess(np.max(abs((ep - em) / (2 * h) - e_dot)), 1e-6)
        self.assertLess(np.max(abs((up_ - um_) / (2 * h) - ud_d)), 1e-6)
        # the first-order forms differ: both corrections are needed
        X = se3.Ad(se3.inv(q.T_root[0]) @ q_ref.T_root[0])
        self.assertGreater(np.max(abs(X @ a_ref[root] - ud_d[blk])), 1e-3)
        self.assertGreater(np.max(abs(v[root] - u_d[blk] - e_dot[blk])), 1e-4)


class IntegrationTest(unittest.TestCase):
    def test_step_stays_on_the_closure_manifold(self):
        case = CASES['manipulator']()
        m = MinimalModel(case.comp)
        q, u, f = case.sample(np.random.default_rng(4))
        for _ in range(20):
            q, u = m.step(q, u, f, 0.005)
        self.assertLess(case.comp.mode.gap(q)[0], 1e-9)


if __name__ == '__main__':
    unittest.main()
