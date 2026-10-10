"""Five hand-linked humanoids: composition, independent engines, local updates and events."""
import unittest

import numpy as np

import conftest  # noqa: F401
import methods as M
from ipd import partition as ip
from ipd import updates as up
from ipd.backends.pin_backend import PinBackend
from systems import CASES


class CircleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = CASES['circle']()
        cls.pb = PinBackend(cls.case.sys)
        rng = np.random.default_rng(7)
        cls.q, cls.u, cls.f = cls.case.sample(rng)
        cls.ev = cls.case.comp.evaluate(cls.q, cls.u)

    def test_dimensions(self):
        s, comp = self.case.sys, self.case.comp
        self.assertEqual(s.K, 5)
        self.assertEqual(s.nbp, 210)
        self.assertEqual(s.nT, 235)
        self.assertEqual(s.na, 105)
        self.assertEqual(comp.d, 60)
        self.assertEqual(comp.m2, 15)
        self.assertEqual(list(comp.d1), [27] * 5)

    def test_two_level_equals_one_level(self):
        case, ev, u = self.case, self.ev, self.u
        kin, E, Ed = M.one_level(case, self.q, u)
        t = ip.assemble(case.sys, kin, E, Ed)
        E2, Ed2 = case.comp.lift(ev, True)
        self.assertLess(np.max(abs(E - E2)), 1e-11)
        self.assertLess(np.max(abs(Ed - Ed2)), 1e-10)
        for name in ('Gamma', 'C', 'g'):
            a, b = getattr(ev, name), getattr(t, name)
            self.assertLess(np.max(abs(a - b)) / np.max(abs(b)), 1e-11, name)

    def test_acceleration_against_reference_and_pinocchio(self):
        case, s = self.case, self.case.sys
        vT = self.ev.v[:s.nT]
        ref = M.reference(case, self.q, vT, self.f, self.pb)
        self.assertLess(ref['rank'], ref['rows'])                      # the chords are redundant
        a, _ = M.ip_two_level(case, self.q, self.u, self.f, self.ev)
        d = M.discrepancy(ref, a)
        self.assertLess(d['delta_a'], 1e-10)
        self.assertLess(d['r_c'], 1e-9)
        solver = self.pb.solver(case.comp.mode)
        a_pin, _ = M.pinocchio_kkt(case, self.q, self.u, self.f, self.pb, solver, vT)
        self.assertLess(M.discrepancy(ref, a_pin)['delta_a'], 1e-9)
        dec = case.ens.composite(stance=case.comp.stance, links=[])
        dec.prepare(case.q0)
        a_dec, _ = M.decentralized(case, dec, self.q, self.ev.v, self.f)
        self.assertGreater(M.discrepancy(ref, a_dec)['delta_a'], 1e-3)

    def test_local_inertial_update_is_exact_and_rank_six(self):
        case, s, comp, ev = self.case, self.case.sys, self.case.comp, self.ev
        body = s.body_index['d2/r_forearm']
        dM = up.point_mass_inertia(3.0, [0.06, 0.0, -0.20], np.diag([2e-3, 2e-3, 1e-3]))
        dG, dC, dg, J = up.inertial_update(comp, ev, body, dM)
        M0 = s.M
        try:
            s.M = M0.copy()
            s.M[body] += dM
            ev2 = comp.evaluate(self.q, self.u)
        finally:
            s.M = M0
        self.assertLess(np.max(abs(ev.Gamma + dG - ev2.Gamma)) / np.max(abs(ev2.Gamma)), 1e-12)
        self.assertLess(np.max(abs(ev.C + dC - ev2.C)) / np.max(abs(ev2.C)), 1e-12)
        self.assertLess(np.max(abs(ev.g + dg - ev2.g)) / np.max(abs(ev2.g)), 1e-12)
        self.assertEqual(np.linalg.matrix_rank(dG, 1e-10), 6)
        import scipy.linalg as sla
        cho = sla.cho_factor(ev.Gamma)
        rhs = np.random.default_rng(0).normal(size=comp.d)
        x = up.woodbury_solve(lambda B: sla.cho_solve(cho, B), J, dM, rhs)
        self.assertLess(np.max(abs(x - np.linalg.solve(ev2.Gamma, rhs))) / np.max(abs(x)), 1e-10)

    def test_every_mode_equals_its_one_level_reduction(self):
        """The same member modules composed into other modes, against the reduction over all chords."""
        case, ens, comp, s = self.case, self.case.ens, self.case.comp, self.case.sys
        rng = np.random.default_rng(3)
        for kw in (dict(links=[]), dict(stance=['left'] * ens.n), dict(stance=['none'] * ens.n, links=[])):
            links = kw.get('links', comp.links)
            other = ens.composite(kw.get('stance'), links, triples={l: comp.triples[l] for l in links})
            other.prepare(self.q)
            u = rng.uniform(-1., 1., other.d) * 0.4
            ev = other.evaluate(self.q, u)
            kin = s.kinematics(self.q)
            E = other.mode.lift(kin)
            s.velocity(kin, E @ u)
            E, Ed = other.mode.lift(kin, derivative=True)
            t = ip.assemble(s, kin, E, Ed)
            self.assertLess(np.max(abs(ev.Gamma - t.Gamma)) / np.max(abs(t.Gamma)), 1e-11)
            self.assertLess(np.max(abs(ev.C @ u + ev.g - t.C @ u - t.g)) / np.max(abs(t.C @ u + t.g)), 1e-10)

    def test_partition_covariance(self):
        case, ens, s, comp = self.case, self.case.ens, self.case.sys, self.case.comp
        anchors = {sid: comp.mode.anchors[i] for i, sid in enumerate(comp.mode.support_ids)}
        alt = {l: next(t for t in ((0, 2, 3), (1, 2, 3), (0, 1, 2), (0, 1, 3)) if t != comp.triples[l]) for l in comp.links}
        other = ens.composite(comp.stance, comp.links, anchors, triples=alt)
        other.prepare(case.q0)
        a1, _ = M.ip_two_level(case, self.q, self.u, self.f, self.ev)
        u2 = self.ev.v[other.mode.independent]
        ev2 = other.evaluate(self.q, u2)
        ud2 = np.linalg.solve(ev2.Gamma, ev2.L.T @ self.f - ev2.C @ u2 - ev2.g - ev2.D @ u2)
        E2, Ed2 = other.lift(ev2, True)
        a2 = E2[:s.nT] @ ud2 + Ed2[:s.nT] @ u2
        self.assertLess(np.max(abs(a1 - a2)) / max(1., np.max(abs(a1))), 1e-9)

    def test_constraint_wrenches_balance_the_weight(self):
        case, s, comp, ev = self.case, self.case.sys, self.case.comp, self.ev
        q = case.q0
        ev0 = comp.evaluate(q, np.zeros(comp.d))
        f = np.linalg.lstsq(ev0.L.T, ev0.g, rcond=None)[0]           # static equilibrium efforts
        lam = up.constraint_wrenches(comp, ev0, f, np.zeros(comp.d))
        total = sum(lam[c][5] for c in lam if c.endswith('_sole'))
        self.assertAlmostEqual(total / (s.mass.sum() * 9.81), 1., places=9)

    def test_closing_transition_removes_energy(self):
        case, ens, s, comp = self.case, self.case.ens, self.case.sys, self.case.comp
        links_open = [l for l in comp.links if l != 'hand_0_4']
        anchors = {sid: comp.mode.anchors[i] for i, sid in enumerate(comp.mode.support_ids)}
        opened = ens.composite(comp.stance, links_open, anchors, triples={l: comp.triples[l] for l in links_open})
        opened.prepare(case.q0)
        uA = np.random.default_rng(1).normal(size=opened.d) * .5
        evA = opened.evaluate(self.q, uA)
        evB = comp.evaluate(self.q)
        w_minus = up.level1_speeds(comp, self.ev, evA.v)
        u_plus, loss, per_member = up.closing_transition(comp, evB, w_minus)
        T_before = .5 * uA @ evA.Gamma @ uA
        T_after = .5 * u_plus @ evB.Gamma @ u_plus
        self.assertGreater(loss, 0.)
        self.assertAlmostEqual((T_before - T_after) / loss, 1., places=8)
        self.assertAlmostEqual(per_member.sum(), loss, places=12)


if __name__ == '__main__':
    unittest.main()
