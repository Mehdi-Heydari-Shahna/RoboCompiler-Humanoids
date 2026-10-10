"""One closed-chain humanoid: record, PACDM acquisition, partition and independent engines."""
import hashlib
import unittest
from pathlib import Path

import numpy as np

import conftest  # noqa: F401
import methods as M
from dancers.humanoid import humanoid
from ipd.backends.pin_backend import PinBackend
from systems import CASES

ROOT = Path(__file__).resolve().parents[1]
PACDM_SHA256 = 'bbd1fb48'          # first eight hex digits of the vendored solver's SHA-256


class RecordTest(unittest.TestCase):
    def test_record_counts(self):
        r = humanoid()
        self.assertEqual(len(r['bodies']), 42)
        self.assertEqual(len(r['joints']), 49)
        self.assertEqual(len(r['actuators']), 21)
        self.assertFalse(any('cut' in j for j in r['joints']))
        mass = sum(b['mass_kg'] for b in r['bodies'])
        self.assertAlmostEqual(mass, 61.72, places=2)

    def test_vendored_pacdm_is_unchanged(self):
        digest = hashlib.sha256((ROOT / 'vendor' / 'pacdm_original.py').read_bytes()).hexdigest()
        self.assertTrue(digest.startswith(PACDM_SHA256), digest)


class DancerDynamicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = {k: CASES[k]() for k in ('dancer_floating', 'dancer_double')}

    def test_dimensions(self):
        self.assertEqual(self.cases['dancer_floating'].comp.d, 27)
        self.assertEqual(self.cases['dancer_double'].comp.d, 15)
        self.assertEqual(self.cases['dancer_double'].sys.na, 21)

    def test_lift_equals_vendored_pacdm_mapping(self):
        for case in self.cases.values():
            comp = case.comp
            q, u, f = case.sample(np.random.default_rng(0))
            E = comp.lift(comp.evaluate(q))
            N, info, graph = comp.mode.mapping_pacdm(q)
            self.assertTrue(info['success'])
            self.assertEqual(info['rank_full'], info['expected_rank'])
            self.assertLess(np.max(abs(E[graph.keep] - N)), 1e-10)

    def test_routes_agree_with_the_reference(self):
        for name, case in self.cases.items():
            s, comp = case.sys, case.comp
            pb = PinBackend(s)
            solver = pb.solver(comp.mode)
            rng = np.random.default_rng(1)
            for _ in range(3):
                q, u, f = case.sample(rng)
                ev = comp.evaluate(q, u)
                vT = ev.v[:s.nT]
                ref = M.reference(case, q, vT, f, pb)
                cache = M.one_level(case, q, u)
                for route, a in (('two-level', M.ip_two_level(case, q, u, f, ev)[0]),
                                 ('one-level', M.ip_one_level(case, q, u, f, cache)[0]),
                                 ('projected tree', M.projected_tree(case, q, u, f, pb, cache, 'analytic')[0]),
                                 ('pinocchio', M.pinocchio_kkt(case, q, u, f, pb, solver, vT)[0])):
                    self.assertLess(M.discrepancy(ref, a)['delta_a'], 1e-9, f'{name}: {route}')
                a_fd = M.projected_tree(case, q, u, f, pb, cache, 'fd')[0]
                self.assertLess(M.discrepancy(ref, a_fd)['delta_a'], 1e-7, name)

    def test_quasi_velocity_bias_differs_from_the_holonomic_formula(self):
        case = self.cases['dancer_double']
        q, u, f = case.sample(np.random.default_rng(2))
        ev = case.comp.evaluate(q, u)
        h = M.lagrangian_bias_fd(case, q, u)
        self.assertGreater(np.max(abs(h - ev.C @ u)) / np.max(abs(ev.C @ u)), 1e-2)

    def test_mujoco_tree_dynamics_and_rigid_limit(self):
        import mujoco
        case = self.cases['dancer_double']
        s, comp = case.sys, case.comp
        pb = PinBackend(s)
        mj = M.MujocoStatewise(case)
        q, u, f = case.sample(np.random.default_rng(3))
        ev = comp.evaluate(q, u)
        vT = ev.v[:s.nT]
        ref = M.reference(case, q, vT, f, pb)
        active = mj.plant.data.eq_active.copy()
        mj.plant.data.eq_active[:] = 0
        a_free = mj.acceleration(q, vT, f)
        mj.plant.data.eq_active[:] = active
        a_tree = np.linalg.solve(ref['M'], -ref['h'])
        self.assertLess(np.max(abs(a_free - a_tree)) / np.max(abs(a_tree)), 1e-7)
        a_lim, _ = mj.rigid_limit(q, vT, f)
        version = tuple(int(x) for x in mujoco.__version__.split('.')[:2])
        if version >= (3, 7):          # the constraint bias contains Jdot v (MuJoCo changelog, 3.7.0)
            self.assertLess(M.discrepancy(ref, a_lim)['delta_a'], 1e-3)
        else:                          # earlier releases leave Jdot v out, in MuJoCo's velocity coordinates
            a_old = M.kkt_without_velocity_product_mujoco(case, ref, vT)
            self.assertLess(np.max(abs(a_lim - a_old)) / max(1., np.max(abs(ref['a']))), 1e-3)
            self.assertGreater(M.discrepancy(ref, a_lim)['delta_a'], 1e-3)


if __name__ == '__main__':
    unittest.main()
