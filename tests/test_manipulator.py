"""Series-parallel manipulator of arXiv:2510.23226: compilation, partition and its identities."""
import unittest

import numpy as np

import conftest  # noqa: F401
import methods as M
from examples import series_parallel as sp
from ipd import partition as ip
from ipd.backends.pin_backend import PinBackend
from ipd.compiler import ModelError, compile_graph
from systems import CASES


class CompileTest(unittest.TestCase):
    def test_record_is_compiled_without_user_cuts(self):
        comp = compile_graph(sp.physical(), 'actuator_space')
        self.assertEqual(comp.cmg['independent_ids'], ['phi', 'theta', 'delta'])
        self.assertEqual(len(comp.cmg['closures']), 1)
        self.assertFalse(comp.cmg['floating'])

    def test_unknown_partition_is_rejected(self):
        with self.assertRaises((ModelError, KeyError)):
            compile_graph(sp.physical(), 'no_such_partition')

    def test_closed_chain_relations_of_the_reference(self):
        case = CASES['manipulator']()
        s, comp = case.sys, case.comp
        rng = np.random.default_rng(1)
        iz, idl = s.index['m/zeta'], s.index['m/delta']
        col = list(comp.mode.independent_ids).index('m/delta')
        for _ in range(8):
            q, u, f = case.sample(rng)
            E = comp.lift(comp.evaluate(q))
            delta = q.theta[idl]
            self.assertLess(abs(q.theta[iz] - sp.zeta_of_delta(delta)), 1e-9)
            self.assertLess(abs(E[iz, col] / sp.dzeta_ddelta(delta) - 1.), 1e-9)


class DynamicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = {k: CASES[k]() for k in ('manipulator', 'manipulator_joint')}

    def test_acceleration_against_reference_and_pinocchio(self):
        for name, case in self.cases.items():
            s, comp = case.sys, case.comp
            pb = PinBackend(s)
            solver = pb.solver(comp.mode)
            rng = np.random.default_rng(2)
            for _ in range(4):
                q, u, f = case.sample(rng)
                ev = comp.evaluate(q, u)
                vT = ev.v[:s.nT]
                ref = M.reference(case, q, vT, f, pb)
                a, _ = M.ip_two_level(case, q, u, f, ev)
                self.assertLess(M.discrepancy(ref, a)['delta_a'], 1e-9, name)
                a_pin, _ = M.pinocchio_kkt(case, q, u, f, pb, solver, vT)
                self.assertLess(M.discrepancy(ref, a_pin)['delta_a'], 1e-8, name)
                a_nc, _ = M.no_curvature(case, q, u, f)
                self.assertGreater(M.discrepancy(ref, a_nc)['delta_a'], 1e-6, name)

    def test_factorization_identity_by_differences(self):
        """``Gammadot - C - C^T = 0`` for the sum and for every body, with ``Gammadot`` from differences."""
        from run_statewise import bodywise_gamma_rate_fd
        case = self.cases['manipulator']
        s, comp = case.sys, case.comp
        rng = np.random.default_rng(3)
        q, u, f = case.sample(rng)
        ev = comp.evaluate(q, u)
        kin, E, Ed = M.one_level(case, q, u)
        t = ip.assemble(s, kin, E, Ed, per_body=True)
        scale = np.max(abs(ev.Gamma))
        Gd = M.gamma_rate_fd(case, q, u)
        self.assertLess(np.max(abs(Gd - ev.C - ev.C.T)) / scale, 1e-6)
        Gdi = bodywise_gamma_rate_fd(case, q, E @ u)
        self.assertLess(np.max(abs(Gdi - t.C_i - np.swapaxes(t.C_i, 1, 2))) / scale, 1e-6)
        # the test can fail: without the curvature term the identity does not hold
        t0 = ip.assemble(s, kin, E, np.zeros_like(Ed), per_body=True)
        self.assertGreater(np.max(abs(Gdi - t0.C_i - np.swapaxes(t0.C_i, 1, 2))) / scale, 1e-3)
        self.assertLess(abs(M.potential_rate_fd(case, q, u) - u @ ev.g) / max(1., abs(u @ ev.g)), 1e-6)

    def test_holonomic_formula_agrees_on_a_fixed_base(self):
        case = self.cases['manipulator']
        q, u, f = case.sample(np.random.default_rng(4))
        ev = case.comp.evaluate(q, u)
        h = M.lagrangian_bias_fd(case, q, u)
        self.assertLess(np.max(abs(h - ev.C @ u)) / np.max(abs(ev.C @ u)), 1e-6)

    def test_regressor_is_linear_in_inertial_parameters(self):
        case = self.cases['manipulator']
        s, comp = case.sys, case.comp
        rng = np.random.default_rng(5)
        q, u, f = case.sample(rng)
        ev = comp.evaluate(q, u)
        kin, E, Ed = M.one_level(case, q, u)
        a, ur = rng.normal(size=comp.d), rng.normal(size=comp.d)
        Y = ip.regressor(s, kin, E, Ed, a, ur)
        ET = E[:s.nT]
        rotor = ET.T @ (s.armature * (ET @ a + Ed[:s.nT] @ ur))
        lhs = ev.Gamma @ a + ev.C @ ur + ev.g
        self.assertLess(np.max(abs(np.einsum('bak,bk->a', Y, ip.inertia_parameters(s)) + rotor - lhs)) / np.max(abs(lhs)), 1e-11)


if __name__ == '__main__':
    unittest.main()
