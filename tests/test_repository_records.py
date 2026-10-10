"""The dynamics backend on a user-supplied excavator record.

The record is not distributed with this repository: set ``ROBOCOMPILER_EXCAVATOR_RECORD``
to its path to run these tests (they are skipped otherwise).  It is compiled in each of its
partition requests and checked against the 50-digit hard-constraint reference and against
Pinocchio's constrained dynamics.
"""
import unittest

import numpy as np

import conftest  # noqa: F401
import methods as M
from examples import excavator as X
from ipd.backends.pin_backend import PinBackend
from ipd.compiler import compile_graph
from systems import EXCAVATOR_CASES, available, excavator


def sample(case, rng):
    """A random state of the case (polished to round-off by the case) and its closure residual."""
    q, u, f = case.sample(rng)
    _, info = case.comp.mode.correct(q, maxiter=0)
    return q, u, f, info


@unittest.skipUnless(X.available(), f'no excavator record (set {X.ENV})')
class ExcavatorRecordTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        g = X.physical()
        parts = [p for c, p in EXCAVATOR_CASES.items() if available(c)]
        cls.built = {p: (g, compile_graph(g, p), excavator(p)) for p in parts}

    def test_record_compiles_in_both_partition_requests(self):
        for partition, (g, graph, case) in self.built.items():
            s, comp = case.sys, case.comp
            self.assertEqual(graph.cmg['independent_ids'], g['partition_requests'][partition])
            self.assertEqual(comp.d, len(g['partition_requests'][partition]))
            self.assertEqual(s.na, len(g['actuators']))
            # every joint beyond a spanning tree of the bodies (ground included) closes an independent loop
            self.assertEqual(len(graph.cmg['closures']), len(g['joints']) - (len(g['bodies']) - 1))
            self.assertFalse(graph.cmg['floating'])
            # the seed of the record closes every loop in this package's frame conventions
            _, info = comp.mode.correct(s.neutral(), maxiter=0)
            self.assertLess(info['residual_inf'], 1e-9, partition)
            self.assertLess(info['gap_m'], 1e-9, partition)

    def test_accelerations_agree_with_both_references(self):
        """Partitioned model against the 50-digit hard-constraint reference and Pinocchio's constrained dynamics."""
        for partition, (g, graph, case) in self.built.items():
            s, comp = case.sys, case.comp
            pb = PinBackend(s)
            solver = pb.solver(comp.mode)
            rng = np.random.default_rng(7)
            for _ in range(5):
                q, u, f, info = sample(case, rng)
                self.assertLess(info['residual_inf'], 1e-11, partition)
                ev = comp.evaluate(q, u)
                vT = ev.v[:s.nT]
                ref = M.reference(case, q, vT, f, pb)
                a, _ = M.ip_two_level(case, q, u, f, ev)
                d = M.discrepancy(ref, a)
                self.assertLess(d['delta_a'], 1e-8, partition)
                self.assertLess(d['delta_f'], 1e-10, partition)
                a_pin, _ = M.pinocchio_kkt(case, q, u, f, pb, solver, vT)
                self.assertLess(M.discrepancy(ref, a_pin)['delta_a'], 1e-8, partition)
                # the comparison can fail: without the constraint curvature the model is wrong
                a_nc, _ = M.no_curvature(case, q, u, f)
                self.assertGreater(M.discrepancy(ref, a_nc)['delta_a'], 1e-5, partition)

    def test_factorization_identity_by_differences(self):
        """``Gammadot - C - C^T = 0`` with ``Gammadot`` from centred differences along the motion."""
        for partition, (g, graph, case) in self.built.items():
            q, u, f, _ = sample(case, np.random.default_rng(11))
            ev = case.comp.evaluate(q, u)
            Gd = M.gamma_rate_fd(case, q, u)
            self.assertLess(np.max(abs(Gd - ev.C - ev.C.T)) / np.max(abs(ev.Gamma)), 1e-6, partition)
            self.assertLess(abs(M.potential_rate_fd(case, q, u) - u @ ev.g) / max(1., abs(u @ ev.g)), 1e-6, partition)


if __name__ == '__main__':
    unittest.main()
