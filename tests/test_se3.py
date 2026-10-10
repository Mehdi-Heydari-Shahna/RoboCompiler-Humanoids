"""Angular-first SE(3) algebra used by the compiled model."""
import unittest

import numpy as np

import conftest  # noqa: F401
from ipd import se3


class Se3Test(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_exp_log_round_trip(self):
        for scale in (1e-9, 1e-3, 0.5, 2.8):
            xi = self.rng.normal(size=(20, 6))
            xi[:, :3] *= scale / np.maximum(1., np.linalg.norm(xi[:, :3], axis=1))[:, None]   # rotation angle below pi
            xi[:, 3:] *= scale
            back = se3.log(se3.exp(xi))
            self.assertLess(np.max(abs(back - xi)), 1e-10 * max(1., scale))

    def test_adjoint_identities(self):
        xi = self.rng.normal(size=6) * 0.7
        T = se3.exp(xi)
        self.assertLess(np.max(abs(se3.Ad(T) @ se3.Ad(se3.inv(T)) - np.eye(6))), 1e-12)
        self.assertLess(np.max(abs(se3.Ad_inv(T) - np.linalg.inv(se3.Ad(T)))), 1e-12)
        from scipy.linalg import expm
        self.assertLess(np.max(abs(se3.Ad(T) - expm(se3.ad(xi)))), 1e-12)

    def test_inverse_left_jacobian_is_the_log_differential(self):
        xi = self.rng.normal(size=6) * 0.4
        T = se3.exp(xi)
        J = np.zeros((6, 6))
        h = 1e-6
        for k in range(6):
            d = np.zeros(6)
            d[k] = h
            J[:, k] = (se3.log(se3.exp(d) @ T) - se3.log(se3.exp(-d) @ T)) / (2 * h)
        self.assertLess(np.max(abs(J - se3.inv_left(xi))), 1e-8)

    def test_momentum_matrix_is_skew_and_dual_to_ad(self):
        P, W = self.rng.normal(size=6), self.rng.normal(size=6)
        Pt = se3.momentum_matrix(P)
        self.assertLess(np.max(abs(Pt + Pt.T)), 1e-15)
        self.assertLess(np.max(abs(Pt @ W - se3.ad(W).T @ P)), 1e-14)

    def test_spatial_inertia_kinetic_energy(self):
        m, c = 3.2, np.array([0.1, -0.2, 0.05])
        Ic = np.diag([0.02, 0.03, 0.04])
        M = se3.spatial_inertia(np.asarray(m), c, Ic)
        V = self.rng.normal(size=6)
        w, v = V[:3], V[3:]
        vc = v + np.cross(w, c)
        self.assertAlmostEqual(.5 * V @ M @ V, .5 * m * vc @ vc + .5 * w @ Ic @ w, places=12)


if __name__ == '__main__':
    unittest.main()
