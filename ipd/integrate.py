"""The compiled dynamics as an ordinary differential equation on a closure mode.

On a closure mode the state is ``x = (q, u)``: a configuration on the closure
manifold and the independent speeds.  The equations of motion

    udot = Gamma^-1 (L^T f - C u - g - D u)

are an ODE in minimal coordinates (no multipliers, no algebraic states).  This module
provides

* ``rate``     the right-hand side;
* ``retract``  a move of the independent coordinates with the dependent ones
               following by the PACDM closure correction;
* ``step``     explicit Runge-Kutta integration on the manifold (Runge-Kutta-Munthe-Kaas
               for the twists of floating roots).

It is used by the integration tests of the state-wise study (energy integral, change of
partition during integration) and by the compiled-ODE plant.
"""
from __future__ import annotations

import numpy as np

from . import se3


class MinimalModel:
    def __init__(self, comp, correct_tol=1e-11, correct_maxiter=30):
        self.comp, self.sys = comp, comp.sys
        self.ind = comp.mode.independent
        self.d, self.na = comp.d, comp.sys.na
        self.tol, self.maxiter = correct_tol, correct_maxiter
        self.evaluations = 0
        # independent coordinates that are components of a root twist
        self.root_blocks = []
        pos = {int(j): i for i, j in enumerate(self.ind)}
        for sl in self.sys.root_slice:
            if sl is not None and all(j in pos for j in range(sl.start, sl.stop)):
                self.root_blocks.append(np.array([pos[j] for j in range(sl.start, sl.stop)], int))

    # ------------------------------------------------------------------ ODE
    def rate(self, q, u, f, tau=None):
        """Independent acceleration ``udot`` and the evaluation it was computed from."""
        ev = self.comp.evaluate(q, u)
        self.evaluations += 1
        rhs = ev.L.T @ f - ev.C @ u - ev.g - ev.D @ u
        if tau is not None:
            rhs = rhs + tau
        return np.linalg.solve(ev.Gamma, rhs), ev

    def retract(self, q, dxi):
        """Configuration after the independent coordinates move by ``dxi`` (velocity coordinates)."""
        s = self.sys
        dv = np.zeros(s.n)
        dv[self.ind] = dxi
        q1, info = self.comp.mode.correct(s.integrate(q, dv), tol=self.tol, maxiter=self.maxiter)
        if not info['success']:
            raise RuntimeError(f'closure correction failed: {info}')
        return q1

    def difference(self, q0, q1):
        """Independent velocity coordinates ``xi`` with ``retract(q0, xi) = q1``."""
        return self.sys.difference(q0, q1)[self.ind]

    def _pull_back(self, theta, u):
        """Stage velocity seen from the start of the step.

        A root twist ``u`` at the stage configuration ``T exp(theta)`` corresponds to the chart
        rate ``J_r^-1(theta) u`` at ``T`` (Runge-Kutta-Munthe-Kaas); scalar coordinates are unchanged.
        """
        if not self.root_blocks:
            return u
        out = u.copy()
        for blk in self.root_blocks:
            out[blk] = se3.inv_left(-theta[blk]) @ u[blk]
        return out

    def step(self, q, u, f, h, scheme='rk4', tau=None):
        """One explicit Runge-Kutta step of length ``h`` with constant efforts.

        ``rk4`` is the classical fourth-order scheme in the chart at ``q`` (the stage
        velocities of root twists are pulled back to that chart, so the order is kept for
        floating roots); ``midpoint`` and ``euler`` are of order two and one.
        """
        a1, _ = self.rate(q, u, f, tau)
        if scheme == 'euler':
            return self.retract(q, h * u), u + h * a1
        if scheme == 'midpoint':
            u2 = u + .5 * h * a1
            a2, _ = self.rate(self.retract(q, .5 * h * u), u2, f, tau)
            return self.retract(q, h * u2), u + h * a2
        k1 = u
        u2 = u + .5 * h * a1
        a2, _ = self.rate(self.retract(q, .5 * h * k1), u2, f, tau)
        k2 = self._pull_back(.5 * h * k1, u2)
        u3 = u + .5 * h * a2
        a3, _ = self.rate(self.retract(q, .5 * h * k2), u3, f, tau)
        k3 = self._pull_back(.5 * h * k2, u3)
        u4 = u + h * a3
        a4, _ = self.rate(self.retract(q, h * k3), u4, f, tau)
        k4 = self._pull_back(h * k3, u4)
        return self.retract(q, h / 6. * (k1 + 2. * k2 + 2. * k3 + k4)), u + h / 6. * (a1 + 2. * a2 + 2. * a3 + a4)
