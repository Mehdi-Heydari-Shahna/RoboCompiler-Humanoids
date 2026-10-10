"""The experiment controller on the compiled minimal-coordinate dynamics.

The controller only exercises the dynamics layer in closed loop.  It is a conventional
model-based tracking law in the independent speeds of a closure mode (the
reference-velocity form of computed-torque control),

    Q = Gamma udot_r + C u_r + g + D u - k_s Gamma s,
    s = u - u_r,   u_r = u_d - Lambda e,

that uses the inertia, the body-wise Coriolis factor and the gravity term of the
partition.

``Allocator`` maps the generalized force ``Q`` to actuator efforts through the
virtual-work-dual port map, ``L^T f = Q``.  When a mode has more ports than
independent speeds, the free part is chosen by a small QP that keeps every support
wrench inside the contact wrench cone of its sole.  Support wrenches follow from the
equations of the dependent coordinates,

    lambda_m = R_m^-T [Gamma1 wdot + C1 w + g1 + D1 w - L1^T f]_dep(m),

so no constraint force enters the equations of motion themselves.
"""
from __future__ import annotations

import numpy as np

from . import se3


class Reference:
    """Sampled reference on the closure manifold: configurations, velocities, accelerations."""

    def __init__(self, t, T_root, theta, v, a, info=None):
        self.t = np.asarray(t, float)
        self.T_root, self.theta, self.v, self.a = T_root, theta, v, a
        self.dt = float(self.t[1] - self.t[0])
        self.info = info or {}

    def index(self, t):
        return int(np.clip(round((t - self.t[0]) / self.dt), 0, len(self.t) - 1))

    def at(self, t):
        from .system import Config
        i = self.index(t)
        return Config(self.T_root[i], self.theta[i]), self.v[i], self.a[i]


def tracking_error(system, independent, q_ref, v_ref, a_ref, q, v, root_slices=None):
    """Error in the independent coordinates, its rate and the reference seen from the state.

    Scalar joints use the coordinate difference.  A floating root uses the body-frame
    logarithm ``e = log(T_ref^-1 T)``.  The reference twist is transported to the current
    root frame, ``u_d = X u_ref`` with ``X = Ad(T^-1 T_ref)``, and then exactly

        d/dt u_d = X udot_ref - ad(u) u_d,        edot = J_r^-1(e) (u - u_d),

    with the right Jacobian ``J_r(e) = J_l(-e)`` of the exponential.  Returns ``e``,
    ``edot``, ``u_d`` and ``d/dt u_d`` restricted to ``independent``.
    """
    root_slices = system.root_slice if root_slices is None else root_slices
    d = system.difference(q_ref, q)
    u_d, ud_d = v_ref.copy(), a_ref.copy()
    ed = np.asarray(v, float) - v_ref
    for k, sl in enumerate(root_slices):
        if sl is None:
            continue
        X = se3.Ad(se3.inv(q.T_root[k]) @ q_ref.T_root[k])
        u_d[sl] = X @ v_ref[sl]
        ud_d[sl] = X @ a_ref[sl] - se3.ad(v[sl]) @ u_d[sl]
        ed[sl] = se3.inv_left(-d[sl]) @ (v[sl] - u_d[sl])
    return d[independent], ed[independent], u_d[independent], ud_d[independent]


class Gains:
    """Feedback gains in the independent speeds: ``Lambda`` (1/s) and ``K = k_s Gamma`` (``k_s`` in 1/s)."""

    def __init__(self, lam=15., ks=45.):
        self.lam, self.ks = lam, ks


def tracking_law(ev, e, e_dot, u, u_d, ud_d, gains, C=None, damping=True):
    """Generalized force of the tracking law and the speed error ``s = u - u_r``.

    ``u_r = u_d - Lambda e`` and ``udot_r = d/dt u_d - Lambda edot`` are exact time
    derivatives of each other along the motion (see ``tracking_error``).
    """
    ur = u_d - gains.lam * e
    udr = ud_d - gains.lam * e_dot
    s = u - ur
    C = ev.C if C is None else C
    Q = ev.Gamma @ (udr - gains.ks * s) + C @ ur + ev.g
    if damping:
        Q = Q + ev.D @ u
    return Q, s


# ---------------------------------------------------------------------- support wrenches and allocation
class SupportCone:
    """Contact wrench cone of a flat rectangular sole, ``[n_x, n_y, n_z, f_x, f_y, f_z]`` in its anchor frame.

    The wrench acts on the sole and is taken about the sole centre; ``z`` is the outward
    normal of the ground, ``X`` and ``Y`` are the half length and half width.  With a
    friction pyramid of coefficient ``mu`` at every point of the rectangle, the set of
    realisable wrenches is (Caron, Pham and Nakamura, ICRA 2015)

        f_z >= 0,   |f_x| <= mu f_z,   |f_y| <= mu f_z,   |n_x| <= Y f_z,   |n_y| <= X f_z,
        |Y f_x - mu n_x| + |X f_y - mu n_y| <= mu (X + Y) f_z + n_z,
        |Y f_x + mu n_x| + |X f_y + mu n_y| <= mu (X + Y) f_z - n_z.

    The last two lines bound the yaw moment; they are eight linear rows.  ``margin``
    shrinks the rectangle and the friction coefficient, and the pyramid is inscribed in
    the friction cone (``mu / sqrt(2)``).
    """
    ROWS = 17

    def __init__(self, half_length, half_width, mu=0.9, margin=0.75, fz_min=15.):
        self.X, self.Y = margin * half_length, margin * half_width
        self.mu = margin * mu / np.sqrt(2.)
        self.fz_min = fz_min
        self.margin, self.mu_assumed = float(margin), float(mu)

    def rows(self, fz_max=np.inf, tight=1.):
        """(A, lower, upper) with ``lower <= A lambda <= upper``.

        ``fz_max`` caps the normal force (a sole that is being unloaded or loaded) and
        ``tight`` in (0, 1] shrinks friction and rectangle of such a sole.
        """
        mu, X, Y = tight * self.mu, tight * self.X, tight * self.Y
        A = np.zeros((self.ROWS, 6))
        lo = np.full(self.ROWS, -np.inf); hi = np.zeros(self.ROWS)
        A[0, 5] = 1.; lo[0], hi[0] = self.fz_min if fz_max > self.fz_min else 0., max(fz_max, 0.)
        k = 1
        for col, lim in ((3, mu), (4, mu), (0, Y), (1, X)):
            A[k, col], A[k, 5] = 1., -lim
            A[k + 1, col], A[k + 1, 5] = -1., -lim
            k += 2
        for sz in (1., -1.):                 # sz = +1: bound from below (uses -n_z), -1: from above
            for sa in (1., -1.):
                for sb in (1., -1.):
                    # sa (Y f_x - sz mu n_x) + sb (X f_y - sz mu n_y) - mu (X + Y) f_z - sz n_z <= 0
                    A[k, 3], A[k, 0] = sa * Y, -sa * sz * mu
                    A[k, 4], A[k, 1] = sb * X, -sb * sz * mu
                    A[k, 5], A[k, 2] = -mu * (X + Y), -sz
                    k += 1
        return A, lo, hi

    def violation(self, lam, fz_max=np.inf, tight=1.):
        """Largest violation of the rows by a wrench (non-positive inside the cone)."""
        A, lo, hi = self.rows(fz_max, tight)
        val = A @ lam
        return float(max(np.max(val - hi), np.max(lo - val)))


DAQP_STATUS = {1: 'solved', 2: 'solved', -1: 'infeasible', -2: 'cycling', -3: 'unbounded', -4: 'iteration limit'}


class Allocator:
    """Actuator-effort allocation ``L^T f = Q`` with effort bounds, support-wrench limits,
    optional interaction-force weighting and optional extra inequality rows.

    For a prescribed motion the level-2 constraint wrenches are affine in the efforts,
    ``lambda_e = lam0_e - Lam_e f``, so the contact wrench cones are linear rows in ``f``
    and interaction forces can be weighed in the objective

        min  f^T W f + w_link sum_links |lambda_e|^2 + w_sup sum_soles |tangential part|^2 + slack |z|^2
        s.t. L^T f + z = Q,  bounds,  rows.

    The solution without active rows is closed form; otherwise the quadratic programme is
    solved by the dual active-set method DAQP in variables scaled by the effort bounds
    (efforts) and by ``1 / sqrt(slack)`` (slack), and its status is reported (``counts``
    accumulates the statuses of all calls).  The slack makes the programme feasible whenever
    bounds and rows are; a positive slack means that the commanded force is not realised.
    When the solver reports no solution the call is marked ``failed`` and the previous
    efforts are held.
    """

    def __init__(self, comp, cones=None, weight=None, slack=1e4, link_weight=0., support_weight=0., transfer_load=None):
        import daqp
        self.daqp = daqp
        self.comp = comp
        s = comp.sys
        self.na, self.d = s.na, comp.d
        self.lower, self.upper = s.effort_lower.copy(), s.effort_upper.copy()
        self.weight = None if weight is None else np.asarray(weight, float)
        self.slack = slack
        self.link_weight = float(link_weight)
        # weight of the tangential forces and moments of the supports (per N^2, moments over 0.1 m)
        self.support_weight = float(support_weight)
        # normal force (N) at which a capped sole gets its full friction and centre-of-pressure limits
        self.transfer_load = transfer_load
        self.cones = cones or {}
        # The wrench of a level-2 module follows from the rows of its own dependent
        # coordinates only if no other module takes one of them as an input, i.e. if all
        # level-2 modules have independent inputs (one layer).
        if any(g.layer != 0 for g in comp.groups2):
            raise NotImplementedError('constraint wrenches of chained level-2 modules')
        self.supports, self.link_modules = [], []
        for g in comp.groups2:
            for row, mid in enumerate(g.module_ids):
                m = comp.modules2[mid]
                if len(m['chords']) == 1:
                    (self.supports if m['chords'][0] < comp.ns else self.link_modules).append((g, row, m['chords'][0]))
        covered = {comp.c_ids[e] for _, _, e in self.supports}
        missing = [c for c in self.cones if c in comp.c_ids[:comp.ns] and c not in covered]
        if missing:
            raise NotImplementedError('support cones need single-chord support modules: ' + ', '.join(missing))
        self.counts = {}

    @property
    def W(self):
        if self.weight is not None:
            return self.weight
        return 1. / np.maximum(np.maximum(abs(self.lower), abs(self.upper)), 1e-9) ** 2

    def set_bounds(self, lower, upper):
        """Change the effort bounds (an actuator with zero bounds is taken out of the allocation)."""
        self.lower, self.upper = np.array(lower, float), np.array(upper, float)

    def _maps(self, ev, wdot, modules):
        comp, s = self.comp, self.comp.sys
        D1 = comp.D1
        r0 = np.zeros(D1)
        LT = np.zeros((D1, s.na))
        for k in range(s.K):
            n1 = comp.d1[k]
            sl = slice(comp.off1[k], comp.off1[k + 1])
            w = ev.w[sl] if ev.w is not None else np.zeros(n1)
            r0[sl] = ev.Gamma1[k][:n1, :n1] @ wdot[sl] + ev.g1[k][:n1] + ev.D1[k][:n1, :n1] @ w
            if ev.C1 is not None:
                r0[sl] += ev.C1[k][:n1, :n1] @ w
            LT[sl, s.member_ports[k]] = ev.L1[k][:, :n1].T
        out = {}
        for g, row, e in modules:
            dep = g.dep[row]
            C = ev.Cbar[e][:, g.cols[row]][g.rows[row]]
            RT = C[:, :g.ndep].T
            out[e] = (np.linalg.solve(RT, r0[dep]), np.linalg.solve(RT, LT[dep]))
        return out

    def support_maps(self, ev, wdot):
        """Affine maps ``lambda_e = lam0_e - Lam_e f`` of every single-chord support module.

        ``wdot`` is the level-1 acceleration ``G udot + Gdot u`` used for the inertial part.
        """
        return self._maps(ev, wdot, self.supports)

    def link_maps(self, ev, wdot):
        """The same affine maps for the hand links (wrench in the link's minus frame)."""
        return self._maps(ev, wdot, self.link_modules)

    def _weights(self, live):
        """Effort weights; a drive that is out of the allocation gets a neutral weight (its effort is fixed at zero)."""
        return np.where(live, self.W, 1.)

    def _objective(self, ev, wdot, maps, W):
        """Quadratic and linear parts of the objective beyond ``f^T W f`` (``None`` when absent)."""
        use_links = self.link_weight > 0. and self.link_modules
        use_supports = self.support_weight > 0. and maps
        if not (use_links or use_supports):
            return None, None
        H = np.diag(W)
        q = np.zeros(self.na)
        if use_links:
            lm = self.link_maps(ev, wdot)
            Lam = np.concatenate([lm[e][1][3:] for e in lm])          # force rows of every link
            lam0 = np.concatenate([lm[e][0][3:] for e in lm])
            H += self.link_weight * Lam.T @ Lam
            q -= self.link_weight * Lam.T @ lam0
        if use_supports:
            sel = np.diag([10., 10., 10., 1., 1.]) @ np.eye(6)[:5]      # moments over 0.1 m, tangential forces
            Lam = np.concatenate([sel @ maps[e][1] for e in maps])
            lam0 = np.concatenate([sel @ maps[e][0] for e in maps])
            H += self.support_weight * Lam.T @ Lam
            q -= self.support_weight * Lam.T @ lam0
        return H, q

    def _closed_form(self, L, Q, live, H, q, W):
        """Minimiser of the objective subject to ``L^T f = Q`` over the live drives (no bounds, no rows)."""
        f = np.zeros(self.na)
        Ll = L[live]
        if H is None:
            Wi = 1. / W[live]
            LW = Ll * Wi[:, None]
            A = Ll.T @ LW                                # weighted minimum-norm solution
            try:
                y = np.linalg.solve(A, Q) if live.all() else np.linalg.lstsq(A, Q, rcond=1e-10)[0]
            except np.linalg.LinAlgError:                # directions without actuation
                y = np.linalg.lstsq(A, Q, rcond=1e-10)[0]
            f[live] = LW @ y
        else:
            n = int(live.sum())
            K = np.block([[H[np.ix_(live, live)], Ll], [Ll.T, np.zeros((self.d, self.d))]])
            f[live] = np.linalg.lstsq(K, np.r_[-q[live], Q], rcond=None)[0][:n]
        return f

    def cone_rows(self, maps, fz_max=None):
        """Cone rows of all supports in the efforts: list of ``(A, lower, upper)`` with ``lower <= A f <= upper``."""
        comp = self.comp
        fz_max = fz_max or {}
        rows = []
        for e, (lam0e, Lame) in maps.items():
            cone = self.cones.get(comp.c_ids[e])
            if cone is None:
                continue
            cap = fz_max.get(comp.c_ids[e], np.inf)
            tight = 1.
            if self.transfer_load and np.isfinite(cap):
                tight = float(np.clip(cap / self.transfer_load, 0.1, 1.))
            A, lo, hi = cone.rows(cap, tight)
            rows.append((A @ (-Lame), lo - A @ lam0e, hi - A @ lam0e))
        return rows

    def solve(self, ev, Q, wdot, fz_max=None, f_prev=None, extra_rows=None):
        """Efforts ``f`` with ``L^T f = Q`` (slack-penalised), bounds, support cones and extra rows.

        ``extra_rows`` is a list of ``(A, lower, upper)`` with ``lower <= A f <= upper``.
        Returns ``f`` and a report: whether the QP ran, its status, the largest slack and the
        level-2 support wrenches of the solution.  The wrench maps use the acceleration
        ``wdot`` of the commanded ``Q``; with a non-zero slack the executed motion differs.
        """
        comp = self.comp
        L = ev.L
        live = (self.upper - self.lower) > 0.
        W = self._weights(live)
        maps = self.support_maps(ev, wdot)
        H, q = self._objective(ev, wdot, maps, W)
        f = self._closed_form(L, Q, live, H, q, W)
        rows = self.cone_rows(maps, fz_max) + list(extra_rows or [])
        ok = bool(np.all(f >= self.lower - 1e-9) and np.all(f <= self.upper + 1e-9)
                  and np.max(abs(L.T @ f - Q)) <= 1e-8 * max(1., np.max(abs(Q))))
        if ok:
            for Af, lo, hi in rows:
                val = Af @ f
                if np.any(val < lo - 1e-7) or np.any(val > hi + 1e-7):
                    ok = False
                    break
        info = dict(qp=False, slack=0., status='closed_form', failed=False)
        if not ok:
            f, info = self._qp(L, Q, rows, f if f_prev is None else f_prev, H, q, W)
        self.counts[info['status']] = self.counts.get(info['status'], 0) + 1
        info['wrenches'] = {comp.c_ids[e]: lam0e - Lame @ f for e, (lam0e, Lame) in maps.items()}
        return f, info

    def _qp(self, L, Q, rows, warm, H, q, W):
        """Slack-penalised allocation by DAQP in scaled variables ``f = S fs`` and ``z = z_s / sqrt(slack)``."""
        na, d = self.na, self.d
        S = np.maximum(np.maximum(abs(self.lower), abs(self.upper)), 1e-9)
        sig = 1. / np.sqrt(self.slack)
        Hf = np.diag(W) if H is None else H
        n = na + d
        Hx = np.zeros((n, n))
        Hx[:na, :na] = S[:, None] * Hf * S[None, :]
        Hx[na:, na:] = np.eye(d)
        gx = np.zeros(n)
        if q is not None:
            gx[:na] = S * q
        nc = sum(len(r[1]) for r in rows)
        A = np.zeros((d + nc, n))
        A[:d, :na] = L.T * S[None, :]
        A[:d, na:] = sig * np.eye(d)
        big = 1e30
        lo = np.empty(n + d + nc); hi = np.empty(n + d + nc)
        lo[:na], hi[:na] = self.lower / S, self.upper / S              # simple bounds of the scaled efforts
        lo[na:n], hi[na:n] = -big, big                                 # the slack is free
        lo[n:n + d] = hi[n:n + d] = Q                                  # L^T f + z = Q
        k = d
        for Af, l, h in rows:
            m = len(l)
            A[k:k + m, :na] = Af * S[None, :]
            lo[n + k:n + k + m] = np.where(np.isfinite(l), l, -big)
            hi[n + k:n + k + m] = np.where(np.isfinite(h), h, big)
            k += m
        sense = np.zeros(n + d + nc, dtype=np.int32)
        sense[n:n + d] = 5                                             # equality rows
        try:
            x, _, exitflag, _ = self.daqp.solve(Hx, gx, A, hi, lo, sense)
            status = DAQP_STATUS.get(int(exitflag), f'daqp exit flag {int(exitflag)}')
        except Exception as exc:
            x, status = None, f'error: {type(exc).__name__}'
        if x is None or status != 'solved' or not np.all(np.isfinite(x)):
            # no solution: keep the efforts of the warm start (the previous efforts when the caller passes them,
            # otherwise the unconstrained solution), clipped to the bounds, and say so
            return np.clip(warm, self.lower, self.upper), dict(qp=True, slack=float('nan'), status=status, failed=True)
        x = np.asarray(x, float)
        f = np.clip(x[:na] * S, self.lower, self.upper)
        return f, dict(qp=True, slack=float(np.max(abs(sig * x[na:]))), status=status, failed=False)
