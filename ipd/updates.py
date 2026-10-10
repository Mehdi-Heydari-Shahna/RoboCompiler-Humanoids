"""Local updates, connection events and constraint wrenches of the compiled model.

*Inertial edits.*  When the spatial inertia of one body changes by ``dM`` (payload,
tool, component variant) only that body's member of the partition changes:

    Gamma' = Gamma + J_i^T dM J_i,      rank <= 6,
    C'     = C + J_i^T (dM Jdot_i - Ptilde(dM V_i) J_i),
    g'     = g - J_i^T dM [0; R_i^T g].

The lift, the body Jacobians and every other body's contribution are untouched,
and ``Gamma'^-1`` follows from ``Gamma^-1`` by a 6x6 solve (Woodbury).

*Connection events.*  Closing additional chords maps the level-1 speeds onto the
range of the new composer lift ``G``.  The inelastic transition is the
``Gamma1``-orthogonal projection; the kinetic energy it removes is a sum over
bodies.  Opening chords leaves the velocity continuous.

*Constraint wrenches.*  The equations of the dependent coordinates of a module give
its constraint wrench without the wrench ever entering the equations of motion.
"""
from __future__ import annotations

import numpy as np

from . import se3


def point_mass_inertia(mass, position, inertia_com=None):
    """Spatial inertia (about the body-frame origin) of a payload fixed at ``position``."""
    return se3.spatial_inertia(np.asarray(mass, float), position, np.zeros((3, 3)) if inertia_com is None else inertia_com)


def body_jacobian(comp, ev, body, rate=False):
    """Compiled body Jacobian ``J_i`` (6, d) of a physical body in the mode's independent speeds."""
    s = comp.sys
    k = s.member_of_body[body]
    n1 = comp.d1[k]
    sl = slice(comp.off1[k], comp.off1[k + 1])
    J = ev.J1[body][:, :n1] @ ev.G[sl]
    if not rate:
        return J
    return J, ev.J1d[body][:, :n1] @ ev.G[sl] + ev.J1[body][:, :n1] @ ev.Gd[sl]


def inertial_update(comp, ev, body, dM):
    """Change of (Gamma, C, g) for a change ``dM`` of the spatial inertia of one body.

    Returns ``dGamma``, ``dC`` (``None`` without velocities), ``dg`` and the body Jacobian.
    """
    s = comp.sys
    grav = np.r_[0., 0., 0., ev.kin.T[body, :3, :3].T @ s.gravity]
    if ev.u is None:
        J = body_jacobian(comp, ev, body)
        return J.T @ dM @ J, None, -J.T @ (dM @ grav), J
    J, Jd = body_jacobian(comp, ev, body, True)
    P = dM @ ev.V[body]
    dC = J.T @ (dM @ Jd - se3.momentum_matrix(P) @ J)
    return J.T @ dM @ J, dC, -J.T @ (dM @ grav), J


def woodbury_solve(Gamma_solve, J, dM, rhs):
    """Solve ``(Gamma + J^T dM J) x = rhs`` from a solver of ``Gamma`` and a small symmetric solve.

    ``Gamma_solve(B)`` returns ``Gamma^-1 B``.  ``dM`` may be indefinite or singular;
    its range is used through an eigen-decomposition, so the inner system has the
    rank of ``dM`` (at most six).
    """
    w, U = np.linalg.eigh((dM + dM.T) / 2)
    keep = abs(w) > 1e-13 * max(1., np.max(abs(w)))
    if not np.any(keep):
        return Gamma_solve(rhs)
    U, w = U[:, keep], w[keep]
    A = J.T @ U                                   # (d, r)
    Y = Gamma_solve(np.c_[rhs, A])
    y, Z = Y[:, 0], Y[:, 1:]
    inner = np.diag(1. / w) + A.T @ Z
    return y - Z @ np.linalg.solve(inner, A.T @ y)


def level1_speeds(comp, ev, v):
    """Level-1 speeds ``w`` (root twists, requested joints, link cut rates) of a full velocity ``v``."""
    s = comp.sys
    w = np.zeros(comp.D1)
    for k in range(s.K):
        w[comp.off1[k]:comp.off1[k + 1]] = v[comp.level1_dofs[k]]
    w[comp.off1[-1]:] = v[comp.chart_dofs]
    return w


def level1_inertia(comp, ev):
    """Block-diagonal level-1 inertia over the level-1 speeds (cut coordinates are massless)."""
    s = comp.sys
    G1 = np.zeros((comp.D1, comp.D1))
    for k in range(s.K):
        n1 = comp.d1[k]
        sl = slice(comp.off1[k], comp.off1[k + 1])
        G1[sl, sl] = ev.Gamma1[k][:n1, :n1]
    return G1


def closing_transition(comp, ev, w_minus):
    """Inelastic transition onto a mode with more closures.

    ``ev`` is an evaluation of the target mode at the event configuration and
    ``w_minus`` the level-1 speeds just before the event (cut-coordinate entries
    are irrelevant).  Returns the post-event independent speeds, the kinetic
    energy removed and its split over the members.
    """
    s = comp.sys
    G1 = level1_inertia(comp, ev)
    u_plus = np.linalg.solve(ev.Gamma, ev.G.T @ (G1 @ w_minus))
    dw = w_minus - ev.G @ u_plus
    loss = np.array([.5 * dw[comp.off1[k]:comp.off1[k + 1]] @ ev.Gamma1[k][:comp.d1[k], :comp.d1[k]]
                     @ dw[comp.off1[k]:comp.off1[k + 1]] for k in range(s.K)])
    return u_plus, float(loss.sum()), loss


def body_energy_loss(comp, ev, w_minus, u_plus):
    """Kinetic energy of the lost motion ``dw = w_minus - G u_plus``, body by body (rotors not included).

    The summands add up to the energy removed by the transition (Carnot); they are not the
    changes of the kinetic energy of the individual bodies, which can have either sign.
    """
    s = comp.sys
    dw = w_minus - ev.G @ u_plus
    out = np.zeros(s.nbp)
    for k in range(s.K):
        idx = s.member_bodies[k]
        dV = np.einsum('bij,j->bi', ev.J1[idx][:, :, :comp.d1[k]], dw[comp.off1[k]:comp.off1[k + 1]])
        out[idx] = .5 * np.einsum('bi,bij,bj->b', dV, s.M[idx], dV)
    return out


def constraint_wrenches(comp, ev, f, udot, tau_ext1=None):
    """Wrenches of the level-2 chords from the equations of their dependent coordinates.

    ``lambda_m = R_m^-T [Gamma1 wdot + C1 w + g1 + D1 w - L1^T f - tau_ext1]_dep(m)`` for
    every module whose inputs are independent.  Returns {chord id: 6-vector in the
    chord's minus frame, ``[moment; force]`` acting on the plus body}.
    """
    s = comp.sys
    wdot = ev.G @ udot + (ev.Gd @ ev.u if ev.Gd is not None else 0.)
    r = np.zeros(comp.D1)
    for k in range(s.K):
        n1 = comp.d1[k]
        sl = slice(comp.off1[k], comp.off1[k + 1])
        w = ev.w[sl] if ev.w is not None else np.zeros(n1)
        r[sl] = ev.Gamma1[k][:n1, :n1] @ wdot[sl] + ev.g1[k][:n1] + ev.D1[k][:n1, :n1] @ w - ev.L1[k][:, :n1].T @ f[s.member_ports[k]]
        if ev.C1 is not None:
            r[sl] += ev.C1[k][:n1, :n1] @ w
    if tau_ext1 is not None:
        r -= tau_ext1
    out = {}
    for g in comp.groups2:
        if g.layer != 0:
            raise NotImplementedError('chained level-2 modules')
        for row, mid in enumerate(g.module_ids):
            m = comp.modules2[mid]
            C = np.concatenate([ev.Cbar[e] for e in m['chords']])[:, g.cols[row]][g.rows[row]]
            lam = np.linalg.solve(C[:, :g.ndep].T, r[g.dep[row]])
            full = np.zeros(6 * len(m['chords']))
            full[g.rows[row]] = lam
            for i, e in enumerate(m['chords']):
                out[comp.c_ids[e]] = full[6 * i:6 * i + 6]
    return out
