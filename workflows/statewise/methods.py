"""Constrained forward dynamics of one state by several routes.

Every function returns the tree acceleration ``a`` (core velocity order: a floating
root contributes the rate of its body twist) for the same configuration ``q``,
independent speeds ``u`` and actuator efforts ``f``.

    ip_two_level        proposed: inertia-partitioned model, member modules and composer
    ip_one_level        the same partition from a single lift over all chords
    projected_analytic  RoboCompiler Eq. (14) with Pinocchio CRBA/RNEA and the analytic curvature
    projected_fd        the same with the curvature from the paper's centred directional
                        difference of the closure Jacobian (its Listing 4)
    pinocchio_kkt       Pinocchio constraintDynamics (proximal KKT) on rigid constraint models
    mujoco              MuJoCo forward dynamics with (soft) equality constraints

Ablations of the proposed model:

    no_curvature        the rate of the lift is dropped
    simplified          inertial terms of the moving actuator bodies and rotors are neglected,
                        their weight is kept
    massless            the actuator bodies and rotors are removed altogether
    decentralized       inter-member links are not modelled

``maximal`` (see ``maximal.py``) is a reference that does not pass through the compiler.
"""
from __future__ import annotations

import numpy as np

from ipd import partition as ip
from ipd import reference as rf
from ipd import se3


# ---------------------------------------------------------------------- proposed model
def ip_two_level(case, q, u, f, ev=None):
    comp, s = case.comp, case.sys
    ev = comp.evaluate(q, u) if ev is None else ev
    ud = np.linalg.solve(ev.Gamma, ev.L.T @ f - ev.C @ u - ev.g - ev.D @ u)
    E, Ed = comp.lift(ev, True)
    return E[:s.nT] @ ud + Ed[:s.nT] @ u, dict(ev=ev, E=E, Ed=Ed, ud=ud)


def one_level(case, q, u):
    """Lift, its rate and the assembled terms of the one-level reduction over all chords."""
    comp, s = case.comp, case.sys
    kin = s.kinematics(q)
    E = comp.mode.lift(kin)
    s.velocity(kin, E @ u)
    E, Ed = comp.mode.lift(kin, derivative=True)
    return kin, E, Ed


def ip_one_level(case, q, u, f, cache=None):
    s = case.sys
    kin, E, Ed = one_level(case, q, u) if cache is None else cache
    t = ip.assemble(s, kin, E, Ed)
    ET = E[:s.nT]
    D = ET.T @ (s.damping[:, None] * ET)
    ud = np.linalg.solve(t.Gamma, ET.T @ (s.B @ f) - t.C @ u - t.g - D @ u)
    return ET @ ud + Ed[:s.nT] @ u, dict(terms=t, ud=ud)


# ---------------------------------------------------------------------- projected tree model of the RoboCompiler paper
def curvature_fd(mode, q, v, h0=1e-5):
    """``c = Edot u`` by the centred directional difference of the closure Jacobian (paper, Listing 4)."""
    s = mode.sys

    def C(qq):
        return mode.jacobian(s.kinematics(qq))[1]

    h = h0 / max(1., float(np.max(abs(v))))
    dC = (C(s.integrate(q, h * v)) - C(s.integrate(q, -h * v))) / (2 * h)
    rows = np.concatenate([[6 * m['chords'][r // 6] + r % 6 for r in m['rows']] for m in mode.modules]) \
        if mode.modules else np.zeros(0, int)
    p = np.concatenate([m['dep'] for m in mode.modules]).astype(int) if mode.modules else np.zeros(0, int)
    c = np.zeros(s.n)
    if len(p):
        c[p] = -np.linalg.solve(C(q)[np.ix_(rows, p)], dC[rows] @ v)
    return c


def projected_tree(case, q, u, f, pb, cache=None, curvature='analytic', h0=1e-5):
    """``M_r = E_T^T M_T E_T`` and ``h_r = E_T^T (h_T + M_T c_T)`` with recursive tree algorithms."""
    s = case.sys
    kin, E, Ed = one_level(case, q, u) if cache is None else cache
    ET = E[:s.nT]
    v = E @ u
    if curvature == 'analytic':
        c = Ed @ u
    elif curvature == 'fd':
        c = curvature_fd(case.comp.mode, q, v, h0)
    else:
        c = np.zeros(s.n)
    cT = c[:s.nT]
    MT = pb.mass(q)
    hT = pb.bias(q, v[:s.nT]) + s.damping * v[:s.nT]
    ud = np.linalg.solve(ET.T @ MT @ ET, ET.T @ (s.B @ f - hT - MT @ cT))
    return ET @ ud + cT, dict(ud=ud)


# ---------------------------------------------------------------------- independent engines
def pinocchio_kkt(case, q, u, f, pb, solver, vT):
    s = case.sys
    return pb.solve(solver, q, vT, s.B @ f - s.damping * vT), {}


class MujocoStatewise:
    """MuJoCo model of a case with ideal supports as welds, contact and Coulomb friction disabled.

    MuJoCo's equality constraints are soft: the constraint force is regularised by the
    impedance ``d`` (at most 0.9999), with a deviation from the rigid solution that is
    proportional to ``(1 - d) / d``.  ``rigid_limit`` extrapolates that deviation to zero.
    Releases before 3.7.0 also leave the velocity-product term ``Jdot v`` out of the
    constraint bias of connect and weld equalities (MuJoCo changelog, 3.7.0).
    """

    def __init__(self, case, loop_solref=0.004):
        import mujoco
        from ipd.backends.mujoco_backend import MujocoPlant
        comp, s = case.comp, case.sys
        welds = [(sid.split('/')[0], sid.split('/')[1], comp.mode.anchors[i]) for i, sid in enumerate(comp.mode.support_ids)]
        self.plant = MujocoPlant(s, q=case.q0, contact=False, links=list(comp.links), support_welds=welds,
                                 loop_solref=loop_solref)
        self.plant.model.dof_frictionloss[:] = 0.        # Coulomb friction is not part of the smooth dynamics compared here
        self.case = case
        self.version = mujoco.__version__

    def acceleration(self, q, vT, f, impedance=0.9999):
        p = self.plant
        p.model.eq_solimp[:, 0] = impedance
        p.model.eq_solimp[:, 1] = impedance
        p.set_effort(f)
        p.set_state(q, vT)
        return p.acceleration(q)

    def __call__(self, q, vT, f):
        return self.acceleration(q, vT, f), {}

    def rigid_limit(self, q, vT, f, impedances=(0.9996, 0.9998, 0.9999)):
        """Polynomial (Richardson) extrapolation of the acceleration to zero regularisation."""
        eps = np.array([(1. - d) / d for d in impedances])
        acc = np.array([self.acceleration(q, vT, f, d) for d in impedances])
        w = np.array([np.prod([-eps[j] / (eps[i] - eps[j]) for j in range(len(eps)) if j != i]) for i in range(len(eps))])
        self.acceleration(q, vT, f)                      # leave the model at its default impedance
        return w @ acc, {}


def kkt_without_velocity_product_mujoco(case, ref, vT, dps=50):
    """Hard-constraint acceleration when ``Jdot v`` is left out in MuJoCo's velocity coordinates.

    MuJoCo's free joint uses the world-frame linear velocity of the root.  With
    ``v_core = X(q) v_mj`` the condition ``J_mj a_mj = 0`` reads ``J (a - Xdot v_mj) = 0``
    in the core coordinates, and ``Xdot v_mj`` is ``-w x v`` in the linear root slots.
    """
    s = case.sys
    shift = np.zeros(s.nT)
    for sl in s.root_slice:
        if sl is not None:
            shift[sl.start + 3:sl.stop] = -np.cross(vT[sl][:3], vT[sl][3:])
    a, *_ = rf.kkt_reference(ref['M'], ref['h'], ref['J'], -ref['J'] @ shift, dps=dps)
    return a


# ---------------------------------------------------------------------- ablations
def no_curvature(case, q, u, f, cache=None):
    s = case.sys
    kin, E, Ed = one_level(case, q, u) if cache is None else cache
    t = ip.assemble(s, kin, E, np.zeros_like(E))
    ET = E[:s.nT]
    D = ET.T @ (s.damping[:, None] * ET)
    ud = np.linalg.solve(t.Gamma, ET.T @ (s.B @ f) - t.C @ u - t.g - D @ u)
    return ET @ ud, {}


def simplified(case, q, u, f, M, armature, gravity_full=True):
    """Actuator bodies and rotors left out of the inertial terms.

    With ``gravity_full`` their weight stays in the gravity vector (only ``Gamma`` and
    ``C`` are simplified); without it the bodies are removed altogether.
    """
    comp, s = case.comp, case.sys
    M0, a0 = s.M, s.armature
    try:
        s.M, s.armature = M, armature
        ev = comp.evaluate(q, u)
    finally:
        s.M, s.armature = M0, a0
    if gravity_full:
        ev.g1, ev.g = comp.gravity(ev)
    return ip_two_level(case, q, u, f, ev)


def decentralized(case, model, q, v, f):
    """Every member on its own compiled model; ``model`` is the mode without inter-member links."""
    s = case.sys
    u = v[model.mode.independent]
    ev = model.evaluate(q, u)
    ud = np.linalg.solve(ev.Gamma, ev.L.T @ f - ev.C @ u - ev.g - ev.D @ u)
    E, Ed = model.lift(ev, True)
    return E[:s.nT] @ ud + Ed[:s.nT] @ u, {}


# ---------------------------------------------------------------------- reference and metrics
def reference(case, q, vT, f, pb, dps=50):
    """Extended-precision hard-constraint reference on Pinocchio's tree terms and physical rows."""
    s = case.sys
    MT = pb.mass(q)
    bias = pb.bias(q, vT) + s.damping * vT
    tau = s.B @ f
    hT = bias - tau
    J, gamma, layout = pb.physical_rows(case.comp.mode, q, vT)
    if J.shape[0]:
        a, lam, rows, r, radius = rf.kkt_reference(MT, hT, J, gamma, dps=dps)
    else:
        a, lam, rows, r, radius = rf.kkt_reference(MT, hT, np.zeros((0, s.nT)), np.zeros(0), dps=dps, rows=np.zeros(0, int))
    return dict(M=MT, h=hT, bias=bias, tau=tau, J=J, gamma=gamma, a=a, rank=int(r), rows=int(J.shape[0]),
                radius=float(radius), layout=layout,
                velocity_residual=float(np.max(abs(J @ vT))) if J.shape[0] else 0.)


def discrepancy(ref, a):
    return rf.discrepancy(ref['M'], a, ref['a'], ref['bias'], ref['tau'], ref['J'] if ref['J'].shape[0] else None,
                          ref['gamma'] if ref['J'].shape[0] else None)


# ---------------------------------------------------------------------- structural identities of the partition
def gamma_rate_fd(case, q, u, h0=1e-5):
    """Directional derivative of ``Gamma`` along the motion ``v = E u`` by centred differences."""
    comp, s = case.comp, case.sys
    ev = comp.evaluate(q)
    v = comp.lift(ev) @ u
    h = h0 / max(1., float(np.max(abs(v))))
    Gp = comp.evaluate(s.integrate(q, h * v)).Gamma
    Gm = comp.evaluate(s.integrate(q, -h * v)).Gamma
    return (Gp - Gm) / (2 * h)


def potential(system, q):
    T = system.poses(q)[:system.nbp]
    pc = T[:, :3, 3] + np.einsum('bij,bj->bi', T[:, :3, :3], system.com)
    return float(-system.mass @ (pc @ system.gravity))


def potential_rate_fd(case, q, u, h0=1e-5):
    comp, s = case.comp, case.sys
    v = comp.lift(comp.evaluate(q)) @ u
    h = h0 / max(1., float(np.max(abs(v))))
    return (potential(s, s.integrate(q, h * v)) - potential(s, s.integrate(q, -h * v))) / (2 * h)


def lagrangian_bias_fd(case, q, u, h0=1e-5):
    """Velocity-product force of the holonomic Lagrangian formula used by the reference paper,

        h = Gammadot u - (1/2) grad_pi (u^T Gamma u),

    with the gradient taken along the independent speed directions (the columns of the
    lift).  It equals ``C u`` when the independent speeds are derivatives of
    coordinates and differs from it when they are quasi-velocities.
    """
    comp, s = case.comp, case.sys
    ev = comp.evaluate(q)
    E = comp.lift(ev)
    h = h0
    grad = np.zeros(comp.d)
    for k in range(comp.d):
        Gp = comp.evaluate(s.integrate(q, h * E[:, k])).Gamma
        Gm = comp.evaluate(s.integrate(q, -h * E[:, k])).Gamma
        grad[k] = (u @ Gp @ u - u @ Gm @ u) / (2 * h)
    return gamma_rate_fd(case, q, u, h0) @ u - .5 * grad
