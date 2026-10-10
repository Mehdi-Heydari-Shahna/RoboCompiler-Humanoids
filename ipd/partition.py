"""Inertia-partitioned minimal-coordinate dynamics on the compiled lift.

For independent speeds ``u`` with ``v = E(q) u`` every rigid body ``i`` has the body
twist ``V_i = J_i u`` with the compiled body Jacobian ``J_i = Ad(inv(T_i)) S_i E``,
where ``S_i`` holds the world screws of the coordinates that move the body.  With
the constant body-frame spatial inertia ``M_i`` the body contributes

    Gamma_i = J_i^T M_i J_i,
    C_i     = J_i^T (M_i Jdot_i - Ptilde(M_i V_i) J_i),
    g_i     = -J_i^T M_i [0; R_i^T g],

and the equations of motion in the independent speeds are

    Gamma udot + C u + g = L^T f + tau_ext,   Gamma = sum_i Gamma_i, ...

``Ptilde`` is skew-symmetric, so ``Gammadot_i - 2 C_i`` is skew-symmetric body by
body.  The speeds need not be derivatives of coordinates: a floating root enters
through its body twist.  Rotor (reflected) inertias are scalar members of the
partition acting on single tree coordinates.
"""
from __future__ import annotations

import numpy as np

from . import se3


class Terms:
    """Assembled terms; per-body arrays are present when requested."""
    __slots__ = ('Gamma', 'C', 'g', 'J', 'Jdot', 'V', 'Gamma_i', 'C_i', 'g_i', 'Gamma_rotor', 'C_rotor',
                 'kinetic_i', 'potential_i')

    def __init__(self):
        for k in self.__slots__:
            setattr(self, k, None)


def body_jacobians(system, kin, E, Edot=None):
    """Compiled body Jacobians ``J_i`` (nbp, 6, d) and, with ``Edot``, their rates."""
    nbp = system.nbp
    AdI = se3.Ad_inv(kin.T[:nbp])
    Js = system.accumulate(kin.S, E)
    J = AdI @ Js
    if Edot is None:
        return J, None
    Jsd = system.accumulate(kin.Sdot, E) + system.accumulate(kin.S, Edot)
    Jd = AdI @ (Jsd - se3.ad(kin.Vs[:nbp]) @ Js)
    return J, Jd


def assemble(system, kin, E, Edot=None, per_body=False, M=None):
    """Inertia-partitioned terms for the speeds that parametrize the columns of ``E``.

    ``kin`` must carry the velocity ``v = E u`` when ``Edot`` is given.  ``M`` can
    override the body spatial inertias (nbp, 6, 6).
    """
    nbp, nT = system.nbp, system.nT
    M = system.M if M is None else M
    d = E.shape[1]
    out = Terms()
    J, Jd = body_jacobians(system, kin, E, Edot)
    out.J, out.Jdot = J, Jd
    MJ = M @ J
    Jf = J.reshape(-1, d)
    ET = E[:nT]
    arm = system.armature
    out.Gamma = Jf.T @ MJ.reshape(-1, d) + ET.T @ (arm[:, None] * ET)
    grav = np.zeros((nbp, 6))
    grav[:, 3:] = np.einsum('bji,j->bi', kin.T[:nbp, :3, :3], system.gravity)
    Mg = np.einsum('bij,bj->bi', M, grav)
    out.g = -Jf.T @ Mg.reshape(-1)
    if Jd is not None:
        V = np.einsum('bij,bj->bi', se3.Ad_inv(kin.T[:nbp]), kin.Vs[:nbp])
        out.V = V
        P = np.einsum('bij,bj->bi', M, V)
        W = M @ Jd - se3.momentum_matrix(P) @ J
        out.C = Jf.T @ W.reshape(-1, d) + ET.T @ (arm[:, None] * Edot[:nT])
    if per_body:
        out.Gamma_i = np.einsum('bia,bic->bac', J, MJ)
        out.g_i = -np.einsum('bia,bi->ba', J, Mg)
        out.Gamma_rotor = np.einsum('ja,jc->jac', ET * arm[:, None], ET)
        if Jd is not None:
            out.C_i = np.einsum('bia,bic->bac', J, W)
            out.C_rotor = np.einsum('ja,jc->jac', ET * arm[:, None], Edot[:nT])
    return out


def energies(system, kin, per_body=False):
    """Kinetic and gravitational potential energy from the body twists of ``kin``."""
    nbp = system.nbp
    V = np.einsum('bij,bj->bi', se3.Ad_inv(kin.T[:nbp]), kin.Vs[:nbp])
    Tb = .5 * np.einsum('bi,bij,bj->b', V, system.M, V)
    rotor = .5 * system.armature * kin.v[:system.nT] ** 2
    pc = kin.T[:nbp, :3, 3] + np.einsum('bij,bj->bi', kin.T[:nbp, :3, :3], system.com)
    Ub = -system.mass * (pc @ system.gravity)
    if per_body:
        return Tb, rotor, Ub
    return float(Tb.sum() + rotor.sum()), float(Ub.sum())


def tree_terms(system, kin):
    """Tree-coordinate mass matrix, Coriolis matrix and gravity (no closure), by body sums.

    This is the special case ``E = [I; 0]`` and is used to recover constraint forces
    and to compare with recursive algorithms.
    """
    E = np.zeros((system.n, system.nT))
    E[:system.nT] = np.eye(system.nT)
    Ed = np.zeros_like(E) if kin.v is not None else None
    return assemble(system, kin, E, Ed)


def regressor(system, kin, E, Edot, a, ur):
    """Body-wise regressor ``Y`` with ``Gamma a + C ur + g = sum_i Y_i phi_i`` (rigid bodies).

    ``phi_i = (m, m c, I_xx, I_xy, I_xz, I_yy, I_yz, I_zz)`` with the rotational inertia
    taken about the body-frame origin.  Returns ``Y`` of shape (nbp, d, 10).
    """
    nbp = system.nbp
    J, Jd = body_jacobians(system, kin, E, Edot)
    V = np.einsum('bij,bj->bi', se3.Ad_inv(kin.T[:nbp]), kin.Vs[:nbp])
    acc = np.einsum('bij,j->bi', J, a) + np.einsum('bij,j->bi', Jd, ur)
    acc[:, 3:] -= np.einsum('bji,j->bi', kin.T[:nbp, :3, :3], system.gravity)
    Wr = np.einsum('bij,j->bi', J, ur)
    Y = np.empty((nbp, E.shape[1], 10))
    basis = inertia_basis()
    for k in range(10):
        Mk = basis[k]
        F = np.einsum('ij,bj->bi', Mk, acc) + np.einsum('bji,bj->bi', se3.ad(Wr), -np.einsum('ij,bj->bi', Mk, V))
        Y[:, :, k] = np.einsum('bia,bi->ba', J, F)
    return Y


def inertia_basis():
    """Basis ``M = sum_k phi_k B_k`` of spatial inertias in the parameters ``phi``."""
    B = np.zeros((10, 6, 6))
    B[0, 3:, 3:] = np.eye(3)
    for k in range(3):
        e = np.zeros(3); e[k] = 1.
        B[1 + k, :3, 3:] = se3.skew(e)
        B[1 + k, 3:, :3] = se3.skew(e).T
    for k, (i, j) in enumerate([(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]):
        B[4 + k, i, j] = 1.
        B[4 + k, j, i] = 1.
    return B


def inertia_parameters(system):
    """Parameter vectors ``phi_i`` (nbp, 10) of the physical bodies."""
    m, c = system.mass, system.com
    Io = system.M[:, :3, :3]
    return np.c_[m, m[:, None] * c, Io[:, 0, 0], Io[:, 0, 1], Io[:, 0, 2], Io[:, 1, 1], Io[:, 1, 2], Io[:, 2, 2]]
