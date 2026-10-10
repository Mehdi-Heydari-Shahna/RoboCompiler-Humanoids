"""Batched SE(3) and spatial-vector algebra, angular-first.

Conventions (identical to the PACDM core in ``vendor/pacdm_original.py``):

* a transform ``T`` is a 4x4 homogeneous matrix ``[[R, p], [0, 1]]``;
* a twist is ``[w; v]`` (angular part first) and a wrench is ``[n; f]``;
* ``Ad(T) = [[R, 0], [skew(p) R, R]]`` maps twists from the child to the parent frame.

Every function accepts arrays with arbitrary leading batch dimensions.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

EYE3 = np.eye(3)
EYE4 = np.eye(4)
EYE6 = np.eye(6)


def skew(a):
    """Cross-product matrix of ``a`` with shape (..., 3) -> (..., 3, 3)."""
    a = np.asarray(a, float)
    out = np.zeros(a.shape[:-1] + (3, 3))
    out[..., 0, 1] = -a[..., 2]
    out[..., 0, 2] = a[..., 1]
    out[..., 1, 0] = a[..., 2]
    out[..., 1, 2] = -a[..., 0]
    out[..., 2, 0] = -a[..., 1]
    out[..., 2, 1] = a[..., 0]
    return out


def inv(T):
    """Inverse of homogeneous transforms (..., 4, 4)."""
    T = np.asarray(T, float)
    out = np.zeros_like(T)
    Rt = np.swapaxes(T[..., :3, :3], -1, -2)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -np.einsum('...ij,...j->...i', Rt, T[..., :3, 3])
    out[..., 3, 3] = 1.
    return out


def Ad(T):
    """Adjoint (..., 6, 6): twist in child coordinates -> twist in parent coordinates."""
    T = np.asarray(T, float)
    R = T[..., :3, :3]
    out = np.zeros(T.shape[:-2] + (6, 6))
    out[..., :3, :3] = R
    out[..., 3:, 3:] = R
    out[..., 3:, :3] = skew(T[..., :3, 3]) @ R
    return out


def Ad_inv(T):
    """Adjoint of the inverse transform, ``Ad(inv(T))``, without forming the inverse."""
    T = np.asarray(T, float)
    Rt = np.swapaxes(T[..., :3, :3], -1, -2)
    out = np.zeros(T.shape[:-2] + (6, 6))
    out[..., :3, :3] = Rt
    out[..., 3:, 3:] = Rt
    out[..., 3:, :3] = -Rt @ skew(T[..., :3, 3])
    return out


def ad(V):
    """Lie bracket matrix (..., 6, 6): ``ad(V) W = V x W`` for twists."""
    V = np.asarray(V, float)
    out = np.zeros(V.shape[:-1] + (6, 6))
    W = skew(V[..., :3])
    out[..., :3, :3] = W
    out[..., 3:, 3:] = W
    out[..., 3:, :3] = skew(V[..., 3:])
    return out


def cross3(a, b):
    """Cross product of (..., 3) arrays (faster than ``np.cross`` on small stacks)."""
    a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]
    b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
    out = np.empty(np.broadcast_shapes(a.shape, b.shape))
    out[..., 0] = a1 * b2 - a2 * b1
    out[..., 1] = a2 * b0 - a0 * b2
    out[..., 2] = a0 * b1 - a1 * b0
    return out


def cross_motion(V, W):
    """Twist cross product ``V x W = ad(V) W`` for (..., 6) arrays."""
    V = np.asarray(V, float)
    W = np.asarray(W, float)
    out = np.empty(np.broadcast_shapes(V.shape, W.shape))
    out[..., :3] = cross3(V[..., :3], W[..., :3])
    out[..., 3:] = cross3(V[..., 3:], W[..., :3]) + cross3(V[..., :3], W[..., 3:])
    return out


def cross_force(V, F):
    """Dual cross product ``V x* F = -ad(V)^T F`` for a twist ``V`` and a wrench ``F``."""
    V = np.asarray(V, float)
    F = np.asarray(F, float)
    out = np.empty(np.broadcast_shapes(V.shape, F.shape))
    out[..., :3] = cross3(V[..., :3], F[..., :3]) + cross3(V[..., 3:], F[..., 3:])
    out[..., 3:] = cross3(V[..., :3], F[..., 3:])
    return out


def momentum_matrix(P):
    """Skew matrix ``Ptilde(P)`` with ``Ptilde(P) W = ad(W)^T P`` (..., 6) -> (..., 6, 6).

    For a momentum ``P = [n; l]`` it is ``[[skew(n), skew(l)], [skew(l), 0]]``.  It is
    skew-symmetric, linear in ``P`` and gives ``V x* P = -Ptilde(P) V``.
    """
    P = np.asarray(P, float)
    out = np.zeros(P.shape[:-1] + (6, 6))
    L = skew(P[..., 3:])
    out[..., :3, :3] = skew(P[..., :3])
    out[..., :3, 3:] = L
    out[..., 3:, :3] = L
    return out


def exp_so3(w):
    """Rotation matrices exp(skew(w)) for (..., 3) rotation vectors."""
    w = np.asarray(w, float)
    t2 = np.sum(w * w, axis=-1)
    t = np.sqrt(t2)
    small = t < 1e-4
    ts = np.where(small, 1., t)
    a = np.where(small, 1 - t2 / 6 + t2 * t2 / 120, np.sin(ts) / ts)
    b = np.where(small, .5 - t2 / 24 + t2 * t2 / 720, (1 - np.cos(ts)) / (ts * ts))
    K = skew(w)
    return EYE3 + a[..., None, None] * K + b[..., None, None] * (K @ K)


def exp(x):
    """SE(3) exponential of twists (..., 6) -> (..., 4, 4)."""
    x = np.asarray(x, float)
    w, v = x[..., :3], x[..., 3:]
    t2 = np.sum(w * w, axis=-1)
    t = np.sqrt(t2)
    small = t < 1e-4
    ts = np.where(small, 1., t)
    a = np.where(small, 1 - t2 / 6 + t2 * t2 / 120, np.sin(ts) / ts)
    b = np.where(small, .5 - t2 / 24 + t2 * t2 / 720, (1 - np.cos(ts)) / (ts * ts))
    c = np.where(small, 1 / 6 - t2 / 120 + t2 * t2 / 5040, (ts - np.sin(ts)) / (ts ** 3))
    K = skew(w)
    K2 = K @ K
    T = np.zeros(x.shape[:-1] + (4, 4))
    T[..., :3, :3] = EYE3 + a[..., None, None] * K + b[..., None, None] * K2
    V = EYE3 + b[..., None, None] * K + c[..., None, None] * K2
    T[..., :3, 3] = np.einsum('...ij,...j->...i', V, v)
    T[..., 3, 3] = 1.
    return T


def log_so3(R):
    """Rotation vectors of (..., 3, 3) rotation matrices on the principal branch."""
    R = np.asarray(R, float)
    u = .5 * np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0],
                       R[..., 1, 0] - R[..., 0, 1]], axis=-1)
    s = np.linalg.norm(u, axis=-1)
    cth = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    theta = np.arctan2(s, cth)
    if np.any(np.pi - theta < 1e-6):
        raise ValueError('SO3 logarithm near pi branch')
    scale = np.where(s < 1e-14, 1., theta / np.where(s < 1e-14, 1., s))
    return u * scale[..., None]


def log(T):
    """SE(3) logarithm (..., 4, 4) -> (..., 6) on the principal branch."""
    T = np.asarray(T, float)
    w = log_so3(T[..., :3, :3])
    t2 = np.sum(w * w, axis=-1)
    theta = np.sqrt(t2)
    small = theta < 1e-4
    ts = np.where(small, 1., theta)
    beta = np.where(small, 1 / 12 + t2 / 720 + t2 * t2 / 30240,
                    (1 - ts / 2 / np.tan(ts / 2)) / (ts * ts))
    K = skew(w)
    Vinv = EYE3 - .5 * K + beta[..., None, None] * (K @ K)
    out = np.empty(T.shape[:-2] + (6,))
    out[..., :3] = w
    out[..., 3:] = np.einsum('...ij,...j->...i', Vinv, T[..., :3, 3])
    return out


def inv_left(x):
    """Inverse left Jacobian of SE(3) at ``x`` (..., 6) -> (..., 6, 6).

    ``d/dt log(D) = inv_left(log D) (Ddot D^-1)^vee``.  A Bernoulli series in
    ``ad(x)`` is used near the identity and the exact exponential formula elsewhere.
    """
    x = np.asarray(x, float)
    Z = ad(x)
    norm = np.linalg.norm(Z, axis=(-2, -1))
    Z2 = Z @ Z
    Z4 = Z2 @ Z2
    out = EYE6 - .5 * Z + Z2 / 12 - Z4 / 720 + (Z4 @ Z2) / 30240
    large = norm > 2e-2
    if np.any(large):
        flat = out.reshape(-1, 6, 6)
        Zf = Z.reshape(-1, 6, 6)
        for i in np.flatnonzero(large.reshape(-1)):
            A = np.zeros((12, 12))
            A[:6, :6] = Zf[i]
            A[:6, 6:] = EYE6
            flat[i] = np.linalg.solve(expm(A)[:6, 6:], EYE6)
        out = flat.reshape(out.shape)
    return out


def left_jacobian(x):
    """Left Jacobian of SE(3) at ``x``: the inverse of :func:`inv_left`."""
    return np.linalg.inv(inv_left(x))


def rot_axis(axis, angle):
    """Rotation matrices about unit ``axis`` (..., 3) by ``angle`` (...)."""
    axis = np.asarray(axis, float)
    angle = np.asarray(angle, float)
    K = skew(axis)
    s = np.sin(angle)[..., None, None]
    c = (1 - np.cos(angle))[..., None, None]
    return EYE3 + s * K + c * (K @ K)


def transform(R=None, p=None):
    """Assemble homogeneous transforms from rotations and translations."""
    R = EYE3 if R is None else np.asarray(R, float)
    p = np.zeros(3) if p is None else np.asarray(p, float)
    shape = np.broadcast_shapes(R.shape[:-2], p.shape[:-1])
    T = np.zeros(shape + (4, 4))
    T[..., :3, :3] = R
    T[..., :3, 3] = p
    T[..., 3, 3] = 1.
    return T


def spatial_inertia(mass, com, inertia_com):
    """Spatial inertia about the body-frame origin, angular-first (..., 6, 6).

    ``M = [[I_c + m skew(c) skew(c)^T, m skew(c)], [m skew(c)^T, m 1]]`` so that the
    kinetic energy of a body with body twist ``V`` is ``V^T M V / 2``.
    """
    mass = np.asarray(mass, float)
    C = skew(com)
    Ct = np.swapaxes(C, -1, -2)
    M = np.zeros(mass.shape + (6, 6))
    m = mass[..., None, None]
    M[..., :3, :3] = np.asarray(inertia_com, float) + m * (C @ Ct)
    M[..., :3, 3:] = m * C
    M[..., 3:, :3] = m * Ct
    M[..., 3:, 3:] = m * EYE3
    return M


def quat_xyzw_from_matrix(R):
    """Unit quaternion (x, y, z, w) of a single rotation matrix."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(np.asarray(R, float)).as_quat()


def matrix_from_quat_xyzw(q):
    """Rotation matrix of a quaternion given as (x, y, z, w)."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(np.asarray(q, float)).as_matrix()
