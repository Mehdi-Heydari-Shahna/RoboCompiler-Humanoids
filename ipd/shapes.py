"""Mass properties of simple solids and small transform helpers for record authoring."""
from __future__ import annotations

import numpy as np


def rot(axis, angle):
    """Rotation matrix about a principal axis ('x', 'y', 'z') or a 3-vector."""
    if isinstance(axis, str):
        axis = {'x': (1., 0., 0.), 'y': (0., 1., 0.), 'z': (0., 0., 1.)}[axis]
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0., -a[2], a[1]], [a[2], 0., -a[0]], [-a[1], a[0], 0.]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def tf(p=(0., 0., 0.), R=None):
    """Homogeneous transform from a translation and an optional rotation."""
    T = np.eye(4)
    T[:3, 3] = p
    if R is not None:
        T[:3, :3] = R
    return T


def frame_from_z(direction):
    """Rotation whose z axis is the given direction (x, y completed deterministically)."""
    z = np.asarray(direction, float)
    z = z / np.linalg.norm(z)
    ref = np.array([1., 0., 0.]) if abs(z[0]) < .9 else np.array([0., 1., 0.])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    return np.c_[x, np.cross(z, x), z]


def box_inertia(mass, size):
    """Inertia about the centre of a solid box with full side lengths ``size``."""
    x, y, z = size
    return mass / 12. * np.diag([y * y + z * z, x * x + z * z, x * x + y * y])


def cylinder_inertia(mass, radius, length, axis='z'):
    """Inertia about the centre of a solid cylinder along a principal axis."""
    lat = mass * (3 * radius ** 2 + length ** 2) / 12.
    ax = mass * radius ** 2 / 2.
    d = [lat, lat, lat]
    d['xyz'.index(axis)] = ax
    return np.diag(d)


def capsule_inertia(mass, radius, length, axis='z'):
    """Inertia about the centre of a solid capsule (cylinder ``length`` plus two hemispheres)."""
    vc = np.pi * radius ** 2 * length
    vs = 4. / 3. * np.pi * radius ** 3
    mc, ms = mass * vc / (vc + vs), mass * vs / (vc + vs)
    ax = mc * radius ** 2 / 2. + ms * 2. * radius ** 2 / 5.
    lat = mc * (3 * radius ** 2 + length ** 2) / 12. + ms * (2. * radius ** 2 / 5. + length ** 2 / 4. + 3. * length * radius / 8.)
    d = [lat, lat, lat]
    d['xyz'.index(axis)] = ax
    return np.diag(d)


def sphere_inertia(mass, radius):
    return 2. / 5. * mass * radius ** 2 * np.eye(3)


def rotate_inertia(I, R):
    """Express an inertia tensor given in a rotated frame ``R`` in the parent axes."""
    return R @ np.asarray(I, float) @ R.T


def body(bid, mass, com, inertia_com, geoms=None):
    """Body record; ``geoms`` are optional drawing/contact primitives in the body frame."""
    rec = {'id': bid, 'mass_kg': float(mass), 'com_m': [float(x) for x in com],
           'inertia_com_kg_m2': np.asarray(inertia_com, float).tolist()}
    if geoms:
        rec['geoms'] = geoms
    return rec


def joint(jid, jtype, body_a, body_b, T_AJ=None, T_BJ=None, axis=(0., 0., 1.), lower=-np.pi, upper=np.pi):
    rec = {'id': jid, 'type': jtype, 'body_a': body_a, 'body_b': body_b,
           'T_AJ': (np.eye(4) if T_AJ is None else np.asarray(T_AJ, float)).tolist(),
           'T_BJ': (np.eye(4) if T_BJ is None else np.asarray(T_BJ, float)).tolist()}
    if jtype in ('revolute', 'prismatic'):
        rec['axis'] = [float(x) for x in axis]
        rec['limits'] = {'lower': float(lower), 'upper': float(upper)}
    return rec
