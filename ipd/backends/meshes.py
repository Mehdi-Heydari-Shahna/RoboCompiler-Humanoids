"""Tessellation of the parametric visual shapes of a physical record (rendering only).

Records describe their visual geometry parametrically, as a URDF describes visuals; the
MJCF exporter turns the descriptions into triangle meshes.  Visual geometry carries no
mass and takes no part in contact.

``loft``            a lofted superellipse: cross-sections ``(z, ax, ay, cx, cy)`` along the
                    local z axis with rounded end caps; used for limb and trunk shells.
``superellipsoid``  ``|x/a|^p + |y/b|^p + |z/c|^p = 1``; a rounded box for ``p`` around 4-8.
"""
from __future__ import annotations

import numpy as np


def _ring(ax, ay, cx, cy, z, n, p):
    t = np.linspace(0., 2. * np.pi, n, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    e = 2. / p
    x = cx + ax * np.sign(c) * np.abs(c) ** e
    y = cy + ay * np.sign(s) * np.abs(s) ** e
    return np.c_[x, y, np.full(n, z)]


def loft(sections, exponent=2.0, n=32, caps=(0.6, 0.6), cap_rings=5):
    """Vertices (k, 3) and faces (m, 3) of a closed lofted shell.

    ``sections`` are rows ``(z, ax, ay[, cx, cy])`` ordered along z.  ``caps`` are the
    dome heights at the two ends as fractions of the end half-width (0 gives a flat end).
    """
    sec = [list(r) + [0.] * (5 - len(r)) for r in sections]
    sec = sorted(sec, key=lambda r: r[0])
    rings = []
    z0, ax0, ay0, cx0, cy0 = sec[0]
    h0 = caps[0] * min(ax0, ay0)
    for k in range(cap_rings, 0, -1):                     # dome at the lower end
        phi = .5 * np.pi * k / (cap_rings + 1)
        rings.append(_ring(ax0 * np.cos(phi), ay0 * np.cos(phi), cx0, cy0, z0 - h0 * np.sin(phi), n, exponent))
    for z, ax, ay, cx, cy in sec:
        rings.append(_ring(ax, ay, cx, cy, z, n, exponent))
    z1, ax1, ay1, cx1, cy1 = sec[-1]
    h1 = caps[1] * min(ax1, ay1)
    for k in range(1, cap_rings + 1):                     # dome at the upper end
        phi = .5 * np.pi * k / (cap_rings + 1)
        rings.append(_ring(ax1 * np.cos(phi), ay1 * np.cos(phi), cx1, cy1, z1 + h1 * np.sin(phi), n, exponent))
    V = np.vstack(rings + [[[cx0, cy0, z0 - h0]], [[cx1, cy1, z1 + h1]]])
    r = len(rings)
    bottom, top = r * n, r * n + 1
    F = []
    for i in range(r - 1):
        a, b = i * n, (i + 1) * n
        for j in range(n):
            j1 = (j + 1) % n
            F.append([a + j, a + j1, b + j1])
            F.append([a + j, b + j1, b + j])
    for j in range(n):
        j1 = (j + 1) % n
        F.append([bottom, j1, j])
        F.append([top, (r - 1) * n + j, (r - 1) * n + j1])
    return V, np.array(F, int)


def superellipsoid(half, exponent=4.0, n=24):
    """Vertices and faces of ``|x/a|^p + |y/b|^p + |z/c|^p = 1`` (a rounded box for large ``p``)."""
    a, b, c = half
    e = 2. / exponent
    u = np.linspace(-.5 * np.pi, .5 * np.pi, n + 1)[1:-1]          # latitude, poles added separately
    v = np.linspace(0., 2. * np.pi, 2 * n, endpoint=False)
    f = lambda w: np.sign(w) * np.abs(w) ** e
    V = []
    for ui in u:
        cu, su = np.cos(ui), np.sin(ui)
        V.append(np.c_[a * f(cu) * f(np.cos(v)), b * f(cu) * f(np.sin(v)), np.full(len(v), c * f(su))])
    V = np.vstack(V + [[[0., 0., -c]], [[0., 0., c]]])
    m, k = len(u), len(v)
    south, north = m * k, m * k + 1
    F = []
    for i in range(m - 1):
        for j in range(k):
            j1 = (j + 1) % k
            F.append([i * k + j, i * k + j1, (i + 1) * k + j1])
            F.append([i * k + j, (i + 1) * k + j1, (i + 1) * k + j])
    for j in range(k):
        j1 = (j + 1) % k
        F.append([south, j1, j])
        F.append([north, (m - 1) * k + j, (m - 1) * k + j1])
    return V, np.array(F, int)


def tessellate(geom):
    """Vertices and faces of a parametric visual geom of type ``loft`` or ``rbox``."""
    if geom['type'] == 'loft':
        return loft(geom['sections'], geom.get('exponent', 2.0), geom.get('n', 32), tuple(geom.get('caps', (0.6, 0.6))))
    if geom['type'] == 'rbox':
        return superellipsoid(geom['half'], geom.get('exponent', 4.0), geom.get('n', 20))
    raise ValueError(f"no tessellation for geom type {geom['type']}")
