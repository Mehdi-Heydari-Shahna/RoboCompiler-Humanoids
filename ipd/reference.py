"""Extended-precision hard-constraint reference for constrained forward dynamics.

Given the double-precision tree quantities ``M``, ``h`` (bias minus applied effort),
the physical constraint rows ``J`` and the drift ``gamma`` of one state, the reference
solves

    [ M    J_r^T ] [  a   ]   [ -h      ]
    [ J_r   0    ] [ -lam ] = [ -gamma_r ]

in ball arithmetic at a chosen number of digits (python-flint), where ``J_r`` is a
maximal set of independent rows chosen by a scaled rank-revealing factorisation.
Only the solve is carried out in extended precision.  The returned radius bounds its
rounding error; it does not include the double-precision error of the inputs
(``M``, ``h``, ``J``, ``gamma`` are double-precision numbers, which limits the
reference itself to about 1e-15 relative) or the choice among redundant rows, which
is estimated separately by ``row_sensitivity``.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import qr


def independent_rows(J, M, tol=1e-9, order=None, zero=1e-12):
    """Indices of a maximal independent row set of ``J`` in the inertia-scaled metric.

    Rows are normalised before the rank-revealing factorisation; rows whose norm is below
    ``zero`` times the largest row norm are structurally zero and are left out first (a
    normalised round-off row would otherwise count as a constraint).
    """
    scale = 1. / np.sqrt(np.diag(M))
    A = J * scale[None, :]
    norms = np.linalg.norm(A, axis=1)
    keep = np.flatnonzero(norms > zero * max(float(norms.max()), 1e-300)) if len(norms) else np.zeros(0, int)
    if order is not None:
        keep = np.array([i for i in order if i in set(keep.tolist())], int)
    A = A[keep] / norms[keep][:, None]
    if not len(keep):
        return np.zeros(0, int), 0
    _, R, piv = qr(A.T, pivoting=True, mode='economic')
    d = abs(np.diag(R))
    r = int(np.sum(d > tol * d[0]))
    return np.sort(keep[piv[:r]]), r


def kkt_reference(M, h, J, gamma, dps=50, tol=1e-9, rows=None):
    """Acceleration ``a`` with ``M a + h = J^T lam`` and ``J a + gamma = 0`` at ``dps`` digits.

    ``h`` is the bias minus the applied tree effort.  Returns ``a`` rounded to
    double precision, the multipliers of the selected rows, the row indices, the
    rank and the largest enclosure radius of ``a``.
    """
    from flint import arb, arb_mat, ctx
    old = ctx.dps
    ctx.dps = dps
    try:
        n = M.shape[0]
        if rows is None:
            rows, r = independent_rows(J, M, tol)
        else:
            r = len(rows)
        Jr, gr = J[rows], gamma[rows]
        K = np.zeros((n + r, n + r))
        K[:n, :n] = M
        K[:n, n:] = Jr.T
        K[n:, :n] = Jr
        rhs = np.r_[-h, -gr]
        Km = arb_mat(n + r, n + r, [arb(float(x)) for x in K.ravel()])
        bm = arb_mat(n + r, 1, [arb(float(x)) for x in rhs])
        x = Km.solve(bm)
        a = np.array([float(x[i, 0].mid()) for i in range(n)])
        lam = -np.array([float(x[n + i, 0].mid()) for i in range(r)])
        radius = max(float(x[i, 0].rad()) for i in range(n))
        return a, lam, np.asarray(rows), r, radius
    finally:
        ctx.dps = old


def row_sensitivity(M, h, J, gamma, a_ref, trials=4, seed=0, dps=50, tol=1e-9):
    """Spread of the reference under alternative independent subsets of redundant rows.

    Returns the largest normalised acceleration and force-equivalent differences
    between ``a_ref`` and references built from randomly re-ordered row selections.
    """
    rng = np.random.default_rng(seed)
    da = df = 0.
    scale_f = max(1., np.max(abs(h)))                   # h here is the bias minus the applied effort
    for _ in range(trials):
        order = rng.permutation(J.shape[0])
        rows, r = independent_rows(J, M, tol, order=order)
        a, _, _, _, _ = kkt_reference(M, h, J, gamma, dps=dps, rows=rows)
        da = max(da, float(np.max(abs(a - a_ref)) / max(1., np.max(abs(a_ref)))))
        df = max(df, float(np.max(abs(M @ (a - a_ref))) / scale_f))
    return da, df


def discrepancy(M, a, a_ref, bias, tau, J=None, gamma=None):
    """Metrics of the RoboCompiler paper (its Eq. 23) for one state.

        delta_a = |a - a_ref|_inf / max(1, |a_ref|_inf),
        delta_f = |M (a - a_ref)|_inf / max(1, |tau|_inf, |h|_inf),
        r_c     = |J a + gamma|_inf,

    with the tree bias ``h`` (velocity-product, gravity and damping forces) and the
    applied tree effort ``tau``.
    """
    out = dict(delta_a=float(np.max(abs(a - a_ref)) / max(1., np.max(abs(a_ref)))),
               delta_f=float(np.max(abs(M @ (a - a_ref))) / max(1., np.max(abs(tau)), np.max(abs(bias)))))
    if J is not None:
        out['r_c'] = float(np.max(abs(J @ a + gamma)))
    return out
