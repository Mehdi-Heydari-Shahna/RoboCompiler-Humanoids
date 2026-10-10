"""Closure modes: PACDM residuals, rank-checked lift and its analytic time derivative.

A ``Mode`` is a fixed-topology closure set of a ``System``: the internal chords of
every member, a subset of the inter-member links and a set of ideal supports (sole
frames welded to world anchors).  Every chord names the coordinates it determines;
chords sharing dependent coordinates form a module, modules whose inputs are
determined by other modules are ordered (cyclic dependencies are merged), and
structurally identical modules of one layer are evaluated as a stack.

For a module with selected rows ``S`` and columns split into dependent ``p`` and
input ``a`` coordinates,

    E[p] = -(S C_p)^-1 (S C_a) E[a],
    Edot[p] = -(S C_p)^-1 S (Cdot E[cols] + C_a Edot[a]),

where ``E`` is the lift ``v = E u`` of the paper and ``Cdot`` is the analytic time
derivative of the path-assembled differential.  ``Edot u`` is the constraint-
curvature term, obtained here without finite differences.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import qr

from . import se3
from .compiler import ModelError, components

RANK_TOL = 1e-10
RANK_TOL_FULL = 1e-6      # rank of a complete module differential; redundant rows are consistent only to the closure tolerance
RCOND_MIN = 1e-10
CLOSURE_TOL = 1e-8


def rank(A, tol=RANK_TOL):
    s = np.linalg.svd(A, compute_uv=False)
    return int(np.sum(s > tol * s[0])) if len(s) and s[0] else 0


def select(Jp):
    """Row selector for a full-column-rank dependent block (pivoted QR, as in PACDM)."""
    n = Jp.shape[1]
    r = rank(Jp)
    if r < n:
        return np.array([], int), 0., r
    rows = qr(Jp.T, pivoting=True, mode='economic')[2][:n]
    rows = np.sort(rows)
    return rows, 1 / np.linalg.cond(Jp[rows], p=1), r


def _scc_merge(deps, inps):
    """Merge modules with cyclic input/dependent relations and layer the result.

    Module ``i`` depends on the module that determines one of its inputs.  Returns a
    list of (member module indices, layer), where layer 0 has independent inputs only.
    """
    n = len(deps)
    owner = {x: i for i, d in enumerate(deps) for x in d}
    succ = [sorted({owner[x] for x in inps[i] if x in owner and owner[x] != i}) for i in range(n)]
    index, low, on, stack, comp = {}, {}, set(), [], {}

    def strong(v):
        index[v] = low[v] = len(index)
        stack.append(v)
        on.add(v)
        for w in succ[v]:
            if w not in index:
                strong(w)
                low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            c = len(set(comp.values()))
            while True:
                w = stack.pop()
                on.discard(w)
                comp[w] = c
                if w == v:
                    break

    for v in range(n):
        if v not in index:
            strong(v)
    groups = {}
    for i in range(n):
        groups.setdefault(comp[i], []).append(i)
    layer = {}

    def depth(c):
        if c not in layer:
            needs = {comp[w] for i in groups[c] for w in succ[i]} - {c}
            layer[c] = 1 + max([depth(x) for x in needs], default=-1)
        return layer[c]

    return [(groups[c], depth(c)) for c in sorted(groups)]


class _Group:
    """Stack of structurally identical modules of one layer."""
    __slots__ = ('ch', 'cols', 'dep', 'inp', 'sig', 'rows', 'ndep', 'layer', 'module_ids', 'b1')


class Mode:
    def __init__(self, system, supports=(), links=(), dependents=None, name='mode'):
        """``supports``: list of (member, sole_site) or (member, sole_site, anchor 4x4).

        ``links``: ids of active inter-member links.  ``dependents`` maps the id of
        every support (``member/site``) and active link to the coordinate ids it
        determines; cut coordinates of a link are always dependent and are added
        automatically.
        """
        s = self.sys = system
        self.name = name
        dependents = dict(dependents or {})
        ch_sys = list(range(s.n_internal)) + [s.chord_index[l] for l in links]
        self.link_ids = list(links)
        self.support_ids, sb2, sP2 = [], [], []
        self.anchors = []
        for sup in supports:
            member, site = sup[0], sup[1]
            b, P = s.site_frame(member, site)
            self.support_ids.append(f'{member}/{site}')
            sb2.append(b); sP2.append(P)
            self.anchors.append(None if len(sup) < 3 else np.asarray(sup[2], float))
        ns = len(self.support_ids)
        self.ids = [s.chord_ids[i] for i in ch_sys] + self.support_ids
        self.m = len(self.ids)
        self.n_sys_chords = len(ch_sys)
        self.b1 = np.r_[s.chord_b1[ch_sys], np.full(ns, s.W, int)].astype(int)
        self.b2 = np.r_[s.chord_b2[ch_sys], np.array(sb2, int)].astype(int)
        self.P1 = np.concatenate([s.chord_P1[ch_sys], np.tile(np.eye(4), (ns, 1, 1))])
        self.P2 = np.concatenate([s.chord_P2[ch_sys], np.array(sP2, float).reshape(-1, 4, 4)])
        self.sigma = np.concatenate([s.sigma[ch_sys],
                                     s.anc[np.array(sb2, int)].astype(np.int8).reshape(-1, s.n)])
        self.support_slice = slice(self.n_sys_chords, self.m)
        # Dependent coordinates of every chord.
        deps = []
        for k, m in enumerate(s.members):
            for p in m.comp.plan['cut_plans']:
                deps.append([s.index[f'{m.name}/{x}'] for x in p['dependent_ids']])
        for l in links:
            cut = s.chord_cut_dofs[s.chord_index[l]]
            if l not in dependents:
                raise ModelError(f'No dependent coordinates given for link {l}')
            deps.append([s.index[x] for x in dependents[l]] + list(cut))
        for sid in self.support_ids:
            if sid not in dependents:
                raise ModelError(f'No dependent coordinates given for support {sid}')
            deps.append([s.index[x] for x in dependents[sid]])
        self.chord_dependents = deps
        active_cols = np.flatnonzero(np.any(self.sigma != 0, axis=0))
        all_dep = [x for d in deps for x in d]
        # Level-1 independent coordinates: root twists and requested joints of the members.
        base = []
        for m in s.members:
            base += [s.index[f'{m.name}/{x}'] for x in m.comp.cmg['independent_ids']]
        self.level1 = np.array(base, int)
        dep_set = set(all_dep)
        self.independent = np.array([x for x in base if x not in dep_set], int)
        self.dependent = np.array(sorted(dep_set), int)
        self.d = len(self.independent)
        self.independent_ids = [s.ids[i] for i in self.independent]
        bad = [s.ids[x] for x in active_cols if x not in dep_set and x not in set(base)]
        if bad:
            raise ModelError('Coordinates on a chord path are neither independent nor determined: ' + ', '.join(bad))
        self.pos = np.full(s.n, -1, int)
        self.pos[self.independent] = np.arange(self.d)
        self._build_modules()
        self.rows_ready = False

    # ------------------------------------------------------------------ structure
    def _build_modules(self):
        s = self.sys
        groups = components(self.chord_dependents)
        mods = []
        for g in groups:
            dep = sorted(set().union(*[set(self.chord_dependents[i]) for i in g]))
            cols = np.flatnonzero(np.any(self.sigma[g] != 0, axis=0))
            inp = [x for x in cols if x not in set(dep)]
            mods.append(dict(chords=list(g), dep=dep, inp=inp))
        merged = []
        for members, layer in _scc_merge([m['dep'] for m in mods], [m['inp'] for m in mods]):
            chords = sorted(c for i in members for c in mods[i]['chords'])
            dep = sorted(set().union(*[set(mods[i]['dep']) for i in members]))
            cols = np.flatnonzero(np.any(self.sigma[chords] != 0, axis=0))
            if not set(dep) <= set(cols.tolist()):
                off = [s.ids[x] for x in dep if x not in set(cols.tolist())]
                raise ModelError('Requested dependent coordinates are not on a path of their module: ' + ', '.join(off))
            inp = [int(x) for x in cols if x not in set(dep)]
            merged.append(dict(chords=chords, dep=[int(x) for x in dep], inp=inp, layer=layer))
        merged.sort(key=lambda m: (m['layer'], m['chords'][0]))
        self.modules = merged
        self.module_sizes = [len(m['dep']) for m in merged]

    def _make_groups(self):
        """Stack modules of equal shape within a layer (rows must already be selected)."""
        buckets = {}
        for i, m in enumerate(self.modules):
            key = (m['layer'], len(m['chords']), len(m['dep']), len(m['inp']), len(m['rows']))
            buckets.setdefault(key, []).append(i)
        self.groups = []
        for key in sorted(buckets):
            ids = buckets[key]
            g = _Group()
            g.module_ids = ids
            g.layer = key[0]
            g.ndep = key[2]
            g.ch = np.array([self.modules[i]['chords'] for i in ids], int)
            g.dep = np.array([self.modules[i]['dep'] for i in ids], int).reshape(len(ids), key[2])
            g.inp = np.array([self.modules[i]['inp'] for i in ids], int).reshape(len(ids), key[3])
            g.cols = np.concatenate([g.dep, g.inp], axis=1)
            g.sig = np.array([self.sigma[self.modules[i]['chords']][:, g.cols[k]] for k, i in enumerate(ids)], float)
            g.rows = np.array([self.modules[i]['rows'] for i in ids], int)
            g.b1 = self.b1[g.ch]
            self.groups.append(g)

    # ------------------------------------------------------------------ anchors
    def set_anchors(self, q, T=None):
        """Weld every support whose anchor is unset at the current pose of its sole frame."""
        T = self.sys.poses(q) if T is None else T
        for i, a in enumerate(self.anchors):
            e = self.n_sys_chords + i
            if a is None:
                self.anchors[i] = T[self.b2[e]] @ self.P2[e]
            self.P1[e] = self.anchors[i]
        return self

    def set_anchor(self, support_id, T_anchor):
        i = self.support_ids.index(support_id)
        self.anchors[i] = np.asarray(T_anchor, float)
        self.P1[self.n_sys_chords + i] = self.anchors[i]

    # ------------------------------------------------------------------ evaluation
    def frames(self, T, defects=None):
        Tm = T[self.b1] @ self.P1
        if defects is not None:
            Tm = Tm @ defects
        Tp = T[self.b2] @ self.P2
        D = se3.inv(Tm) @ Tp
        rho = se3.log(D)
        return rho, se3.inv_left(rho) @ se3.Ad_inv(Tm), D

    def jacobian(self, kin, A=None, rho=None):
        """Dense residual and differential ``C`` (6m, n) of all active chords."""
        if A is None:
            rho, A, _ = self.frames(kin.T)
        C = np.einsum('eij,jn->ein', A, kin.S) * self.sigma[:, None, :]
        return rho.reshape(-1), C.reshape(6 * self.m, self.sys.n)

    def _module_jac(self, g, A, S):
        """Stacked module differentials (g, 6 nch, ncols) with dependent columns first."""
        Sc = np.moveaxis(S[:, g.cols], 0, 1)                       # (g, 6, ncols)
        C = np.einsum('gcij,gjn->gcin', A[g.ch], Sc) * g.sig[:, :, None, :]
        return C.reshape(len(g.ch), -1, g.cols.shape[1])

    def select_rows(self, q):
        """Rank-check every module at ``q`` and fix its row selector."""
        kin = self.sys.kinematics(q)
        rho, A, D = self.frames(kin.T)
        info = []
        for m in self.modules:
            g = _Group()
            g.ch = np.array([m['chords']], int)
            g.cols = np.array([m['dep'] + m['inp']], int)
            g.sig = self.sigma[m['chords']][:, g.cols[0]][None].astype(float)
            C = self._module_jac(g, A, kin.S)[0]
            Jp = C[:, :len(m['dep'])]
            rows, rc, rp = select(Jp)
            rf = rank(C, RANK_TOL_FULL)
            # The lift spans the whole kernel of the module differential only if its rank is
            # the number of dependent coordinates: a larger rank means that the rows also
            # constrain the inputs, i.e. the requested partition is not a partition.
            ok = rp == len(m['dep']) and rc >= RCOND_MIN and rf == rp
            m['rows'] = rows.tolist() if ok else list(range(len(m['dep'])))
            info.append(dict(chords=[self.ids[c] for c in m['chords']], dependent=len(m['dep']), inputs=len(m['inp']),
                             residual_rows=C.shape[0], rank_full=rf, rank_dependent=rp, rcond=float(rc), ok=bool(ok)))
        self.rank_info = info
        if not all(i['ok'] for i in info):
            bad = [i for i in info if not i['ok']]
            raise ModelError(f'Rank or conditioning does not support the requested partition: {bad[0]}')
        self._make_groups()
        self.rows_ready = True
        return info

    def solutions(self, kin, derivative=False, frames=None, position=None):
        """Per group, the partial maps ``X = -(S C_p)^-1 S C_a`` and their rates.

        For every stacked module, ``v_p = X v_a`` and ``d/dt X = -(S C_p)^-1 S (Cdot_p X + Cdot_a)``.
        ``position`` may pass the result of an earlier call at the same configuration,
        so that only the rates are computed.
        """
        rho, A, D = self.frames(kin.T) if frames is None else frames
        out = []
        for i, g in enumerate(self.groups):
            if position is None:
                C = self._module_jac(g, A, kin.S)
                Cs = np.take_along_axis(C, g.rows[:, :, None], axis=1)
                R = Cs[:, :, :g.ndep]
                X = -np.linalg.solve(R, Cs[:, :, g.ndep:])
            else:
                _, X, _, R = position[i]
            Xd = None
            if derivative:
                Vm = kin.Vs[g.b1]                                   # (g, nch, 6) twists of the minus frames
                Sc = np.moveaxis(kin.S[:, g.cols], 0, 1)
                Sd = np.moveaxis(kin.Sdot[:, g.cols], 0, 1)
                W = Sd[:, None] - se3.ad(Vm) @ Sc[:, None]          # (g, nch, 6, ncols)
                Cd = (A[g.ch] @ W) * g.sig[:, :, None, :]
                Cd = np.take_along_axis(Cd.reshape(len(g.ch), -1, g.cols.shape[1]), g.rows[:, :, None], axis=1)
                Xd = -np.linalg.solve(R, Cd[:, :, :g.ndep] @ X + Cd[:, :, g.ndep:])
            out.append((g, X, Xd, R))
        return out

    def lift(self, kin, derivative=False, frames=None):
        """Lift ``E`` (n, d) with ``v = E u``; with ``derivative`` also ``Edot`` along ``kin.v``."""
        s = self.sys
        E = np.zeros((s.n, self.d))
        E[self.independent, np.arange(self.d)] = 1.
        Ed = np.zeros_like(E) if derivative else None
        for g, X, Xd, _ in self.solutions(kin, derivative, frames):
            Einp = E[g.inp]
            E[g.dep] = X @ Einp
            if derivative:
                Ed[g.dep] = Xd @ Einp + X @ Ed[g.inp]
        return (E, Ed) if derivative else E

    def residual(self, q):
        """Full closure residual (m, 6) and the comparison transforms."""
        rho, A, D = self.frames(self.sys.poses(q))
        return rho, D

    def gap(self, q):
        """Largest residual component and largest translational endpoint gap."""
        rho, D = self.residual(q)
        return float(np.max(abs(rho))) if self.m else 0., float(np.max(np.linalg.norm(D[:, :3, 3], axis=1))) if self.m else 0.

    def correct(self, q, tol=1e-11, maxiter=12, full=True):
        """Newton correction of the dependent coordinates on the selected rows.

        The independent coordinates are held.  Returns the corrected configuration
        and a report; ``success`` requires the complete residual (all rows) to pass
        the physical closure tolerance when ``full`` is set.
        """
        s = self.sys
        info = dict(success=False, iterations=0)
        for it in range(maxiter + 1):
            kin = s.kinematics(q)
            rho, A, D = self.frames(kin.T)
            S = kin.S
            worst = 0.
            dth = np.zeros(s.n)
            for g in self.groups:
                C = self._module_jac(g, A, S)
                Cs = np.take_along_axis(C, g.rows[:, :, None], axis=1)
                r = np.take_along_axis(rho[g.ch].reshape(len(g.ch), -1), g.rows, axis=1)
                worst = max(worst, float(np.max(abs(r))))
                rhs = r + np.einsum('gij,gj->gi', Cs[:, :, g.ndep:], dth[g.inp])
                dth[g.dep] = -np.linalg.solve(Cs[:, :, :g.ndep], rhs[:, :, None])[:, :, 0]
            info.update(iterations=it, selected_residual_inf=worst)
            if worst <= tol:
                break
            q = s.integrate(q, dth)
        info['residual_inf'] = float(np.max(abs(rho))) if self.m else 0.
        info['gap_m'] = float(np.max(np.linalg.norm(D[:, :3, 3], axis=1))) if self.m else 0.
        info['success'] = bool(worst <= tol and (not full or info['residual_inf'] <= CLOSURE_TOL))
        return q, info

    # ------------------------------------------------------------------ PACDM (vendored) interface
    def pacdm_graph(self, q):
        """Adapter exposing this mode to the unchanged PACDM solver in ``vendor``."""
        return _PacdmGraph(self, q)

    def acquire(self, q):
        """Defect-homotopy acquisition of a feasible configuration with the PACDM solver.

        The independent coordinates of ``q`` are held; the artificial defect is
        removed by continuation.  The result is polished and rank-checked.
        """
        from vendor.pacdm_original import PACDM
        graph = self.pacdm_graph(q)
        x0 = q.theta[graph.keep].copy()
        x, info = PACDM(graph).acquire(x0[graph.active], x0)
        if not info['success']:
            raise ModelError(f'PACDM acquisition failed: {info}')
        out = q.copy()
        out.theta[graph.keep] = x
        self.select_rows(out)
        out, ci = self.correct(out)
        if not ci['success']:
            raise ModelError(f'Polish after acquisition failed: {ci}')
        return out, dict(acquisition=info, polish=ci)

    def mapping_pacdm(self, q):
        """Lift and acceptance report from the unchanged PACDM solver (dense, for checks)."""
        from vendor.pacdm_original import PACDM
        graph = self.pacdm_graph(q)
        N, info = PACDM(graph).mapping(q.theta[graph.keep].copy())
        return N, info, graph


class _PacdmGraph:
    """Duck-typed graph for ``vendor.pacdm_original.PACDM`` on a fixed set of root poses.

    The coordinate vector is ``theta`` of the configuration.  Unused coordinates
    (cut coordinates of inactive links) are removed; floating roots that the mode
    keeps independent are held at their pose.
    """
    def __init__(self, mode, q):
        s = mode.sys
        self.mode, self.q = mode, q.copy()
        for k, sl in enumerate(s.root_slice):
            if sl is not None and np.any(np.isin(np.arange(sl.start, sl.stop), mode.dependent)):
                raise ModelError('The PACDM adapter needs every floating root to be independent')
        self.keep = np.sort(np.r_[mode.independent, mode.dependent])
        self.n = len(self.keep)
        loc = np.full(s.n, -1, int)
        loc[self.keep] = np.arange(self.n)
        self.active = loc[mode.independent]
        self.passive = loc[mode.dependent]
        self.lower = s.lower[self.keep].copy()
        self.upper = s.upper[self.keep].copy()
        self.lower[self.active] = -np.inf
        self.upper[self.active] = np.inf

    def _config(self, x):
        q = self.q.copy()
        q.theta[self.keep] = x
        for sl in self.mode.sys.root_slice:
            if sl is not None:
                q.theta[sl] = 0.
        return q

    def residual(self, x, defects=None):
        mode = self.mode
        kin = mode.sys.kinematics(self._config(x))
        rho, A, D = mode.frames(kin.T, defects)
        _, C = mode.jacobian(kin, A, rho)
        return rho.reshape(-1), C[:, self.keep], D
