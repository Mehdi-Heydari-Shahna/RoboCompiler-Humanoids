"""Two-level compiled dynamics of an ensemble: reusable member modules and a composer.

Level 1 (member modules).  Each member is reduced over its own internal chords to
its level-1 speeds ``w_k`` (root twist and the requested joints).  Its compiled body
Jacobians and its inertia partition

    Gamma1_k = sum_{i in member k} J1_i^T M_i J1_i,   C1_k,   g1_k

depend only on that member's state.  They do not depend on which supports or
inter-member links are closed.

Level 2 (composition).  Supports and links compare frames of bodies, so their
differential in the level-1 speeds is the difference of level-1 body Jacobians,

    Cbar_e = A_e (J1s_{b2} - J1s_{b1})    (+ cut-coordinate columns of a link),

and only the lift ``w = G u`` of these few rows depends on the topology.  Then

    Gamma = sum_k G_k^T Gamma1_k G_k,
    C     = sum_k G_k^T (C1_k G_k + Gamma1_k Gdot_k),
    g     = sum_k G_k^T g1_k,

which is the same model as a one-level reduction over all chords (``ipd.mode.Mode``
with ``ipd.partition.assemble``); the two are compared in the tests.
"""
from __future__ import annotations

import numpy as np

from . import se3
from .compiler import ModelError, components
from .mode import Mode, _Group, _scc_merge, rank, select, RANK_TOL_FULL, RCOND_MIN


class Evaluation:
    """Container of one evaluation; attributes are filled according to the request."""

    def __init__(self):
        self.__dict__.update(q=None, kin=None, u=None, v=None, w=None, E1=None, E1d=None, G=None, Gd=None,
                             Gamma1=None, C1=None, g1=None, D1=None, L1=None, Gamma=None, C=None, g=None,
                             D=None, L=None, J1=None, J1d=None, Js1=None, Jsd1=None, V=None, frames1=None,
                             frames2=None, Cbar=None)


class Composite:
    def __init__(self, system, supports=(), links=(), dependents=None, name='mode'):
        s = self.sys = system
        self.name = name
        if not hasattr(s, '_free_mode'):
            s._free_mode = Mode(s, name='free')
        self.free = s._free_mode
        self.mode = Mode(s, supports, links, dependents, name)
        self.supports = [(x[0], x[1]) for x in supports]
        self.links = list(links)
        K = s.K
        # ---- level-1 layout: member-local columns
        self.d1 = np.array([len(m.comp.cmg['independent_ids']) for m in s.members], int)
        self.d1max = int(self.d1.max())
        self.off1 = np.r_[0, np.cumsum(self.d1)]
        self.loc = np.full(s.n, -1, int)
        self.member_of_dof = np.full(s.n, -1, int)
        self.level1_dofs = []
        for k, m in enumerate(s.members):
            dofs = np.array([s.index[f'{m.name}/{x}'] for x in m.comp.cmg['independent_ids']], int)
            self.loc[dofs] = np.arange(len(dofs))
            self.level1_dofs.append(dofs)
            self.member_of_dof[s.tree_slice[k]] = k
            self.member_of_dof[s.cut_slice[k]] = k
        self.chart_dofs = np.array([j for l in self.links for j in s.chord_cut_dofs[s.chord_index[l]]], int)
        self.nchart = len(self.chart_dofs)
        self.D1 = int(self.off1[-1]) + self.nchart
        self.d1_index = np.full(s.n, -1, int)               # coordinate -> column in the level-1 space
        for k in range(K):
            self.d1_index[self.level1_dofs[k]] = self.off1[k] + np.arange(self.d1[k])
        self.d1_index[self.chart_dofs] = self.off1[-1] + np.arange(self.nchart)
        for g in self.free.modules:
            if any(self.loc[j] < 0 for j in g['inp']):
                raise ModelError('Internal modules must have level-1 independent inputs')
        self.tree_dofs = [np.arange(sl.start, sl.stop) for sl in s.tree_slice]
        self.member_dofs = [np.r_[np.arange(a.start, a.stop), np.arange(b.start, b.stop)]
                            for a, b in zip(s.tree_slice, s.cut_slice)]
        # structural sparsity of the level-1 lift in local columns
        E1s = np.zeros((s.n, self.d1max), bool)
        for k in range(K):
            E1s[self.level1_dofs[k], np.arange(self.d1[k])] = True
        for g in self.free.modules:
            E1s[np.ix_(g['dep'], self.loc[g['inp']])] = True
        self._E1s = E1s
        self._E1_identity = np.zeros((s.n, self.d1max))
        for k in range(K):
            self._E1_identity[self.level1_dofs[k], np.arange(self.d1[k])] = 1.
        # ---- level-2 chords
        ns, nl = len(self.supports), len(self.links)
        self.m2 = ns + nl
        names = [m.name for m in s.members]
        sb2, sP2, sk = [], [], []
        for member, site in self.supports:
            b, P = s.site_frame(member, site)
            sb2.append(b); sP2.append(P); sk.append(names.index(member))
        le = [s.chord_index[l] for l in self.links]
        self.c_b1 = np.r_[np.full(ns, s.W, int), s.chord_b1[le]].astype(int)        # frame used for the pose
        self.c_pb1 = np.r_[np.full(ns, -1, int), np.array([s.chord_ref_slot[e] for e in le], int)].astype(int)
        self.c_b2 = np.r_[np.array(sb2, int), s.chord_b2[le]].astype(int)
        self.c_P1 = np.concatenate([np.tile(np.eye(4), (ns, 1, 1)), s.chord_P1[le]])
        self.c_P2 = np.concatenate([np.array(sP2, float).reshape(-1, 4, 4), s.chord_P2[le]])
        self.c_k2 = np.r_[np.array(sk, int), s.member_of_body[s.chord_b2[le]]].astype(int)
        self.c_k1 = np.r_[np.full(ns, -1, int), np.array([s.member_of_body[s.chord_ref_slot[e]] for e in le], int)].astype(int)
        self.c_ids = [f'{a}/{b}' for a, b in self.supports] + self.links
        self.ns, self.nl = ns, nl
        # dependents in level-1 columns
        dependents = dict(dependents or {})
        deps = []
        for cid in self.c_ids[:ns]:
            deps.append([int(self.d1_index[s.index[x]]) for x in dependents[cid]])
        for i, l in enumerate(self.links):
            chart = [int(self.off1[-1]) + 3 * i + a for a in range(3)]
            deps.append([int(self.d1_index[s.index[x]]) for x in dependents[l]] + chart)
        if any(x < 0 for d in deps for x in d):
            raise ModelError('A requested dependent coordinate is not a level-1 speed')
        self.deps2 = deps
        dep_set = set(x for d in deps for x in d)
        self.independent2 = np.array([x for x in range(int(self.off1[-1])) if x not in dep_set], int)
        self.d = len(self.independent2)
        if self.d != self.mode.d:
            raise ModelError('Two-level and one-level partitions disagree')
        # structural supports of the level-2 rows
        sup = []
        for e in range(self.m2):
            cols = set()
            k2 = self.c_k2[e]
            anc2 = s.anc[self.c_b2[e]]
            cols |= {int(self.off1[k2]) + c for c in np.flatnonzero(np.any(E1s[anc2], axis=0))}
            if self.c_k1[e] >= 0:
                k1 = self.c_k1[e]
                anc1 = s.anc[self.c_pb1[e]]
                cols |= {int(self.off1[k1]) + c for c in np.flatnonzero(np.any(E1s[anc1], axis=0))}
                i = e - ns
                cols |= {int(self.off1[-1]) + 3 * i + a for a in range(3)}
            sup.append(sorted(cols))
        for e, d in enumerate(deps):
            if not set(d) <= set(sup[e]):
                raise ModelError(f'Chord {self.c_ids[e]}: a requested dependent coordinate is not on its path')
        mods = []
        for g in components(deps):
            dep = sorted(set().union(*[set(deps[i]) for i in g]))
            cols = sorted(set().union(*[set(sup[i]) for i in g]))
            mods.append(dict(chords=list(g), dep=dep, inp=[x for x in cols if x not in set(dep)]))
        merged = []
        for members, layer in _scc_merge([m['dep'] for m in mods], [m['inp'] for m in mods]):
            chords = sorted(c for i in members for c in mods[i]['chords'])
            dep = sorted(set().union(*[set(mods[i]['dep']) for i in members]))
            cols = sorted(set().union(*[set(sup[c]) for c in chords]))
            merged.append(dict(chords=chords, dep=dep, inp=[x for x in cols if x not in set(dep)], layer=layer))
        merged.sort(key=lambda m: (m['layer'], m['chords'][0]))
        self.modules2 = merged
        self.ready = False
        self.follow_anchors = False

    # ------------------------------------------------------------------ anchors and acceptance
    def prepare(self, q, acquire=False):
        """Weld supports at the current sole frames (unless anchors were given), optionally
        acquire a feasible configuration with PACDM, fix the row selectors and check ranks."""
        self.mode.set_anchors(q)
        info = None
        if acquire:
            q, info = self.mode.acquire(q)
        else:
            self.mode.select_rows(q)
        if not self.free.rows_ready:
            self.free.select_rows(q)
        self._sync_anchors()
        self._select_rows2(q)
        self.ready = True
        return q, info

    def _sync_anchors(self):
        for i in range(self.ns):
            self.c_P1[i] = self.mode.anchors[i]

    def set_anchor(self, support_id, T_anchor):
        self.mode.set_anchor(support_id, T_anchor)
        self._sync_anchors()

    def _select_rows2(self, q):
        ev = self.evaluate(q, _raw=True)
        info = []
        for m in self.modules2:
            C = ev.Cbar[m['chords']].reshape(-1, self.D1)[:, m['dep'] + m['inp']]
            rows, rc, rp = select(C[:, :len(m['dep'])])
            rf = rank(C, RANK_TOL_FULL)
            ok = rp == len(m['dep']) and rc >= RCOND_MIN and rf == rp
            m['rows'] = rows.tolist()
            info.append(dict(chords=[self.c_ids[c] for c in m['chords']], dependent=len(m['dep']), inputs=len(m['inp']),
                             residual_rows=C.shape[0], rank_full=rf, rank_dependent=rp, rcond=float(rc), ok=bool(ok)))
            if not ok:
                raise ModelError(f'Rank or conditioning does not support the requested partition: {info[-1]}')
        self.rank_info2 = info
        buckets = {}
        for i, m in enumerate(self.modules2):
            buckets.setdefault((m['layer'], len(m['chords']), len(m['dep']), len(m['inp']), len(m['rows'])), []).append(i)
        self.groups2 = []
        for key in sorted(buckets):
            ids = buckets[key]
            g = _Group()
            g.module_ids, g.layer, g.ndep = ids, key[0], key[2]
            g.ch = np.array([self.modules2[i]['chords'] for i in ids], int)
            g.dep = np.array([self.modules2[i]['dep'] for i in ids], int).reshape(len(ids), key[2])
            g.inp = np.array([self.modules2[i]['inp'] for i in ids], int).reshape(len(ids), key[3])
            g.cols = np.concatenate([g.dep, g.inp], axis=1)
            g.rows = np.array([self.modules2[i]['rows'] for i in ids], int)
            self.groups2.append(g)

    # ------------------------------------------------------------------ evaluation
    def _level1(self, kin, fr, position=None):
        """Level-1 lift in member-local columns; with ``position`` only its rate is computed."""
        s = self.sys
        if position is None:
            sols = self.free.solutions(kin, False, fr)
            E1 = self._E1_identity.copy()
            for g, X, _, _ in sols:
                E1[g.dep[:, :, None], self.loc[g.inp][:, None, :]] = X
            return E1, sols
        E1d = np.zeros((s.n, self.d1max))
        for g, X, Xd, _ in self.free.solutions(kin, True, fr, position):
            E1d[g.dep[:, :, None], self.loc[g.inp][:, None, :]] = Xd
        return E1d, None

    def _frames2(self, T):
        Tm = T[self.c_b1] @ self.c_P1
        Tp = T[self.c_b2] @ self.c_P2
        D = se3.inv(Tm) @ Tp
        rho = se3.log(D)
        return rho, se3.inv_left(rho) @ se3.Ad_inv(Tm), D

    def _cbar(self, A, J2, J1, Sc):
        """Dense level-2 differential (m2, 6, D1).

        ``J2`` and ``J1`` are the world-frame level-1 Jacobians of the plus and minus
        bodies of every chord (m2, 6, d1max) and ``Sc`` the screws of the link cut
        coordinates (nl, 6, 3); for the rate they hold the corresponding rates.
        """
        C = np.zeros((self.m2, 6, self.D1))
        AJ2 = A @ J2
        for e in range(self.m2):
            k2 = self.c_k2[e]
            C[e, :, self.off1[k2]:self.off1[k2] + self.d1[k2]] = AJ2[e, :, :self.d1[k2]]
        if self.nl:
            ns = self.ns
            AJ1 = A[ns:] @ J1
            C[ns:, :, self.off1[-1]:] = 0.
            ASc = A[ns:] @ Sc
            for i in range(self.nl):
                k1 = self.c_k1[ns + i]
                C[ns + i, :, self.off1[k1]:self.off1[k1] + self.d1[k1]] -= AJ1[i, :, :self.d1[k1]]
                C[ns + i, :, self.off1[-1] + 3 * i:self.off1[-1] + 3 * i + 3] = -ASc[i]
        return C

    def gravity(self, ev, M=None):
        """Level-1 and composed gravity vectors for the spatial inertias ``M`` (default: the system's)."""
        s = self.sys
        M = s.M if M is None else M
        T = ev.kin.T
        grav = np.zeros((s.nbp, 6))
        grav[:, 3:] = np.einsum('bji,j->bi', T[:s.nbp, :3, :3], s.gravity)
        Mg = np.einsum('bij,bj->bi', M, grav)
        g1 = np.zeros((s.K, self.d1max))
        g = np.zeros(self.d)
        for k in range(s.K):
            idx = s.member_bodies[k]
            g1[k] = -ev.J1[idx].reshape(-1, self.d1max).T @ Mg[idx].reshape(-1)
            g += ev.G[self.off1[k]:self.off1[k + 1]].T @ g1[k][:self.d1[k]]
        return g1, g

    def evaluate(self, q, u=None, matrices=True, _raw=False, curvature=True):
        """Compiled dynamics at ``q`` (and independent speeds ``u``).

        Without ``u`` only position-level quantities are produced (lift, inertia,
        gravity, port map).  With ``u`` the Coriolis factor and all rates are added.
        ``curvature=False`` is an ablation: the rates of both lifts (``E1dot``, ``Gdot``)
        are set to zero everywhere, i.e. the constraint-curvature term is left out.
        """
        s = self.sys
        ev = Evaluation()
        ev.q = q
        kin = ev.kin = s.kinematics(q)
        fr1 = self.free.frames(kin.T)
        E1, pos1 = self._level1(kin, fr1)
        Js1 = s.accumulate(kin.S, E1)
        if self.follow_anchors and self.ns:
            # ideal supports welded at the current sole frames (velocity-level support constraint)
            self.c_P1[:self.ns] = kin.T[self.c_b2[:self.ns]] @ self.c_P2[:self.ns]
        rho2, A2, D2 = self._frames2(kin.T)
        ev.frames1, ev.frames2 = fr1, (rho2, A2, D2)
        ns = self.ns
        chart = self.chart_dofs.reshape(-1, 3)
        Sc = np.moveaxis(kin.S[:, chart], 0, 1) if self.nl else None
        Cbar = ev.Cbar = self._cbar(A2, Js1[self.c_b2], Js1[self.c_pb1[ns:]] if self.nl else None, Sc)
        if _raw:
            return ev
        G = np.zeros((self.D1, self.d))
        G[self.independent2, np.arange(self.d)] = 1.
        sols2 = []
        for g in self.groups2:
            C = np.take_along_axis(Cbar[g.ch].reshape(len(g.ch), -1, self.D1), g.cols[:, None, :], axis=2)
            Cs = np.take_along_axis(C, g.rows[:, :, None], axis=1)
            R = Cs[:, :, :g.ndep]
            X = -np.linalg.solve(R, Cs[:, :, g.ndep:])
            G[g.dep] = X @ G[g.inp]
            sols2.append((g, R, X))
        ev.E1, ev.G, ev.Js1 = E1, G, Js1
        Gk = [G[self.off1[k]:self.off1[k + 1]] for k in range(s.K)]
        AdI = se3.Ad_inv(kin.T[:s.nbp])
        J1 = ev.J1 = AdI @ Js1
        M = s.M
        MJ = M @ J1
        K, d1 = s.K, self.d1max
        arm, damp = s.armature, s.damping
        Gamma1 = np.zeros((K, d1, d1)); g1 = np.zeros((K, d1)); Dm = np.zeros((K, d1, d1))
        grav = np.zeros((s.nbp, 6))
        grav[:, 3:] = np.einsum('bji,j->bi', kin.T[:s.nbp, :3, :3], s.gravity)
        Mg = np.einsum('bij,bj->bi', M, grav)
        L1 = []
        for k in range(K):
            idx = s.member_bodies[k]
            Jf = J1[idx].reshape(-1, d1)
            ET = E1[s.tree_slice[k]]
            a = arm[s.tree_slice[k]]
            Gamma1[k] = Jf.T @ MJ[idx].reshape(-1, d1) + ET.T @ (a[:, None] * ET)
            g1[k] = -Jf.T @ Mg[idx].reshape(-1)
            Dm[k] = ET.T @ (damp[s.tree_slice[k]][:, None] * ET)
            L1.append(s.B[s.tree_slice[k]][:, s.member_ports[k]].T @ ET)
        ev.Gamma1, ev.g1, ev.D1, ev.L1 = Gamma1, g1, Dm, L1
        d = self.d
        Gamma = np.zeros((d, d)); gv = np.zeros(d); D = np.zeros((d, d)); L = np.zeros((s.na, d))
        for k in range(K):
            n1 = self.d1[k]
            Gamma += Gk[k].T @ Gamma1[k][:n1, :n1] @ Gk[k]
            gv += Gk[k].T @ g1[k][:n1]
            D += Gk[k].T @ Dm[k][:n1, :n1] @ Gk[k]
            L[s.member_ports[k]] = L1[k][:, :n1] @ Gk[k]
        ev.Gamma, ev.g, ev.D, ev.L = Gamma, gv, D, L
        if u is None:
            return ev
        # ---- velocity level
        u = ev.u = np.asarray(u, float)
        w = ev.w = G @ u
        v = np.zeros(s.n)
        for k in range(K):
            md = self.member_dofs[k]
            v[md] = E1[md, :self.d1[k]] @ w[self.off1[k]:self.off1[k + 1]]
        v[self.chart_dofs] = w[self.off1[-1]:]
        ev.v = v
        s.velocity(kin, v)
        E1d, _ = self._level1(kin, fr1, pos1)
        if not curvature:
            E1d = np.zeros_like(E1d)
        Jsd1 = s.accumulate(kin.Sdot, E1) + s.accumulate(kin.S, E1d)
        ev.E1d, ev.Jsd1 = E1d, Jsd1
        J1d = ev.J1d = AdI @ (Jsd1 - se3.ad(kin.Vs[:s.nbp]) @ Js1)
        V = ev.V = np.einsum('bij,bj->bi', AdI, kin.Vs[:s.nbp])
        P = np.einsum('bij,bj->bi', M, V)
        Wm = M @ J1d - se3.momentum_matrix(P) @ J1
        C1 = np.zeros((K, d1, d1))
        for k in range(K):
            idx = s.member_bodies[k]
            ET, ETd = E1[s.tree_slice[k]], E1d[s.tree_slice[k]]
            C1[k] = J1[idx].reshape(-1, d1).T @ Wm[idx].reshape(-1, d1) + ET.T @ (arm[s.tree_slice[k]][:, None] * ETd)
        ev.C1 = C1
        # level-2 rate
        adV = se3.ad(kin.Vs[self.c_b1])
        J2d = Jsd1[self.c_b2] - adV @ Js1[self.c_b2]
        J1d_ = Scd = None
        if self.nl:
            J1d_ = Jsd1[self.c_pb1[ns:]] - adV[ns:] @ Js1[self.c_pb1[ns:]]
            Scd = np.moveaxis(kin.Sdot[:, chart], 0, 1) - adV[ns:] @ Sc
        Cbd = self._cbar(A2, J2d, J1d_, Scd)
        Gd = np.zeros_like(G)
        for g, R, X in (sols2 if curvature else ()):
            Cd = np.take_along_axis(Cbd[g.ch].reshape(len(g.ch), -1, self.D1), g.cols[:, None, :], axis=2)
            Cd = np.take_along_axis(Cd, g.rows[:, :, None], axis=1)
            Xd = -np.linalg.solve(R, Cd[:, :, :g.ndep] @ X + Cd[:, :, g.ndep:])
            Gd[g.dep] = Xd @ G[g.inp] + X @ Gd[g.inp]
        ev.Gd = Gd
        C = np.zeros((d, d))
        for k in range(K):
            n1 = self.d1[k]
            Gdk = Gd[self.off1[k]:self.off1[k + 1]]
            C += Gk[k].T @ (C1[k][:n1, :n1] @ Gk[k] + Gamma1[k][:n1, :n1] @ Gdk)
        ev.C = C
        return ev

    # ------------------------------------------------------------------ full lift (for checks and task maps)
    def lift(self, ev, derivative=False):
        """Dense one-level lift ``E = E1 G`` (n, d) and optionally its rate."""
        s = self.sys
        E = np.zeros((s.n, self.d))
        Ed = np.zeros_like(E) if derivative else None
        for k in range(s.K):
            md, sl = self.member_dofs[k], slice(self.off1[k], self.off1[k + 1])
            E[md] = ev.E1[md, :self.d1[k]] @ ev.G[sl]
            if derivative:
                Ed[md] = ev.E1d[md, :self.d1[k]] @ ev.G[sl] + ev.E1[md, :self.d1[k]] @ ev.Gd[sl]
        E[self.chart_dofs] = ev.G[self.off1[-1]:]
        if derivative:
            Ed[self.chart_dofs] = ev.Gd[self.off1[-1]:]
        return (E, Ed) if derivative else E

    def frame_jacobian(self, ev, body, P=None, rate=False):
        """World-aligned Jacobian of a frame on ``body`` in the independent speeds.

        Rows are ``[angular; linear]`` of the frame origin in world axes.  With
        ``rate`` the product ``Jdot u`` is returned as well.
        """
        s = self.sys
        k = s.member_of_body[body]
        sl = slice(self.off1[k], self.off1[k + 1])
        Tf = ev.kin.T[body] if P is None else ev.kin.T[body] @ P
        X = np.eye(6)
        X[3:, :3] = -se3.skew(Tf[:3, 3])
        Js = ev.Js1[body][:, :self.d1[k]] @ ev.G[sl]
        J = X @ Js
        if not rate:
            return J
        Jsd = ev.Jsd1[body][:, :self.d1[k]] @ ev.G[sl] + ev.Js1[body][:, :self.d1[k]] @ ev.Gd[sl]
        Vw = Js @ ev.u
        vel = Vw[3:] + np.cross(Vw[:3], Tf[:3, 3])
        Xd = np.zeros((6, 6))
        Xd[3:, :3] = -se3.skew(vel)
        return J, X @ (Jsd @ ev.u) + Xd @ Vw
