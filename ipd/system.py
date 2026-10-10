"""Numerical multibody system assembled from compiled mechanism graphs.

A ``System`` joins one or more compiled mechanisms (``ipd.compiler.Compilation``) and
the interconnections between them (hand links, grasps) into one augmented forest:

* every physical body and every retained cut-joint coordinate is a tree node, the
  cut coordinates living on massless frames, so each chord is a 6-row SE(3)
  comparison of two frames;
* the velocity vector is ``v = [tree speeds of all members, cut speeds, link chart
  speeds]``; a floating root contributes its body twist ``[w; v]`` (not the
  derivative of any coordinate), a scalar joint contributes its rate;
* kinematics is evaluated level by level on stacked arrays: body poses ``T``, world
  screws ``S`` of all coordinates (``V_b^world = sum_{j <= b} S_j v_j``), their rates
  ``Sdot_j = V_child(j) x S_j`` and tree-accumulated sums.

The closure differential of chord ``e`` is path-assembled (PACDM):

    C_e = inv_left(rho_e) Ad(inv(T_e^-)) S diag(sigma_e),

with ``sigma_e`` the signed ancestor incidence of the chord.  Its time derivative
along a motion has the same incidence, with ``S`` replaced by
``Sdot - ad(V_e^-) S``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from . import se3
from .compiler import Compilation, ModelError

FIXED, REVOLUTE, PRISMATIC, ROOT, BASE = 0, 1, 2, 3, 4


@dataclass
class Member:
    """One compiled mechanism placed in the world."""
    name: str
    comp: Compilation
    T_root: np.ndarray


class Config:
    """Configuration: root poses and scalar coordinates aligned with the velocity index.

    ``theta`` has one entry per velocity coordinate; the six entries of a floating
    root are unused (its pose is ``T_root[k]``).
    """
    __slots__ = ('T_root', 'theta')

    def __init__(self, T_root, theta):
        self.T_root = np.array(T_root, float)
        self.theta = np.array(theta, float)

    def copy(self):
        return Config(self.T_root.copy(), self.theta.copy())


class Kinematics:
    """Result of one kinematics pass (see ``System.kinematics``)."""
    __slots__ = ('q', 'T', 'S', 'v', 'Vs', 'Sdot')


class System:
    def __init__(self, members, links=(), name='system'):
        self.name = name
        self.members = list(members)
        self.K = len(self.members)
        if len({m.name for m in self.members}) != self.K:
            raise ModelError('Duplicate member name')
        self.gravity = np.asarray(self.members[0].comp.cmg['gravity_m_s2'], float)
        ids, cut_ids = [], []
        self.tree_slice, self.cut_slice, self.root_slice, self.floating = [], [], [], []
        for m in self.members:
            c = m.comp.cmg
            start = len(ids)
            ids += [f'{m.name}/{x}' for x in c['coordinate_ids']]
            self.tree_slice.append(slice(start, len(ids)))
            self.floating.append(bool(c['floating']))
            self.root_slice.append(slice(start, start + 6) if c['floating'] else None)
        self.nT = len(ids)
        for m in self.members:
            start = self.nT + len(cut_ids)
            cut_ids += [f'{m.name}/{x}' for x in m.comp.cmg['cut_coordinate_ids']]
            self.cut_slice.append(slice(start, self.nT + len(cut_ids)))
        self.links = [dict(l) for l in links]
        link_ids = []
        for l in self.links:
            link_ids += [f"{l['id']}__cut_{a}" for a in 'xyz']
        self.ids = ids + cut_ids + link_ids
        self.n = len(self.ids)
        self.index = {c: i for i, c in enumerate(self.ids)}
        if len(self.index) != self.n:
            raise ModelError('Duplicate coordinate id')
        self._build_bodies()
        self._build_chords()
        self._build_levels()
        self._build_actuation()

    # ------------------------------------------------------------------ structure
    def _build_bodies(self):
        names, parent, jtype, dof, A, Binv, xi = [], [], [], [], [], [], []
        mass, com, inertia, member_of = [], [], [], []
        self.body_index = {}
        self.member_bodies = []
        for k, m in enumerate(self.members):
            c = m.comp.cmg
            first = len(names)
            edge = {e['follower_body']: e for e in c['joints']}
            records = {b['id']: b for b in c['bodies']}
            for bid in c['body_order']:
                gid = f'{m.name}/{bid}'
                self.body_index[gid] = len(names)
                names.append(gid)
                b = records[bid]
                massless = bool(b.get('massless', False))
                mass.append(0. if massless else float(b['mass_kg']))
                com.append(np.zeros(3) if massless else np.asarray(b['com_m'], float))
                inertia.append(np.zeros((3, 3)) if massless else np.asarray(b['inertia_com_kg_m2'], float))
                member_of.append(k)
                if bid == c['root_body']:
                    parent.append(-1)
                    jtype.append(ROOT if c['floating'] else BASE)
                    dof.append(self.tree_slice[k].start if c['floating'] else -1)
                    A.append(np.eye(4)); Binv.append(np.eye(4)); xi.append(np.zeros(6))
                    continue
                e = edge[bid]
                parent.append(self.body_index[f"{m.name}/{e['base_body']}"])
                A.append(np.asarray(e['T_BJ'], float))
                Binv.append(se3.inv(np.asarray(e['T_FJ'], float)))
                if e['type'] == 'fixed':
                    jtype.append(FIXED); dof.append(-1); xi.append(np.zeros(6))
                else:
                    jtype.append(REVOLUTE if e['type'] == 'revolute' else PRISMATIC)
                    dof.append(self.index[f"{m.name}/{e['id']}"])
                    axis = np.asarray(e['axis'], float)
                    xi.append(np.r_[axis, 0, 0, 0] if e['type'] == 'revolute' else np.r_[0, 0, 0, axis])
            self.member_bodies.append(np.arange(first, len(names)))
        self.nbp = len(names)
        self._virtual = dict(names=[], parent=[], jtype=[], dof=[], A=[], xi=[])
        self._phys = dict(names=names, parent=parent, jtype=jtype, dof=dof, A=A, Binv=Binv, xi=xi)
        self.mass = np.array(mass)
        self.com = np.array(com)
        self.inertia_com = np.array(inertia)
        self.member_of_body = np.array(member_of)
        self.M = se3.spatial_inertia(self.mass, self.com, self.inertia_com)

    def _add_virtual(self, name, parent, jtype, dof, A, xi):
        v = self._virtual
        v['names'].append(name); v['parent'].append(parent); v['jtype'].append(jtype)
        v['dof'].append(dof); v['A'].append(np.asarray(A, float)); v['xi'].append(np.asarray(xi, float))
        return self.nbp + len(v['names']) - 1

    def _build_chords(self):
        """Chord table: internal cuts of every member, then inter-member links."""
        ch = dict(id=[], kind=[], b1=[], P1=[], b2=[], P2=[], member=[], cut_dofs=[], ref_body=[], ref_slot=[])
        ex = np.r_[1., 0, 0, 0, 0, 0]; ey = np.r_[0, 1., 0, 0, 0, 0]; ez = np.r_[0, 0, 1., 0, 0, 0]

        def chain(prefix, body1, T1, dofs, kind, axis):
            """Massless cut-joint frames attached to ``body1`` at ``T1``; returns the last frame."""
            if kind == 'fixed':
                return body1, T1, -1
            if kind in ('revolute', 'prismatic'):
                a = np.asarray(axis, float)
                tw = np.r_[a, 0, 0, 0] if kind == 'revolute' else np.r_[0, 0, 0, a]
                b = self._add_virtual(prefix, body1, REVOLUTE if kind == 'revolute' else PRISMATIC, dofs[0], T1, tw)
                return b, np.eye(4), -1
            bx = self._add_virtual(prefix + '_x', body1, REVOLUTE, dofs[0], T1, ex)
            by = self._add_virtual(prefix + '_y', bx, REVOLUTE, dofs[1], np.eye(4), ey)
            bz = self._add_virtual(prefix + '_z', by, REVOLUTE, dofs[2], np.eye(4), ez)
            return bz, np.eye(4), bx

        for k, m in enumerate(self.members):
            for c in m.comp.cmg['closures']:
                dofs = [self.index[f'{m.name}/{x}'] for x in c['cut_coordinates']]
                b1 = self.body_index[f"{m.name}/{c['body1']}"]
                b2 = self.body_index[f"{m.name}/{c['body2']}"]
                last, P1, ref = chain(f"{m.name}/{c['id']}__frame", b1, np.asarray(c['T_1J'], float), dofs, c['type'], c['axis'])
                ch['id'].append(f"{m.name}/{c['id']}"); ch['kind'].append(c['type']); ch['b1'].append(last)
                ch['P1'].append(P1); ch['b2'].append(b2); ch['P2'].append(np.asarray(c['T_2J'], float))
                ch['member'].append(k); ch['cut_dofs'].append(dofs); ch['ref_body'].append(ref); ch['ref_slot'].append(b1)
        self.n_internal = len(ch['id'])
        names = {m.name: k for k, m in enumerate(self.members)}
        for l in self.links:
            k1, k2 = names[l['member1']], names[l['member2']]
            s1 = self._site(k1, l['site1']); s2 = self._site(k2, l['site2'])
            dofs = [self.index[f"{l['id']}__cut_{a}"] for a in 'xyz']
            T1, T2 = np.eye(4), np.eye(4)
            T1[:3, 3], T2[:3, 3] = s1['point_m'], s2['point_m']
            b1 = self.body_index[f"{l['member1']}/{s1['body']}"]
            b2 = self.body_index[f"{l['member2']}/{s2['body']}"]
            last, P1, ref = chain(f"{l['id']}__frame", b1, T1, dofs, 'spherical', None)
            ch['id'].append(l['id']); ch['kind'].append('spherical'); ch['b1'].append(last); ch['P1'].append(P1)
            ch['b2'].append(b2); ch['P2'].append(T2); ch['member'].append(-1); ch['cut_dofs'].append(dofs)
            ch['ref_body'].append(ref); ch['ref_slot'].append(b1)
        self.chord_ids = ch['id']
        self.chord_index = {c: i for i, c in enumerate(ch['id'])}
        self.chord_kind = ch['kind']
        self.chord_member = np.array(ch['member'], int)
        self.chord_cut_dofs = ch['cut_dofs']
        self.chord_ref_body = ch['ref_body']
        self.chord_ref_slot = ch['ref_slot']
        self._chord_b1 = np.array(ch['b1'], int)
        self._chord_b2 = np.array(ch['b2'], int)
        self._chord_P1 = np.array(ch['P1'], float).reshape(-1, 4, 4)
        self._chord_P2 = np.array(ch['P2'], float).reshape(-1, 4, 4)
        self.nch = len(ch['id'])

    def _site(self, k, site_id):
        for key in ('point_sites', 'sole_sites'):
            for s in self.members[k].comp.cmg[key]:
                if s['id'] == site_id:
                    return s
        raise ModelError(f'Unknown site {site_id} on member {self.members[k].name}')

    def _build_levels(self):
        p, v = self._phys, self._virtual
        self.body_names = p['names'] + v['names']
        self.nb = len(self.body_names)
        self.W = self.nb                      # index of the world frame in pose arrays
        parent = np.array(p['parent'] + v['parent'], int)
        self.parent = np.where(parent < 0, self.W, parent)
        self.jtype = np.array(p['jtype'] + v['jtype'], int)
        self.dof = np.array(p['dof'] + v['dof'], int)
        self.A = np.array(p['A'] + v['A'], float).reshape(-1, 4, 4)
        self.Binv = np.array(p['Binv'] + [np.eye(4)] * len(v['names']), float).reshape(-1, 4, 4)
        self.xi = np.array(p['xi'] + v['xi'], float).reshape(-1, 6)
        self.Xfixed = self.A @ self.Binv
        level = np.zeros(self.nb, int)
        for b in range(self.nb):              # parents precede children by construction
            level[b] = 0 if self.parent[b] == self.W else level[self.parent[b]] + 1
            if self.parent[b] != self.W and self.parent[b] >= b:
                raise ModelError('Body order is not topological')
        self.level = level
        self.rev = np.flatnonzero(self.jtype == REVOLUTE)
        self.pri = np.flatnonzero(self.jtype == PRISMATIC)
        self.roots = np.flatnonzero(self.jtype == ROOT)
        self.root_member = self.member_of_body[self.roots] if len(self.roots) else np.zeros(0, int)
        self.bases = np.flatnonzero(self.jtype == BASE)
        self.base_member = self.member_of_body[self.bases] if len(self.bases) else np.zeros(0, int)
        self.scalar = np.flatnonzero((self.jtype == REVOLUTE) | (self.jtype == PRISMATIC))
        self.scalar_dof = self.dof[self.scalar]
        self.dof_child = np.full(self.n, -1, int)
        self.dof_child[self.scalar_dof] = self.scalar
        for b in self.roots:
            self.dof_child[self.dof[b]:self.dof[b] + 6] = b
        self.levels = []                      # per level: (all bodies, scalar bodies, their dofs, fixed bodies)
        for lv in range(1, level.max() + 1 if self.nb else 0):
            idx = np.flatnonzero(level == lv)
            sc = idx[(self.jtype[idx] == REVOLUTE) | (self.jtype[idx] == PRISMATIC)]
            fx = idx[self.jtype[idx] == FIXED]
            self.levels.append((idx, sc, self.dof[sc], fx))
        self.levels_phys = []                 # the same, restricted to physical bodies
        for idx, sc, dofs, fx in self.levels:
            pi, ps = idx[idx < self.nbp], sc[sc < self.nbp]
            if len(pi):
                self.levels_phys.append((pi, ps, self.dof[ps], fx[fx < self.nbp]))
        # Ancestor incidence: anc[b, j] is True when coordinate j moves body b.
        anc = np.zeros((self.nb + 1, self.n), bool)
        for b in range(self.nb):
            if self.parent[b] != self.W:
                anc[b] = anc[self.parent[b]]
            if self.jtype[b] in (REVOLUTE, PRISMATIC):
                anc[b, self.dof[b]] = True
            elif self.jtype[b] == ROOT:
                anc[b, self.dof[b]:self.dof[b] + 6] = True
        self.anc = anc
        b1 = np.where(self._chord_b1 < 0, self.W, self._chord_b1)
        self.chord_b1, self.chord_b2 = b1, self._chord_b2
        self.chord_P1, self.chord_P2 = self._chord_P1, self._chord_P2
        self.sigma = anc[self.chord_b2].astype(np.int8) - anc[b1].astype(np.int8)
        # Joint limits (branch bounds) for every scalar coordinate.
        self.lower = np.full(self.n, -np.pi); self.upper = np.full(self.n, np.pi)
        for k, m in enumerate(self.members):
            for e in m.comp.cmg['joints']:
                if e['type'] != 'fixed':
                    i = self.index[f"{m.name}/{e['id']}"]
                    self.lower[i], self.upper[i] = e['limits']['lower'], e['limits']['upper']
            for c in m.comp.cmg['closures']:
                if c['type'] in ('revolute', 'prismatic') and c['limits']:
                    i = self.index[f"{m.name}/{c['cut_coordinates'][0]}"]
                    self.lower[i], self.upper[i] = c['limits']['lower'], c['limits']['upper']
                elif c['type'] == 'prismatic':
                    i = self.index[f"{m.name}/{c['cut_coordinates'][0]}"]
                    self.lower[i], self.upper[i] = -np.inf, np.inf
        for sl in self.root_slice:
            if sl is not None:
                self.lower[sl], self.upper[sl] = -np.inf, np.inf

    def _build_actuation(self):
        """Actuator ports: ``ldot = B^T v_T`` and tree effort ``tau_T = B f``."""
        ports, rows, cols, gear, lo, hi = [], [], [], [], [], []
        self.armature = np.zeros(self.nT); self.damping = np.zeros(self.nT); self.frictionloss = np.zeros(self.nT)
        self.member_ports = []
        for k, m in enumerate(self.members):
            c = m.comp.cmg
            first = len(ports)
            for a in c['actuators']:
                ports.append(f"{m.name}/{a['id']}")
                rows.append(self.index[f"{m.name}/{a['joint_id']}"]); cols.append(len(ports) - 1)
                gear.append(float(a.get('gear', 1.))); lo.append(a['effort_bounds'][0]); hi.append(a['effort_bounds'][1])
            self.member_ports.append(np.arange(first, len(ports)))
            for cid in c['coordinate_ids'][6 if c['floating'] else 0:]:
                i = self.index[f'{m.name}/{cid}']
                self.armature[i] = c['armature'][cid]; self.damping[i] = c['damping'][cid]
                self.frictionloss[i] = c['frictionloss'][cid]
        self.port_ids = ports
        self.na = len(ports)
        self.B = np.zeros((self.nT, self.na))
        self.B[rows, cols] = gear
        self.port_dof = np.array(rows, int)
        self.effort_lower = np.array(lo, float); self.effort_upper = np.array(hi, float)

    # ------------------------------------------------------------------ configuration
    def neutral(self):
        """Seed configuration: member seed poses and seed joint values, cut coordinates lifted."""
        theta = np.zeros(self.n)
        for m in self.members:
            for jid, value in m.comp.cmg['seed'].get('joints', {}).items():
                i = self.index.get(f'{m.name}/{jid}')      # seeds of joints that became chords are lifted
                if i is not None:
                    theta[i] = value
        q = Config(np.array([m.T_root for m in self.members], float), theta)
        self.set_references(q)
        return self.lift(q)

    def integrate(self, q, dv):
        """Retraction ``q (+) dv``: root poses move by the body-twist exponential."""
        out = Config(q.T_root.copy(), q.theta + dv)
        for k, sl in enumerate(self.root_slice):
            if sl is not None:
                out.theta[sl] = 0.
                if np.any(dv[sl]):
                    out.T_root[k] = q.T_root[k] @ se3.exp(dv[sl])
        return out

    def difference(self, q0, q1):
        """Velocity-coordinate displacement ``d`` with ``q0 (+) d = q1``."""
        d = q1.theta - q0.theta
        for k, sl in enumerate(self.root_slice):
            if sl is not None:
                d[sl] = se3.log(se3.inv(q0.T_root[k]) @ q1.T_root[k])
        return d

    def set_references(self, q):
        """Re-centre the rotation chart of every spherical chord at configuration ``q``."""
        if not hasattr(self, '_A0'):
            self._A0 = self.A.copy()
        T = self.poses(q)
        for e in range(self.nch):
            bx = self.chord_ref_body[e]
            if bx < 0:
                continue
            b1 = self.chord_ref_slot[e]
            T1 = T[b1] @ self._A0[bx]
            T2 = T[self.chord_b2[e]] @ self.chord_P2[e]
            A = self._A0[bx].copy()
            A[:3, :3] = self._A0[bx][:3, :3] @ (T1[:3, :3].T @ T2[:3, :3])
            self.A[bx] = A
        self.Xfixed = self.A @ self.Binv

    def lift(self, q, T=None):
        """Set the cut coordinates from the current body poses (no physical coordinate changes)."""
        q = q.copy()
        T = self.poses(q) if T is None else T
        if not hasattr(self, '_lift_index'):
            groups = {'spherical': [], 'revolute': [], 'prismatic': []}
            for e in range(self.nch):
                if self.chord_cut_dofs[e]:
                    groups[self.chord_kind[e]].append(e)
            self._lift_index = {}
            for kind, es in groups.items():
                first = np.array([self.dof_child[self.chord_cut_dofs[e][0]] for e in es], int)
                ncut = 3 if kind == 'spherical' else 1
                self._lift_index[kind] = (np.array(es, int), first, self.parent[first] if len(es) else first,
                                          np.array([self.chord_cut_dofs[e] for e in es], int).reshape(len(es), ncut))
        for kind, (es, first, par, dofs) in self._lift_index.items():
            if not len(es):
                continue
            rel = se3.inv(T[par] @ self.A[first]) @ (T[self.chord_b2[es]] @ self.chord_P2[es])
            if kind == 'spherical':
                q.theta[dofs] = Rotation.from_matrix(rel[:, :3, :3]).as_euler('XYZ')
            elif kind == 'revolute':
                q.theta[dofs[:, 0]] = np.einsum('ei,ei->e', self.xi[first, :3], se3.log_so3(rel[:, :3, :3]))
            else:
                q.theta[dofs[:, 0]] = np.einsum('ei,ei->e', self.xi[first, 3:], rel[:, :3, 3])
        return q

    # ------------------------------------------------------------------ kinematics
    def poses(self, q):
        """World poses of all bodies; the extra last entry is the world frame."""
        th = q.theta
        X = self.Xfixed.copy()
        if len(self.rev):
            Q = np.zeros((len(self.rev), 4, 4))
            Q[:, :3, :3] = se3.rot_axis(self.xi[self.rev, :3], th[self.dof[self.rev]])
            Q[:, 3, 3] = 1.
            X[self.rev] = self.A[self.rev] @ Q @ self.Binv[self.rev]
        if len(self.pri):
            Q = np.tile(np.eye(4), (len(self.pri), 1, 1))
            Q[:, :3, 3] = self.xi[self.pri, 3:] * th[self.dof[self.pri], None]
            X[self.pri] = self.A[self.pri] @ Q @ self.Binv[self.pri]
        T = np.empty((self.nb + 1, 4, 4))
        T[self.W] = np.eye(4)
        if len(self.roots):
            T[self.roots] = q.T_root[self.root_member]
        if len(self.bases):
            T[self.bases] = q.T_root[self.base_member]
        for idx, _, _, _ in self.levels:
            T[idx] = T[self.parent[idx]] @ X[idx]
        return T

    def kinematics(self, q, v=None):
        """Poses and world screws; with ``v`` also world body twists and screw rates."""
        k = Kinematics()
        k.q = q
        T = k.T = self.poses(q)
        S = np.zeros((6, self.n))
        F = T[self.parent[self.scalar]] @ self.A[self.scalar]      # joint frames before the joint motion
        axis_w = np.einsum('bij,bj->bi', F[:, :3, :3], self.xi[self.scalar, :3])
        axis_v = np.einsum('bij,bj->bi', F[:, :3, :3], self.xi[self.scalar, 3:])
        S[:3, self.scalar_dof] = axis_w.T
        S[3:, self.scalar_dof] = (se3.cross3(F[:, :3, 3], axis_w) + axis_v).T
        for b in self.roots:
            S[:, self.dof[b]:self.dof[b] + 6] = se3.Ad(T[b])
        k.S = S
        k.v = k.Vs = k.Sdot = None
        return k if v is None else self.velocity(k, v)

    def velocity(self, k, v):
        """Add world body twists and screw rates for the velocity ``v`` to a kinematics result."""
        v = np.asarray(v, float)
        S = k.S
        k.v = v
        Vs = np.zeros((self.nb + 1, 6))
        for b in self.roots:
            Vs[b] = S[:, self.dof[b]:self.dof[b] + 6] @ v[self.dof[b]:self.dof[b] + 6]
        for idx, sc, dofs, fx in self.levels:
            Vs[idx] = Vs[self.parent[idx]]
            if len(sc):
                Vs[sc] += (S[:, dofs] * v[dofs]).T
        k.Vs = Vs
        Sdot = np.zeros((6, self.n))
        Sdot[:, self.scalar_dof] = se3.cross_motion(Vs[self.scalar], S[:, self.scalar_dof].T).T
        for b in self.roots:
            sl = slice(self.dof[b], self.dof[b] + 6)
            Sdot[:, sl] = se3.ad(Vs[b]) @ S[:, sl]
        k.Sdot = Sdot
        return k

    def accumulate(self, S, X, physical=True):
        """Tree-accumulated sums ``out[b] = sum_{j <= b} S[:, j] X[j, :]``.

        ``S`` is (6, n) and ``X`` is (n, d).  The result is the world-frame Jacobian of
        every body with respect to whatever speeds parametrize the rows of ``X``; it
        has shape (nbp, 6, d) for the physical bodies, or (nb + 1, 6, d) for all
        frames with the world entry zero.  Rows of ``X`` may use member-local columns.
        """
        d = X.shape[1]
        out = np.zeros((self.nbp if physical else self.nb + 1, 6, d))
        for b in self.roots:
            sl = slice(self.dof[b], self.dof[b] + 6)
            out[b] = S[:, sl] @ X[sl]
        for idx, sc, dofs, fx in (self.levels_phys if physical else self.levels):
            out[idx] = out[self.parent[idx]]
            if len(sc):
                out[sc] += S[:, dofs].T[:, :, None] * X[dofs][:, None, :]
        return out

    # ------------------------------------------------------------------ chords
    def chord_frames(self, T, chords=None, anchors=None, defects=None):
        """Residuals and differential prefactors of chords.

        Returns ``rho`` (m, 6), ``A`` (m, 6, 6) with ``C_e = A_e S diag(sigma_e)``,
        the comparison ``Delta`` (m, 4, 4) and the minus frames ``Tm``.
        """
        ch = np.arange(self.nch) if chords is None else np.asarray(chords, int)
        Tm = T[self.chord_b1[ch]] @ self.chord_P1[ch]
        if defects is not None:
            Tm = Tm @ defects
        Tp = T[self.chord_b2[ch]] @ self.chord_P2[ch]
        D = se3.inv(Tm) @ Tp
        rho = se3.log(D)
        return rho, se3.inv_left(rho) @ se3.Ad_inv(Tm), D, Tm

    def body_twists(self, kin):
        """Body-frame twists ``V_b = Ad(inv(T_b)) V_b^world`` of the physical bodies."""
        return np.einsum('bij,bj->bi', se3.Ad_inv(kin.T[:self.nbp]), kin.Vs[:self.nbp])

    def site_frame(self, member, site_id):
        """(body index, constant frame in the body) of a named sole or point site."""
        k = [m.name for m in self.members].index(member)
        s = self._site(k, site_id)
        P = np.eye(4)
        if 'frame_translation_m' in s:
            P[:3, 3] = s['frame_translation_m']; P[:3, :3] = s['frame_rotation']
        else:
            P[:3, 3] = s['point_m']
        return self.body_index[f"{member}/{s['body']}"], P
