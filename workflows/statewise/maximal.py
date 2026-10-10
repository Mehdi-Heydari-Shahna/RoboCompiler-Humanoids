"""Maximal-coordinate constrained dynamics straight from the physical records.

This reference does not pass through the compiler: it uses no spanning tree, no cut,
no lift and no multibody library.  Every body of every record is a free rigid body
(Newton-Euler equations at its centre of mass, in world axes) and every *physical
joint* of the record is a set of holonomic constraints between its two attachment
frames, written from the record's ``T_AJ``, ``T_BJ`` and axis.  Hand links are point
coincidences of two palm sites and ideal supports weld a sole frame to a world
anchor.  Actuator efforts and drive data (rotor inertia, damping) act on the
coordinate of their joint, whichever role the compiler gave that joint.

Unknowns are ``x = [alpha_i; a_c,i]`` of every body: angular acceleration and
acceleration of the centre of mass in world axes.  With the constraint rows
``g'' = G x + gamma`` the accelerations solve

    [ M   G_r^T ] [ x  ]   [ rhs    ]
    [ G_r   0   ] [ -l ] = [ -gamma ],

for a maximal independent row set ``G_r``.

What is shared with the model under test: the state only (body poses and twists are
read from the kinematics of the compiled model at the sampled state) and the applied
efforts.  The position and velocity residuals of the record's joints at that state
are returned as well; they check the compiled kinematics against the record.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import qr

cross = np.cross


def skew(a):
    return np.array([[0., -a[2], a[1]], [a[2], 0., -a[0]], [-a[1], a[0], 0.]])


def tangents(n):
    """Two unit vectors that complete ``n`` to an orthonormal triad."""
    ref = np.array([1., 0., 0.]) if abs(n[0]) < 0.9 else np.array([0., 1., 0.])
    t1 = cross(n, ref)
    t1 /= np.linalg.norm(t1)
    return t1, cross(n, t1)


class MaximalSystem:
    """Free bodies, joints as constraints; build with ``add_record`` and solve once per state."""

    def __init__(self, gravity):
        self.g = np.asarray(gravity, float)
        self.bodies = {}                          # key -> dict(R, p, c, w, vc, slot)
        self.n = 0
        self.blocks = []                          # (slot, world inertia, mass, omega)
        self.rows, self.gamma, self.g0, self.g1 = [], [], [], []
        self.forces = []                          # (row, force along the row)
        self.rotors = []                          # (row, gamma, armature)
        self.bodies['__world__'] = dict(R=np.eye(3), p=np.zeros(3), c=np.zeros(3), w=np.zeros(3), vc=np.zeros(3), slot=None)

    # ------------------------------------------------------------------ bodies
    def add_body(self, key, record, T, V, fixed=False):
        """``T`` is the world pose of the body frame, ``V = [w; v_O]`` its world-frame twist."""
        R, p = T[:3, :3], T[:3, 3]
        if fixed:
            self.bodies[key] = dict(R=R, p=p, c=p.copy(), w=np.zeros(3), vc=np.zeros(3), slot=None)
            return
        c = p + R @ np.asarray(record['com_m'], float)
        w = V[:3]
        vc = V[3:] + cross(w, c)                  # v_O is the velocity of the body point at the world origin
        Iw = R @ np.asarray(record['inertia_com_kg_m2'], float) @ R.T
        self.bodies[key] = dict(R=R, p=p, c=c, w=w, vc=vc, slot=self.n)
        self.blocks.append((self.n, Iw, float(record['mass_kg']), w))
        self.n += 6

    # ------------------------------------------------------------------ elementary constraints
    # each returns (rows (m, n), gamma (m,), velocity residual (m,), position residual (m,))
    def _row(self, m=1):
        return np.zeros((m, self.n))

    def point(self, a, Pa, b, Pb):
        """Coincidence of the world points ``Pa`` (fixed in ``a``) and ``Pb`` (fixed in ``b``)."""
        A, B = self.bodies[a], self.bodies[b]
        ra, rb = Pa - A['c'], Pb - B['c']
        rows = self._row(3)
        if B['slot'] is not None:
            rows[:, B['slot']:B['slot'] + 3] = -skew(rb)
            rows[:, B['slot'] + 3:B['slot'] + 6] = np.eye(3)
        if A['slot'] is not None:
            rows[:, A['slot']:A['slot'] + 3] = skew(ra)
            rows[:, A['slot'] + 3:A['slot'] + 6] = -np.eye(3)
        gam = cross(B['w'], cross(B['w'], rb)) - cross(A['w'], cross(A['w'], ra))
        vel = (B['vc'] + cross(B['w'], rb)) - (A['vc'] + cross(A['w'], ra))
        return rows, gam, vel, Pb - Pa

    def perpendicular(self, a, ea, b, fb):
        """``ea . fb = 0`` for a direction ``ea`` fixed in ``a`` and ``fb`` fixed in ``b``."""
        A, B = self.bodies[a], self.bodies[b]
        row = self._row()
        if A['slot'] is not None:
            row[0, A['slot']:A['slot'] + 3] = cross(ea, fb)
        if B['slot'] is not None:
            row[0, B['slot']:B['slot'] + 3] = cross(fb, ea)
        wa, wb = A['w'], B['w']
        gam = cross(wa, cross(wa, ea)) @ fb + 2. * cross(wa, ea) @ cross(wb, fb) + ea @ cross(wb, cross(wb, fb))
        vel = cross(wa, ea) @ fb + ea @ cross(wb, fb)
        return row, np.array([gam]), np.array([vel]), np.array([ea @ fb])

    def along(self, a, ea, Pa, b, Pb):
        """``ea . (Pb - Pa)`` for a direction ``ea`` fixed in ``a``: a constraint or a prismatic coordinate."""
        A, B = self.bodies[a], self.bodies[b]
        ra, rb, D = Pa - A['c'], Pb - B['c'], Pb - Pa
        wa, wb = A['w'], B['w']
        Dd = (B['vc'] + cross(wb, rb)) - (A['vc'] + cross(wa, ra))
        row = self._row()
        if A['slot'] is not None:
            row[0, A['slot']:A['slot'] + 3] = cross(ea, D) - cross(ra, ea)
            row[0, A['slot'] + 3:A['slot'] + 6] = -ea
        if B['slot'] is not None:
            row[0, B['slot']:B['slot'] + 3] = cross(rb, ea)
            row[0, B['slot'] + 3:B['slot'] + 6] = ea
        gam = cross(wa, cross(wa, ea)) @ D + 2. * cross(wa, ea) @ Dd + ea @ (cross(wb, cross(wb, rb)) - cross(wa, cross(wa, ra)))
        vel = cross(wa, ea) @ D + ea @ Dd
        return row, np.array([gam]), np.array([vel]), np.array([ea @ D])

    def _add(self, item):
        rows, gam, vel, pos = item
        self.rows.append(rows); self.gamma.append(gam); self.g1.append(vel); self.g0.append(pos)

    # ------------------------------------------------------------------ joints of a record
    def joint(self, a, b, kind, T_AJ, T_BJ, axis=None, effort=0., damping=0., armature=0.):
        """Constraints of one physical joint; effort and drive data act on its coordinate."""
        A, B = self.bodies[a], self.bodies[b]
        TA, TB = np.asarray(T_AJ, float), np.asarray(T_BJ, float)
        Pa, Ra = A['p'] + A['R'] @ TA[:3, 3], A['R'] @ TA[:3, :3]
        Pb, Rb = B['p'] + B['R'] @ TB[:3, 3], B['R'] @ TB[:3, :3]
        if kind == 'spherical':
            self._add(self.point(a, Pa, b, Pb))
            return
        if kind == 'fixed':
            self._add(self.point(a, Pa, b, Pb))
            ex, ey, ez = np.eye(3)
            for u, v in ((ex, ey), (ey, ez), (ez, ex)):
                self._add(self.perpendicular(a, Ra @ u, b, Rb @ v))
            return
        ax = np.asarray(axis, float)
        t1, t2 = tangents(ax)
        na, nb = Ra @ ax, Rb @ ax
        if kind == 'revolute':
            self._add(self.point(a, Pa, b, Pb))
            self._add(self.perpendicular(a, Ra @ t1, b, nb))
            self._add(self.perpendicular(a, Ra @ t2, b, nb))
            row = np.zeros(self.n)                        # coordinate rate: n . (w_b - w_a)
            if B['slot'] is not None:
                row[B['slot']:B['slot'] + 3] = na
            if A['slot'] is not None:
                row[A['slot']:A['slot'] + 3] = -na
            gam = cross(A['w'], na) @ (B['w'] - A['w'])
            rate = na @ (B['w'] - A['w'])
        elif kind == 'prismatic':
            self._add(self.perpendicular(a, Ra @ t1, b, nb))
            self._add(self.perpendicular(a, Ra @ t2, b, nb))
            self._add(self.perpendicular(a, Ra @ t1, b, Rb @ t2))
            self._add(self.along(a, Ra @ t1, Pa, b, Pb))
            self._add(self.along(a, Ra @ t2, Pa, b, Pb))
            r, g, v, _ = self.along(a, na, Pa, b, Pb)       # coordinate: displacement along the axis
            row, gam, rate = r[0], g[0], v[0]
        else:
            raise ValueError(kind)
        if effort or damping:
            self.forces.append((row, float(effort) - float(damping) * rate))
        if armature:
            self.rotors.append((row, gam, float(armature)))

    def add_record(self, prefix, record, pose, twist, efforts):
        """All bodies and joints of one physical record.

        ``pose`` and ``twist`` map body ids to world poses and world-frame twists;
        ``efforts`` maps actuator ids to their effort.
        """
        ground = record.get('ground')
        for b in record['bodies']:
            self.add_body(prefix + b['id'], b, pose[b['id']], twist[b['id']], fixed=b['id'] == ground)
        self._pending = getattr(self, '_pending', [])
        drive = record.get('drive', {})
        act = {a['joint_id']: float(a.get('gear', 1.)) * efforts[a['id']] for a in record.get('actuators', [])}
        for j in record['joints']:
            d = drive.get(j['id'], {})
            self._pending.append(dict(a=prefix + j['body_a'], b=prefix + j['body_b'], kind=j['type'], T_AJ=j['T_AJ'], T_BJ=j['T_BJ'],
                                      axis=j.get('axis'), effort=act.get(j['id'], 0.), damping=d.get('damping', 0.),
                                      armature=d.get('armature', 0.)))
            # Coulomb friction ('frictionloss') acts in the simulator only; like the compiled model and the
            # other references, this reference compares the smooth dynamics and leaves it out.

    def link_points(self, a, point_a, b, point_b):
        """Point coincidence of two body-fixed points (a hand link)."""
        self._links = getattr(self, '_links', [])
        self._links.append((a, np.asarray(point_a, float), b, np.asarray(point_b, float)))

    def weld_to_world(self, body, T_frame_in_body, T_anchor):
        """Ideal support: a frame of ``body`` coincides with a world anchor."""
        self._welds = getattr(self, '_welds', [])
        self._welds.append((body, np.asarray(T_frame_in_body, float), np.asarray(T_anchor, float)))

    # ------------------------------------------------------------------ solve
    def solve(self, tol=1e-9):
        """Body accelerations and a report (rank, residuals of the record's joints at the state)."""
        for j in getattr(self, '_pending', []):               # rows need the final number of unknowns
            self.joint(**j)
        for a, pa, b, pb in getattr(self, '_links', []):
            A, B = self.bodies[a], self.bodies[b]
            self._add(self.point(a, A['p'] + A['R'] @ pa, b, B['p'] + B['R'] @ pb))
        for body, P, anchor in getattr(self, '_welds', []):
            self.joint('__world__', body, 'fixed', anchor, P)
        n = self.n
        M = np.zeros((n, n))
        rhs = np.zeros(n)
        for slot, Iw, mass, w in self.blocks:
            M[slot:slot + 3, slot:slot + 3] = Iw
            M[slot + 3:slot + 6, slot + 3:slot + 6] = mass * np.eye(3)
            rhs[slot:slot + 3] = -cross(w, Iw @ w)
            rhs[slot + 3:slot + 6] = mass * self.g
        for row, force in self.forces:
            rhs += row * force
        for row, gam, arm in self.rotors:                     # rotor: arm * (row . x + gam) along the coordinate
            M += arm * np.outer(row, row)
            rhs -= arm * gam * row
        G = np.vstack(self.rows) if self.rows else np.zeros((0, n))
        gam = np.concatenate(self.gamma) if self.gamma else np.zeros(0)
        norms = np.linalg.norm(G, axis=1)
        keep = np.flatnonzero(norms > 1e-12 * max(1e-300, norms.max())) if len(norms) else np.zeros(0, int)
        r = 0
        rows = np.zeros(0, int)
        if len(keep):
            Gn = G[keep] / norms[keep][:, None]
            _, Rq, piv = qr(Gn.T, pivoting=True, mode='economic')
            d = abs(np.diag(Rq))
            r = int(np.sum(d > tol * d[0]))
            rows = np.sort(keep[piv[:r]])
        K = np.block([[M, G[rows].T], [G[rows], np.zeros((r, r))]])
        b = np.r_[rhs, -gam[rows]]
        y = np.linalg.solve(K, b)
        Kl = K.astype(np.longdouble)                           # one step of iterative refinement
        y = y + np.linalg.solve(K, np.asarray(b.astype(np.longdouble) - Kl @ y.astype(np.longdouble), float))
        x = y[:n]
        report = dict(unknowns=int(n), rows=int(G.shape[0]), rank=int(r),
                      position_residual=float(np.max(abs(np.concatenate(self.g0)))) if self.g0 else 0.,
                      velocity_residual=float(np.max(abs(np.concatenate(self.g1)))) if self.g1 else 0.,
                      acceleration_residual_all_rows=float(np.max(abs(G @ x + gam))) if len(gam) else 0.)
        return x, report


# ---------------------------------------------------------------------- interface to the state-wise study
def records_of(case):
    """(member name, physical record) of every member of a study case."""
    if case.ens is not None:
        return [(m.name, r) for m, r in zip(case.sys.members, case.ens.records)]
    return [(m.name, m.comp.physical) for m in case.sys.members]


def body_accelerations_of_model(case, kin, a_full):
    """``[alpha; a_c]`` of the physical bodies (record order) from the tree acceleration of the compiled model."""
    s = case.sys
    Aw = s.accumulate(kin.S, a_full[:, None])[:, :, 0] + s.accumulate(kin.Sdot, kin.v[:, None])[:, :, 0]
    out = []
    for name, record in records_of(case):
        ground = record.get('ground')
        for b in record['bodies']:
            if b['id'] == ground:
                continue
            i = s.body_index[f"{name}/{b['id']}"]
            T = kin.T[i]
            c = T[:3, 3] + T[:3, :3] @ s.com[i]
            w = kin.Vs[i][:3]
            vc = kin.Vs[i][3:] + cross(w, c)
            out.append(np.r_[Aw[i][:3], Aw[i][3:] + cross(Aw[i][:3], c) + cross(w, vc)])
    return np.concatenate(out)


def reference(case, kin, f):
    """Maximal-coordinate body accelerations for the state in ``kin`` (with velocities) and efforts ``f``."""
    s, comp = case.sys, case.comp
    mx = MaximalSystem(s.gravity)
    for name, record in records_of(case):
        pose = {b['id']: kin.T[s.body_index[f"{name}/{b['id']}"]] for b in record['bodies']}
        twist = {b['id']: kin.Vs[s.body_index[f"{name}/{b['id']}"]] for b in record['bodies']}
        efforts = {a['id']: f[s.port_ids.index(f"{name}/{a['id']}")] for a in record.get('actuators', [])}
        mx.add_record(name + '/', record, pose, twist, efforts)
    by_id = {l['id']: l for l in s.links}
    sites = {}
    for name, record in records_of(case):
        for site in record.get('point_sites', []):
            sites[(name, site['id'])] = site
    for lid in comp.links:
        l = by_id[lid]
        s1, s2 = sites[(l['member1'], l['site1'])], sites[(l['member2'], l['site2'])]
        mx.link_points(f"{l['member1']}/{s1['body']}", s1['point_m'], f"{l['member2']}/{s2['body']}", s2['point_m'])
    for i, sid in enumerate(comp.mode.support_ids):
        member, site = sid.split('/')
        b, P = s.site_frame(member, site)
        mx.weld_to_world(s.body_names[b], P, comp.mode.anchors[i])
    return mx.solve()


def compare(case, kin, a_full, f):
    """Normalised difference of the body accelerations of the compiled model and the maximal-coordinate reference."""
    x_ref, report = reference(case, kin, f)
    x = body_accelerations_of_model(case, kin, a_full)
    report['delta_body'] = float(np.max(abs(x - x_ref)) / max(1., np.max(abs(x_ref))))
    return report
