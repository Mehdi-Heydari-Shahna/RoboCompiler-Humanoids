"""Pinocchio view of a compiled ``System`` (independent recursive and constrained dynamics).

The Pinocchio model is built programmatically from the same compiled records: one
free-flyer per floating root, one unaligned scalar joint per tree coordinate, bodies
on fixed joints merged onto their supporting joint.  Cut coordinates are not part of
this model; chords become rigid constraint models (3-D point pairs for pins and
spherical connections, 6-D welds for fixed chords and supports).

Velocity conventions: the core uses the root body twist ``[w; v]`` in the root frame,
Pinocchio uses ``[v; w]`` in the same frame, so the two tree velocity vectors differ
by a constant permutation ``v_pin = P v_T``.
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin

from .. import se3
from ..system import BASE, FIXED, PRISMATIC, REVOLUTE, ROOT


def _se3(T):
    return pin.SE3(np.array(T[:3, :3]), np.array(T[:3, 3]))


class PinBackend:
    def __init__(self, system, armature=True):
        s = self.sys = system
        model = pin.Model()
        model.gravity = pin.Motion(np.r_[s.gravity, 0., 0., 0.])
        joint_of = np.zeros(s.nbp, int)            # supporting Pinocchio joint of each body
        Mjb = np.zeros((s.nbp, 4, 4))              # body frame expressed in that joint frame
        idx_v = np.full(s.nT, -1, int)
        idx_q = np.full(s.nT, -1, int)
        self.root_joint = {}
        for b in range(s.nbp):
            name = s.body_names[b]
            inertia = pin.Inertia(float(s.mass[b]), np.array(s.com[b]), np.array(s.inertia_com[b]))
            t = s.jtype[b]
            if t == ROOT:
                k = s.member_of_body[b]
                jid = model.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), name)
                joint_of[b], Mjb[b] = jid, np.eye(4)
                self.root_joint[k] = jid
                iv, iq = model.joints[jid].idx_v, model.joints[jid].idx_q
                sl = s.root_slice[k]
                idx_v[sl] = np.r_[iv + 3, iv + 4, iv + 5, iv, iv + 1, iv + 2]   # core [w; v] -> pin [v; w]
                idx_q[sl] = -2
            elif t == BASE:
                k = s.member_of_body[b]
                joint_of[b], Mjb[b] = 0, np.array(s.members[k].T_root, float)
            elif t == FIXED:
                p = s.parent[b]
                joint_of[b], Mjb[b] = joint_of[p], Mjb[p] @ s.Xfixed[b]
            else:
                p = s.parent[b]
                axis = s.xi[b, :3] if t == REVOLUTE else s.xi[b, 3:]
                jm = pin.JointModelRevoluteUnaligned(np.array(axis)) if t == REVOLUTE else \
                    pin.JointModelPrismaticUnaligned(np.array(axis))
                jid = model.addJoint(int(joint_of[p]), jm, _se3(Mjb[p] @ s.A[b]), name)
                joint_of[b], Mjb[b] = jid, s.Binv[b]
                idx_v[s.dof[b]] = model.joints[jid].idx_v
                idx_q[s.dof[b]] = model.joints[jid].idx_q
            if s.mass[b] > 0:
                model.appendBodyToJoint(int(joint_of[b]), inertia, _se3(Mjb[b]))
        self.model, self.joint_of, self.Mjb = model, joint_of, Mjb
        self.idx_v, self.idx_q = idx_v, idx_q
        if model.nv != s.nT:
            raise ValueError('Pinocchio model and compiled tree have different velocity dimensions')
        self.P = np.zeros((model.nv, s.nT))        # v_pin = P v_T
        self.P[idx_v, np.arange(s.nT)] = 1.
        self.armature = s.armature.copy() if armature else np.zeros(s.nT)
        self.data = model.createData()

    # ------------------------------------------------------------------ state maps
    def q_pin(self, q):
        s = self.sys
        out = pin.neutral(self.model)
        for k, jid in self.root_joint.items():
            iq = self.model.joints[jid].idx_q
            T = q.T_root[k]
            out[iq:iq + 3] = T[:3, 3]
            out[iq + 3:iq + 7] = se3.quat_xyzw_from_matrix(T[:3, :3])
        sc = self.idx_q >= 0
        out[self.idx_q[sc]] = q.theta[:s.nT][sc]
        return out

    def v_pin(self, vT):
        return self.P @ np.asarray(vT, float)[:self.sys.nT]

    # ------------------------------------------------------------------ recursive tree dynamics
    def mass(self, q):
        """Tree mass matrix in the core velocity order (CRBA), armature included."""
        M = pin.crba(self.model, self.data, self.q_pin(q))
        M = np.triu(M) + np.triu(M, 1).T
        return self.P.T @ M @ self.P + np.diag(self.armature)

    def bias(self, q, vT):
        """Coriolis, centrifugal and gravity generalized force (RNEA with zero acceleration)."""
        return self.P.T @ pin.nonLinearEffects(self.model, self.data, self.q_pin(q), self.v_pin(vT))

    def gravity(self, q):
        return self.P.T @ pin.computeGeneralizedGravity(self.model, self.data, self.q_pin(q))

    def inverse(self, q, vT, aT):
        tau = pin.rnea(self.model, self.data, self.q_pin(q), self.v_pin(vT), self.v_pin(aT))
        return self.P.T @ tau + self.armature * np.asarray(aT)[:self.sys.nT]

    def coriolis(self, q, vT):
        C = pin.computeCoriolisMatrix(self.model, self.data, self.q_pin(q), self.v_pin(vT))
        return self.P.T @ C @ self.P

    def body_pose(self, q, b):
        """World pose of physical body ``b`` from Pinocchio forward kinematics."""
        pin.forwardKinematics(self.model, self.data, self.q_pin(q))
        return self.data.oMi[int(self.joint_of[b])].homogeneous @ self.Mjb[b]

    # ------------------------------------------------------------------ constraints
    def constraint_models(self, mode):
        """Rigid constraint models for all chords of ``mode`` and their row layout."""
        s = self.sys
        models, layout = [], []

        def frame(body, P):
            if body == s.W:
                return 0, P
            return int(self.joint_of[body]), self.Mjb[body] @ P

        for e in range(mode.m):
            if e < mode.n_sys_chords:
                gid = s.chord_index[mode.ids[e]]
                kind = s.chord_kind[gid]
                cut = s.chord_cut_dofs[gid]
                if cut:
                    first = s.dof_child[cut[0]]
                    b1, P1 = s.parent[first], s._A0[first] if hasattr(s, '_A0') else s.A[first]
                    axis = s.xi[first][:3]
                else:
                    b1, P1, axis = mode.b1[e], mode.P1[e], None
            else:
                kind, b1, P1, axis = 'fixed', mode.b1[e], mode.P1[e], None
            b2, P2 = mode.b2[e], mode.P2[e]
            j1, F1 = frame(b1, P1)
            j2, F2 = frame(b2, P2)
            if kind == 'fixed':
                models.append(pin.RigidConstraintModel(pin.ContactType.CONTACT_6D, self.model, j1, _se3(F1), j2, _se3(F2),
                                                       pin.ReferenceFrame.LOCAL))
                layout.append((e, 6))
            elif kind == 'spherical':
                models.append(pin.RigidConstraintModel(pin.ContactType.CONTACT_3D, self.model, j1, _se3(F1), j2, _se3(F2),
                                                       pin.ReferenceFrame.LOCAL))
                layout.append((e, 3))
            elif kind == 'revolute':
                # A pin is two coincident point pairs on its axis (five independent rows).
                for shift in (0., 0.1):
                    O1, O2 = np.eye(4), np.eye(4)
                    O1[:3, 3] = shift * axis
                    O2[:3, 3] = shift * axis
                    models.append(pin.RigidConstraintModel(pin.ContactType.CONTACT_3D, self.model, j1, _se3(F1 @ O1),
                                                           j2, _se3(F2 @ O2), pin.ReferenceFrame.LOCAL))
                    layout.append((e, 3))
            else:
                raise NotImplementedError(f'No Pinocchio constraint model for a {kind} chord')
        return models, layout

    def solver(self, mode, mu=1e-9, iterations=200, accuracy=1e-13):
        """Reusable proximal KKT solver for the chords of ``mode`` (models, data and settings)."""
        models, layout = self.constraint_models(mode)
        datas = [m.createData() for m in models]
        data = self.model.createData()
        self.model.armature = self.P @ self.armature
        pin.initConstraintDynamics(self.model, data, models)
        self.model.armature = np.zeros(self.model.nv)
        return dict(models=models, datas=datas, data=data, settings=pin.ProximalSettings(accuracy, mu, iterations), layout=layout)

    def solve(self, solver, q, vT, tauT):
        """Constrained acceleration with a prepared solver (core tree order)."""
        self.model.armature = self.P @ self.armature
        a = pin.constraintDynamics(self.model, solver['data'], self.q_pin(q), self.v_pin(vT), self.P @ np.asarray(tauT, float),
                                   solver['models'], solver['datas'], solver['settings'])
        self.model.armature = np.zeros(self.model.nv)
        return self.P.T @ a

    # ------------------------------------------------------------------ physical constraint rows (analytic)
    def _row_frames(self, mode):
        """Point and orientation comparisons of every chord as pairs of frames on Pinocchio joints."""
        s = self.sys
        A0 = getattr(s, '_A0', s.A)
        out = []

        def frame(body, P):
            if body == s.W:
                return 0, np.array(P, float)
            return int(self.joint_of[body]), self.Mjb[body] @ P

        for e in range(mode.m):
            if e < mode.n_sys_chords:
                gid = s.chord_index[mode.ids[e]]
                kind, cut = s.chord_kind[gid], s.chord_cut_dofs[gid]
                if cut:
                    first = s.dof_child[cut[0]]
                    b1, P1, axis = s.parent[first], A0[first], s.xi[first][:3]
                else:
                    b1, P1, axis = mode.b1[e], mode.P1[e], None
            else:
                kind, b1, P1, axis = 'fixed', mode.b1[e], mode.P1[e], None
            b2, P2 = mode.b2[e], mode.P2[e]
            if kind == 'fixed':
                out.append((e, 'weld', frame(b1, P1), frame(b2, P2)))
            elif kind == 'spherical':
                out.append((e, 'point', frame(b1, P1), frame(b2, P2)))
            elif kind == 'revolute':
                for shift in (0., 0.1):
                    O = np.eye(4)
                    O[:3, 3] = shift * axis
                    out.append((e, 'point', frame(b1, P1 @ O), frame(b2, P2 @ O)))
            else:
                raise NotImplementedError(f'No physical rows for a {kind} chord')
        return out

    def physical_rows(self, mode, q, vT):
        """Physical constraint rows ``J_c`` (core tree order) and analytic drift ``Jdot_c v``.

        Every comparison is written in world axes: coincident points give three rows
        (difference of the origin velocities), welds add the difference of the
        angular velocities.  The drift is the corresponding difference of classical
        and angular accelerations at zero joint acceleration.
        """
        model = self.model
        data = model.createData()
        qp, vp = self.q_pin(q), self.v_pin(vT)
        pin.forwardKinematics(model, data, qp, vp, np.zeros(model.nv))
        pin.computeJointJacobians(model, data, qp)
        LWA = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        rows, drift, layout = [], [], []

        def point(jid, F):
            """World-aligned Jacobian [linear; angular] and velocity-product acceleration of a frame."""
            if jid == 0:
                return np.zeros((6, model.nv)), np.zeros(6)
            J = pin.getJointJacobian(model, data, jid, LWA)
            oMi = data.oMi[jid]
            r = oMi.rotation @ F[:3, 3]
            Jp = J.copy()
            Jp[:3] = J[:3] - se3.skew(r) @ J[3:]
            v, a = data.v[jid], data.a[jid]
            w = oMi.rotation @ v.angular
            al = oMi.rotation @ a.angular
            a_joint = oMi.rotation @ (a.linear + np.cross(v.angular, v.linear))     # classical acceleration of the joint origin
            acc = a_joint + np.cross(al, r) + np.cross(w, np.cross(w, r))
            return Jp, np.r_[acc, al]

        for e, kind, (j1, F1), (j2, F2) in self._row_frames(mode):
            J1, a1 = point(j1, F1)
            J2, a2 = point(j2, F2)
            n = 6 if kind == 'weld' else 3
            rows.append((J2 - J1)[:n])
            drift.append((a2 - a1)[:n])
            layout.append((e, n))
        return np.vstack(rows) @ self.P, np.concatenate(drift), layout
