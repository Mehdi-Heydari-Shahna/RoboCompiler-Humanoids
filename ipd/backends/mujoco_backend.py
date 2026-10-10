"""MuJoCo view of a compiled ``System``: MJCF export and state maps.

The MJCF is generated from the same compiled records as the inertia-partitioned
model.  Tree joints become hinge/slide joints with their armature and damping,
every floating root gets a free joint, chords become ``connect`` equalities between
sites (a pin is two point pairs on its axis, a ball joint and a hand link one), and
the actuator ports become force/torque motors.  Supports are not equalities: soles
touch the floor through native unilateral frictional contact.

Velocity conventions: MuJoCo's free joint uses the world-frame linear velocity and
the body-frame angular velocity; the core uses the body twist ``[w; v]``.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from .. import se3
from . import meshes
from ..system import BASE, Config, FIXED, PRISMATIC, REVOLUTE, ROOT

# Materials of the visual geometry: (rgba, specular, shininess, reflectance, emission)
MATERIALS = {
    'shell': ('0.86 0.87 0.88 1', 0.55, 0.55, 0.0, 0.0),
    'shell_grey': ('0.66 0.68 0.71 1', 0.5, 0.5, 0.0, 0.0),
    'graphite': ('0.16 0.17 0.19 1', 0.45, 0.55, 0.0, 0.0),
    'joint': ('0.10 0.105 0.115 1', 0.6, 0.75, 0.0, 0.0),
    'metal': ('0.50 0.53 0.57 1', 0.9, 0.85, 0.0, 0.0),
    'chrome': ('0.84 0.85 0.87 1', 1.0, 1.0, 0.15, 0.0),
    'rubber': ('0.055 0.055 0.06 1', 0.08, 0.1, 0.0, 0.0),
    'visor': ('0.015 0.02 0.03 1', 1.0, 1.0, 0.3, 0.0),
    'grey': ('0.55 0.57 0.60 1', 0.3, 0.3, 0.0, 0.0),
}
ACCENTS = {
    'red': '0.82 0.20 0.18 1', 'orange': '0.95 0.52 0.10 1', 'green': '0.16 0.64 0.42 1',
    'blue': '0.18 0.46 0.86 1', 'purple': '0.56 0.36 0.80 1',
}
LEGACY_COLORS = {'dark': 'joint', 'steel': 'metal', 'chrome': 'chrome', 'skin': 'shell', 'grey': 'grey'}


class PlainScene:
    """Default scene: a plain floor plane with the given contact data and a headlight."""

    floor = dict(friction=0.9, torsional=0.02, rolling=0.001, solref=(0.006, 1.0), solimp=(0.95, 0.99, 0.001), condim=3)
    assets = {}

    def visual(self, vis, offscreen):
        ET.SubElement(vis, 'global', offwidth=str(offscreen[0]), offheight=str(offscreen[1]))
        ET.SubElement(vis, 'headlight', ambient='0.45 0.45 0.45', diffuse='0.55 0.55 0.55', specular='0.1 0.1 0.1')

    def assets_xml(self, asset):
        ET.SubElement(asset, 'material', name='floor', rgba='0.85 0.86 0.88 1')

    def world(self, world):
        f = self.floor
        ET.SubElement(world, 'light', pos='0 0 6', dir='0 0 -1', diffuse='0.5 0.5 0.5', castshadow='false')
        ET.SubElement(world, 'geom', name='floor', type='plane', size='8 8 0.1', material='floor', contype='1', conaffinity='1',
                      condim=str(f['condim']), friction=f"{f['friction']} {f['torsional']} {f['rolling']}",
                      solref=' '.join(map(str, f['solref'])), solimp=' '.join(map(str, f['solimp'])))


def fmt(x):
    return ' '.join(f'{float(v):.10g}' for v in np.ravel(x))


def _quat_wxyz(R):
    q = se3.quat_xyzw_from_matrix(R)
    return np.r_[q[3], q[:3]]


def export_mjcf(system, path=None, timestep=1e-3, loop_solref=0.004, friction=None, floor=True, gravity=True,
                offscreen=(1280, 800), q=None, link_solref=None, support_welds=None, contact=True, links=None, scene=None):
    """Write the MJCF of ``system``; returns the XML string.

    ``q`` (a ``Config``) fixes the initial root poses; joint values start at zero and
    are set through ``MujocoPlant.set_state``.  ``scene`` supplies the floor (its contact
    data are also used for the soles), lights, sky and textures (``PlainScene`` when
    omitted); ``friction`` overrides the friction coefficient of the scene.
    """
    scene = scene or PlainScene()
    fl = dict(scene.floor)
    if friction is not None:
        fl['friction'] = float(friction)
    s = system
    root = ET.Element('mujoco', model=s.name)
    ET.SubElement(root, 'compiler', angle='radian', inertiafromgeom='false', autolimits='true')
    option = ET.SubElement(root, 'option', timestep=fmt(timestep), integrator='implicitfast', cone='elliptic', impratio='10',
                           solver='Newton', iterations='60', tolerance='1e-10',
                           gravity=fmt(s.gravity if gravity else np.zeros(3)))
    if not contact:
        ET.SubElement(option, 'flag', contact='disable')
    ET.SubElement(root, 'size', memory='300M')
    vis = ET.SubElement(root, 'visual')
    scene.visual(vis, offscreen)
    asset = ET.SubElement(root, 'asset')
    scene.assets_xml(asset)
    for name, (rgba, spec, shin, refl, emis) in MATERIALS.items():
        ET.SubElement(asset, 'material', name=name, rgba=rgba, specular=fmt(spec), shininess=fmt(shin), reflectance=fmt(refl),
                      emission=fmt(emis))
    for name, rgba in ACCENTS.items():
        ET.SubElement(asset, 'material', name='accent_' + name, rgba=rgba, specular='0.45', shininess='0.55')
        ET.SubElement(asset, 'material', name='led_' + name, rgba=rgba, specular='0.2', shininess='0.3', emission='0.9')
    default = ET.SubElement(root, 'default')
    ET.SubElement(default, 'geom', contype='0', conaffinity='0', density='0', group='1')
    ET.SubElement(default, 'site', size='0.008', rgba='0 0 0 0')
    foot = ET.SubElement(default, 'default', **{'class': 'sole'})
    ET.SubElement(foot, 'geom', contype='1', conaffinity='1', condim=str(fl['condim']), group='0',
                  friction=f"{fl['friction']} {fl['torsional']} {fl['rolling']}",
                  solref=' '.join(map(str, fl['solref'])), solimp=' '.join(map(str, fl['solimp'])))
    world = ET.SubElement(root, 'worldbody')
    if floor:
        scene.floor = fl
        scene.world(world)
    sole_bodies = {}
    for k, m in enumerate(s.members):
        for site in m.comp.cmg['sole_sites']:
            sole_bodies[s.body_index[f"{m.name}/{site['body']}"]] = site
    records = {}
    for m in s.members:
        for b in m.comp.cmg['bodies']:
            records[f"{m.name}/{b['id']}"] = b
    elements = {}
    T0 = s.poses(q if q is not None else s.neutral())
    for b in range(s.nbp):
        name = s.body_names[b]
        t = s.jtype[b]
        if t in (ROOT, BASE):
            T = T0[b]
            e = ET.SubElement(world, 'body', name=name, pos=fmt(T[:3, 3]), quat=fmt(_quat_wxyz(T[:3, :3])))
            if t == ROOT:
                ET.SubElement(e, 'freejoint', name=name + '__root')
        else:
            X0 = s.Xfixed[b]
            e = ET.SubElement(elements[s.parent[b]], 'body', name=name, pos=fmt(X0[:3, 3]), quat=fmt(_quat_wxyz(X0[:3, :3])))
            if t != FIXED:
                B = se3.inv(s.Binv[b])
                axis = B[:3, :3] @ (s.xi[b, :3] if t == REVOLUTE else s.xi[b, 3:])
                j = s.dof[b]
                ET.SubElement(e, 'joint', name=s.ids[j], type='hinge' if t == REVOLUTE else 'slide', pos=fmt(B[:3, 3]),
                              axis=fmt(axis), range=fmt([s.lower[j], s.upper[j]]), armature=fmt(s.armature[j]),
                              damping=fmt(s.damping[j]), frictionloss=fmt(s.frictionloss[j]))
        elements[b] = e
        if s.mass[b] > 0:
            I = s.inertia_com[b]
            ET.SubElement(e, 'inertial', pos=fmt(s.com[b]), mass=fmt(s.mass[b]),
                          fullinertia=fmt([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]))
        for i, gm in enumerate(records.get(name, {}).get('geoms', [])):
            mat = gm.get('material') or LEGACY_COLORS.get(gm.get('color', 'grey'), None)
            if mat is None:
                mat = 'accent_' + gm['color'] if gm.get('color') in ACCENTS else 'grey'
            attrs = dict(name=f'{name}__g{i}', material=mat)
            if b in sole_bodies and gm['type'] == 'box':
                attrs['class'] = 'sole'
            kind = gm['type']
            if kind == 'capsule':
                ET.SubElement(e, 'geom', type='capsule', fromto=fmt(np.r_[gm['from'], gm['to']]), size=fmt(gm['radius']), **attrs)
            elif kind == 'box':
                ET.SubElement(e, 'geom', type='box', pos=fmt(gm['center']), size=fmt(np.asarray(gm['size']) / 2), **attrs)
            elif kind == 'sphere':
                ET.SubElement(e, 'geom', type='sphere', pos=fmt(gm['center']), size=fmt(gm['radius']), **attrs)
            elif kind == 'ellipsoid':
                ET.SubElement(e, 'geom', type='ellipsoid', pos=fmt(gm['center']), size=fmt(gm['half']), **attrs)
            elif kind == 'cylinder':
                a = np.asarray(gm['center'], float)
                d = np.asarray(gm['axis'], float) / np.linalg.norm(gm['axis']) * gm['half_length']
                ET.SubElement(e, 'geom', type='cylinder', fromto=fmt(np.r_[a - d, a + d]), size=fmt(gm['radius']), **attrs)
            elif kind in ('loft', 'rbox'):
                V, F = meshes.tessellate(gm)
                mesh_name = f'{name}__m{i}'
                ET.SubElement(asset, 'mesh', name=mesh_name, vertex=fmt(np.round(V, 6)), face=' '.join(map(str, F.ravel())),
                              smoothnormal='true')
                ET.SubElement(e, 'geom', type='mesh', mesh=mesh_name, pos=fmt(gm.get('center', [0., 0., 0.])), **attrs)
            else:
                raise ValueError(f'unknown visual geom type {kind}')
    for k, m in enumerate(s.members):
        for site in m.comp.cmg['sole_sites']:
            e = elements[s.body_index[f"{m.name}/{site['body']}"]]
            ET.SubElement(e, 'site', name=f"{m.name}/{site['id']}", pos=fmt(site['frame_translation_m']),
                          quat=fmt(_quat_wxyz(np.asarray(site['frame_rotation']))))
        for site in m.comp.cmg['point_sites']:
            e = elements[s.body_index[f"{m.name}/{site['body']}"]]
            ET.SubElement(e, 'site', name=f"{m.name}/{site['id']}", pos=fmt(site['point_m']))
    equality = ET.SubElement(root, 'equality')
    A0 = getattr(s, '_A0', s.A)
    for e_id in range(s.nch):
        cid, kind = s.chord_ids[e_id], s.chord_kind[e_id]
        cut = s.chord_cut_dofs[e_id]
        if cut:
            first = s.dof_child[cut[0]]
            b1, P1, axis = s.parent[first], A0[first], s.xi[first][:3]
        else:
            b1, P1, axis = s.chord_b1[e_id], s.chord_P1[e_id], None
        b2, P2 = s.chord_b2[e_id], s.chord_P2[e_id]
        is_link = e_id >= s.n_internal
        solref = (link_solref or loop_solref) if is_link else loop_solref
        pars = dict(solref=f'{solref} 1', solimp='0.9999 0.9999 0.001')
        if kind == 'fixed':
            ET.SubElement(elements[b1], 'site', name=cid + '__a', pos=fmt(P1[:3, 3]), quat=fmt(_quat_wxyz(P1[:3, :3])))
            ET.SubElement(elements[b2], 'site', name=cid + '__b', pos=fmt(P2[:3, 3]), quat=fmt(_quat_wxyz(P2[:3, :3])))
            ET.SubElement(equality, 'weld', name=cid, site1=cid + '__a', site2=cid + '__b', **pars)
            continue
        shifts = (0., 0.1) if kind == 'revolute' else (0.,)
        for n, shift in enumerate(shifts):
            tag = cid if n == 0 else f'{cid}__axis'
            off = np.zeros(3) if axis is None else shift * axis
            ET.SubElement(elements[b1], 'site', name=tag + '__a', pos=fmt(P1[:3, 3] + P1[:3, :3] @ off))
            ET.SubElement(elements[b2], 'site', name=tag + '__b', pos=fmt(P2[:3, 3] + P2[:3, :3] @ off))
            ET.SubElement(equality, 'connect', name=tag, site1=tag + '__a', site2=tag + '__b', **pars)
    if links is not None:
        for el in list(equality):
            base = el.get('name').split('__')[0]
            if base in [l['id'] for l in s.links] and base not in links:
                equality.remove(el)
    for member, site, anchor in (support_welds or []):
        # ideal support for state-wise comparisons: the sole frame welded to a world anchor
        name = f'{member}/{site}'
        ET.SubElement(world, 'site', name=name + '__anchor', pos=fmt(anchor[:3, 3]), quat=fmt(_quat_wxyz(anchor[:3, :3])))
        ET.SubElement(equality, 'weld', name=name + '__weld', site1=name + '__anchor', site2=name,
                      solref=f'{loop_solref} 1', solimp='0.9999 0.9999 0.001')
    actuator = ET.SubElement(root, 'actuator')
    for i, pid in enumerate(s.port_ids):
        j = s.port_dof[i]
        ET.SubElement(actuator, 'motor', name=pid, joint=s.ids[j], gear=fmt(s.B[j, i]),
                      ctrlrange=fmt([s.effort_lower[i], s.effort_upper[i]]), ctrllimited='true')
    ET.indent(root)
    xml = ET.tostring(root, encoding='unicode')
    if path is not None:
        with open(path, 'w') as f:
            f.write(xml + '\n')
    return xml


class MujocoPlant:
    """MuJoCo model of a compiled system with state maps to the core conventions."""

    def __init__(self, system, xml=None, **export):
        import mujoco
        self.mj = mujoco
        s = self.sys = system
        scene = export.get('scene') or PlainScene()
        export['scene'] = scene
        self.xml = xml or export_mjcf(system, **export)
        self.model = mujoco.MjModel.from_xml_string(self.xml, dict(scene.assets))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.qadr = np.full(s.nT, -1, int)
        self.vadr = np.full(s.nT, -1, int)
        self.root_q, self.root_v = {}, {}
        for b in range(s.nbp):
            t = s.jtype[b]
            if t == ROOT:
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, s.body_names[b] + '__root')
                self.root_q[s.member_of_body[b]] = int(m.jnt_qposadr[jid])
                self.root_v[s.member_of_body[b]] = int(m.jnt_dofadr[jid])
            elif t in (REVOLUTE, PRISMATIC):
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, s.ids[s.dof[b]])
                self.qadr[s.dof[b]] = int(m.jnt_qposadr[jid])
                self.vadr[s.dof[b]] = int(m.jnt_dofadr[jid])
        self.scalar = np.flatnonzero(self.qadr >= 0)
        self.act = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, p) for p in s.port_ids], int)
        self.eq = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_EQUALITY, i): i for i in range(m.neq)}
        self.body_id = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in s.body_names[:s.nbp]], int)
        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        if m.nv != s.nT:
            raise ValueError('MuJoCo model and compiled tree have different velocity dimensions')

    # ------------------------------------------------------------------ state maps
    def set_state(self, q, vT=None):
        d = self.data
        s = self.sys
        for k, a in self.root_q.items():
            T = q.T_root[k]
            d.qpos[a:a + 3] = T[:3, 3]
            d.qpos[a + 3:a + 7] = _quat_wxyz(T[:3, :3])
        d.qpos[self.qadr[self.scalar]] = q.theta[self.scalar]
        if vT is not None:
            vT = np.asarray(vT, float)
            for k, a in self.root_v.items():
                sl = s.root_slice[k]
                R = q.T_root[k][:3, :3]
                d.qvel[a:a + 3] = R @ vT[sl][3:]
                d.qvel[a + 3:a + 6] = vT[sl][:3]
            d.qvel[self.vadr[self.scalar]] = vT[self.scalar]
        else:
            d.qvel[:] = 0.
        self.mj.mj_forward(self.model, d)

    def get_state(self, template):
        """Core configuration (cut coordinates lifted) and tree velocity from ``qpos``/``qvel``."""
        d = self.data
        s = self.sys
        q = template.copy()
        v = np.zeros(s.n)
        for k, a in self.root_q.items():
            quat = d.qpos[a + 3:a + 7]
            R = se3.matrix_from_quat_xyzw(np.r_[quat[1:], quat[0]])
            q.T_root[k] = se3.transform(R, d.qpos[a:a + 3])
            b = self.root_v[k]
            sl = s.root_slice[k]
            v[sl] = np.r_[d.qvel[b + 3:b + 6], R.T @ d.qvel[b:b + 3]]
        q.theta[self.scalar] = d.qpos[self.qadr[self.scalar]]
        v[self.scalar] = d.qvel[self.vadr[self.scalar]]
        return q, v

    def acceleration(self, q):
        """Tree acceleration in the core convention from ``qacc`` (body-twist rate for roots)."""
        d = self.data
        s = self.sys
        a = np.zeros(s.nT)
        for k, b in self.root_v.items():
            R = q.T_root[k][:3, :3]
            w = d.qvel[b + 3:b + 6]
            vloc = R.T @ d.qvel[b:b + 3]
            a[s.root_slice[k]] = np.r_[d.qacc[b + 3:b + 6], R.T @ d.qacc[b:b + 3] - np.cross(w, vloc)]
        a[self.scalar] = d.qacc[self.vadr[self.scalar]]
        return a

    def set_effort(self, f):
        self.data.ctrl[self.act] = f

    def step(self, n=1):
        for _ in range(n):
            self.mj.mj_step(self.model, self.data)

    def set_link(self, link_id, active):
        self.data.eq_active[self.eq[link_id]] = 1 if active else 0

    # ------------------------------------------------------------------ measurements
    def equality_gaps(self):
        """Distance between the two sites of every connect equality (m)."""
        m, d = self.model, self.data
        out = {}
        for name, i in self.eq.items():
            if m.eq_type[i] == self.mj.mjtEq.mjEQ_CONNECT and d.eq_active[i]:
                out[name] = float(np.linalg.norm(d.site_xpos[m.eq_obj1id[i]] - d.site_xpos[m.eq_obj2id[i]]))
        return out

    def equality_forces(self):
        """Constraint force of every active connect equality (N), in constraint coordinates."""
        m, d = self.model, self.data
        out = {}
        for r in range(d.nefc):
            if d.efc_type[r] == self.mj.mjtConstraint.mjCNSTR_EQUALITY:
                name = self.mj.mj_id2name(m, self.mj.mjtObj.mjOBJ_EQUALITY, d.efc_id[r])
                out.setdefault(name, []).append(float(d.efc_force[r]))
        return {k: np.array(v) for k, v in out.items()}

    def foot_wrenches(self, sole_frames):
        """Floor reaction on each sole body about its sole-frame origin, in world axes.

        ``sole_frames`` maps a key to (core body index, 4x4 sole frame in the body).
        Returns {key: (force (3), moment (3), number of contacts)}.
        """
        m, d = self.model, self.data
        out = {}
        keys = {int(self.body_id[b]): (key, P) for key, (b, P) in sole_frames.items()}
        acc = {key: [np.zeros(3), np.zeros(3), 0] for key in sole_frames}
        wrench = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            for body, sign in ((b1, -1.), (b2, 1.)):
                if body in keys and (c.geom1 == self.floor or c.geom2 == self.floor):
                    self.mj.mj_contactForce(m, d, i, wrench)
                    f = sign * (c.frame.reshape(3, 3).T @ wrench[:3])
                    key, P = keys[body]
                    origin = d.xpos[body] + d.xmat[body].reshape(3, 3) @ P[:3, 3]
                    acc[key][0] += f
                    acc[key][1] += np.cross(c.pos - origin, f)
                    acc[key][2] += 1
        for key, (f, n, cnt) in acc.items():
            out[key] = (f, n, cnt)
        return out
