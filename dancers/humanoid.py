"""Parametric closed-chain humanoid ("dancer") as a physical mechanism record.

One function authors the complete physical description: 42 rigid bodies with inertias
from their solids, 49 physical joints with both attachment frames, 21 actuator ports
with effort bounds and drive data, named sole, palm and shoulder sites, and the visual
shape of every body (shells, joint housings, hands and head; rendering only).

Mechanism (per side): three rotary hip drives, a knee driven by a linear actuator
through a planar triangle, an ankle with two passive axes driven by two linear struts
(universal joint on the shank, ball joint on the foot), three rotary shoulder drives
and an elbow driven by a linear actuator; a rotary waist joins pelvis and torso.  The
joint graph has eight independent cycles per robot.  No joint is marked as a cut; the
rod-end pins of the knee and elbow actuators and the ball joints of the ankle struts
are ideal bearings (no drive data), all other joints carry drive data.

Drives.  Every actuator is a motor with a transmission: a gearbox of ratio ``N`` for the
rotary drives, a ball screw of lead ``l`` (``N = 2 pi / l``) for the linear ones.  The
joint-level data follow from the motor: reflected rotor inertia ``J_m N^2`` (the
armature, a member of the inertia partition), peak effort, no-load speed, viscous and
Coulomb friction.  The values are representative of current electric humanoids; they
are not the data of a particular product.  Coulomb friction acts in the simulator only;
the compiled model contains the reflected inertia and the viscous term.

``variant`` scales the design, so the five dancers are five physical records of one
architecture with different link dimensions and inertias.
"""
from __future__ import annotations

import numpy as np

from ipd.shapes import body, box_inertia, capsule_inertia, joint, sphere_inertia, tf

NOMINAL = dict(
    hip_half_width=0.095, hip_drop=0.06, thigh=0.38, shank=0.38, ankle_height=0.07,
    foot_length=0.22, foot_width=0.10, foot_forward=0.035, waist_height=0.09,
    shoulder_half_width=0.20, shoulder_height=0.32, upper_arm=0.27, forearm=0.25, palm=0.29,
    torso_height=0.40, head_radius=0.09,
)

# Representative electric drives: motor rotor inertia (kg m^2), gearbox ratio or screw lead (m),
# joint-level peak effort (N m or N), no-load speed (rad/s or m/s), viscous (N m s/rad or N s/m)
# and Coulomb friction (N m or N, simulator only).
DRIVES = {
    'hip_pitch': dict(kind='rotary', rotor=1.2e-4, ratio=25., peak=220., no_load=15.0, damping=0.5, coulomb=6.6),
    'hip_roll': dict(kind='rotary', rotor=1.2e-4, ratio=25., peak=150., no_load=15.0, damping=0.5, coulomb=4.5),
    'hip_yaw': dict(kind='rotary', rotor=6.0e-5, ratio=25., peak=80., no_load=16.0, damping=0.5, coulomb=2.4),
    'waist_yaw': dict(kind='rotary', rotor=6.0e-5, ratio=25., peak=80., no_load=16.0, damping=0.5, coulomb=2.4),
    'shoulder_pitch': dict(kind='rotary', rotor=6.0e-5, ratio=30., peak=90., no_load=18.0, damping=0.3, coulomb=2.7),
    'shoulder_roll': dict(kind='rotary', rotor=6.0e-5, ratio=30., peak=90., no_load=18.0, damping=0.3, coulomb=2.7),
    'shoulder_yaw': dict(kind='rotary', rotor=4.0e-5, ratio=30., peak=60., no_load=18.0, damping=0.3, coulomb=1.8),
    'knee_stroke': dict(kind='linear', rotor=2.0e-5, lead=0.005, peak=4500., no_load=0.40, damping=40., coulomb=135.),
    'ankle_stroke': dict(kind='linear', rotor=1.0e-5, lead=0.004, peak=2500., no_load=0.35, damping=30., coulomb=75.),
    'elbow_stroke': dict(kind='linear', rotor=0.8e-5, lead=0.004, peak=1500., no_load=0.40, damping=20., coulomb=45.),
}
BEARING = dict(damping=0.05, coulomb=0.05)     # passive revolute joints with bearings that carry data


def reflected_inertia(spec):
    """Armature of a drive: ``J_m N^2`` with ``N`` the gear ratio or ``2 pi / lead``."""
    n = spec['ratio'] if spec['kind'] == 'rotary' else 2. * np.pi / spec['lead']
    return spec['rotor'] * n * n


def _mirror(p, side):
    return [p[0], side * p[1], p[2]]


def humanoid(name='dancer', scale=1.0, arm_scale=1.0, leg_scale=1.0, mass_scale=None, color='blue'):
    """Physical record of one humanoid.

    ``scale`` scales every length, ``arm_scale`` and ``leg_scale`` additionally scale the
    limb segments, and masses follow the cube of the length scale unless ``mass_scale``
    is given.  Actuator ports, drives, limits and the joint graph are unchanged; ``color``
    is the accent colour of the shells.
    """
    s = float(scale)
    sa, sl = s * float(arm_scale), s * float(leg_scale)
    ms = s ** 3 if mass_scale is None else float(mass_scale)
    g = {k: v * s for k, v in NOMINAL.items()}
    for k in ('thigh', 'shank'):
        g[k] = NOMINAL[k] * sl
    for k in ('upper_arm', 'forearm', 'palm'):
        g[k] = NOMINAL[k] * sa
    bodies, joints, actuators, drive, seed = [], [], [], {}, {}
    y = (0., 1., 0.); x = (1., 0., 0.); z = (0., 0., 1.)
    accent = 'accent_' + color

    def add_body(bid, mass, com, inertia, geoms):
        bodies.append(body(bid, mass * ms, com, np.asarray(inertia) * ms, geoms))

    # ------------------------------------------------------------- visual shapes (body frame, metres)
    def cap(a, b, r, mat):
        return {'type': 'capsule', 'from': list(map(float, a)), 'to': list(map(float, b)), 'radius': float(r), 'material': mat}

    def box(center, size, mat):
        return {'type': 'box', 'center': list(map(float, center)), 'size': list(map(float, size)), 'material': mat}

    def sph(center, r, mat):
        return {'type': 'sphere', 'center': list(map(float, center)), 'radius': float(r), 'material': mat}

    def cyl(center, axis, r, half, mat):
        return {'type': 'cylinder', 'center': list(map(float, center)), 'axis': list(map(float, axis)), 'radius': float(r),
                'half_length': float(half), 'material': mat}

    def ell(center, half, mat):
        return {'type': 'ellipsoid', 'center': list(map(float, center)), 'half': list(map(float, half)), 'material': mat}

    def rbox(center, half, mat, exponent=4.):
        return {'type': 'rbox', 'center': list(map(float, center)), 'half': list(map(float, half)), 'exponent': float(exponent),
                'material': mat}

    def shell(sections, mat, exponent=2.6, caps=(0.55, 0.55)):
        """Lofted shell along the local z axis; rows ``(z, ax, ay, cx, cy)``."""
        return {'type': 'loft', 'sections': [list(map(float, r)) for r in sections], 'exponent': float(exponent),
                'caps': list(caps), 'material': mat}

    # ------------------------------------------------------------- drives
    def drive_data(jid, spec):
        drive[jid] = {'armature': reflected_inertia(spec), 'damping': spec['damping'], 'frictionloss': spec['coulomb']}

    def motor(spec):
        out = {k: spec[k] for k in ('kind', 'rotor', 'peak', 'no_load', 'damping', 'coulomb')}
        out['rotor_kg_m2'] = out.pop('rotor')
        if spec['kind'] == 'rotary':
            out['gear_ratio'] = spec['ratio']
        else:
            out['screw_lead_m'] = spec['lead']
        out['reflected_inertia'] = reflected_inertia(spec)
        return out

    def rotary(jid, a, b, T_AJ, axis, lo, hi, kind, q0=0.):
        spec = DRIVES[kind]
        joints.append(joint(jid, 'revolute', a, b, T_AJ=T_AJ, axis=axis, lower=lo, upper=hi))
        actuators.append({'id': jid + '_drive', 'joint_id': jid, 'gear': 1., 'effort_bounds': [-spec['peak'], spec['peak']],
                          'effort_unit': 'Nm', 'motor': motor(spec)})
        drive_data(jid, spec)
        seed[jid] = q0

    def passive(jid, a, b, T_AJ=None, T_BJ=None, axis=y, lo=-3.1, hi=3.1, bearing=True, q0=0., kind='revolute'):
        """Passive joint.  ``bearing=False`` marks an ideal bearing without drive data: the compiler
        may choose such a joint to close its loop (every loop needs one)."""
        joints.append(joint(jid, kind, a, b, T_AJ=T_AJ, T_BJ=T_BJ, axis=axis, lower=lo, upper=hi))
        if kind == 'revolute':
            if bearing:
                drive[jid] = {'armature': 0., 'damping': BEARING['damping'], 'frictionloss': BEARING['coulomb']}
            seed[jid] = q0

    def linear(jid, a, b, retracted, lo, hi, kind, q0):
        spec = DRIVES[kind]
        joints.append(joint(jid, 'prismatic', a, b, T_AJ=tf((0., 0., -retracted)), axis=(0., 0., -1.), lower=lo, upper=hi))
        actuators.append({'id': jid + '_drive', 'joint_id': jid, 'gear': 1., 'effort_bounds': [-spec['peak'], spec['peak']],
                          'effort_unit': 'N', 'motor': motor(spec)})
        drive_data(jid, spec)
        seed[jid] = q0

    def actuator_bodies(prefix, r_barrel, len_barrel, m_barrel, len_rod, m_rod):
        add_body(prefix + '_barrel', m_barrel, [0., 0., -len_barrel / 2], capsule_inertia(m_barrel, r_barrel, len_barrel),
                 [cap([0, 0, 0], [0, 0, -len_barrel], r_barrel, 'metal'),
                  cyl([0, 0, -0.012 * s], z, r_barrel * 1.25, 0.014 * s, 'joint')])
        add_body(prefix + '_rod', m_rod, [0., 0., len_rod / 2], capsule_inertia(m_rod, r_barrel * .55, len_rod),
                 [cap([0, 0, 0], [0, 0, len_rod], r_barrel * .5, 'chrome')])

    # ---------------------------------------------------------------- trunk
    add_body('pelvis', 7.0, [0., 0., 0.02 * s], box_inertia(7.0, (0.16 * s, 0.28 * s, 0.14 * s)),
             [rbox([-0.005 * s, 0, 0.012 * s], (0.085 * s, 0.128 * s, 0.062 * s), 'shell', 4.5),
              rbox([-0.005 * s, 0, 0.07 * s], (0.088 * s, 0.131 * s, 0.011 * s), accent, 7.),
              cyl([0, 0, g['waist_height'] - 0.012 * s], z, 0.068 * s, 0.022 * s, 'joint')]
             + [cyl([0, sg * g['hip_half_width'], -g['hip_drop'] + 0.032 * s], z, 0.046 * s, 0.03 * s, 'joint') for sg in (1, -1)])
    th, hr = g['torso_height'], g['head_radius']
    m_chest, m_head = 12.5, 2.5
    c_chest, c_head = np.array([0., 0., 0.03 * s + th / 2]), np.array([0., 0., 0.03 * s + th + hr * 1.15])
    com_t = (m_chest * c_chest + m_head * c_head) / (m_chest + m_head)
    I_t = box_inertia(m_chest, (0.16 * s, 0.30 * s, th)) + sphere_inertia(m_head, hr)
    for m_, c_ in ((m_chest, c_chest), (m_head, c_head)):
        dvec = c_ - com_t
        I_t = I_t + m_ * (dvec @ dvec * np.eye(3) - np.outer(dvec, dvec))
    zt = lambda f: 0.03 * s + f * th              # height in the torso frame as a fraction of the chest
    torso_vis = [
        shell([(zt(0.02), 0.080 * s, 0.122 * s, 0.004 * s), (zt(0.30), 0.094 * s, 0.150 * s, 0.010 * s),
               (zt(0.62), 0.104 * s, 0.176 * s, 0.012 * s), (zt(0.86), 0.096 * s, 0.168 * s, 0.008 * s)], 'shell', 3.4, (0.35, 0.45)),
        rbox([0.094 * s, 0, zt(0.66)], (0.016 * s, 0.112 * s, 0.075 * s), 'graphite', 5.),
        rbox([0.110 * s, 0, zt(0.72)], (0.004 * s, 0.065 * s, 0.007 * s), 'led_' + color, 6.),
        rbox([-0.112 * s, 0, zt(0.55)], (0.040 * s, 0.118 * s, 0.125 * s), 'graphite', 5.),
        cyl([0, 0, zt(1.0) + 0.028 * s], z, 0.036 * s, 0.032 * s, 'joint'),
        ell(c_head, (0.094 * s, 0.084 * s, 0.104 * s), 'shell'),
        ell(c_head + np.array([0.036 * s, 0., 0.006 * s]), (0.066 * s, 0.074 * s, 0.046 * s), 'visor'),
    ] + [cyl(c_head + np.array([0., sg * 0.083 * s, 0.]), y, 0.030 * s, 0.012 * s, 'graphite') for sg in (1, -1)] \
      + [cyl([0, sg * (g['shoulder_half_width'] - 0.028 * s), g['shoulder_height']], y, 0.056 * s, 0.034 * s, 'joint')
         for sg in (1, -1)]
    add_body('torso', m_chest + m_head, com_t, I_t, torso_vis)
    rotary('waist_yaw', 'pelvis', 'torso', tf((0., 0., g['waist_height'])), z, -1.0, 1.0, 'waist_yaw')

    # ---------------------------------------------------------------- legs and arms
    for side, sgn in (('l', 1.), ('r', -1.)):
        p = side + '_'
        L, Ls = g['thigh'], g['shank']
        fl, fw, ah, ff = g['foot_length'], g['foot_width'], g['ankle_height'], g['foot_forward']
        # ---- leg chain
        add_body(p + 'hip_yaw_link', 1.2, [0., 0., 0.], sphere_inertia(1.2, 0.05 * s), [cyl([0, 0, 0], x, 0.05 * s, 0.045 * s, 'joint')])
        add_body(p + 'hip_roll_link', 1.2, [0., 0., 0.], sphere_inertia(1.2, 0.05 * s), [])
        add_body(p + 'thigh', 5.5, [0., 0., -0.45 * L], capsule_inertia(5.5, 0.055 * s, L * .85),
                 [cyl([0, 0, 0], y, 0.060 * s, 0.050 * s, 'joint'),
                  shell([(-0.16 * L, 0.054 * s, 0.064 * s, 0.022 * s), (-0.42 * L, 0.057 * s, 0.061 * s, 0.024 * s),
                         (-0.74 * L, 0.047 * s, 0.051 * s, 0.019 * s), (-0.87 * L, 0.041 * s, 0.045 * s, 0.015 * s)], 'shell'),
                  rbox([0.077 * s, 0, -0.42 * L], (0.006 * s, 0.030 * s, 0.075 * s), accent, 6.),
                  cyl([0, 0, -L], y, 0.050 * s, 0.052 * s, 'joint')])
        add_body(p + 'shank', 3.4, [0., 0., -0.42 * Ls], capsule_inertia(3.4, 0.042 * s, Ls * .85),
                 [shell([(-0.12 * Ls, 0.044 * s, 0.048 * s, 0.020 * s), (-0.40 * Ls, 0.046 * s, 0.046 * s, 0.022 * s),
                         (-0.78 * Ls, 0.035 * s, 0.038 * s, 0.016 * s), (-0.88 * Ls, 0.031 * s, 0.034 * s, 0.012 * s)], 'shell'),
                  cyl([0, 0, -Ls], y, 0.034 * s, 0.040 * s, 'joint')])
        add_body(p + 'ankle_link', 0.2, [0., 0., 0.], sphere_inertia(0.2, 0.03 * s), [sph([0, 0, 0], 0.028 * s, 'joint')])
        add_body(p + 'foot', 1.1, [ff * .8, 0., -ah + 0.02 * s], box_inertia(1.1, (fl, fw, 0.035 * s)),
                 [box([ff, 0, -ah + 0.015 * s], (fl, fw, 0.03 * s), 'rubber'),
                  rbox([ff - 0.006 * s, 0, -ah + 0.047 * s], (0.094 * s, 0.045 * s, 0.021 * s), 'shell', 4.),
                  rbox([ff + 0.084 * s, 0, -ah + 0.041 * s], (0.022 * s, 0.046 * s, 0.015 * s), accent, 4.)])
        rotary(p + 'hip_yaw', 'pelvis', p + 'hip_yaw_link', tf((0., sgn * g['hip_half_width'], -g['hip_drop'])), z,
               -0.7, 0.7, 'hip_yaw')
        rotary(p + 'hip_roll', p + 'hip_yaw_link', p + 'hip_roll_link', None, x, -0.6, 0.6, 'hip_roll')
        rotary(p + 'hip_pitch', p + 'hip_roll_link', p + 'thigh', None, y, -1.9, 0.7, 'hip_pitch', q0=-0.38)
        passive(p + 'knee', p + 'thigh', p + 'shank', tf((0., 0., -L)), None, y, 0.03, 2.2, q0=0.76)
        passive(p + 'ankle_pitch', p + 'shank', p + 'ankle_link', tf((0., 0., -Ls)), None, y, -0.9, 0.9, q0=-0.38)
        passive(p + 'ankle_roll', p + 'ankle_link', p + 'foot', None, None, x, -0.5, 0.5)
        # ---- knee actuator: barrel pinned behind the thigh, rod pinned behind the shank
        ka, kb = np.array([-0.055 * s, 0., -0.06 * s]), np.array([-0.055 * s, 0., -0.10 * s])
        actuator_bodies(p + 'knee', 0.022 * s, 0.20 * s, 0.6, 0.20 * s, 0.4)
        passive(p + 'knee_barrel_pivot', p + 'thigh', p + 'knee_barrel', tf(ka), None, y)
        linear(p + 'knee_stroke', p + 'knee_barrel', p + 'knee_rod', 0.22 * s, -0.005 * s, 0.215 * s, 'knee_stroke', 0.1 * s)
        passive(p + 'knee_rod_pin', p + 'knee_rod', p + 'shank', None, tf(kb), y, bearing=False)
        # ---- ankle struts: universal joint on the shank, ball joint on the foot
        for tag, off in (('o', sgn * 0.04 * s), ('i', -sgn * 0.04 * s)):
            q = p + 'ankle_' + tag
            top, bottom = np.array([-0.05 * s, off, -0.10 * s]), np.array([-0.05 * s, off, -0.01 * s])
            add_body(q + '_cross', 0.03, [0., 0., 0.], sphere_inertia(0.03, 0.012 * s), [sph([0, 0, 0], 0.013 * s, 'joint')])
            actuator_bodies(q, 0.014 * s, 0.17 * s, 0.35, 0.15 * s, 0.25)
            passive(q + '_u1', p + 'shank', q + '_cross', tf(top), None, x, -1.2, 1.2)
            passive(q + '_u2', q + '_cross', q + '_barrel', None, None, y, -1.2, 1.2)
            linear(q + '_stroke', q + '_barrel', q + '_rod', 0.20 * s, -0.005 * s, 0.17 * s, 'ankle_stroke', 0.08 * s)
            passive(q + '_ball', q + '_rod', p + 'foot', None, tf(bottom), kind='spherical')
        # ---- arm chain
        sh = tf((0., sgn * g['shoulder_half_width'], g['shoulder_height']))
        ua, fa, pm = g['upper_arm'], g['forearm'], g['palm']
        add_body(p + 'shoulder_link1', 0.7, [0., 0., 0.], sphere_inertia(0.7, 0.045 * s), [cyl([0, 0, 0], x, 0.048 * s, 0.040 * s, 'joint')])
        add_body(p + 'shoulder_link2', 0.7, [0., 0., 0.], sphere_inertia(0.7, 0.045 * s), [])
        add_body(p + 'upper_arm', 1.8, [0., 0., -0.45 * ua], capsule_inertia(1.8, 0.038 * s, ua * .8),
                 [shell([(-0.13 * ua, 0.038 * s, 0.042 * s, 0.012 * s), (-0.50 * ua, 0.041 * s, 0.040 * s, 0.014 * s),
                         (-0.84 * ua, 0.033 * s, 0.034 * s, 0.010 * s)], 'shell', 2.4),
                  cyl([0, 0, -ua], y, 0.036 * s, 0.040 * s, 'joint')])
        m_fa, m_hand = 0.9, 0.4
        c_fa, c_hand = np.array([0., 0., -0.45 * fa]), np.array([0., 0., -pm])
        com_f = (m_fa * c_fa + m_hand * c_hand) / (m_fa + m_hand)
        I_f = capsule_inertia(m_fa, 0.030 * s, fa * .8) + sphere_inertia(m_hand, 0.04 * s)
        for m_, c_ in ((m_fa, c_fa), (m_hand, c_hand)):
            dvec = c_ - com_f
            I_f = I_f + m_ * (dvec @ dvec * np.eye(3) - np.outer(dvec, dvec))
        th_y = sgn * 0.040 * s                     # thumb on the inner side of the hand
        hand = [rbox([0, 0, -pm + 0.012 * s], (0.016 * s, 0.037 * s, 0.040 * s), 'graphite', 4.)]
        hand += [cap([0.004 * s, yy * s, -pm - 0.024 * s], [0.020 * s, yy * s, -pm - 0.066 * s], 0.0085 * s, 'graphite')
                 for yy in (-0.027, -0.009, 0.009, 0.027)]
        hand += [cap([0.010 * s, th_y, -pm + 0.004 * s], [0.032 * s, th_y * 1.2, -pm - 0.028 * s], 0.0095 * s, 'graphite')]
        add_body(p + 'forearm', m_fa + m_hand, com_f, I_f,
                 [shell([(-0.12 * fa, 0.035 * s, 0.036 * s, 0.008 * s), (-0.50 * fa, 0.034 * s, 0.034 * s, 0.008 * s),
                         (-0.88 * fa, 0.028 * s, 0.029 * s, 0.004 * s)], 'shell', 2.3),
                  cyl([0, 0, -fa + 0.004 * s], z, 0.031 * s, 0.009 * s, accent)] + hand)
        rotary(p + 'shoulder_pitch', 'torso', p + 'shoulder_link1', sh, y, -3.0, 1.2, 'shoulder_pitch', q0=-0.15)
        lo, hi = (-0.3, 2.8) if sgn > 0 else (-2.8, 0.3)
        rotary(p + 'shoulder_roll', p + 'shoulder_link1', p + 'shoulder_link2', None, x, lo, hi, 'shoulder_roll', q0=sgn * 0.55)
        rotary(p + 'shoulder_yaw', p + 'shoulder_link2', p + 'upper_arm', None, z, -1.5, 1.5, 'shoulder_yaw')
        passive(p + 'elbow', p + 'upper_arm', p + 'forearm', tf((0., 0., -ua)), None, (0., -1., 0.), 0.08, 2.1, q0=0.7)
        # ---- elbow actuator: barrel behind the upper arm, rod on a lever behind the elbow
        ea, eb = np.array([-0.04 * s, 0., ua * -0.22]), np.array([-0.04 * s, 0., 0.02 * s])
        actuator_bodies(p + 'elbow', 0.014 * s, 0.13 * s, 0.3, 0.12 * s, 0.2)
        passive(p + 'elbow_barrel_pivot', p + 'upper_arm', p + 'elbow_barrel', tf(ea), None, y)
        linear(p + 'elbow_stroke', p + 'elbow_barrel', p + 'elbow_rod', 0.14 * s, -0.005 * s, 0.135 * s, 'elbow_stroke', 0.06 * s)
        passive(p + 'elbow_rod_pin', p + 'elbow_rod', p + 'forearm', None, tf(eb), y, bearing=False)

    sole_sites, point_sites = [], []
    for side, sgn in (('left', 1.), ('right', -1.)):
        p = side[0] + '_'
        hx, hy = g['foot_length'] / 2, g['foot_width'] / 2
        centre = np.array([g['foot_forward'], 0., -g['ankle_height']])
        sole_sites.append({'id': side + '_sole', 'body': p + 'foot', 'frame_translation_m': centre.tolist(),
                           'frame_rotation': np.eye(3).tolist(),
                           'corners_m': [[sx * hx, sy * hy, 0.] for sx in (-1, 1) for sy in (-1, 1)],
                           'half_extent_m': [hx, hy]})
        point_sites.append({'id': side + '_palm', 'body': p + 'forearm', 'point_m': [0., 0., -g['palm']]})
        point_sites.append({'id': side + '_shoulder', 'body': 'torso',
                            'point_m': [0., sgn * g['shoulder_half_width'] * .8, g['shoulder_height'] + 0.07 * s]})
    rev = [j['id'] for j in joints if j['type'] == 'revolute']
    main = ['waist_yaw'] + [f'{sd}_{n}' for sd in 'lr' for n in
                            ('hip_yaw', 'hip_roll', 'hip_pitch', 'knee', 'ankle_pitch', 'ankle_roll',
                             'shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow')]
    assert set(main) <= set(rev)
    record = {
        'schema': 'cmg.physical-mechanism-graph/1.0', 'name': name, 'floating_root': 'pelvis',
        'gravity_m_s2': [0., 0., -9.81], 'bodies': bodies, 'joints': joints, 'actuators': actuators,
        'drive': drive, 'sole_sites': sole_sites, 'point_sites': point_sites,
        'partition_requests': {'joint_space': main, 'actuator_space': [a['joint_id'] for a in actuators]},
        'seed': {'joints': seed},
        'design': {'scale': s, 'arm_scale': float(arm_scale), 'leg_scale': float(leg_scale), 'mass_scale': ms,
                   'color': color, 'geometry_m': g},
        'provenance': {'origin': 'authored for this package; no third-party model data',
                       'units': 'SI', 'frames': 'x forward, y left, z up; body frames at the proximal joint',
                       'drives': 'representative motor and transmission data; not a particular product'},
    }
    return record


LEG_JOINTS = ('hip_yaw', 'hip_roll', 'hip_pitch', 'knee', 'ankle_pitch', 'ankle_roll')
ARM_JOINTS = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow')
