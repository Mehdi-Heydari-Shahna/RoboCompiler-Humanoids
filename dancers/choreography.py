"""Choreography for the hand-linked formations and its reference on the closure manifold.

A dance is a list of smooth primitives acting on formation-relative quantities of
every dancer: pelvis offset (forward, lateral, vertical), pelvis yaw/pitch/roll,
waist yaw, and the height and reach of the hand-link points.  Phase lags between
dancers turn a primitive into a canon (a wave travelling around the circle or along
the line).

The reference is generated on the compiled model.  Pelvis poses and waists are
independent coordinates and are set directly.  Each hand-link point is a task of
the right arm that holds it (three equations, four joints) resolved with the
compiled frame Jacobian; the redundant direction keeps a nominal elbow posture.
The left arms, the legs and all actuator loops then follow from the PACDM closure
correction, so every sample satisfies all chords of the mode.
"""
from __future__ import annotations

import numpy as np

from ipd import se3
from ipd.control import Reference
from ipd.shapes import rot

from .humanoid import ARM_JOINTS


def smoothstep(x):
    """C2 step from 0 to 1 on [0, 1]."""
    x = np.clip(x, 0., 1.)
    return x * x * x * (10. - 15. * x + 6. * x * x)


def window(t, t0, t1, ramp=0.6):
    """C2 window that is 1 on [t0 + ramp, t1 - ramp] and 0 outside [t0, t1]."""
    return smoothstep((t - t0) / ramp) * smoothstep((t1 - t) / ramp)


def wave(t, t0, t1, period, phase=0., ramp=0.8):
    """Raised cosine oscillation in [0, 1] inside a window (0 at the window ends)."""
    return window(t, t0, t1, ramp) * .5 * (1. - np.cos(2. * np.pi * (t - t0) / period - phase))


def swing(t, t0, t1, period, phase=0., ramp=0.8):
    """Sine oscillation in [-1, 1] inside a window."""
    return window(t, t0, t1, ramp) * np.sin(2. * np.pi * (t - t0) / period - phase)


SEGMENTS = (('settle', 1.5), ('bounce canon', 7.5), ('arm wave', 13.5), ('breathing', 19.5), ('sway and twist', 25.5),
            ('bow', 1e9))


class Dance:
    """Formation-relative motion of every dancer as functions of time (thirty seconds).

    settle 0-1.5 s, bounce canon (knee bounces travelling around the formation, 0.75 s period),
    arm wave (the linked hands rise by 0.30 m in a travelling wave, 1.25 s period), breathing
    (common lean with the hands, 1.5 s), sway and twist (lateral sway, pelvis and waist twist,
    1.5 s), bow.
    """

    def __init__(self, n, formation, duration=30., scale=1.0):
        self.n, self.formation, self.duration, self.scale = n, formation, duration, scale
        self.lag = 2. * np.pi * np.arange(n) / n if formation == 'circle' else np.pi * np.arange(n) / (n - 1)
        # a closed ring carries part of a common lean through its hand links; open formations lean less
        self.lean = {'circle': 1.0, 'semicircle': 0.85, 'line': 0.7}[formation]

    def pelvis(self, t):
        """(n, 6): forward, lateral, vertical offsets (m) and yaw, pitch, roll (rad) in the formation frame."""
        n, a, lag, T = self.n, self.scale, self.lag, self.duration
        out = np.zeros((n, 6))
        # bounce canon: quick knee bounces in a travelling wave, with a sideways rock
        out[:, 2] -= 0.050 * a * wave(t, 1.5, 7.5, 0.75, lag, 0.5)
        out[:, 1] += 0.020 * a * swing(t, 1.5, 7.5, 1.5, lag, 0.5)
        out[:, 5] += 0.025 * a * swing(t, 1.5, 7.5, 1.5, lag, 0.5)
        # arm wave: a small lift with the hands
        out[:, 2] += 0.012 * a * wave(t, 7.5, 13.5, 1.25, lag, 0.5)
        # breathing: everyone leans in and out together, with a shallow plie
        out[:, 0] += 0.024 * a * self.lean * swing(t, 13.5, 19.5, 1.5, 0., 0.6)
        out[:, 2] -= 0.030 * a * wave(t, 13.5, 19.5, 0.75, 0., 0.6)
        out[:, 4] += 0.040 * a * self.lean * swing(t, 13.5, 19.5, 1.5, 0., 0.6)
        # sway and twist: the whole formation rocks sideways and turns
        out[:, 1] += 0.040 * a * swing(t, 19.5, 25.5, 1.5, 0., 0.6)
        out[:, 3] += 0.12 * a * swing(t, 19.5, 25.5, 1.5, np.pi / 2, 0.6)
        out[:, 5] += 0.025 * a * swing(t, 19.5, 25.5, 1.5, 0., 0.6)
        out[:, 2] -= 0.020 * a * wave(t, 19.5, 25.5, 0.75, 0., 0.6)
        # closing bow
        out[:, 4] += 0.14 * a * wave(t, 25.5, T - 0.3, T - 0.3 - 25.5, 0., 0.9)
        out[:, 0] -= 0.030 * a * wave(t, 25.5, T - 0.3, T - 0.3 - 25.5, 0., 0.9)
        out[:, 2] -= 0.030 * a * wave(t, 25.5, T - 0.3, T - 0.3 - 25.5, 0., 0.9)
        return out

    def waist(self, t):
        return 0.30 * self.scale * swing(t, 19.5, 25.5, 1.5, np.pi / 2, 0.6) * np.ones(self.n)

    def hands(self, t):
        """(n, 3): rise (m), reach toward the formation centre (m) and lateral shift of the link held by
        the right hand of each dancer, in addition to following the two pelvises."""
        n, a, lag = self.n, self.scale, self.lag
        out = np.zeros((n, 3))
        out[:, 0] += 0.30 * a * wave(t, 7.5, 13.5, 1.25, lag + np.pi / n, 0.5)
        out[:, 0] += 0.18 * a * wave(t, 13.5, 19.5, 1.5, np.pi / 2, 0.6)
        out[:, 1] += 0.04 * a * wave(t, 13.5, 19.5, 1.5, np.pi / 2, 0.6)
        out[:, 0] += 0.06 * a * wave(t, 19.5, 25.5, 0.75, 0., 0.6)
        return out

    def free_arm(self, t):
        """(n, 4) joint offsets of an arm that holds no link (ends of a line)."""
        n, a = self.n, self.scale
        out = np.zeros((n, 4))
        out[:, 1] += 0.60 * a * wave(t, 7.5, 13.5, 1.25, self.lag, 0.5)
        out[:, 3] += 0.35 * a * wave(t, 7.5, 13.5, 1.25, self.lag, 0.5)
        out[:, 1] += 0.35 * a * wave(t, 13.5, 19.5, 1.5, np.pi / 2, 0.6)
        out[:, 1] += 0.20 * a * wave(t, 19.5, 25.5, 0.75, 0., 0.6)
        return out

    def segment(self, t):
        """Name of the choreography segment at time ``t`` (for captions)."""
        return next(name for name, t1 in SEGMENTS if t < t1)


def _relative_pose(p):
    """Transform of the pelvis relative to its formation pose from the six offsets."""
    T = np.eye(4)
    T[:3, 3] = p[:3]
    T[:3, :3] = rot('z', p[3]) @ rot('y', p[4]) @ rot('x', p[5])
    return T


def generate(ens, manager, dance, dt=0.002, ik_iterations=3, posture_gain=4., progress=None):
    """Reference of ``dance`` in the closure mode held by ``manager`` (a ``PartitionManager``).

    Both palms of every hand link are driven to the same choreographed point by
    resolved-rate inverse kinematics of their four arm joints; the PACDM correction
    then closes all chords exactly.  The reference is stored in full coordinates and
    does not depend on which arm joints the current partition treats as dependent.
    Velocities and accelerations are centred differences in velocity coordinates.
    """
    s = ens.system
    comp = manager.comp
    q0 = manager.q
    N = int(round(dance.duration / dt)) + 1
    t = np.arange(N) * dt
    T0 = q0.T_root.copy()
    idx = lambda name: s.index[name]
    waist = np.array([idx(f'd{k}/waist_yaw') for k in range(ens.n)])
    rarm = [np.array([idx(f'd{k}/r_{j}') for j in ARM_JOINTS]) for k in range(ens.n)]
    larm = [np.array([idx(f'd{k}/l_{j}') for j in ARM_JOINTS]) for k in range(ens.n)]
    waist0 = q0.theta[waist].copy()
    theta_nom = q0.theta.copy()
    holds = []                                      # (link, right-hand dancer, left-hand dancer)
    for l in ens.links:
        if l['id'] in comp.links:
            holds.append((l, int(l['member2'][1:]), int(l['member1'][1:])))
    by_right = {kr for _, kr, _ in holds}
    by_left = {kl for _, _, kl in holds}
    rpalm = {k: s.site_frame(f'd{k}', 'right_palm') for k in range(ens.n)}
    lpalm = {k: s.site_frame(f'd{k}', 'left_palm') for k in range(ens.n)}
    Tp = s.poses(q0)
    p_link0 = {l['id']: (Tp[rpalm[kr][0]] @ rpalm[kr][1])[:3, 3].copy() for l, kr, kl in holds}
    centre = np.mean(T0[:, :3, 3], axis=0)
    inward, side = {}, {}
    for l, kr, kl in holds:
        d_in = centre - p_link0[l['id']]
        d_in[2] = 0.
        nrm = np.linalg.norm(d_in)
        # in a line the "centre" direction is the common facing direction of the dancers
        inward[l['id']] = d_in / nrm if ens.formation != 'line' and nrm > 1e-6 else T0[kr][:3, 0].copy()
        side[l['id']] = np.cross([0., 0., 1.], inward[l['id']])
    T_root = np.empty((N,) + T0.shape)
    theta = np.empty((N, s.n))
    q = q0.copy()
    worst = dict(residual=0., iterations=0, limit_margin=np.inf, ik_error=0.)
    roots = np.concatenate([np.arange(sl.start, sl.stop) for sl in s.root_slice if sl is not None])
    scalar = np.setdiff1d(np.arange(s.nT), roots)
    eye4 = np.eye(4)

    def ik(arm, frame, target, kin, first, nominal):
        b, P = frame
        p = (kin.T[b] @ P)[:3, 3]
        Sa = kin.S[:, arm]
        J = Sa[3:] + np.cross(Sa[:3].T, p).T                # palm velocity per arm joint rate
        Jp = J.T @ np.linalg.inv(J @ J.T + 1e-9 * np.eye(3))
        step = Jp @ (target - p)
        if first:
            step += (eye4 - Jp @ J) @ (posture_gain * dt * (nominal - q.theta[arm]))
        q.theta[arm] = np.clip(q.theta[arm] + step, s.lower[arm] + 0.03, s.upper[arm] - 0.03)
        return float(np.linalg.norm(target - p))

    for i in range(N):
        pel = dance.pelvis(t[i])
        for k in range(ens.n):
            q.T_root[k] = T0[k] @ _relative_pose(pel[k])
        q.theta[waist] = waist0 + dance.waist(t[i])
        free = dance.free_arm(t[i])
        for k in range(ens.n):
            if k not in by_right:
                q.theta[rarm[k]] = theta_nom[rarm[k]] + np.array([1., -1., -1., 1.]) * free[k]
            if k not in by_left:
                q.theta[larm[k]] = theta_nom[larm[k]] + free[k]
        if holds:
            hands = dance.hands(t[i])
            targets = {}
            for l, kr, kl in holds:
                shift = .5 * ((q.T_root[kr][:3, 3] - T0[kr][:3, 3]) + (q.T_root[kl][:3, 3] - T0[kl][:3, 3]))
                targets[l['id']] = (p_link0[l['id']] + shift + np.array([0., 0., hands[kr, 0]])
                                    + hands[kr, 1] * inward[l['id']] + hands[kr, 2] * side[l['id']])
            err = 0.
            for it in range(ik_iterations):
                kin = s.kinematics(q)
                err = 0.
                for l, kr, kl in holds:
                    err = max(err, ik(rarm[kr], rpalm[kr], targets[l['id']], kin, it == 0, theta_nom[rarm[kr]]))
                    err = max(err, ik(larm[kl], lpalm[kl], targets[l['id']], kin, it == 0, theta_nom[larm[kl]]))
            worst['ik_error'] = max(worst['ik_error'], err)
        comp, _ = manager.update(q, t[i])
        q, ci = comp.mode.correct(q, tol=1e-11, maxiter=25)
        if not ci['success']:
            raise RuntimeError(f'closure correction failed at t={t[i]:.3f}: {ci}')
        worst['residual'] = max(worst['residual'], ci['residual_inf'])
        worst['iterations'] = max(worst['iterations'], ci['iterations'])
        worst['limit_margin'] = min(worst['limit_margin'], float(np.min(np.minimum(q.theta[scalar] - s.lower[scalar],
                                                                                 s.upper[scalar] - q.theta[scalar]))))
        T_root[i], theta[i] = q.T_root, q.theta
        if progress is not None and i % 1000 == 0:
            progress(i, N)
    v, a = differentiate(s, T_root, theta, dt)
    info = dict(worst, formation=ens.formation, mode=comp.name, partition_switches=len(manager.switches),
                min_arm_rcond=float(manager.min_rcond), switch_times=[sw[0] for sw in manager.switches])
    return Reference(t, T_root, theta, v, a, info=info)


def differentiate(system, T_root, theta, dt):
    """Velocity-coordinate rates of a sampled trajectory by centred differences."""
    N = len(theta)
    v = np.zeros_like(theta)
    v[1:-1] = (theta[2:] - theta[:-2]) / (2 * dt)
    v[0], v[-1] = (theta[1] - theta[0]) / dt, (theta[-1] - theta[-2]) / dt
    for k, sl in enumerate(system.root_slice):
        if sl is None:
            continue
        Tk = T_root[:, k]
        fwd = se3.log(se3.inv(Tk[:-1]) @ Tk[1:]) / dt          # body twist over each interval
        v[1:-1, sl] = .5 * (fwd[1:] + fwd[:-1])
        v[0, sl], v[-1, sl] = fwd[0], fwd[-1]
    a = np.zeros_like(v)
    a[1:-1] = (v[2:] - v[:-2]) / (2 * dt)
    a[0], a[-1] = a[1], a[-2]
    return v, a
