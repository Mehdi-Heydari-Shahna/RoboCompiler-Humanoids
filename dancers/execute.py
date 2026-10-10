"""Closed-loop execution of a dance reference: policies, logging and metrics.

Policies (all share the reference; the model-based ones share law, gains and allocation
and differ only in the model they are given):

* ``ip``            compiled inertia-partitioned model of the complete mode (proposed);
* ``recursive``     same reduction with tree terms from Pinocchio CRBA/RNEA/Coriolis
                    (the projected-tree dynamics of the RoboCompiler paper);
* ``decentralized`` every dancer uses its own compiled model; hand links are not modelled;
* ``simplified``    the inertial terms of the moving actuator bodies and of the rotors are
                    left out of ``Gamma`` and ``C``; their weight stays in ``g``;
* ``massless``      the actuator bodies and rotors are left out altogether (also their weight);
* ``no_curvature``  the constraint-curvature terms (rates of both lifts) are omitted;
* ``stale``         the model keeps the inertias it was compiled with after a payload change;
* ``gravity_pd``    the same feedback coordinates and allocation, but only compiled gravity
                    and constant diagonal gains (no inertia or Coriolis model);
* ``actuator_pd``   independent PD loops at the actuator ports with hand-set gains; no model
                    (only the friction feed-forward that every policy receives).
"""
from __future__ import annotations

import time

import numpy as np

from ipd import se3
from ipd.control import Allocator, Gains, SupportCone, tracking_error, tracking_law
from ipd.updates import constraint_wrenches

from .stage import FLOOR

ACTUATOR_PD = {            # (Kp, Kd) per actuator kind: N m/rad, N m s/rad or N/m, N s/m
    'hip_yaw': (1500., 40.), 'hip': (4000., 80.), 'waist': (900., 25.), 'shoulder': (450., 12.),
    'knee_stroke': (4.0e5, 4.0e3), 'ankle': (2.0e5, 2.0e3), 'elbow_stroke': (6.0e4, 6.0e2),
}


def pd_gains(system, scale=1.):
    """Hand-set actuator PD gains by actuator kind."""
    kp, kd = np.zeros(system.na), np.zeros(system.na)
    for i, pid in enumerate(system.port_ids):
        key = next(k for k in ('knee_stroke', 'elbow_stroke', 'ankle', 'hip_yaw', 'hip', 'waist', 'shoulder') if k in pid)
        kp[i], kd[i] = ACTUATOR_PD[key]
    return scale * kp, np.sqrt(scale) * kd


POLICIES = ('ip', 'recursive', 'decentralized', 'simplified', 'massless', 'no_curvature', 'gravity_pd', 'actuator_pd')


def build_policy(ens, comp, ref, q0, name='ip', **kw):
    """Policy by name; the variants differ only in the model handed to the same law and allocation."""
    kind = name
    if name == 'decentralized':
        model = ens.composite(stance=getattr(comp, 'stance', None), links=[])
        model.prepare(q0)
        kw['comp_model'], kind = model, 'ip'
    elif name == 'recursive':
        from ipd.backends.pin_backend import PinBackend
        kw['pin'] = PinBackend(ens.system)
    elif name in ('simplified', 'massless'):
        from .ensemble import simplified_inertia
        kw['model_M'], kw['model_armature'] = simplified_inertia(ens.system)
        kw['gravity_full'] = name == 'simplified'
        kind = 'ip'
    elif name == 'no_curvature':
        kw['curvature'], kind = False, 'ip'
    elif name not in POLICIES:
        raise ValueError(f'unknown policy {name}')
    policy = Policy(ens, comp, ref, kind=kind, **kw)
    policy.name = name
    return policy


def payload_inertia(system, body, mass, point):
    """Inertial data of ``body`` after a point payload ``mass`` is fixed at ``point`` (body frame).

    Returns the new mass, centre of mass, inertia about the new centre of mass and the
    change ``dM`` of the body-frame spatial inertia.
    """
    m0, c0, I0 = system.mass[body], system.com[body], system.inertia_com[body]
    point = np.asarray(point, float)
    m1 = m0 + mass
    c1 = (m0 * c0 + mass * point) / m1
    I1 = I0.copy()
    for m_, c_ in ((m0, c0), (mass, point)):
        d = c_ - c1
        I1 = I1 + m_ * (d @ d * np.eye(3) - np.outer(d, d))
    dM = se3.spatial_inertia(np.asarray(mass, float), point, np.zeros((3, 3)))
    return m1, c1, I1, dM


FRICTION_FEEDFORWARD = 0.85       # fraction of the identified Coulomb friction of the drives compensated by feed-forward
FRICTION_SMOOTHING = dict(revolute=0.05, prismatic=0.005)      # rad/s and m/s: width of the sign function (tanh)


class Policy:
    def __init__(self, ens, comp, ref, kind='ip', gains=None, model_M=None, model_armature=None, gravity_full=False,
                 curvature=True, friction=FLOOR['friction'], comp_model=None, pin=None, load_schedule=None, pd_scale=1.,
                 follow_anchors=True, link_weight=0., support_weight=0., transfer_load=None,
                 friction_feedforward=FRICTION_FEEDFORWARD):
        self.ens, self.comp, self.ref, self.kind = ens, comp, ref, kind
        self.name = kind
        self.sys = s = ens.system
        self.gains = gains or Gains()
        # the model used by the controller may be a different closure mode (decentralized) or inertia set
        self.model = comp_model or comp
        self.model.follow_anchors = bool(follow_anchors)
        self.M = model_M
        self.armature = model_armature
        self.gravity_full = bool(gravity_full)      # keep the weight of bodies whose inertia the model leaves out
        self.curvature = bool(curvature)
        self.pin = pin
        self.load_schedule = load_schedule
        self.kp, self.kd = pd_gains(s, pd_scale)
        G0 = np.diag(self.model.evaluate(ref.at(0.)[0]).Gamma)
        self.kp0, self.kd0 = self.gains.ks * self.gains.lam * G0, (self.gains.ks + self.gains.lam) * G0
        cones = {}
        for k, m in enumerate(s.members):
            for site in m.comp.cmg['sole_sites']:
                hx, hy = site['half_extent_m']
                cones[f"{m.name}/{site['id']}"] = SupportCone(hx, hy, mu=friction)
        self.alloc = Allocator(self.model, cones, link_weight=link_weight, support_weight=support_weight,
                               transfer_load=transfer_load)
        self.f_prev = None
        self.timing = dict(evaluate=0., allocate=0., calls=0, qp=0)
        # every policy compensates the identified Coulomb friction of the drives along the reference velocity;
        # the compiled model itself carries no Coulomb friction
        from ipd.system import PRISMATIC
        prism = np.zeros(s.n, bool)
        for b in range(s.nbp):
            if s.jtype[b] == PRISMATIC:
                prism[s.dof[b]] = True
        self.f_coulomb = float(friction_feedforward) * s.frictionloss[s.port_dof]
        self.v_smooth = np.where(prism[s.port_dof], FRICTION_SMOOTHING['prismatic'], FRICTION_SMOOTHING['revolute'])

    def _evaluate(self, q, u):
        s = self.sys
        if self.M is None and self.armature is None:
            return self.model.evaluate(q, u, curvature=self.curvature)
        M0, a0 = s.M, s.armature
        try:
            if self.M is not None:
                s.M = self.M
            if self.armature is not None:
                s.armature = self.armature
            ev = self.model.evaluate(q, u, curvature=self.curvature)
        finally:
            s.M, s.armature = M0, a0
        if self.gravity_full:
            ev.g1, ev.g = self.model.gravity(ev)
        return ev

    def __call__(self, t, q, v):
        f, info = self._command(t, q, v)
        if np.any(self.f_coulomb):
            v_ref = self.ref.at(t)[1]
            f = np.clip(f + self.f_coulomb * np.tanh(v_ref[self.sys.port_dof] / self.v_smooth), self.alloc.lower, self.alloc.upper)
        return f, info

    def _command(self, t, q, v):
        s, model = self.sys, self.model
        ind = model.mode.independent
        q_ref, v_ref, a_ref = self.ref.at(t)
        u = v[ind]
        fz_max = self.load_schedule(t) if self.load_schedule else None
        if self.kind == 'actuator_pd':
            ell = q.theta[s.port_dof]; ell_ref = q_ref.theta[s.port_dof]
            f = self.kp * (ell_ref - ell) + self.kd * (v_ref[s.port_dof] - v[s.port_dof])
            return np.clip(f, self.alloc.lower, self.alloc.upper), dict(qp=False, slack=0., wrenches={}, status='none')
        e, e_dot, u_d, ud_d = tracking_error(s, ind, q_ref, v_ref, a_ref, q, v)
        if self.kind == 'gravity_pd':
            t0 = time.perf_counter()
            ev = self.model.evaluate(q)
            Q = ev.g - self.kp0 * e - self.kd0 * (u - u_d)
            t1 = time.perf_counter()
            f, info = self.alloc.solve(ev, Q, np.zeros(self.model.D1), fz_max=fz_max, f_prev=self.f_prev)
            self.f_prev = f
            self.timing['evaluate'] += t1 - t0; self.timing['allocate'] += time.perf_counter() - t1; self.timing['calls'] += 1
            self.timing['qp'] += int(info['qp'])
            info.update(e=e, s=u - u_d + self.gains.lam * e)
            return f, info
        t0 = time.perf_counter()
        ev = self._evaluate(q, u)
        C = ev.C
        if self.kind == 'recursive':
            ev.Gamma, C, ev.g = self._recursive(ev, q, u)
        t1 = time.perf_counter()
        Q, sl = tracking_law(ev, e, e_dot, u, u_d, ud_d, self.gains, C=C)
        ud = np.linalg.solve(ev.Gamma, Q - C @ u - ev.g - ev.D @ u)
        wdot = ev.G @ ud + ev.Gd @ u
        f, info = self.alloc.solve(ev, Q, wdot, fz_max=fz_max, f_prev=self.f_prev)
        self.f_prev = f
        info['Q_defect'] = float(np.max(abs(ev.L.T @ f - Q)))
        t2 = time.perf_counter()
        self.timing['evaluate'] += t1 - t0; self.timing['allocate'] += t2 - t1; self.timing['calls'] += 1
        self.timing['qp'] += int(info['qp'])
        info.update(e=e, s=sl)
        if model.nl:
            ud_f = np.linalg.solve(ev.Gamma, ev.L.T @ f - C @ u - ev.g - ev.D @ u)
            lam = constraint_wrenches(model, ev, f, ud_f)
            info['link_force_model'] = {l: float(np.linalg.norm(lam[l][3:])) for l in model.links}
        return f, info

    def _recursive(self, ev, q, u):
        """Reduced terms from Pinocchio's recursive tree algorithms and the same lift."""
        s = self.sys
        E, Ed = self.model.lift(ev, True)
        ET, ETd = E[:s.nT], Ed[:s.nT]
        vT = ev.v[:s.nT]
        MT = self.pin.mass(q)
        CT = self.pin.coriolis(q, vT)
        return ET.T @ MT @ ET, ET.T @ (CT @ ET + MT @ ETd), ET.T @ self.pin.gravity(q)


FALL_HEIGHT, FALL_UPRIGHT = 0.45, 0.6          # pelvis height (m) and cosine of the pelvis tilt below which a dancer has fallen


def run(ens, comp, ref, policy, plant, duration=None, control_dt=0.002, log_dt=0.01, events=None,
        verbose=False, frame_dt=1. / 30.):
    """Execute ``policy`` on a plant (``dancers.plants``) and log tracking, closure, contact and effort.

    ``events`` is a list of (time, callable(plant, policy)) applied once, e.g. a payload that appears.
    The policy sees the measured state (``plant.measured``; with noise on the realistic plant), the
    log records the true state.

    Timing of the logged forces.  At control step ``k`` the plant is read (its state and the
    constraint forces of the period that just ended, i.e. under the efforts of step ``k-1``);
    then the policy computes the efforts of step ``k`` and the wrenches the model expects
    them to produce.  ``fz_model`` and ``link_force_model`` logged at step ``k`` are therefore
    the predictions made at step ``k-1``: they refer to the same period as the measured
    ``fz`` and ``link_force``.

    A run stops when a dancer falls, when a state or an effort is not finite, or when the
    plant reports a problem (``plant.health()``); the reason is returned as ``failure``.
    """
    s = ens.system
    duration = duration or ref.t[-1]
    n_ctrl = int(round(duration / control_dt))
    every = max(1, int(round(log_dt / control_dt)))
    events = sorted(events or [], key=lambda x: x[0])
    keys = ('t', 'pelvis_err', 'pelvis_rot_err', 'e_inf', 's_inf', 'effort_frac', 'gap', 'fz', 'cop', 'friction_ratio',
            'link_force', 'link_gap', 'f', 'qp', 'slack', 'pelvis', 'pelvis_ref', 'fz_model', 'tilt', 'height', 'power',
            'palm', 'palm_ref', 'link_force_model', 'Q_defect', 'sole', 'sole_ref', 'stance',
            'support_gap')
    log = {k: [] for k in keys}
    soles = [f'd{k}/{side}_sole' for k in range(ens.n) for side in ('left', 'right')]
    palms = [s.site_frame(f'd{k}', f'{side}_palm') for k in range(ens.n) for side in ('left', 'right')]
    link_ids = [l['id'] for l in ens.links]
    sole_frames = [s.site_frame(f'd{k}', f'{side}_sole') for k in range(ens.n) for side in ('left', 'right')]
    measured = getattr(plant, 'measured', None)
    failure = ''
    slack_max, qp_failed = 0., 0                     # over all control steps (the log keeps every ``every``-th step)
    infeasible, saturated = 0, 0
    predicted = dict(fz=[np.nan] * len(soles), link=[np.nan] * len(link_ids))
    frames_T, frames_theta, frames_t, next_frame = [], [], [], 0.
    t0 = time.time()
    for i in range(n_ctrl):
        t = i * control_dt
        while events and events[0][0] <= t + 1e-12:
            events.pop(0)[1](plant, policy)
        q, v = plant.state()
        if not (np.all(np.isfinite(q.T_root)) and np.all(np.isfinite(q.theta)) and np.all(np.isfinite(v))):
            failure = 'state not finite'
            break
        if frame_dt and t + 1e-9 >= next_frame:
            frames_T.append(q.T_root.copy()); frames_theta.append(q.theta[:s.nT].copy()); frames_t.append(t)
            next_frame += frame_dt
        qm, vm = measured(q, v) if measured is not None else (q, v)
        f, info = policy(t, qm, vm)
        if not np.all(np.isfinite(f)):
            failure = 'effort not finite'
            break
        if np.isfinite(info.get('slack', 0.)):
            slack_max = max(slack_max, float(info.get('slack', 0.)))
        qp_failed += int(bool(info.get('failed', False)))
        infeasible += int(not np.isfinite(info.get('slack', 0.)) or info.get('slack', 0.) > INFEASIBLE_SLACK)
        saturated += int(np.max(abs(f) / s.effort_upper) >= SATURATED)
        f_act = f
        Tm = q.T_root
        if i % every == 0:
            q_ref, v_ref, _ = ref.at(t)
            Tr = q_ref.T_root
            perr = np.linalg.norm(Tm[:, :3, 3] - Tr[:, :3, 3], axis=1)
            rerr = np.linalg.norm(se3.log_so3(np.swapaxes(Tr[:, :3, :3], 1, 2) @ Tm[:, :3, :3]), axis=1)
            ms = plant.measure()
            Tb, Tbr = s.poses(q), s.poses(q_ref)
            log['t'].append(t); log['pelvis_err'].append(perr); log['pelvis_rot_err'].append(rerr)
            log['e_inf'].append(float(np.max(abs(info['e']))) if 'e' in info else np.nan)
            log['s_inf'].append(float(np.max(abs(info['s']))) if 's' in info else np.nan)
            log['effort_frac'].append(float(np.max(abs(f_act) / s.effort_upper)))
            for k in ('gap', 'fz', 'cop', 'friction_ratio', 'link_force', 'link_gap', 'support_gap'):
                log[k].append(ms[k])
            log['f'].append(f_act.copy()); log['qp'].append(int(info.get('qp', False))); log['slack'].append(info.get('slack', 0.))
            log['pelvis'].append(Tm[:, :3, 3].copy()); log['pelvis_ref'].append(Tr[:, :3, 3].copy())
            log['fz_model'].append(list(predicted['fz'])); log['link_force_model'].append(list(predicted['link']))
            log['tilt'].append(np.degrees(np.arccos(np.clip(Tm[:, 2, 2], -1, 1))))
            log['height'].append(Tm[:, 2, 3].copy())
            log['power'].append(float(f_act @ v[s.port_dof]))
            log['palm'].append(np.array([(Tb[b] @ P)[:3, 3] for b, P in palms]))
            log['palm_ref'].append(np.array([(Tbr[b] @ P)[:3, 3] for b, P in palms]))
            log['sole'].append(np.array([(Tb[b] @ P)[:3, 3] for b, P in sole_frames]))
            log['sole_ref'].append(np.array([(Tbr[b] @ P)[:3, 3] for b, P in sole_frames]))
            log['stance'].append({'both': 0, 'left': 1, 'right': 2}.get(info.get('stance', 'both'), 0))
            log['Q_defect'].append(info.get('Q_defect', np.nan))
            if verbose and i % (every * 100) == 0:
                print('  t=%5.2f  pelvis err %.3f mm  e %.1e  gap %.1e  effort %.2f  minFz %.0f' % (
                    t, 1e3 * perr.max(), log['e_inf'][-1], log['gap'][-1], log['effort_frac'][-1], np.nanmin(ms['fz'])), flush=True)
        if np.any(Tm[:, 2, 3] < FALL_HEIGHT) or np.any(Tm[:, 2, 2] < FALL_UPRIGHT):
            failure = 'fell'
            break
        lam = info.get('wrenches', {})
        lfm = info.get('link_force_model', {})
        predicted = dict(fz=[lam[k][5] if k in lam else np.nan for k in soles], link=[lfm.get(l, np.nan) for l in link_ids])
        plant.advance(f_act, control_dt)
        problem = plant.health() if hasattr(plant, 'health') else ''
        if problem:
            failure = problem
            break
    out = {k: np.array(v, float) for k, v in log.items()}
    out['failure'] = failure
    out['fallen'] = bool(failure)
    out['frames_t'], out['frames_T_root'], out['frames_theta'] = np.array(frames_t), np.array(frames_T), np.array(frames_theta)
    out['wall_s'] = time.time() - t0
    out['slack_max_all_steps'], out['qp_failed_steps'] = slack_max, qp_failed
    out['infeasible_steps'], out['saturated_steps'] = infeasible, saturated
    out['sole_keys'] = soles
    out['effort_upper'] = s.effort_upper.copy()
    out['timing'] = dict(policy.timing) if hasattr(policy, 'timing') else {}
    out['qp_status'] = dict(getattr(getattr(policy, 'alloc', None), 'counts', {}) or {})
    out['control_steps'] = i + (0 if failure else 1)
    return out


LOADED_SOLE_N = 50.      # centre-of-pressure and friction statistics use soles that carry at least this normal force
CONTACT_FROM_S = 0.5     # contact statistics start here: the plants start with unloaded contacts
INFEASIBLE_SLACK = 0.1   # N or N m: an allocation with a larger slack does not realise the commanded generalized force
SATURATED = 0.99         # a drive at this fraction of its effort bound or above counts as saturated


def metrics(log, t_min=0.):
    """Scalar summary of a run (tracking, closure, contact and effort).

    All statistics cover the logged samples with ``t >= t_min``; the default is the whole
    run, including the initial settling of the plant.  The only exception are the contact
    statistics (normal force, centre of pressure, friction ratio), which start at
    ``CONTACT_FROM_S`` because a plant with contact starts with unloaded soles.
    ``max_slack`` is the largest allocation slack of all control steps when the log carries
    it (``slack_max_all_steps``), otherwise of the logged samples; ``qp_failed_steps``
    counts the control steps in which the allocation had no solution and held the efforts.
    ``infeasible_fraction`` and ``effort_saturated_fraction`` are the fractions of all control steps
    with an allocation slack above ``INFEASIBLE_SLACK`` and with a drive at ``SATURATED`` of its bound
    (of the logged samples when the log does not carry the counts).
    """
    t = log['t']
    m = t >= t_min
    pe = log['pelvis_err'][m]
    mc = t >= max(t_min, min(CONTACT_FROM_S, .5 * t[-1]))
    fz = log['fz'][mc]
    has_e = bool(np.isfinite(log['e_inf'][m]).any())
    has_link = bool(np.isfinite(log['link_force'][m]).any())
    loaded = np.nan_to_num(fz, nan=0.) > LOADED_SOLE_N
    cop = np.where(loaded[:, :, None], np.abs(log['cop'][mc]), 0.)
    fric = np.where(loaded, log['friction_ratio'][mc], 0.)
    sole_d = log['sole'][m] - log['sole_ref'][m]
    sole_err = np.linalg.norm(sole_d, axis=2)
    status = dict(log.get('qp_status', {}))
    solved = sum(v for k, v in status.items() if k in ('closed_form', 'solved'))
    slack = np.asarray(log['slack'][m], float)
    slack_logged = float(np.nanmax(slack)) if np.isfinite(slack).any() else 0.
    return dict(
        completed=bool(not log['fallen']), failure=str(log.get('failure', '')), simulated_s=float(t[-1]),
        window_s=[float(t[m][0]), float(t[m][-1])],
        pelvis_rms_mm=float(1e3 * np.sqrt(np.mean(pe ** 2))), pelvis_peak_mm=float(1e3 * pe.max()),
        pelvis_final_mm=float(1e3 * log['pelvis_err'][-1].max()),
        pelvis_rot_rms_deg=float(np.degrees(np.sqrt(np.mean(log['pelvis_rot_err'][m] ** 2)))),
        pelvis_rot_peak_deg=float(np.degrees(np.max(log['pelvis_rot_err'][m]))),
        e_rms=float(np.sqrt(np.nanmean(log['e_inf'][m] ** 2))) if has_e else float('nan'),
        e_peak=float(np.nanmax(log['e_inf'][m])) if has_e else float('nan'),
        loop_gap_peak_um=float(1e6 * np.nanmax(log['gap'][m])),
        link_gap_peak_um=float(1e6 * np.nanmax(log['link_gap'][m])) if np.isfinite(log['link_gap'][m]).any() else float('nan'),
        support_gap_peak_um=float(1e6 * np.nanmax(log['support_gap'][m])) if np.isfinite(log['support_gap'][m]).any() else float('nan'),
        link_force_peak_N=float(np.nanmax(log['link_force'][m])) if has_link else float('nan'),
        link_force_rms_N=float(np.sqrt(np.nanmean(log['link_force'][m] ** 2))) if has_link else float('nan'),
        effort_peak_fraction=float(log['effort_frac'][m].max()),
        effort_saturated_fraction=(float(log['saturated_steps']) / max(1, int(log['control_steps'])) if 'saturated_steps' in log and t_min <= 0.
                                   else float(np.mean(log['effort_frac'][m] >= SATURATED))),
        effort_rms_fraction=float(np.sqrt(np.mean((log['f'][m] / log['effort_upper']) ** 2))),
        min_fz_N=float(np.nanmin(fz)), max_friction_ratio=float(fric.max()),
        max_cop_x_m=float(cop[:, :, 0].max()), max_cop_y_m=float(cop[:, :, 1].max()), loaded_sole_N=LOADED_SOLE_N,
        contact_window_s=[float(t[mc][0]), float(t[mc][-1])],
        palm_rms_mm=float(1e3 * np.sqrt(np.mean(np.sum((log['palm'][m] - log['palm_ref'][m]) ** 2, axis=2)))),
        sole_error_peak_mm=float(1e3 * sole_err.max()), sole_error_final_mm=float(1e3 * sole_err[-1].max()),
        sole_error_horizontal_peak_mm=float(1e3 * np.linalg.norm(sole_d[:, :, :2], axis=2).max()),
        sole_error_vertical_peak_mm=float(1e3 * np.abs(sole_d[:, :, 2]).max()),
        max_tilt_deg=float(log['tilt'][m].max()), qp_fraction=float(np.mean(log['qp'][m])),
        max_slack=float(log['slack_max_all_steps']) if 'slack_max_all_steps' in log and t_min <= 0. else slack_logged,
        max_slack_logged=slack_logged, qp_failed_steps=int(log.get('qp_failed_steps', 0)),
        infeasible_fraction=(float(log['infeasible_steps']) / max(1, int(log['control_steps'])) if 'infeasible_steps' in log and t_min <= 0.
                             else float(np.mean(np.nan_to_num(slack, nan=np.inf) > INFEASIBLE_SLACK))),
        qp_status=status, qp_not_solved=int(sum(status.values()) - solved),
        actuator_work_J=float(np.trapezoid(log['power'][m], t[m])), wall_s=float(log['wall_s']),
    )
