"""Travelling formation: the hand-linked dancers side-step together.

The whole formation moves to the dancers' right: a closed ring and a semicircle turn
about their centre, a line translates.  One step cycle is

    shift   the pelvises move over the left feet and the right feet are unloaded;
    swing   the right feet lift, travel by one step and land (left feet carry the dancers);
    shift   the pelvises move over the right feet, the left feet are unloaded;
    swing   the left feet follow.

Every support change is a change of closure mode (physical reconfiguration): the
member modules of the five dancers are untouched and only the composer is re-solved.
The reference is generated kinematically with all ten sole frames following their
anchor trajectories, so that it is one continuous motion on the closure manifold;
which soles carry load is decided by the support schedule.
"""
from __future__ import annotations

import numpy as np

from ipd import se3
from ipd.control import Reference
from ipd.shapes import rot

from .choreography import differentiate, smoothstep


def bump(x):
    """Smooth lift profile on [0, 1]: zero value and slope at both ends, one at the middle."""
    x = np.clip(x, 0., 1.)
    return 16. * x * x * (1. - x) * (1. - x)


class StepDance:
    def __init__(self, ens, q0, cycles=5, step=0.10, clearance=0.045, shift=0.82, t_settle=1.5, t_shift=1.3, t_swing=0.65,
                 t_end=2.0, unload=0.35, reload=0.30):
        """``step`` is the arc length travelled per cycle (m), ``shift`` the fraction of the
        way from the midpoint between the feet to the stance foot that the pelvis moves."""
        self.ens, self.formation = ens, ens.formation
        self.cycles, self.step, self.clearance, self.shift = int(cycles), float(step), float(clearance), float(shift)
        self.t_settle, self.t_shift, self.t_swing, self.t_end = t_settle, t_shift, t_swing, t_end
        self.unload, self.reload = unload, reload
        self.cycle = 2. * (t_shift + t_swing)
        self.duration = t_settle + self.cycles * self.cycle + t_end
        s = ens.system
        T = s.poses(q0)
        self.T0 = q0.T_root.copy()
        self.sole_ids = [f'd{k}/{side}_sole' for k in range(ens.n) for side in ('left', 'right')]
        self.sole0 = {}
        for k in range(ens.n):
            for side in ('left', 'right'):
                b, P = s.site_frame(f'd{k}', f'{side}_sole')
                self.sole0[f'd{k}/{side}_sole'] = T[b] @ P
        xy = self.T0[:, :2, 3]
        if self.formation == 'line':
            self.centre, self.radius = None, None
            self.travel = -self.T0[0, :3, 1]                # the dancers' right
        else:
            self.centre = np.mean(xy, axis=0) if self.formation == 'circle' else self._arc_centre(xy)
            self.radius = float(np.mean(np.linalg.norm(xy - self.centre, axis=1)))
        # half distance between the feet along the direction of travel
        self.half = float(np.mean([.5 * np.linalg.norm(self.sole0[f'd{k}/left_sole'][:2, 3] - self.sole0[f'd{k}/right_sole'][:2, 3])
                                   for k in range(ens.n)]))

    @staticmethod
    def _arc_centre(xy):
        """Centre of the circle through the pelvis positions of an arc formation (least squares)."""
        A = np.c_[2 * xy, np.ones(len(xy))]
        sol = np.linalg.lstsq(A, np.sum(xy ** 2, axis=1), rcond=None)[0]
        return sol[:2]

    # ------------------------------------------------------------------ timing
    def advances(self, t):
        """Arc lengths travelled by the right feet, the left feet, and the lift fractions of both."""
        aR = aL = liftR = liftL = 0.
        for c in range(self.cycles):
            t0 = self.t_settle + c * self.cycle
            xR = (t - t0 - self.t_shift) / self.t_swing
            xL = (t - t0 - 2. * self.t_shift - self.t_swing) / self.t_swing
            aR += self.step * smoothstep(xR)
            aL += self.step * smoothstep(xL)
            liftR += bump(xR) if 0. < xR < 1. else 0.
            liftL += bump(xL) if 0. < xL < 1. else 0.
        return aR, aL, liftR, liftL

    def weight(self, t):
        """Position of the pelvis between the feet: -shift over the left foot, +shift over the right foot.

        The transfers are slow enough for the centre of pressure to stay inside the
        support (they end before the unloading of the next swing foot ends).
        """
        ts = self.t_shift * 0.75
        w = 0.
        for c in range(self.cycles):
            t0 = self.t_settle + c * self.cycle
            if t < t0:
                break
            start = 0. if c == 0 else 1.                      # from the centre, later from the right foot
            w = start + (-1. - start) * smoothstep((t - t0) / ts)
            w += 2. * smoothstep((t - t0 - self.t_shift - self.t_swing) / ts)
        t_last = self.t_settle + self.cycles * self.cycle
        if t >= t_last:
            w = 1. - smoothstep((t - t_last) / (self.t_end * 0.75))
        return self.shift * w

    def stance(self, t):
        """Load-bearing soles: 'both', 'left' or 'right' (the same for every dancer)."""
        for c in range(self.cycles):
            t0 = self.t_settle + c * self.cycle
            a = t0 + self.t_shift
            if a - 0.02 <= t < a + self.t_swing + 0.04:
                return 'left'
            b = t0 + 2. * self.t_shift + self.t_swing
            if b - 0.02 <= t < b + self.t_swing + 0.04:
                return 'right'
        return 'both'

    def load_caps(self, t, weight_N):
        """Upper bounds of the normal forces of the soles that are about to lift or have just landed."""
        caps = {}
        for c in range(self.cycles):
            t0 = self.t_settle + c * self.cycle
            for side, lift in (('right', t0 + self.t_shift), ('left', t0 + 2. * self.t_shift + self.t_swing)):
                land = lift + self.t_swing + 0.04
                if lift - self.unload <= t < lift - 0.02:
                    cap = weight_N * (1. - smoothstep((t - (lift - self.unload)) / (self.unload - 0.05)))
                elif land <= t < land + self.reload:
                    cap = weight_N * smoothstep((t - land) / self.reload)
                else:
                    continue
                for k in range(self.ens.n):
                    caps[f'd{k}/{side}_sole'] = float(cap)
        return caps

    # ------------------------------------------------------------------ geometry
    def move(self, T, arc):
        """Rigid motion of the formation by the arc length ``arc`` applied to a world frame."""
        out = T.copy()
        if self.formation == 'line':
            out[:3, 3] = T[:3, 3] + arc * self.travel
            return out
        R = rot('z', arc / self.radius)
        c = np.r_[self.centre, 0.]
        out[:3, :3] = R @ T[:3, :3]
        out[:3, 3] = c + R @ (T[:3, 3] - c)
        return out

    def targets(self, t):
        """Pelvis poses (n, 4, 4) and sole anchors {id: 4x4} at time ``t``."""
        aR, aL, liftR, liftL = self.advances(t)
        w = self.weight(t)
        sR, sL = self.half + aR, -self.half + aL
        sP = .5 * (sR + sL) + .5 * w * (sR - sL)
        pelvis = np.array([self.move(self.T0[k], sP) for k in range(self.ens.n)])
        anchors = {}
        for k in range(self.ens.n):
            for side, a, lift in (('left', aL, liftL), ('right', aR, liftR)):
                A = self.move(self.sole0[f'd{k}/{side}_sole'], a)
                A[2, 3] += self.clearance * lift
                anchors[f'd{k}/{side}_sole'] = A
        return pelvis, anchors

    def segment(self, t):
        st = self.stance(t)
        if t < self.t_settle:
            return 'settle'
        if t >= self.t_settle + self.cycles * self.cycle:
            return 'centre'
        return {'both': 'weight shift', 'left': 'right feet step', 'right': 'left feet follow'}[st]


def generate_steps(ens, manager, dance, dt=0.002, progress=None):
    """Reference of a ``StepDance`` on the closure manifold (all soles follow their anchors)."""
    s = ens.system
    comp = manager.comp
    q = manager.q.copy()
    N = int(round(dance.duration / dt)) + 1
    t = np.arange(N) * dt
    T_root = np.empty((N,) + q.T_root.shape)
    theta = np.empty((N, s.n))
    worst = dict(residual=0., iterations=0, limit_margin=np.inf)
    roots = np.concatenate([np.arange(sl.start, sl.stop) for sl in s.root_slice if sl is not None])
    scalar = np.setdiff1d(np.arange(s.nT), roots)
    for i in range(N):
        pelvis, anchors = dance.targets(t[i])
        q.T_root[:] = pelvis
        for sid, A in anchors.items():
            comp.set_anchor(sid, A)
        q, ci = comp.mode.correct(q, tol=1e-11, maxiter=30)
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
    info = dict(worst, formation=ens.formation, cycles=dance.cycles, step_m=dance.step, clearance_m=dance.clearance,
                duration_s=dance.duration, travel_m=dance.cycles * dance.step,
                turn_deg=float(np.degrees(dance.cycles * dance.step / dance.radius)) if dance.radius else 0.)
    return Reference(t, T_root, theta, v, a, info=info)


class SteppingPolicy:
    """One policy per support mode, selected by the support schedule of the dance.

    The member modules are shared by all modes; switching a mode replaces the composer.
    """

    def __init__(self, ens, ref, dance, q0, name='ip', triples=None, support_weight=2e-4, **kw):
        from .execute import build_policy
        self.ens, self.ref, self.dance, self.name = ens, ref, dance, name
        self.sys = ens.system
        self.weight = 9.81 * max(float(self.sys.mass[idx].sum()) for idx in self.sys.member_bodies)
        self.policies, self.comps = {}, {}
        for st in ('both', 'left', 'right'):
            comp = ens.composite(stance=[st] * ens.n, triples=triples)
            comp.prepare(q0)
            self.comps[st] = comp
            self.policies[st] = build_policy(ens, comp, ref, q0, name, support_weight=support_weight,
                                             transfer_load=.5 * self.weight,
                                             load_schedule=lambda t: self.dance.load_caps(t, self.weight), **kw)
        self.current = 'both'
        self.switches = []

    @property
    def alloc(self):
        """Allocator of the current support mode (bounds are shared by all modes through ``set_bounds``)."""
        return _Allocators(self.policies)

    @property
    def timing(self):
        out = dict(evaluate=0., allocate=0., calls=0, qp=0)
        for p in self.policies.values():
            for k in out:
                out[k] += p.timing[k]
        return out

    def __call__(self, t, q, v):
        st = self.dance.stance(t)
        if st != self.current:
            self.switches.append((float(t), self.current, st))
            self.policies[st].f_prev = self.policies[self.current].f_prev
            self.current = st
        f, info = self.policies[st](t, q, v)
        info['stance'] = st
        return f, info


class _Allocators:
    """The allocators of all support modes seen as one (merged status counts, common bounds)."""

    def __init__(self, policies):
        self._p = policies

    @property
    def counts(self):
        out = {}
        for p in self._p.values():
            for k, v in p.alloc.counts.items():
                out[k] = out.get(k, 0) + v
        return out

    @property
    def cones(self):
        return getattr(self._p['both'].alloc, 'cones', {})

    @property
    def lower(self):
        return self._p['both'].alloc.lower

    @property
    def upper(self):
        return self._p['both'].alloc.upper

    def set_bounds(self, lower, upper):
        for p in self._p.values():
            p.alloc.set_bounds(lower, upper)


class StepSession:
    """Ensemble, stepping dance and its reference (cached on disk)."""

    def __init__(self, formation='circle', partition='joint_space', dt=0.002, cache=None, verbose=False, **dance_kw):
        import hashlib
        import json
        from pathlib import Path
        from .ensemble import Ensemble, PartitionManager
        from .session import ROOT, source_hash
        self.formation = formation
        self.ens = Ensemble(formation, partition)
        self.sys = self.ens.system
        self.manager = PartitionManager(self.ens, self.sys.neutral(), acquire=True)
        self.q0 = self.manager.q.copy()
        self.comp = self.manager.comp
        self.dance = StepDance(self.ens, self.q0, **dance_kw)
        own = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        key = hashlib.sha256(json.dumps([formation, partition, dt, sorted(dance_kw.items()), source_hash(), own]).encode()).hexdigest()[:16]
        cache = Path(cache) if cache else ROOT / 'results' / 'cache'
        path = cache / f'steps_{formation}_{key}.npz'
        if path.exists():
            z = np.load(path, allow_pickle=False)
            self.ref = Reference(z['t'], z['T_root'], z['theta'], z['v'], z['a'], info=json.loads(str(z['info'])))
        else:
            progress = (lambda i, n: print(f'  stepping reference {formation}: {i}/{n}', flush=True)) if verbose else None
            gen = PartitionManager(self.ens, self.sys.neutral(), acquire=True)
            self.ref = generate_steps(self.ens, gen, self.dance, dt=dt, progress=progress)
            cache.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, t=self.ref.t, T_root=self.ref.T_root, theta=self.ref.theta, v=self.ref.v,
                                a=self.ref.a, info=json.dumps(self.ref.info))
        self.reference_path = path


def step_windows(dance, side):
    """Swing intervals ``(lift, land)`` of the soles of one side from the schedule of the dance."""
    out = []
    for c in range(dance.cycles):
        t0 = dance.t_settle + c * dance.cycle
        lift = t0 + dance.t_shift if side == 'right' else t0 + 2. * dance.t_shift + dance.t_swing
        out.append((lift, lift + dance.t_swing))
    return out


EDGE = 0.95      # a loaded sole is "at its edge" or "at the friction limit" beyond this fraction of the limit


def step_metrics(log, dance):
    """Stepping-specific summary: travel, foot clearance, landing error and stance-foot slip.

    * clearance: apex rise of the sole centre above its height at lift-off, per step;
    * landing error: horizontal distance between a sole and its reference 0.15 s after the
      reference landing;
    * slip: largest horizontal displacement of a sole within a window in which the schedule
      loads it fully (no load cap, not swinging), relative to the start of that window;
    * settling: horizontal displacement of a sole between its landing and the end of its
      re-loading (the cap is being raised), reported separately from slip;
    * turn: mean rotation of the pelvis positions about the centre of the formation (the
      ring centre or the arc centre); a line translates and has no turn.
    """
    t = log['t']
    sole, ref = log['sole'], log['sole_ref']
    n = sole.shape[1]
    clearance, landing, slip, settle = [], [], [], []
    at = lambda x: int(np.clip(np.searchsorted(t, x), 0, len(t) - 1))
    for j in range(n):
        side = 'left' if j % 2 == 0 else 'right'
        windows = step_windows(dance, side)
        edges = [t[0]]
        for lift, land in windows:
            if lift > t[-1]:
                break
            a, b = at(lift), at(land)
            if land <= t[-1] and b > a:                        # only swings that the run completed
                clearance.append(float(np.max(sole[a:b + 1, j, 2]) - sole[a, j, 2]))
            if land + 0.15 <= t[-1]:
                k = at(land + 0.15)
                landing.append(float(np.linalg.norm(sole[k, j, :2] - ref[k, j, :2])))
            loaded_from = land + 0.04 + dance.reload
            if loaded_from <= t[-1]:
                a, b = at(land + 0.04), at(loaded_from)
                settle.append(float(np.max(np.linalg.norm(sole[a:b + 1, j, :2] - sole[a, j, :2], axis=1))))
            edges += [lift - dance.unload, loaded_from]
        edges.append(t[-1])
        for a, b in zip(edges[0::2], edges[1::2]):               # fully loaded windows of this sole
            a, b = at(a), at(min(b, t[-1]))
            if b - a > 5:
                slip.append(float(np.max(np.linalg.norm(sole[a:b + 1, j, :2] - sole[a, j, :2], axis=1))))
    pel, pel_ref = log['pelvis'], log['pelvis_ref']
    out = dict(steps=len(clearance), clearance_min_mm=1e3 * min(clearance) if clearance else None,
               clearance_reference_mm=1e3 * dance.clearance, landing_error_max_mm=1e3 * max(landing) if landing else None,
               stance_slip_max_mm=1e3 * max(slip) if slip else None, landing_settle_max_mm=1e3 * max(settle) if settle else None,
               travel_mean_m=float(np.mean(np.linalg.norm(pel[-1, :, :2] - pel[0, :, :2], axis=1))),
               travel_reference_m=float(np.mean(np.linalg.norm(pel_ref[-1, :, :2] - pel_ref[0, :, :2], axis=1))),
               turn_deg=None, turn_reference_deg=None)
    if dance.centre is not None:
        c = np.asarray(dance.centre, float)
        ang = lambda P: np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0])
        wrap = lambda d: (d + np.pi) % (2. * np.pi) - np.pi
        out['turn_deg'] = float(np.degrees(np.mean(wrap(ang(pel[-1]) - ang(pel[0])))))
        out['turn_reference_deg'] = float(np.degrees(np.mean(wrap(ang(pel_ref[-1]) - ang(pel_ref[0])))))
    return out


def contact_metrics(log, records, friction):
    """Contact of the loaded soles on the plant (soles in the order d0 left, d0 right, d1 left, ...).

    * ``max_cop_x_fraction``, ``max_cop_y_fraction``: largest distance of the centre of pressure
      from the sole centre, as a fraction of the half length and the half width of that sole;
    * ``cop_edge_fraction``: fraction of the loaded sole samples with the centre of pressure
      beyond ``EDGE`` of the half extent in either direction;
    * ``friction_limit_fraction``: fraction of the loaded sole samples with a friction use
      (tangential over normal floor reaction) beyond ``EDGE`` of the floor coefficient ``friction``.

    A sole is loaded above ``LOADED_SOLE_N``; the statistics start at ``CONTACT_FROM_S``.
    """
    from .execute import CONTACT_FROM_S, LOADED_SOLE_N
    t = log['t']
    half = np.array([{st['id']: st for st in rec['sole_sites']}[f'{side}_sole']['half_extent_m']
                     for rec in records for side in ('left', 'right')], float)
    mc = t >= min(CONTACT_FROM_S, .5 * t[-1])
    loaded = np.nan_to_num(log['fz'][mc], nan=0.) > LOADED_SOLE_N
    cop_frac = np.where(loaded[:, :, None], np.abs(np.nan_to_num(log['cop'][mc])) / half[None], 0.)
    fric = np.where(loaded, np.nan_to_num(log['friction_ratio'][mc]), 0.)
    n_loaded = max(1, int(loaded.sum()))
    return dict(max_cop_x_fraction=float(cop_frac[:, :, 0].max()), max_cop_y_fraction=float(cop_frac[:, :, 1].max()),
                cop_edge_fraction=float(np.sum(loaded & (cop_frac.max(axis=2) > EDGE)) / n_loaded),
                friction_limit_fraction=float(np.sum(loaded & (fric > EDGE * friction)) / n_loaded),
                contact_edge_threshold=EDGE)
