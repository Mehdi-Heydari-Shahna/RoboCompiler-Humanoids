"""Closed-loop execution of the hand-linked formation dances.

One compiled ensemble (five closed-chain humanoids, hand links, sole supports) is
executed on three plants,

    mujoco       the realistic plant: MuJoCo with soft equality constraints, rubber-on-wood
                 contact, joint Coulomb friction, actuator lag and torque-speed limits and
                 noise on the measured state (``dancers.plants``)
    pinocchio    Pinocchio constrained dynamics (hard KKT with a Baumgarte corrector), ideal sole welds, no friction
    minimal_ode  the compiled inertia-partitioned model integrated as an ODE

under several policies that share the reference, the feedback law and the effort
allocation and differ only in the model they are given (see ``dancers.execute``).

Scenarios (MuJoCo plant):

    nominal      the reference dance
    ideal        the reference dance on MuJoCo without Coulomb friction, actuator dynamics and
                 noise (and without friction compensation): what the realistic plant adds to the
                 tracking error
    payload      a point payload appears in one hand; policies ``stale`` (model not told)
                 and ``updated`` (the body's partition member is replaced)

Every run writes ``<out>/<formation>/<scenario>__<plant>__<policy>.json`` (metrics,
settings, provenance) and a compressed ``.npz`` log next to it.  A run that raises an
exception leaves a JSON with ``status = "exception"`` so that it cannot go unnoticed.
"""
from __future__ import annotations

import os

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')         # small dense problems: BLAS threads only add contention

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dancers import plants                                                        # noqa: E402
from dancers.execute import (CONTACT_FROM_S, FRICTION_FEEDFORWARD, LOADED_SOLE_N, POLICIES, build_policy, metrics,  # noqa: E402
                             payload_inertia, run)
from dancers.session import Session, source_hash                                  # noqa: E402
from ipd.control import Gains                                                     # noqa: E402
from ipd.provenance import stamp                                                  # noqa: E402

PLANTS = {'mujoco': plants.MujocoWorld, 'pinocchio': plants.PinocchioKKT, 'minimal_ode': plants.MinimalODE}
SCENARIOS = ('nominal', 'ideal', 'payload')
PAYLOAD = dict(time=1.0, dancer=2, body='r_forearm', site='right_palm', mass_kg=2.0)
NOISE_SEED = 20261008            # the same noise sequence for every policy (paired comparison)
PINOCCHIO_BAUMGARTE = (400., 40.)  # constraint corrector of the Pinocchio plant (1/s^2, 1/s)


def cone_info(policy):
    """Settings of the support cones of a policy (identical for all soles), for the result files."""
    alloc = getattr(policy, 'alloc', None)
    cones = getattr(alloc, 'cones', None) or {}
    for c in cones.values():
        return dict(margin=c.margin, friction_assumed=c.mu_assumed, pyramid_coefficient=float(c.mu), min_normal_force_N=float(c.fz_min))
    return None


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if np.isfinite(x) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), indent=2, sort_keys=True) + '\n')


def failed_result(exc, settings):
    """Result record of a run that raised an exception."""
    import traceback
    return dict(status='exception', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc(),
                settings=settings, provenance=stamp())


def run_case(se, plant_name, policy_name, scenario='nominal', duration=None, out=None, gains=None, verbose=False, tag=None):
    """Execute one policy on one plant in one scenario; returns the metrics dictionary."""
    ens, s, comp, ref, q0 = se.ens, se.sys, se.comp, se.ref, se.q0
    settings = dict(formation=se.formation, plant=plant_name, policy=policy_name, scenario=scenario,
                    control_dt=0.002, reference=dict(ref.info), reference_source_hash=source_hash()[:16])
    plant_kw, policy_kw, events = {}, {}, []
    if plant_name == 'mujoco':
        plant_kw.update(timestep=1e-3, seed=NOISE_SEED)
        if scenario == 'ideal':
            plant_kw.update(actuator=False, noise=False, coulomb=False)
    if plant_name == 'pinocchio':
        # rigid constraints integrated at acceleration level drift off the closure manifold; without a corrector the
        # drift of the thirty-second dances corrupts the redundant constraint rows of the proximal solve
        plant_kw.update(baumgarte=PINOCCHIO_BAUMGARTE)
    g = Gains() if gains is None else gains
    policy_kw['gains'] = g
    if plant_name != 'mujoco':
        policy_kw['follow_anchors'] = False          # ideal supports: the anchors are exact
    if plant_name != 'mujoco' or scenario == 'ideal':
        policy_kw['friction_feedforward'] = 0.       # plants without Coulomb friction get no friction compensation
    name = policy_name
    if scenario == 'payload':
        if policy_name not in ('stale', 'updated', 'gravity_pd', 'actuator_pd'):
            raise ValueError('the payload scenario runs the policies stale, updated, gravity_pd and actuator_pd')
        if plant_name != 'mujoco':
            raise ValueError('the payload scenario is implemented for the MuJoCo plant')
        name = 'ip' if policy_name in ('stale', 'updated') else policy_name
    policy = build_policy(ens, comp, ref, q0, name, **policy_kw)
    plant = PLANTS[plant_name](ens, comp if plant_name == 'mujoco' else _own_mode(ens, comp, q0), q0, **plant_kw)
    if scenario == 'payload':
        k = PAYLOAD['dancer']
        body = s.body_index[f"d{k}/{PAYLOAD['body']}"]
        _, P = s.site_frame(f'd{k}', PAYLOAD['site'])
        m1, c1, I1, dM = payload_inertia(s, body, PAYLOAD['mass_kg'], P[:3, 3])
        M_new = s.M.copy()
        M_new[body] = M_new[body] + dM

        def attach(pl, po):
            pl.set_body_inertia(body, m1, c1, I1)
            if policy_name == 'updated':
                po.M = M_new                          # one member of the partition is replaced
        events.append((PAYLOAD['time'], attach))
        settings['payload'] = dict(PAYLOAD)
    settings.update(plant_settings=plant_kw, gains=dict(lam=g.lam, ks=g.ks), plant_info=plant.info(), support_cone=cone_info(policy),
                    friction_feedforward=float(policy_kw.get('friction_feedforward', FRICTION_FEEDFORWARD)))
    t0 = time.time()
    log = run(ens, comp, ref, policy, plant, duration=duration, events=events, verbose=verbose)
    m = metrics(log)
    calls = max(1, policy.timing['calls'])
    m.update(evaluate_ms=1e3 * policy.timing['evaluate'] / calls, allocate_ms=1e3 * policy.timing['allocate'] / calls,
             wall_s=time.time() - t0)
    if hasattr(plant, 'worst_residual'):
        m['closure_residual_max'] = float(plant.worst_residual)
    # model predictions against the plant; both columns of the log refer to the same control period
    # (after the first half second, in which the contacts of the plant build up)
    settled = log['t'] >= CONTACT_FROM_S
    d = log['link_force_model'] - log['link_force']
    ok = np.isfinite(d).all(axis=1) & settled
    m['link_force_model_rms_error_N'] = float(np.sqrt(np.mean(d[ok] ** 2))) if ok.any() and d.shape[1] else None
    m['link_force_model_peak_error_N'] = float(np.max(abs(d[ok]))) if ok.any() and d.shape[1] else None
    dz = log['fz_model'] - log['fz']
    okz = np.isfinite(dz).all(axis=1) & settled
    m['support_force_model_rms_error_N'] = float(np.sqrt(np.mean(dz[okz] ** 2))) if okz.any() else None
    m['support_force_model_peak_error_N'] = float(np.max(abs(dz[okz]))) if okz.any() else None
    m['generalized_force_defect_max'] = float(np.nanmax(log['Q_defect'])) if np.isfinite(log['Q_defect']).any() else None
    result = dict(status='ok', metrics=m, settings=settings, provenance=stamp())
    if out is not None:
        stem = Path(out) / se.formation / f"{scenario if tag is None else tag}__{plant_name}__{policy_name}"
        write_json(stem.with_suffix('.json'), result)
        np.savez_compressed(stem.with_suffix('.npz'), **{k: v for k, v in log.items() if isinstance(v, np.ndarray)})
    return result


def _own_mode(ens, comp, q0):
    """A separate evaluator of the same closure mode for a plant (same anchors, same partition)."""
    anchors = {sid: comp.mode.anchors[i] for i, sid in enumerate(comp.mode.support_ids)}
    own = ens.composite(comp.stance, comp.links, anchors, triples=comp.triples)
    own.prepare(q0)
    return own


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--formation', default='circle', choices=('circle', 'semicircle', 'line'))
    ap.add_argument('--plant', default='mujoco', choices=sorted(PLANTS))
    ap.add_argument('--policies', nargs='+', default=['ip'])
    ap.add_argument('--scenario', default='nominal', choices=SCENARIOS)
    ap.add_argument('--duration', type=float, default=None, help='seconds to execute (default: the whole dance)')
    ap.add_argument('--partition', default='joint_space', choices=('joint_space', 'actuator_space'))
    ap.add_argument('--tag', default=None, help='file-name prefix instead of the scenario name')
    ap.add_argument('--out', default=str(ROOT / 'results' / 'dance'))
    ap.add_argument('--cache', default=None, help='directory of cached references')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args(argv)
    se = Session(args.formation, partition=args.partition, cache=args.cache, verbose=args.verbose)
    ok = True
    for name in args.policies:
        try:
            r = run_case(se, args.plant, name, args.scenario, args.duration, args.out, verbose=args.verbose, tag=args.tag)
            m = r['metrics']
            print('RESULT %s %s %s %s  completed=%s %s  pelvis rms %.4f mm peak %.4f mm  palm rms %.4f mm  effort peak %.2f  '
                  'min Fz %.0f N  qp %.3f  slack %.1e  not solved %d  wall %.0f s' % (
                      args.formation, args.scenario, args.plant, name, m['completed'], m['failure'], m['pelvis_rms_mm'],
                      m['pelvis_peak_mm'], m['palm_rms_mm'], m['effort_peak_fraction'], m['min_fz_N'], m['qp_fraction'],
                      m['max_slack'], m['qp_not_solved'], m['wall_s']), flush=True)
        except Exception as exc:                           # keep the remaining policies of a suite running
            import traceback
            traceback.print_exc()
            print('RESULT', args.formation, args.scenario, args.plant, name, 'EXCEPTION', flush=True)
            stem = Path(args.out) / args.formation / f"{args.tag or args.scenario}__{args.plant}__{name}"
            write_json(stem.with_suffix('.json'), failed_result(exc, dict(formation=args.formation, plant=args.plant,
                                                                         policy=name, scenario=args.scenario)))
            ok = False
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
