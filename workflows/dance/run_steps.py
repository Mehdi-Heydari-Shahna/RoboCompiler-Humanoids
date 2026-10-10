"""Closed-loop execution of the travelling formations (side steps with support-mode switching).

The five hand-linked dancers step to their right together: a ring and a semicircle
turn about their centre, a line translates.  Every lift-off and landing changes the
closure mode; the policy keeps one composer per support mode on top of the same five
member modules.  The plant is the realistic MuJoCo plant of ``dancers.plants`` (native
unilateral frictional contact, joint friction, actuator dynamics, measurement noise).

Every run writes ``<out>/<formation>/steps__mujoco__<policy>.json`` and a ``.npz`` log.
"""
from __future__ import annotations

import os

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')         # small dense problems: BLAS threads only add contention

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from dancers import plants                                                # noqa: E402
from dancers.execute import FRICTION_FEEDFORWARD, metrics, run            # noqa: E402
from dancers.session import source_hash                                   # noqa: E402
from dancers.stage import FLOOR                                           # noqa: E402
from dancers.stepping import SteppingPolicy, StepSession, contact_metrics, step_metrics    # noqa: E402
from ipd.provenance import stamp                                          # noqa: E402
from run_dance import NOISE_SEED, cone_info, failed_result, write_json    # noqa: E402

POLICIES = ('ip', 'decentralized', 'simplified', 'massless', 'no_curvature', 'gravity_pd', 'actuator_pd')


def run_case(se, name, duration=None, out=None, verbose=False, tag='steps', noise_scale=1.):
    ens, q0, ref, dance = se.ens, se.q0, se.ref, se.dance
    friction = FLOOR['friction']
    policy = SteppingPolicy(ens, ref, dance, q0, name, triples=se.comp.triples, friction=friction)
    noise = True if noise_scale == 1. else {k: noise_scale * v for k, v in plants.NOISE.items()}
    plant = plants.MujocoWorld(ens, se.comp, q0, timestep=1e-3, seed=NOISE_SEED, noise=noise)
    t0 = time.time()
    log = run(ens, se.comp, ref, policy, plant, duration=duration, verbose=verbose)
    m = metrics(log)
    m.update(step_metrics(log, dance))
    m.update(contact_metrics(log, ens.records, friction))
    calls = max(1, policy.timing['calls'])
    m.update(evaluate_ms=1e3 * policy.timing['evaluate'] / calls, allocate_ms=1e3 * policy.timing['allocate'] / calls,
             wall_s=time.time() - t0, mode_switches=len(policy.switches))
    m['qp_status_by_mode'] = {st: dict(p.alloc.counts) for st, p in policy.policies.items()}
    result = dict(status='ok', metrics=m, provenance=stamp(),
                  settings=dict(formation=se.formation, plant='mujoco', policy=name, scenario='steps', control_dt=0.002,
                                noise_scale=float(noise_scale),
                                plant_info=plant.info(), support_cone=cone_info(policy), friction_feedforward=FRICTION_FEEDFORWARD,
                                gains=dict(lam=policy.policies['both'].gains.lam, ks=policy.policies['both'].gains.ks),
                                reference=dict(ref.info), source_hash=source_hash()[:16], friction=friction,
                                independent_speeds={st: int(c.d) for st, c in policy.comps.items()},
                                level2_chords={st: int(c.m2) for st, c in policy.comps.items()},
                                dance=dict(cycles=dance.cycles, step_m=dance.step, clearance_m=dance.clearance, shift=dance.shift,
                                           t_shift_s=dance.t_shift, t_swing_s=dance.t_swing)))
    if out is not None:
        stem = Path(out) / se.formation / f'{tag}__mujoco__{name}'
        write_json(stem.with_suffix('.json'), result)
        np.savez_compressed(stem.with_suffix('.npz'), **{k: v for k, v in log.items() if isinstance(v, np.ndarray)})
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--formation', default='circle', choices=('circle', 'semicircle', 'line'))
    ap.add_argument('--policies', nargs='+', default=['ip'], choices=POLICIES)
    ap.add_argument('--duration', type=float, default=None)
    ap.add_argument('--cycles', type=int, default=None, help='step cycles (default: that of dancers.stepping.StepDance)')
    ap.add_argument('--out', default=str(ROOT / 'results' / 'dance'))
    ap.add_argument('--cache', default=None)
    ap.add_argument('--noise-scale', type=float, default=1., help='multiplies the measurement noise of the plant')
    ap.add_argument('--tag', default='steps', help='file-name prefix of the results')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args(argv)
    se = StepSession(args.formation, cache=args.cache, verbose=args.verbose,
                     **({} if args.cycles is None else {'cycles': args.cycles}))
    ok = True
    for name in args.policies:
        try:
            m = run_case(se, name, args.duration, args.out, args.verbose, tag=args.tag, noise_scale=args.noise_scale)['metrics']
            r3 = lambda x: None if x is None else round(x, 3)
            print('RESULT %s steps mujoco %s  completed=%s %s  pelvis rms %.3f mm peak %.3f mm  turn %s/%s deg  travel %.3f/%.3f m  '
                  'clearance min %s mm  landing error %s mm  slip %s mm  settle %s mm  effort peak %.2f  slack %.1e  wall %.0f s' % (
                      args.formation, name, m['completed'], m['failure'], m['pelvis_rms_mm'], m['pelvis_peak_mm'], r3(m['turn_deg']),
                      r3(m['turn_reference_deg']), m['travel_mean_m'], m['travel_reference_m'], r3(m['clearance_min_mm']),
                      r3(m['landing_error_max_mm']), r3(m['stance_slip_max_mm']), r3(m['landing_settle_max_mm']),
                      m['effort_peak_fraction'], m['max_slack'], m['wall_s']), flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print('RESULT', args.formation, 'steps mujoco', name, 'EXCEPTION', flush=True)
            if args.out is not None:
                write_json(Path(args.out) / args.formation / f'{args.tag}__mujoco__{name}.json',
                           failed_result(exc, dict(formation=args.formation, plant='mujoco', policy=name, scenario='steps')))
            ok = False
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
