"""Inverse-dynamics feasibility audit of the dance references on the compiled model.

For sampled reference states the generalized force of the motion is allocated to the
actuator ports with the support-wrench limits of the controller, and the efforts,
the slack of the allocation and the support margins are recorded.  The travelling
formations are audited with the composer of each support mode and the load caps of
the schedule.

Every audit is run twice: with the composed model (hand links modelled, so the
allocation may use them) and with per-dancer models (no hand links).  The second one
tells whether a choreography can be executed by every dancer on its own.

Writes ``<out>/<formation>/audit__<dance>.json``.
"""
from __future__ import annotations

import os

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')         # small dense problems: BLAS threads only add contention

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from dancers.audit import audit, summary                      # noqa: E402
from dancers.stage import FLOOR                               # noqa: E402
from ipd.control import Allocator, SupportCone                # noqa: E402
from ipd.system import Config                                 # noqa: E402
from ipd.provenance import stamp                              # noqa: E402
from run_dance import write_json                              # noqa: E402


def audit_steps(se, every=10, links=None):
    """``links=None`` audits the composed model, ``links=[]`` the per-dancer models."""
    ens, s, ref, dance = se.ens, se.sys, se.ref, se.dance
    cones = {}
    for m in s.members:
        for site in m.comp.cmg['sole_sites']:
            cones[f"{m.name}/{site['id']}"] = SupportCone(*site['half_extent_m'], mu=FLOOR['friction'])
    weight = 9.81 * max(float(s.mass[idx].sum()) for idx in s.member_bodies)
    comps, allocs = {}, {}
    for st in ('both', 'left', 'right'):
        c = ens.composite(stance=[st] * ens.n, links=links, triples=se.comp.triples if links is None else None)
        c.prepare(se.q0)
        c.follow_anchors = True
        comps[st], allocs[st] = c, Allocator(c, cones, support_weight=2e-4, transfer_load=.5 * weight)
    rows = []
    for i in range(0, len(ref.t), every):
        t = float(ref.t[i])
        st = dance.stance(t)
        comp = comps[st]
        q = Config(ref.T_root[i], ref.theta[i])
        ind = comp.mode.independent
        u, ud = ref.v[i][ind], ref.a[i][ind]
        ev = comp.evaluate(q, u)
        Q = ev.Gamma @ ud + ev.C @ u + ev.g + ev.D @ u
        f, info = allocs[st].solve(ev, Q, ev.G @ ud + ev.Gd @ u, fz_max=dance.load_caps(t, weight))
        lam = info['wrenches']
        loaded = [k for k in lam if lam[k][5] > 50.]
        rows.append(dict(t=t, effort=float(np.max(abs(f) / s.effort_upper)), slack=float(np.nan_to_num(info['slack'], nan=np.inf)),
                         qp=bool(info['qp']), failed=bool(info['failed']),
                         min_fz=float(min(lam[k][5] for k in lam)),
                         max_cop_x=float(max((abs(lam[k][1] / lam[k][5]) for k in loaded), default=0.)),
                         max_cop_y=float(max((abs(lam[k][0] / lam[k][5]) for k in loaded), default=0.)),
                         max_friction=float(max((np.hypot(lam[k][3], lam[k][4]) / lam[k][5] for k in loaded), default=0.))))
    infeasible = [r for r in rows if r['failed'] or r['slack'] > 1e-3]
    return dict(effort_peak=max(r['effort'] for r in rows), max_slack=max(r['slack'] for r in rows),
                infeasible_fraction=len(infeasible) / len(rows), first_infeasible_s=infeasible[0]['t'] if infeasible else None,
                qp_fraction=float(np.mean([r['qp'] for r in rows])), min_fz_N=min(r['min_fz'] for r in rows),
                max_cop_x_m=max(r['max_cop_x'] for r in rows), max_cop_y_m=max(r['max_cop_y'] for r in rows),
                max_friction_ratio=max(r['max_friction'] for r in rows), samples=len(rows))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--formations', nargs='+', default=['circle', 'semicircle', 'line'])
    ap.add_argument('--dances', nargs='+', default=['nominal', 'steps'], choices=('nominal', 'steps'))
    ap.add_argument('--out', default=str(ROOT / 'results' / 'dance'))
    ap.add_argument('--cache', default=None)
    args = ap.parse_args(argv)
    show = lambda formation, dance, label, r: print(
        'AUDIT %-10s %-7s %-10s effort peak %.2f  slack %.1e  infeasible %.3f  min Fz %.0f N  CoP x %.3f y %.3f m  friction %.2f' % (
            formation, dance, label, r['effort_peak'], r['max_slack'], r['infeasible_fraction'], r['min_fz_N'], r['max_cop_x_m'],
            r['max_cop_y_m'], r['max_friction_ratio']), flush=True)
    for formation in args.formations:
        if 'nominal' in args.dances:
            from dancers.session import Session
            se = Session(formation, cache=args.cache)
            own = se.ens.composite(stance=se.comp.stance, links=[])
            own.prepare(se.q0)
            out = dict(reference=dict(se.ref.info), provenance=stamp())
            for label, comp in (('composed', se.comp), ('per_dancer', own)):
                out[label] = summary(audit(se.ens, comp, se.ref, every=25, cones=True))
                show(formation, 'nominal', label, out[label])
            write_json(Path(args.out) / formation / 'audit__nominal.json', out)
        if 'steps' in args.dances:
            from dancers.stepping import StepSession
            se = StepSession(formation, cache=args.cache)
            out = dict(reference=dict(se.ref.info), provenance=stamp())
            for label, links in (('composed', None), ('per_dancer', [])):
                out[label] = audit_steps(se, links=links)
                show(formation, 'steps', label, out[label])
            write_json(Path(args.out) / formation / 'audit__steps.json', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
