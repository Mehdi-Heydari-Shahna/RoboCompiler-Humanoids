"""Feasibility audit of a reference on the compiled model (inverse dynamics along the motion).

For sampled reference states the generalized force ``Q = Gamma udot + C u + g + D u`` is
allocated to the actuator ports and the support wrenches are recovered from the
dependent-coordinate equations.  The audit reports effort use and the unilateral,
friction and centre-of-pressure margins that an ideal-support execution would need.
"""
from __future__ import annotations

import numpy as np

from .stage import FLOOR
from ipd.control import Allocator, SupportCone
from ipd.system import Config


def audit(ens, comp, ref, every=25, cones=True, friction=FLOOR['friction'], t_range=None):
    s = ens.system
    ind = comp.mode.independent
    cone = {}
    if cones:
        for m in s.members:
            for site in m.comp.cmg['sole_sites']:
                hx, hy = site['half_extent_m']
                cone[f"{m.name}/{site['id']}"] = SupportCone(hx, hy, mu=friction)
    alloc = Allocator(comp, cone)
    rows = []
    for i in range(0, len(ref.t), every):
        if t_range is not None and not (t_range[0] <= ref.t[i] <= t_range[1]):
            continue
        q = Config(ref.T_root[i], ref.theta[i])
        u, ud = ref.v[i][ind], ref.a[i][ind]
        ev = comp.evaluate(q, u)
        Q = ev.Gamma @ ud + ev.C @ u + ev.g + ev.D @ u
        wdot = ev.G @ ud + ev.Gd @ u
        f, info = alloc.solve(ev, Q, wdot)
        lam = info['wrenches']
        fz = np.array([lam[k][5] for k in sorted(lam)])
        safe = np.maximum(fz, 1e-9)
        copx = np.array([-lam[k][1] for k in sorted(lam)]) / safe
        copy = np.array([lam[k][0] for k in sorted(lam)]) / safe
        fric = np.array([np.hypot(lam[k][3], lam[k][4]) for k in sorted(lam)]) / safe
        rows.append(dict(t=float(ref.t[i]), effort=float(np.max(abs(f) / s.effort_upper)), min_fz=float(fz.min()),
                         max_cop_x=float(np.max(abs(copx))), max_cop_y=float(np.max(abs(copy))), max_friction=float(fric.max()),
                         qp=bool(info['qp']), failed=bool(info['failed']),
                         slack=float(np.nan_to_num(info['slack'], nan=np.inf)), cop_x=copx, fz=fz))
    return rows


def summary(rows):
    infeasible = [r for r in rows if r['failed'] or r['slack'] > 1e-3]
    return dict(effort_peak=max(r['effort'] for r in rows), min_fz_N=min(r['min_fz'] for r in rows),
                max_cop_x_m=max(r['max_cop_x'] for r in rows), max_cop_y_m=max(r['max_cop_y'] for r in rows),
                max_friction_ratio=max(r['max_friction'] for r in rows), qp_fraction=float(np.mean([r['qp'] for r in rows])),
                max_slack=max(r['slack'] for r in rows), infeasible_fraction=len(infeasible) / len(rows),
                first_infeasible_s=infeasible[0]['t'] if infeasible else None, samples=len(rows))
