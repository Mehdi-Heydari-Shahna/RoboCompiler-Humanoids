"""The 3-DoF series-parallel manipulator of arXiv:2510.23226 (its Section IV) compiled from its record.

The reference derives the closed-chain relation between the actuated screw and the
passive joint by hand (its Eq. 28).  Here the same relation is obtained from the
physical record alone: the closure is detected, cut and corrected by the compiler, and
the lift of the compiled model is compared with the printed derivative at random
configurations.
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

from examples import series_parallel as sp                       # noqa: E402
from ipd.compiler import compile_graph                           # noqa: E402
from ipd.composite import Composite                              # noqa: E402
from ipd.integrate import MinimalModel                           # noqa: E402
from ipd.provenance import stamp                                 # noqa: E402
from ipd.system import Member, System                            # noqa: E402


def build(partition='actuator_space', payload_kg=0.):
    comp_g = compile_graph(sp.physical(payload_kg), partition)
    s = System([Member('m', comp_g, np.eye(4))], name='series_parallel')
    comp = Composite(s, name=f'manipulator:{partition}')
    q, info = comp.prepare(s.neutral(), acquire=True)
    return s, comp, q, info


def write_json(path, obj):
    def conv(x):
        if isinstance(x, dict):
            return {str(k): conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        if isinstance(x, np.ndarray):
            return conv(x.tolist())
        if isinstance(x, (np.floating, float)):
            return float(x) if np.isfinite(x) else None
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.bool_):
            return bool(x)
        return x
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = dict(obj)
    obj['_provenance'] = stamp()
    path.write_text(json.dumps(conv(obj), indent=2, sort_keys=True) + '\n')


# ---------------------------------------------------------------------- compile
def study_compile(out):
    s, comp, q, info = build('actuator_space')
    m = MinimalModel(comp)
    rng = np.random.default_rng(0)
    iz, idl = s.index['m/zeta'], s.index['m/delta']
    rec = dict(zeta=[], dzeta=[], residual=[])
    for _ in range(64):
        qq = m.retract(q, rng.uniform(-1, 1, 3) * np.array([0.5, 0.4, 0.09]))
        ev = comp.evaluate(qq)
        E = comp.lift(ev)
        delta = qq.theta[idl]
        rec['zeta'].append(abs(qq.theta[iz] - sp.zeta_of_delta(delta)))
        rec['dzeta'].append(abs(E[iz, 2] - sp.dzeta_ddelta(delta)) / abs(sp.dzeta_ddelta(delta)))
        rec['residual'].append(comp.mode.gap(qq)[0])
    result = dict(bodies=int(s.nbp), tree_speeds=int(s.nT), independent=comp.mode.independent_ids, chords=comp.mode.ids,
                  acquisition=dict(steps=int(info['acquisition']['accepted_steps']), residual=float(info['polish']['residual_inf'])),
                  zeta_of_delta_max_abs_error_rad=float(max(rec['zeta'])),
                  dzeta_ddelta_max_rel_error=float(max(rec['dzeta'])), closure_residual_max=float(max(rec['residual'])),
                  configurations=64)
    print('[compile] zeta(delta): %.1e rad   d zeta / d delta (Eq. 28): %.1e relative   closure residual %.1e' % (
        result['zeta_of_delta_max_abs_error_rad'], result['dzeta_ddelta_max_rel_error'], result['closure_residual_max']), flush=True)
    write_json(Path(out) / 'compile.json', result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(ROOT / 'results' / 'manipulator'))
    args = ap.parse_args(argv)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    study_compile(args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
