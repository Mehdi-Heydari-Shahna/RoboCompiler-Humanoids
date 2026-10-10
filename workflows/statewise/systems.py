"""Systems and closure modes of the state-wise studies."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dancers.ensemble import Ensemble, PartitionManager, VARIANTS          # noqa: E402
from examples import excavator as excavator_record                         # noqa: E402
from examples import series_parallel                                       # noqa: E402
from ipd.compiler import compile_graph                                     # noqa: E402
from ipd.composite import Composite                                        # noqa: E402
from ipd.system import Member, System                                      # noqa: E402


class Case:
    """One system in one closure mode, assembled by PACDM, with a sampler of feasible states."""

    def __init__(self, name, system, comp, q0, scale_q, scale_u, effort_fraction, ens=None):
        self.name, self.sys, self.comp, self.q0 = name, system, comp, q0
        self.scale_q, self.scale_u, self.effort_fraction, self.ens = scale_q, scale_u, effort_fraction, ens

    def sample(self, rng):
        """Random feasible state near the assembled configuration: (q, u, f)."""
        s, comp = self.sys, self.comp
        for _ in range(20):
            dv = np.zeros(s.n)
            dv[comp.mode.independent] = rng.uniform(-1., 1., comp.d) * self.scale_q
            q, ci = comp.mode.correct(s.integrate(self.q0, dv), maxiter=40)
            margin = np.min(np.minimum(q.theta[:s.nT] - s.lower[:s.nT], s.upper[:s.nT] - q.theta[:s.nT]))
            if ci['success'] and margin > 1e-3:
                break
        else:
            raise RuntimeError(f'{self.name}: could not sample a feasible state')
        u = rng.uniform(-1., 1., comp.d) * self.scale_u
        f = rng.uniform(-1., 1., s.na) * s.effort_upper * self.effort_fraction
        return q, u, f


def manipulator(partition='actuator_space'):
    comp_g = compile_graph(series_parallel.physical(), partition)
    s = System([Member('m', comp_g, np.eye(4))], name='series_parallel')
    comp = Composite(s, name=f'manipulator:{partition}')
    q, _ = comp.prepare(s.neutral(), acquire=True)
    scale_q = np.array([0.4, 0.3, 0.04 if partition == 'actuator_space' else 0.25])
    return Case(f'manipulator/{partition}', s, comp, q, scale_q, np.array([0.8, 0.8, 0.15 if partition == 'actuator_space' else 0.8]), 0.05)


class PolishedCase(Case):
    """A case whose samples are polished to round-off on the closure manifold.

    For machines whose body masses span several orders of magnitude the reduced inertia
    is badly conditioned, so a closure residual of 1e-12 would already show in the
    accelerations.
    """

    def sample(self, rng):
        q, u, f = super().sample(rng)
        q, info = self.comp.mode.correct(q, tol=1e-14, maxiter=6)
        return q, u, f


def excavator(partition='joint_space'):
    graph = compile_graph(excavator_record.physical(), partition)
    s = System([Member('x', graph, np.eye(4))], name='excavator')
    comp = Composite(s, name=f'excavator:{partition}')
    q, _ = comp.prepare(s.neutral(), acquire=True)
    d = comp.d
    case = PolishedCase(f'excavator/{partition}', s, comp, q, np.full(d, 0.05), np.full(d, 0.3), 0.5)
    case.record_sha256 = excavator_record.record_sha256()
    return case


def single_dancer(stance='both', partition='joint_space', variant=2):
    ens = Ensemble('line', partition, variants=(VARIANTS[variant],))
    man = PartitionManager(ens, ens.system.neutral(), stance=[stance], acquire=True)
    d = man.comp.d
    scale_q = np.full(d, 0.06)
    scale_q[:6] = [0.05, 0.05, 0.05, 0.03, 0.03, 0.03]
    return Case(f'dancer/{stance}', ens.system, man.comp, man.q, scale_q, 0.5, 0.05, ens)


def ensemble(formation='circle', stance='both', links=None, partition='joint_space', tag=None):
    ens = Ensemble(formation, partition)
    active = None if links is None else links
    man = PartitionManager(ens, ens.system.neutral(), stance=[stance] * ens.n, links=active, acquire=True)
    d = man.comp.d
    return Case(tag or f'{formation}/{stance}', ens.system, man.comp, man.q, np.full(d, 0.03), 0.4, 0.04, ens)


CASES = {
    'manipulator': lambda: manipulator('actuator_space'),
    'manipulator_joint': lambda: manipulator('joint_space'),
    'excavator': lambda: excavator('joint_space'),
    'excavator_cylinder': lambda: excavator('cylinder_space'),
    'dancer_floating': lambda: single_dancer('none'),
    'dancer_single': lambda: single_dancer('left'),
    'dancer_double': lambda: single_dancer('both'),
    'circle': lambda: ensemble('circle'),
    'semicircle': lambda: ensemble('semicircle'),
    'line': lambda: ensemble('line'),
    'circle_single': lambda: ensemble('circle', 'left', tag='circle/left stance'),
    'circle_actuator': lambda: ensemble('circle', partition='actuator_space', tag='circle/actuator space'),
}
EXCAVATOR_CASES = {'excavator': 'joint_space', 'excavator_cylinder': 'cylinder_space'}


def available(name):
    """False for an excavator case without a supplied record (or whose partition request the record lacks)."""
    if name not in EXCAVATOR_CASES:
        return True
    if not excavator_record.available():
        return False
    return EXCAVATOR_CASES[name] in excavator_record.physical().get('partition_requests', {})
