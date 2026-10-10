"""Build a formation, its closure mode and its dance reference (cached on disk)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ipd.control import Reference

from .choreography import Dance, generate
from .ensemble import Ensemble, PartitionManager

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ('dancers/humanoid.py', 'dancers/ensemble.py', 'dancers/choreography.py', 'ipd/compiler.py', 'ipd/system.py',
           'ipd/mode.py', 'ipd/composite.py')


def source_hash():
    h = hashlib.sha256()
    for rel in SOURCES:
        h.update((ROOT / rel).read_bytes())
    return h.hexdigest()


class Session:
    """Ensemble, partition manager (assembled by PACDM) and the dance reference."""

    def __init__(self, formation='circle', partition='joint_space', duration=30., dt=0.002, scale=1.0,
                 cache=None, verbose=False, adaptive=True):
        self.formation = formation
        self.ens = Ensemble(formation, partition)
        self.sys = self.ens.system
        self.manager = PartitionManager(self.ens, self.sys.neutral(), acquire=True, adaptive=adaptive)
        self.q0 = self.manager.q.copy()
        self.assembly = self.manager.info
        self.dance = Dance(self.ens.n, formation, duration=duration, scale=scale)
        key = hashlib.sha256(json.dumps([formation, partition, duration, dt, scale, source_hash()]).encode()).hexdigest()[:16]
        cache = Path(cache) if cache else ROOT / 'results' / 'cache'
        path = cache / f'reference_{formation}_{key}.npz'
        if path.exists():
            z = np.load(path, allow_pickle=False)
            self.ref = Reference(z['t'], z['T_root'], z['theta'], z['v'], z['a'], info=json.loads(str(z['info'])))
        else:
            progress = (lambda i, n: print(f'  reference {formation}: {i}/{n}', flush=True)) if verbose else None
            gen = PartitionManager(self.ens, self.sys.neutral(), acquire=True, adaptive=adaptive)
            self.ref = generate(self.ens, gen, self.dance, dt=dt, progress=progress)
            cache.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, t=self.ref.t, T_root=self.ref.T_root, theta=self.ref.theta, v=self.ref.v,
                                a=self.ref.a, info=json.dumps(self.ref.info))
        self.reference_path = path

    @property
    def comp(self):
        return self.manager.comp
