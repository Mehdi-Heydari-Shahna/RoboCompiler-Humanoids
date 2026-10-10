"""An excavator record supplied by the user (not distributed with this repository).

The excavator cases of the state-wise studies and ``tests/test_repository_records.py``
read the record from a private path, given by the environment variable
``ROBOCOMPILER_EXCAVATOR_RECORD`` or by ``--excavator-record PATH`` of ``reproduce.py``
and ``workflows/statewise/run_statewise.py``.  Without a record, the excavator cases
are skipped.

A record in ``cmg.physical-mechanism-graph/1.0`` is used as it is.  A record in
``cmg.physical-closed-chain/1.0`` is adapted by ``adapt_closed_chain_record``, which only
adds what the schema of this package requires and such a record does not carry:

* an entry for the ground body,
* joint limits (the record's own branch window about its seed),
* effort bounds (placeholders: the record has none; they only scale random efforts in the
  state-wise studies).

Bodies, inertias, joints, attachment frames, axes, actuated joints, seed and partition
requests are those of the record.
"""
from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path

ENV = 'ROBOCOMPILER_EXCAVATOR_RECORD'
EFFORT_PLACEHOLDER = 1.0e4


class RecordUnavailable(FileNotFoundError):
    """No excavator record was supplied."""


def record_path(path=None):
    """``path``, else the value of ``ROBOCOMPILER_EXCAVATOR_RECORD``, else None."""
    p = path or os.environ.get(ENV)
    return Path(p).expanduser().resolve() if p else None


def available(path=None):
    """True when a record was supplied (a wrong path still fails loudly when it is read)."""
    return record_path(path) is not None


def record(path=None):
    """The supplied record as stored on disk."""
    p = record_path(path)
    if p is None:
        raise RecordUnavailable(f'no excavator record: set {ENV} or pass --excavator-record')
    return json.loads(p.read_bytes())


def record_sha256(path=None):
    """SHA-256 of the supplied record file (for the provenance of private runs)."""
    return hashlib.sha256(record_path(path).read_bytes()).hexdigest()


def adapt_closed_chain_record(g):
    """``cmg.physical-closed-chain/1.0`` -> ``cmg.physical-mechanism-graph/1.0`` (schema additions only)."""
    if g.get('schema') != 'cmg.physical-closed-chain/1.0':
        raise ValueError(f"unexpected schema {g.get('schema')}")
    rec = {'schema': 'cmg.physical-mechanism-graph/1.0', 'name': g['name'], 'ground': g['ground'],
           'gravity_m_s2': g['gravity_m_s2'], 'seed': deepcopy(g['seed']),
           'partition_requests': deepcopy(g['partition_requests'])}
    bodies = [] if g['ground'] in {b['id'] for b in g['bodies']} else [{'id': g['ground'], 'massless': True, 'mass_kg': 0.}]
    bodies += [{k: b[k] for k in ('id', 'mass_kg', 'com_m', 'inertia_com_kg_m2')} for b in g['bodies']]
    window = {'revolute': g['branch_window']['revolute_rad'], 'prismatic': g['branch_window']['prismatic_m']}
    joints = []
    for j in g['joints']:
        r = {k: j[k] for k in ('id', 'type', 'body_a', 'body_b', 'T_AJ', 'T_BJ')}
        if j['type'] in window:
            seed = g['seed']['joints'][j['id']]
            r.update(axis=j['axis'], limits={'lower': seed - window[j['type']], 'upper': seed + window[j['type']]})
        joints.append(r)
    actuators = [{'id': a['id'], 'joint_id': a['joint_id'], 'gear': 1., 'effort_unit': a['unit'],
                  'effort_bounds': [-EFFORT_PLACEHOLDER, EFFORT_PLACEHOLDER]} for a in g['actuators']]
    rec.update(bodies=bodies, joints=joints, actuators=actuators)
    return rec


def physical(path=None):
    """The record ready for ``ipd.compiler.compile_graph`` (adapted when it is a closed-chain record)."""
    g = record(path)
    return g if g.get('schema') == 'cmg.physical-mechanism-graph/1.0' else adapt_closed_chain_record(g)
