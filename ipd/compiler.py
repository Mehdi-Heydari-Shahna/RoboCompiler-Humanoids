"""Graph compilation of a physical mechanism record.

The input holds physical records only: rigid bodies with inertias, joints with their
two attachment frames, actuator ports, drive data and named sites.  No joint is
marked as a cut and no path, partition, module or coordinate order may be supplied.
``compile_graph`` generates

* a spanning tree and the physical joints that close it (chords),
* a directed tree with a depth-first coordinate order,
* the massless cut-joint coordinates retained on every chord, so that each chord is
  an SE(3) path-pose comparison ``Delta_e = inv(T_e^-) T_e^+ = I``,
* the signed tree-path incidence of every chord (common ancestors cancel),
* the candidate independent/dependent partition of a requested partition name and
  the dependency modules obtained from shared dependent coordinates.

Numerical rank, row selection and the lift are evaluated later on a ``System``
(see ``ipd.system``); this module is purely structural and needs only NumPy.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

SCHEMA_PHYSICAL = 'cmg.physical-mechanism-graph/1.0'
SCHEMA_COMPILED = 'cmg.compiled-mechanism-graph/1.0'

FORBIDDEN = {'coordinate_ids', 'independent_ids', 'dependent_ids', 'modules', 'paths', 'tree',
             'tree_ids', 'cuts', 'cut_joint_ids', 'closures', 'jacobian', 'chart_columns',
             'partitions', 'sparsity', 'rows', 'incidence'}
JOINT_TYPES = ('fixed', 'revolute', 'prismatic', 'spherical')
CUT_DOFS = {'fixed': 0, 'revolute': 1, 'prismatic': 1, 'spherical': 3}
ROOT_IDS = ('wx', 'wy', 'wz', 'vx', 'vy', 'vz')


class ModelError(ValueError):
    """The physical record is inconsistent or cannot be compiled."""


def _is_transform(T, tol=1e-9):
    T = np.asarray(T, float)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        return False
    R = T[:3, :3]
    return bool(np.max(abs(R @ R.T - np.eye(3))) < tol and abs(np.linalg.det(R) - 1) < tol
                and np.max(abs(T[3] - [0, 0, 0, 1])) < tol)


def _check_inertia(body):
    I = np.asarray(body['inertia_com_kg_m2'], float)
    if I.shape != (3, 3) or np.max(abs(I - I.T)) > 1e-9 * max(1., np.max(abs(I))):
        raise ModelError(f"Body {body['id']}: inertia is not symmetric")
    w = np.linalg.eigvalsh((I + I.T) / 2)
    if w[0] <= 0:
        raise ModelError(f"Body {body['id']}: inertia is not positive definite")
    if w[0] + w[1] < w[2] * (1 - 1e-9):
        raise ModelError(f"Body {body['id']}: inertia violates the triangle inequality")


def validate(physical):
    """Check a physical record and return (bodies, joints) dictionaries keyed by id."""
    if FORBIDDEN & set(physical):
        raise ModelError('Input must not pre-author trees, cuts, paths, partitions, modules or '
                         'coordinate orders: ' + ', '.join(sorted(FORBIDDEN & set(physical))))
    bodies = {}
    for b in physical['bodies']:
        if b['id'] in bodies:
            raise ModelError(f"Duplicate body id {b['id']}")
        if '/' in b['id']:
            raise ModelError(f"Body id {b['id']} must not contain '/'")
        massless = bool(b.get('massless', False))
        if massless:
            if float(b.get('mass_kg', 0.)) != 0.:
                raise ModelError(f"Body {b['id']} is declared massless but has mass")
        else:
            if not float(b['mass_kg']) > 0:
                raise ModelError(f"Body {b['id']}: mass must be positive")
            _check_inertia(b)
        bodies[b['id']] = b
    joints = {}
    for j in physical['joints']:
        if j['id'] in joints or j['id'] in bodies:
            raise ModelError(f"Duplicate id {j['id']}")
        if j['type'] not in JOINT_TYPES:
            raise ModelError(f"Joint {j['id']}: unsupported type {j['type']}")
        for key in ('body_a', 'body_b'):
            if j[key] not in bodies:
                raise ModelError(f"Joint {j['id']}: unknown body {j[key]}")
        if j['body_a'] == j['body_b']:
            raise ModelError(f"Joint {j['id']}: connects a body to itself")
        if not (_is_transform(j['T_AJ']) and _is_transform(j['T_BJ'])):
            raise ModelError(f"Joint {j['id']}: attachment frames are not rigid transforms")
        if j['type'] in ('revolute', 'prismatic'):
            if abs(np.linalg.norm(j['axis']) - 1) > 1e-9:
                raise ModelError(f"Joint {j['id']}: axis is not a unit vector")
            lim = j['limits']
            if not lim['lower'] < lim['upper']:
                raise ModelError(f"Joint {j['id']}: empty limit interval")
        joints[j['id']] = j
    root = physical.get('floating_root', physical.get('ground'))
    if root not in bodies:
        raise ModelError('floating_root or ground must name a body')
    if 'floating_root' in physical and 'ground' in physical:
        raise ModelError('Give either floating_root or ground, not both')
    used = set()
    for a in physical.get('actuators', []):
        j = joints.get(a['joint_id'])
        if j is None or j['type'] not in ('revolute', 'prismatic'):
            raise ModelError(f"Actuator {a['id']}: needs an existing scalar joint")
        if a['joint_id'] in used:
            raise ModelError(f"Actuator {a['id']}: joint already has an actuator port")
        used.add(a['joint_id'])
        if float(a.get('gear', 1.)) == 0.:
            raise ModelError(f"Actuator {a['id']}: zero gear")
        lo, hi = a['effort_bounds']
        if not lo < hi:
            raise ModelError(f"Actuator {a['id']}: empty effort interval")
    for key in ('sole_sites', 'point_sites'):
        for s in physical.get(key, []):
            if s['body'] not in bodies:
                raise ModelError(f"Site {s['id']}: unknown body {s['body']}")
    return bodies, joints


def _depths(root, bodies, joints):
    """Breadth-first body depths in the complete (cyclic) joint graph."""
    adj = {b: [] for b in bodies}
    for j in joints.values():
        adj[j['body_a']].append(j['body_b'])
        adj[j['body_b']].append(j['body_a'])
    depth = {root: 0}
    frontier = [root]
    while frontier:
        nxt = []
        for b in frontier:
            for c in sorted(adj[b]):
                if c not in depth:
                    depth[c] = depth[b] + 1
                    nxt.append(c)
        frontier = nxt
    missing = sorted(set(bodies) - set(depth))
    if missing:
        raise ModelError('Bodies not connected to the root: ' + ', '.join(missing))
    return depth


def spanning_tree(root, bodies, joints, keep):
    """Kruskal tree selection; returns (tree joint ids, chord joint ids).

    Edge classes, in the order they are offered to the tree: fixed joints, joints the
    partition keeps independent, that carry an actuator port or that carry drive data
    (rotor inertia, damping, friction loss), prismatic joints, revolute joints,
    spherical joints.  Ties are resolved by graph depth and then by id, so the
    selection does not depend on the record order.

    A joint of the second class never becomes a chord: its coordinate has to exist in
    the tree for its port or its drive data to act on it in every generated view.  A
    loop made of such joints only is refused.
    """
    depth = _depths(root, bodies, joints)

    def klass(j):
        if j['type'] == 'fixed':
            return 0
        if j['id'] in keep:
            return 1
        return {'prismatic': 2, 'revolute': 3, 'spherical': 4}[j['type']]

    order = sorted(joints.values(), key=lambda j: (klass(j), min(depth[j['body_a']], depth[j['body_b']]),
                                                   max(depth[j['body_a']], depth[j['body_b']]), j['id']))
    parent = {b: b for b in bodies}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree, chords = [], []
    for j in order:
        a, b = find(j['body_a']), find(j['body_b'])
        if a == b:
            if j['type'] == 'fixed' or j['id'] in keep:
                raise ModelError(f"Joint {j['id']} closes a loop but must stay in the tree (it is fixed, requested "
                                 'as independent, actuated or carries drive data); every loop needs one joint '
                                 'without port and without drive data')
            chords.append(j['id'])
        else:
            parent[a] = b
            tree.append(j['id'])
    for j in tree:
        if joints[j]['type'] == 'spherical':
            raise ModelError(f"Spherical joint {j} would be a tree joint; only spherical chords are supported")
    return tree, chords


DRIVE_KEYS = ('armature', 'damping', 'frictionloss')


def drive_joints(physical, joints):
    """Ids of the joints that carry non-zero drive data; the record is checked on the way."""
    out = set()
    for jid, data in physical.get('drive', {}).items():
        if jid not in joints:
            raise ModelError(f'Drive data for unknown joint {jid}')
        unknown = set(data) - set(DRIVE_KEYS)
        if unknown:
            raise ModelError(f'Joint {jid}: unknown drive data ' + ', '.join(sorted(unknown)))
        values = [float(data.get(k, 0.)) for k in DRIVE_KEYS]
        if min(values) < 0.:
            raise ModelError(f'Joint {jid}: negative drive data')
        if max(values) > 0.:
            if joints[jid]['type'] not in ('revolute', 'prismatic'):
                raise ModelError(f'Joint {jid}: drive data need a scalar joint')
            out.add(jid)
    return out


def components(supports):
    """Connected components of index sets that share at least one element."""
    parent = list(range(len(supports)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner = {}
    for i, s in enumerate(supports):
        for x in s:
            if x in owner:
                a, b = find(i), find(owner[x])
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                owner[x] = i
    groups = {}
    for i in range(len(supports)):
        groups.setdefault(find(i), []).append(i)
    return [groups[k] for k in sorted(groups)]


@dataclass
class Compilation:
    """Compiled mechanism: generated graph data (``cmg``), structural plan and source."""
    cmg: dict
    plan: dict
    physical: dict
    partition: str = ''
    _index: dict = field(default_factory=dict, repr=False)

    def coordinate_index(self, cid):
        return self._index[cid]


def physical_sha256(physical):
    return hashlib.sha256(json.dumps(physical, sort_keys=True).encode()).hexdigest()


def compile_graph(physical, partition=None, independent=None):
    """Compile one physical mechanism record.

    ``partition`` names an entry of ``physical['partition_requests']``; ``independent``
    may give the independent joint ids directly.  With neither, the first request is
    used.  The root twist of a floating mechanism is always independent here; support
    and interconnection closures are added when mechanisms are composed.
    """
    physical = deepcopy(physical)
    bodies, joints = validate(physical)
    floating = 'floating_root' in physical
    root = physical['floating_root'] if floating else physical['ground']
    requests = physical.get('partition_requests', {})
    if independent is None:
        if partition is None:
            if not requests:
                raise ModelError('No partition request in the record')
            partition = sorted(requests)[0] if 'joint_space' not in requests else 'joint_space'
        if partition not in requests:
            raise ModelError(f'Unknown partition request {partition}')
        independent = list(requests[partition])
    else:
        partition = partition or 'custom'
    if len(set(independent)) != len(independent):
        raise ModelError('Repeated id in the independent request')
    for j in independent:
        if j not in joints or joints[j]['type'] not in ('revolute', 'prismatic'):
            raise ModelError(f'Independent request {j} is not a scalar joint')
    actuated = {a['joint_id'] for a in physical.get('actuators', [])}
    driven = drive_joints(physical, joints)
    tree_ids, chord_ids = spanning_tree(root, bodies, joints, set(independent) | actuated | driven)

    # Directed tree, depth-first preorder with children visited in joint-id order.
    adj = {b: [] for b in bodies}
    for jid in tree_ids:
        j = joints[jid]
        adj[j['body_a']].append((j, j['body_b'], 1))
        adj[j['body_b']].append((j, j['body_a'], -1))
    directed, order, parent = [], [root], {root: None}

    def walk(body):
        for j, child, direction in sorted(adj[body], key=lambda x: x[0]['id']):
            if child in parent:
                continue
            parent[child] = body
            e = {'id': j['id'], 'type': j['type'], 'base_body': body, 'follower_body': child,
                 'T_BJ': deepcopy(j['T_AJ'] if direction == 1 else j['T_BJ']),
                 'T_FJ': deepcopy(j['T_BJ'] if direction == 1 else j['T_AJ'])}
            if j['type'] != 'fixed':
                # A joint traversed from body_b to body_a keeps its physical coordinate: the
                # motion seen from body_b is the inverse one, i.e. the same value about the
                # negated axis, so the limits are those of the record.
                e['axis'] = (direction * np.asarray(j['axis'], float)).tolist()
                e['limits'] = deepcopy(j['limits'])
                e['reversed'] = direction == -1
            directed.append(e)
            order.append(child)
            walk(child)

    import sys
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 4 * len(bodies) + 100))
    walk(root)
    coordinate_ids = ([f'root_{x}' for x in ROOT_IDS] if floating else []) + \
        [e['id'] for e in directed if e['type'] != 'fixed']
    nt = len(coordinate_ids)

    # Chords.  The cut joint keeps its own coordinates on a chain of massless frames
    # attached to body_a, so closure is the full SE(3) equality of two frames.
    closures, cut_coordinate_ids = [], []
    for jid in sorted(chord_ids):
        j = joints[jid]
        nd = CUT_DOFS[j['type']]
        names = [f'{jid}__cut'] if nd == 1 else [f'{jid}__cut_{a}' for a in 'xyz'[:nd]]
        closures.append({'id': jid, 'type': j['type'], 'body1': j['body_a'], 'body2': j['body_b'],
                         'T_1J': deepcopy(j['T_AJ']), 'T_2J': deepcopy(j['T_BJ']),
                         'axis': list(j.get('axis', [0., 0., 1.])), 'cut_coordinates': names,
                         'limits': deepcopy(j.get('limits'))})
        cut_coordinate_ids += names
    for c in physical.get('loop_cuts', []):
        if c['type'] != 'point_coincidence':
            raise ModelError(f"Loop cut {c['id']}: unsupported type {c['type']}")
        T1, T2 = np.eye(4), np.eye(4)
        T1[:3, 3], T2[:3, 3] = c['point1_m'], c['point2_m']
        names = [f"{c['id']}__cut_{a}" for a in 'xyz']
        closures.append({'id': c['id'], 'type': 'spherical', 'body1': c['body1'], 'body2': c['body2'],
                         'T_1J': T1.tolist(), 'T_2J': T2.tolist(), 'axis': [0., 0., 1.],
                         'cut_coordinates': names, 'limits': None})
        cut_coordinate_ids += names

    # Ancestor coordinate sets and signed chord incidence (common ancestors cancel).
    edge_of = {e['follower_body']: e for e in directed}

    def ancestors(body):
        out = set(coordinate_ids[:6]) if floating else set()
        while body != root:
            e = edge_of[body]
            if e['type'] != 'fixed':
                out.add(e['id'])
            body = e['base_body']
        return out

    index = {c: i for i, c in enumerate(coordinate_ids + cut_coordinate_ids)}
    independent_ids = (coordinate_ids[:6] if floating else []) + list(independent)
    for c in independent:
        if c not in index or index[c] >= nt:
            raise ModelError(f'Independent request {c} is not a tree coordinate')
    indep = set(independent_ids)
    cut_plans = []
    for c in closures:
        a1, a2 = ancestors(c['body1']), ancestors(c['body2'])
        path = sorted((a1 ^ a2), key=index.get)
        dependent = [x for x in path if x not in indep] + c['cut_coordinates']
        if not dependent:
            raise ModelError(f"Chord {c['id']} has no dependent coordinate")
        cut_plans.append({'cut': c['id'], 'type': c['type'], 'body1': c['body1'], 'body2': c['body2'],
                          'minus_path': sorted(a1 - a2, key=index.get), 'plus_path': sorted(a2 - a1, key=index.get),
                          'dependent_ids': dependent, 'input_ids': [x for x in path if x in indep],
                          'residual_rows': 6})
    covered = set().union(*[set(p['dependent_ids']) for p in cut_plans]) if cut_plans else set()
    undetermined = [c for c in coordinate_ids if c not in indep and c not in covered]
    if undetermined:
        raise ModelError('Coordinates neither independent nor determined by a chord: ' + ', '.join(undetermined))
    groups = components([p['dependent_ids'] for p in cut_plans])
    modules = []
    for g in groups:
        dep = sorted(set().union(*[set(cut_plans[i]['dependent_ids']) for i in g]), key=index.get)
        inp = sorted(set().union(*[set(cut_plans[i]['input_ids']) for i in g]) - set(dep), key=index.get)
        modules.append({'cuts': [cut_plans[i]['cut'] for i in g], 'dependent_ids': dep, 'input_ids': inp,
                        'residual_rows': 6 * len(g)})

    drive = physical.get('drive', {})
    cmg = {
        'schema': SCHEMA_COMPILED, 'name': physical.get('name', 'mechanism'), 'root_body': root,
        'floating': floating, 'body_order': order,
        'bodies': [deepcopy(bodies[b]) for b in order], 'joints': directed,
        'coordinate_ids': coordinate_ids, 'cut_coordinate_ids': cut_coordinate_ids,
        'independent_ids': independent_ids, 'closures': closures,
        'actuators': sorted(deepcopy(physical.get('actuators', [])), key=lambda a: a['id']),
        'armature': {c: float(drive.get(c, {}).get('armature', 0.)) for c in coordinate_ids[6 if floating else 0:]},
        'damping': {c: float(drive.get(c, {}).get('damping', 0.)) for c in coordinate_ids[6 if floating else 0:]},
        'frictionloss': {c: float(drive.get(c, {}).get('frictionloss', 0.)) for c in coordinate_ids[6 if floating else 0:]},
        'sole_sites': deepcopy(physical.get('sole_sites', [])),
        'point_sites': deepcopy(physical.get('point_sites', [])),
        'gravity_m_s2': list(physical.get('gravity_m_s2', [0., 0., -9.81])),
        'seed': deepcopy(physical.get('seed', {'joints': {}})),
    }
    moving = [e['id'] for e in directed if e['type'] != 'fixed']
    plan = {
        'partition': partition, 'physical_bodies': len(bodies), 'physical_joints': len(joints),
        'tree_joint_ids': sorted(tree_ids), 'cut_joint_ids': sorted(chord_ids),
        'cycle_rank': len(joints) - len(bodies) + 1, 'tree_coordinates': nt,
        'cut_coordinates': len(cut_coordinate_ids), 'augmented_coordinates': nt + len(cut_coordinate_ids),
        'augmented_residual_rows': 6 * len(closures), 'independent_ids': independent_ids,
        'mobility_expected': len(independent_ids), 'floating_base_coordinates': 6 if floating else 0,
        'physical_actuators': len(cmg['actuators']), 'cut_plans': cut_plans, 'modules': modules,
        'module_sizes': [len(m['dependent_ids']) for m in modules],
        'largest_module_block': max([len(m['dependent_ids']) for m in modules], default=0),
        'moving_joint_ids': moving, 'physical_input_sha256': physical_sha256(physical),
        'note': 'Structural plan only. Rank, row selection and conditioning are checked numerically.',
    }
    if len(closures) != plan['cycle_rank'] + len(physical.get('loop_cuts', [])):
        raise ModelError('Chord count differs from the cycle rank of the joint graph')
    return Compilation(cmg=cmg, plan=plan, physical=physical, partition=partition, _index=index)
