"""Five similar humanoids in circle, semicircle and line formations.

The dancers are five physical records of one architecture (``dancers.humanoid``) with
different link dimensions and inertias.  Each is compiled on its own; the ensemble
joins the compiled mechanisms through hand links (point connections between a left
palm and the right palm of the neighbour) and ideal sole supports.  Formation and
support mode change only the interconnection records.
"""
from __future__ import annotations

import numpy as np

from ipd.compiler import compile_graph
from ipd.composite import Composite
from ipd.shapes import rot, tf
from ipd.system import Member, System

from .humanoid import ARM_JOINTS, LEG_JOINTS, humanoid

VARIANTS = (
    dict(scale=0.94, arm_scale=1.00, leg_scale=1.00, color='red'),
    dict(scale=0.98, arm_scale=1.03, leg_scale=0.99, color='orange'),
    dict(scale=1.00, arm_scale=1.00, leg_scale=1.00, color='green'),
    dict(scale=1.03, arm_scale=0.98, leg_scale=1.01, color='blue'),
    dict(scale=1.06, arm_scale=1.00, leg_scale=1.00, color='purple'),
)
FORMATIONS = ('circle', 'semicircle', 'line')
SPACING = 1.05      # nominal distance between neighbouring pelvis centres (m)


def placement(formation, n=5, spacing=SPACING):
    """Pelvis ground positions (n, 2) and yaw angles (n,); the left neighbour of k is k-1."""
    if formation == 'circle':
        R = spacing / (2 * np.sin(np.pi / n))
        phi = 2 * np.pi * np.arange(n) / n
    elif formation == 'semicircle':
        step = np.pi / n
        R = spacing / (2 * np.sin(step / 2))
        phi = -np.pi / 2 + step * (np.arange(n) + .5)
    elif formation == 'line':
        return np.c_[np.zeros(n), -spacing * (np.arange(n) - (n - 1) / 2)], np.zeros(n)
    else:
        raise ValueError(formation)
    return R * np.c_[np.cos(phi), np.sin(phi)], phi + np.pi


def link_records(formation, n=5):
    """Hand links: the left palm of dancer k holds the right palm of dancer k-1."""
    ks = range(n) if formation == 'circle' else range(1, n)
    return [{'id': f'hand_{k}_{(k - 1) % n}', 'member1': f'd{k}', 'site1': 'left_palm',
             'member2': f'd{(k - 1) % n}', 'site2': 'right_palm'} for k in ks]


def standing_height(comp):
    """Pelvis height that puts the soles of the seed posture on the ground."""
    s = System([Member('x', comp, np.eye(4))])
    q = s.neutral()
    T = s.poses(q)
    b, P = s.site_frame('x', 'left_sole')
    return -float((T[b] @ P)[2, 3])


ARM_NOMINAL = dict(shoulder_pitch=-0.25, shoulder_roll=0.65, shoulder_yaw=0.0, elbow=1.20)
ARM_BOUNDS = dict(shoulder_pitch=(-1.5, 0.6), shoulder_roll=(0.08, 1.5), shoulder_yaw=(-1.2, 1.2), elbow=(0.45, 1.7))


def arm_seed(comps, xy, yaw, links, heights):
    """Seed arm postures that bring the linked palms together with clearly bent elbows.

    A bounded least-squares fit over the arm joints of all dancers shares the
    reach between both partners of every link and stays close to a nominal
    posture.  It only supplies seeds: exact closure is established by PACDM.
    Returns {coordinate id: value} and the largest remaining palm gap.
    """
    from scipy.optimize import least_squares
    members = [Member(f'd{k}', c, tf((xy[k, 0], xy[k, 1], heights[k]), rot('z', yaw[k]))) for k, c in enumerate(comps)]
    s = System(members, links)
    q0 = s.neutral()
    frames = [(s.site_frame(l['member1'], l['site1']), s.site_frame(l['member2'], l['site2'])) for l in links]
    ids, x0, lo, hi, sign = [], [], [], [], []
    for m in s.members:
        for sd, sg in (('l', 1.), ('r', -1.)):
            for j in ARM_JOINTS:
                mirrored = sg if j in ('shoulder_roll', 'shoulder_yaw') else 1.
                ids.append(s.index[f'{m.name}/{sd}_{j}']); sign.append(mirrored)
                x0.append(ARM_NOMINAL[j]); lo.append(ARM_BOUNDS[j][0]); hi.append(ARM_BOUNDS[j][1])
    ids, sign, x0 = np.array(ids), np.array(sign), np.array(x0)

    def gaps(x):
        q = q0.copy()
        q.theta[ids] = sign * x
        T = s.poses(q)
        return np.array([(T[b1] @ P1)[:3, 3] - (T[b2] @ P2)[:3, 3] for (b1, P1), (b2, P2) in frames])

    res = least_squares(lambda x: np.r_[gaps(x).ravel() * 30., (x - x0) * 0.25], x0, bounds=(lo, hi), xtol=1e-10, ftol=1e-12)
    values = {s.ids[i].split('/', 1): float(v) for i, v in zip(ids, sign * res.x)} if False else \
        {s.ids[i]: float(v) for i, v in zip(ids, sign * res.x)}
    return values, float(np.max(np.linalg.norm(gaps(res.x), axis=1))) if links else 0.


def consistent_seed(record):
    """Close the internal loops of one record at its joint-space seed with PACDM.

    Returns the seed values of every tree coordinate, so that the same assembled
    posture can seed any partition (actuator strokes follow the joint angles).
    """
    from ipd.mode import Mode
    comp = compile_graph(record, 'joint_space')
    s = System([Member('x', comp, np.eye(4))])
    mode = Mode(s)
    q, _ = mode.acquire(s.neutral())
    return {cid.split('/', 1)[1]: float(q.theta[i]) for i, cid in enumerate(s.ids[:s.nT]) if not cid.startswith('x/root_')}


class Ensemble:
    def __init__(self, formation='circle', partition='joint_space', variants=VARIANTS, spacing=SPACING):
        self.formation, self.partition = formation, partition
        self.n = len(variants)
        self.records = [humanoid(name=f'd{k}', **v) for k, v in enumerate(variants)]
        self.links = link_records(formation, self.n)
        xy, yaw = placement(formation, self.n, spacing)
        self.xy, self.yaw = xy, yaw
        heights = [standing_height(compile_graph(r, 'joint_space')) for r in self.records]
        values, gap = arm_seed([compile_graph(r, 'joint_space') for r in self.records], xy, yaw, self.links, heights)
        self.seed_gap = gap
        for cid, value in values.items():
            k, jid = cid.split('/', 1)
            self.records[int(k[1:])]['seed']['joints'][jid] = value
        for r in self.records:
            r['seed']['joints'].update(consistent_seed(r))
        self.comps = [compile_graph(r, partition) for r in self.records]
        members = [Member(f'd{k}', c, tf((xy[k, 0], xy[k, 1], heights[k]), rot('z', yaw[k])))
                   for k, c in enumerate(self.comps)]
        self.system = System(members, self.links, name=f'dancers_{formation}')
        self.names = [m.name for m in members]

    # ------------------------------------------------------------------ partitions of a mode
    def leg_dependents(self, k, side):
        p = f'd{k}/{side[0]}_'
        if self.partition == 'joint_space':
            return [p + j for j in LEG_JOINTS]
        return [p + 'hip_yaw', p + 'hip_roll', p + 'hip_pitch', p + 'knee_stroke', p + 'ankle_o_stroke', p + 'ankle_i_stroke']

    # ------------------------------------------------------------------ arm partitions of hand links
    def arm_jacobian(self, kin, link):
        """Palm velocity of the left arm that holds ``link`` per rate of its four arm joints (3, 4)."""
        s = self.system
        k = link['member1']
        b, P = s.site_frame(k, link['site1'])
        dofs = [s.index[f'{k}/l_{j}'] for j in ARM_JOINTS]
        p = (kin.T[b] @ P)[:3, 3]
        Sa = kin.S[:, dofs]
        return Sa[3:] + np.cross(Sa[:3].T, p).T

    @staticmethod
    def triple_rcond(J, triple):
        sv = np.linalg.svd(J[:, list(triple)], compute_uv=False)
        return float(sv[-1] / sv[0])

    def best_triple(self, J):
        """Best-conditioned three of the four arm joints as dependents of a hand link."""
        from itertools import combinations
        scores = {t: self.triple_rcond(J, t) for t in combinations(range(4), 3)}
        best = max(scores, key=scores.get)
        return best, scores[best]

    def link_dependents(self, link, triple=(0, 1, 3)):
        p = f"{link['member1']}/l_"
        if self.partition == 'joint_space':
            return [p + ARM_JOINTS[i] for i in triple]
        return [p + 'shoulder_pitch', p + 'shoulder_roll', p + 'elbow_stroke']

    def composite(self, stance=None, links=None, anchors=None, name=None, triples=None):
        """Closure mode: ``stance[k]`` is 'both', 'left', 'right' or 'none' for dancer k.

        ``links`` selects active hand links (all by default).  ``anchors`` may map
        ``'dk/side_sole'`` to a world frame; other supports are welded where they are.
        ``triples`` maps a link id to the three arm joints (indices into ``ARM_JOINTS``)
        that the link determines; the default is (pitch, roll, elbow).
        """
        stance = ['both'] * self.n if stance is None else list(stance)
        links = [l['id'] for l in self.links] if links is None else list(links)
        triples = dict(triples or {})
        supports, dependents = [], {}
        for k, st in enumerate(stance):
            for side in ('left', 'right'):
                if st in ('both', side):
                    sid = f'd{k}/{side}_sole'
                    supports.append((f'd{k}', f'{side}_sole') if not anchors or sid not in anchors
                                    else (f'd{k}', f'{side}_sole', anchors[sid]))
                    dependents[sid] = self.leg_dependents(k, side)
        by_id = {l['id']: l for l in self.links}
        for l in links:
            dependents[l] = self.link_dependents(by_id[l], triples.get(l, (0, 1, 3)))
        comp = Composite(self.system, supports, links, dependents, name or f'{self.formation}:{"".join(s[0] for s in stance)}')
        comp.stance, comp.triples = stance, {l: tuple(triples.get(l, (0, 1, 3))) for l in links}
        return comp


class PartitionManager:
    """Computational reconfiguration of the hand-link partitions.

    Each hand link determines three of the four joints of the left arm that holds
    it.  Every such choice is singular somewhere, so the conditioning of the chosen
    triple is monitored and the partition is re-selected when it degrades.  The
    physical mechanism, its closures and its state are unchanged by a switch; only
    the independent speeds are re-chosen (``u = F u_new``).
    """

    def __init__(self, ens, q, stance=None, links=None, anchors=None, threshold=0.10, ratio=1.5, period=5,
                 acquire=False, adaptive=True):
        self.ens, self.threshold, self.ratio, self.period, self.adaptive = ens, threshold, ratio, period, adaptive
        self.stance, self.links = stance, links
        self.by_id = {l['id']: l for l in ens.links}
        self.calls, self.switches, self.min_rcond = 0, [], np.inf
        active = [l['id'] for l in ens.links] if links is None else list(links)
        triples = {}
        if adaptive and ens.partition == 'joint_space' and active:
            kin = ens.system.kinematics(q)
            triples = {l: self.ens.best_triple(self.ens.arm_jacobian(kin, self.by_id[l]))[0] for l in active}
        self.comp = ens.composite(stance, links, anchors, triples=triples)
        self.q, self.info = self.comp.prepare(q, acquire=acquire)

    def anchors(self):
        m = self.comp.mode
        return {sid: m.anchors[i] for i, sid in enumerate(m.support_ids)}

    def update(self, q, t=0., kin=None, force=False):
        """Return the current mode, re-partitioned at ``q`` if a link triple is ill-conditioned."""
        self.calls += 1
        if not self.adaptive or self.ens.partition != 'joint_space' or not self.comp.links:
            return self.comp, False
        if not force and self.calls % self.period:
            return self.comp, False
        kin = kin or self.ens.system.kinematics(q)
        new = dict(self.comp.triples)
        changed = False
        for l in self.comp.links:
            J = self.ens.arm_jacobian(kin, self.by_id[l])
            rc = self.ens.triple_rcond(J, self.comp.triples[l])
            self.min_rcond = min(self.min_rcond, rc)
            if rc < self.threshold:
                best, rb = self.ens.best_triple(J)
                if best != self.comp.triples[l] and rb > self.ratio * rc:
                    new[l], changed = best, True
        if changed:
            comp = self.ens.composite(self.comp.stance, self.comp.links, self.anchors(), triples=new)
            comp.prepare(q)
            self.switches.append((float(t), {l: new[l] for l in new if new[l] != self.comp.triples[l]}))
            self.comp = comp
        return self.comp, changed


def simplified_inertia(system):
    """Model inertias with the actuator inertial effects left out.

    The moving actuator bodies (barrels, rods, universal-joint crosses) get zero
    spatial inertia and the reflected inertia of the linear drives is removed.  This
    is the reduced model of a design that treats its actuators as ideal force sources.
    """
    M = system.M.copy()
    arm = system.armature.copy()
    for b, name in enumerate(system.body_names[:system.nbp]):
        if name.endswith(('_barrel', '_rod', '_cross')):
            M[b] = 0.
    for i, cid in enumerate(system.ids[:system.nT]):
        if cid.endswith('_stroke'):
            arm[i] = 0.
    return M, arm
