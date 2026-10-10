"""LaTeX tables, number macros and the evidence folder of the paper, from saved run data.

Run after the studies:  python workflows/report/make_tables.py [--results results]

* ``paper/generated/*.tex``  table bodies and ``numbers.tex`` (one macro per quoted number);
* ``evidence/``              the JSON summaries of all studies (no logs), ``SUMMARY.md`` and a
                             SHA-256 manifest (checked by ``verify_evidence.py``).

Every number in the paper is read from these files; nothing is typed by hand.  The
generator stops when the files it reads do not all carry the same code hash, when a run
that the paper needs is missing, or when a run ended with an exception
(``--allow-mixed`` and ``--allow-missing`` turn the checks into printed notes).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FORMATIONS = ('circle', 'semicircle', 'line')
FORMATION_NAME = {'circle': 'Ring', 'semicircle': 'Semicircle', 'line': 'Line'}
POLICIES = (('ip', 'Proposed (partitioned)'), ('recursive', 'Projected tree~\\eqref{eq:rc}'),
            ('decentralized', 'Per-robot models'), ('no_curvature', 'No curvature'),
            ('simplified', 'No actuator inertia'), ('massless', 'Massless actuators'),
            ('gravity_pd', 'Gravity + PD'), ('actuator_pd', 'Actuator PD'))
IDEAL = ('ip', 'decentralized', 'no_curvature', 'simplified', 'massless', 'gravity_pd')
SYSTEM_ROWS = (('manipulator', 'Manipulator, actuator space'), ('manipulator_joint', 'Manipulator, joint space'),
               ('excavator', 'Excavator, joint space'), ('excavator_cylinder', 'Excavator, cylinder space'),
               ('dancer_floating', 'Humanoid, no support'), ('dancer_single', 'Humanoid, one sole'),
               ('dancer_double', 'Humanoid, two soles'), ('circle', 'Ring of five, all soles'),
               ('circle_single', 'Ring, left soles'), ('circle_actuator', 'Ring, actuator space'),
               ('semicircle', 'Semicircle of five'), ('line', 'Line of five'))
COST_ROWS = (('manipulator', 'Manipulator'), ('excavator', 'Excavator'), ('dancer_double', 'Humanoid'),
             ('circle', 'Ring of five'))
MAIN_MUJOCO = '3.15.0'


# ---------------------------------------------------------------------- formatting
def sci(x, digits=1):
    """A number in LaTeX scientific notation (``$2.1\\times10^{-10}$``)."""
    if x is None or not np.isfinite(x):
        return '--'
    if x == 0:
        return '0'
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    if round(abs(m), digits) >= 10:
        m, e = m / 10, e + 1
    return f'${m:.{digits}f}\\!\\times\\!10^{{{e}}}$'


def num(x, digits=None):
    """A moderate number with a sensible number of digits, scientific notation for very small or large values."""
    if x is None or not np.isfinite(x):
        return '--'
    a = abs(x)
    if a == 0:
        return '0'
    if digits is not None:
        return f'{x:.{digits}f}'
    if a >= 1000 or a < 1e-3:
        return sci(x)
    if a >= 100:
        return f'{x:.0f}'
    if a >= 10:
        return f'{x:.1f}'
    if a >= 1:
        return f'{x:.2f}'
    return f'{x:.3f}'


def ratio(x):
    return '--' if x is None or not np.isfinite(x) else f'{x:.1f}'


def pct(x, digits=0):
    if x is None or not np.isfinite(x):
        return '--'
    v = 100. * x
    if 0 < v < 0.5 and digits == 0:
        return '<1' if v >= 0.05 else f'{v:.2f}'
    if 0 < v < 0.5 * 10. ** -digits:             # a non-zero fraction is never printed as zero
        return f'<{10. ** -digits:g}'
    return f'{v:.{digits}f}'


# ---------------------------------------------------------------------- inputs
class Data:
    """Result files with their provenance; records what is missing or failed."""

    def __init__(self, results):
        self.root = Path(results)
        self.hashes, self.versions, self.problems, self.used = {}, {}, [], []

    def get(self, rel, required=True):
        p = self.root / rel
        if not p.exists():
            if required and f'missing: {rel}' not in self.problems:
                self.problems.append(f'missing: {rel}')
            return None
        r = json.loads(p.read_text())
        prov = r.get('provenance') or r.get('_provenance') or {}
        self.hashes[rel] = prov.get('code_sha256')
        self.versions[rel] = prov.get('versions', {})
        if rel not in self.used:
            self.used.append(rel)
        if r.get('status') == 'exception':
            if not any(x.startswith(f'exception: {rel}:') for x in self.problems):
                self.problems.append(f"exception: {rel}: {r.get('error')}")
            return None
        return r

    def run(self, formation, scenario, plant, policy, required=True):
        return self.get(f'dance/{formation}/{scenario}__{plant}__{policy}.json', required)

    def metrics(self, formation, scenario, plant, policy, required=True):
        r = self.run(formation, scenario, plant, policy, required)
        return None if r is None else r['metrics']

    def check(self, allow_mixed=False, allow_missing=False):
        issues = [] if allow_missing else list(self.problems)
        if allow_missing:
            for p in self.problems:
                print('NOTE', p)
        hashes = set(self.hashes.values())
        if len(hashes) > 1 or None in hashes:
            groups = {}
            for rel, h in self.hashes.items():
                groups.setdefault(str(h)[:12], []).append(rel)
            text = 'result files carry different code hashes: ' + '; '.join(f'{k}: {len(v)} files (e.g. {v[0]})' for k, v in groups.items())
            if allow_mixed:
                print('NOTE', text)
            else:
                issues.append(text)
        for rel, v in self.versions.items():
            if v.get('mujoco') not in (None, MAIN_MUJOCO):
                issues.append(f"{rel}: MuJoCo {v.get('mujoco')} instead of {MAIN_MUJOCO}")
        if issues:
            raise SystemExit('make_tables: ' + '\n  '.join([''] + issues))


# ---------------------------------------------------------------------- numbers
class Numbers:
    def __init__(self):
        self.values = {}

    def __setitem__(self, key, value):
        if not key.isalpha():
            raise ValueError(f'macro names are letters only: {key}')
        self.values[key] = value

    def tex(self):
        lines = ['% Generated by workflows/report/make_tables.py from the result files; do not edit.']
        for k in sorted(self.values):
            lines.append(f'\\newcommand{{\\{k}}}{{{self.values[k]}}}')
        return '\n'.join(lines) + '\n'


def word(name):
    """Macro-safe CamelCase word of a policy or case name."""
    return ''.join(w.capitalize() for w in name.replace('-', '_').split('_'))


def maxroute(cons, route, key='delta_a', cases=None):
    vals = [c['routes'][route][key]['max'] for n, c in cons.items() if route in c['routes'] and (cases is None or n in cases)]
    return max(vals) if vals else None


def minroute(cons, route, key='delta_a', cases=None):
    vals = [c['routes'][route][key]['min'] for n, c in cons.items() if route in c['routes'] and (cases is None or n in cases)]
    return min(vals) if vals else None


def statewise_numbers(d, N):
    sysj = d.get('statewise/systems.json')
    cons = d.get('statewise/consistency.json')
    struct = d.get('statewise/structure.json')
    cov = d.get('statewise/covariance.json')
    consv = d.get('statewise/conservation.json')
    upd = d.get('statewise/updates.json')
    comp = d.get('statewise/composition.json')
    tim = d.get('statewise/timing.json')
    d.get('statewise/quasi_velocity.json')
    d.get('statewise/reconfiguration.json')
    d.get('manipulator/compile.json')
    if sysj:
        c = sysj['cases']
        N['HumBodies'] = c['dancer_floating']['record_bodies']
        N['HumJoints'] = c['dancer_floating']['record_joints']
        N['HumPorts'] = c['dancer_floating']['actuators']
        N['HumLoops'] = c['dancer_floating']['internal_loops']
        N['HumMassKg'] = num(c['dancer_floating']['mass_kg'], 1)
        exc = c.get('excavator')                  # absent without a private excavator record
        N['ExcBodies'] = exc['record_bodies'] if exc else '--'
        N['ExcJoints'] = exc['record_joints'] if exc else '--'
        N['RingBodies'] = c['circle']['record_bodies']
        N['RingChords'] = c['circle']['chords']
        N['RingSpeeds'] = c['circle']['independent_speeds']
        N['RingPorts'] = c['circle']['actuators']
        N['KneeReflectedKg'] = num(sysj['humanoid_drives']['knee_stroke']['reflected_inertia'], 0)
    if cons:
        cons = {k: v for k, v in cons.items() if not k.startswith('_')}
        N['ConsCases'] = len(cons)
        N['ConsStatesTotal'] = sum(c['states'] for c in cons.values())
        N['ConsMaxIp'] = sci(maxroute(cons, 'ip_two_level'))
        N['ConsMaxIpDf'] = sci(maxroute(cons, 'ip_two_level', 'delta_f'))
        N['ConsMaxIpOne'] = sci(maxroute(cons, 'ip_one_level'))
        N['ConsMaxPin'] = sci(maxroute(cons, 'pinocchio_kkt'))
        N['ConsMaxProjAnalytic'] = sci(maxroute(cons, 'projected_analytic'))
        N['ConsMaxProjFd'] = sci(maxroute(cons, 'projected_fd'))
        N['ConsMaxMaximal'] = sci(max(c['maximal_coordinates']['delta_body']['max'] for c in cons.values() if c['maximal_coordinates']))
        N['ConsTwoVsOne'] = sci(max(c['two_level_vs_one_level_max'] for c in cons.values() if c['two_level_vs_one_level_max'] is not None))
        for route, key in (('no_curvature', 'NoCurv'), ('simplified', 'Simplified'), ('massless', 'Massless'),
                           ('decentralized', 'Decentralized'), ('mujoco', 'Mujoco'), ('mujoco_rigid_limit', 'MujocoRigid')):
            N[f'Cons{key}Min'] = sci(minroute(cons, route))
            N[f'Cons{key}Max'] = sci(maxroute(cons, route))
        N['ConsDigits'] = cons['circle']['reference']['digits']
        worst = max(cons, key=lambda k: cons[k]['routes']['ip_two_level']['delta_a']['max'])
        N['ConsWorstCase'] = {'excavator': 'the excavator in joint space', 'excavator_cylinder': 'the excavator in cylinder space'}.get(
            worst, worst.replace('_', ' '))
        N['ConsWorstCond'] = sci(cons[worst]['cond_gamma_max'], 0)
        N['ConsRingIp'] = sci(cons['circle']['routes']['ip_two_level']['delta_a']['max'])
        N['ConsRingProjFd'] = sci(cons['circle']['routes']['projected_fd']['delta_a']['max'])
    if struct:
        struct = {k: v for k, v in struct.items() if not k.startswith('_')}
        N['StructSkewMax'] = sci(max(c['skew_fd']['max'] for c in struct.values()))
        N['StructSkewBodyMax'] = sci(max(c['skew_bodywise_fd']['max'] for c in struct.values()))
        N['StructRegressorMax'] = sci(max(c['regressor']['max'] for c in struct.values()))
        N['StructPotentialMax'] = sci(max(c['potential_rate']['max'] for c in struct.values()))
    if cov:
        N['CovAccMax'] = sci(cov['acceleration_difference']['max'])
    if consv:
        consv = {k: v for k, v in consv.items() if not k.startswith('_')}
        r = [c['ratio'] for c in consv.values()]
        N['ConsvRatio'] = f'{min(r):.0f}--{max(r):.0f}' if round(min(r)) != round(max(r)) else f'{min(r):.0f}'
    if upd:
        e = upd['inertial_edit']
        N['UpdPayloadKg'] = num(e['payload_kg'], 0)
        N['UpdLocalUs'] = num(1e6 * e['local_update_s']['median'], 0)
        N['UpdFullMs'] = num(1e3 * e['full_evaluation_s']['median'], 1)
        N['UpdWoodburyUs'] = num(1e6 * e['woodbury_s']['median'], 0)
        N['UpdRefactorUs'] = num(1e6 * e['refactor_s']['median'], 0)
        N['UpdGammaErr'] = sci(max(e['gamma']['max'], e['coriolis']['max'], e['gravity']['max']))
        w = upd['constraint_wrenches']
        N['WrenchErrN'] = sci(max(w['link_force_N']['max'], w['support_force_N']['max']))
    if comp:
        mr = comp['member_reuse']
        reuse = mr['gamma1_difference'] / mr['gamma1_scale']
        N['CompMemberReuse'] = sci(reuse) if reuse > 0. else '0'
        modes = comp['modes']
        N['CompModes'] = len(modes)
        N['CompModeBuildMs'] = num(1e3 * max(m['build_s'] for m in modes.values()), 1)
        N['CompModePrepMs'] = num(1e3 * max(m['prepare_s'] for m in modes.values()), 1)
        N['CompModeTotalMs'] = num(1e3 * max(m['build_s'] + m['prepare_s'] for m in modes.values()), 1)
        N['CompTwoVsOne'] = sci(max(m['two_level_vs_one_level'] for m in modes.values()))
    if tim:
        r = tim['circle']
        N['TimeRingIpMs'] = num(1e3 * r['ip_two_level_full_s'], 1)
        N['TimeRingIpPosMs'] = num(1e3 * r['ip_two_level_position_s'], 1)
        N['TimeRingProjAnalyticMs'] = num(1e3 * r['projected_tree_analytic_s'], 1)
        N['TimeRingProjFdMs'] = num(1e3 * r['projected_tree_fd_s'], 1)
        N['TimeRingPinMs'] = num(1e3 * r['pinocchio_constraint_dynamics_s'], 1)
        N['TimeRingMjMs'] = num(1e3 * r['mujoco_forward_s'], 2)
    return dict(systems=sysj, consistency=cons, timing=tim, composition=comp, updates=upd)


def dance_numbers(d, N):
    m = {(f, p): d.metrics(f, 'nominal', 'mujoco', p) for f in FORMATIONS for p, _ in POLICIES if p != 'recursive'}
    m[('circle', 'recursive')] = d.metrics('circle', 'nominal', 'mujoco', 'recursive')
    ideal = {p: d.metrics('circle', 'ideal', 'mujoco', p) for p in IDEAL}
    pin = {f: d.metrics(f, 'nominal', 'pinocchio', 'ip', required=f == 'circle') for f in FORMATIONS}
    noisy = d.run('semicircle', 'steps_noise2', 'mujoco', 'ip')
    if noisy:
        x = noisy['metrics']
        N['StepsNoiseTwoCompleted'] = 'yes' if x['completed'] else 'no'
        N['StepsNoiseTwoPelvisRmsMm'] = num(x['pelvis_rms_mm'], 2)
        N['StepsNoiseTwoInfeasiblePct'] = pct(x['infeasible_fraction'])
        N['StepsNoiseTwoSaturatedPct'] = pct(x['effort_saturated_fraction'])
    pay = {p: d.run('circle', 'payload', 'mujoco', p) for p in ('updated', 'stale')}
    steps = {(f, p): d.run(f, 'steps', 'mujoco', p) for f in FORMATIONS for p in ('ip', 'decentralized')}
    audits = {f: d.get(f'dance/{f}/audit__nominal.json') for f in FORMATIONS}
    saudits = {f: d.get(f'dance/{f}/audit__steps.json') for f in FORMATIONS}
    ref = d.run('circle', 'nominal', 'mujoco', 'ip')
    if ref:
        st = ref['settings']
        pi = st['plant_info']
        N['FloorFriction'] = num(pi['floor_friction'], 1)
        N['ContactTimeMs'] = num(1e3 * pi['floor_contact']['solref'][0], 0)
        nz = pi['noise']
        N['NoiseJointMrad'] = num(1e3 * nz['joint_rad'], 1)
        N['NoiseJointRateMradS'] = num(1e3 * nz['joint_rate_rad_s'], 0)
        N['NoiseRootMm'] = num(1e3 * nz['root_m'], 2)
        N['NoiseRootRateMmS'] = num(1e3 * nz['root_rate_m_s'], 0)
        N['NoiseScrewUm'] = num(1e6 * nz['joint_m'], 0)
        N['NoiseScrewRateMmS'] = num(1e3 * nz['joint_rate_m_s'], 0)
        N['NoiseRootMrad'] = num(1e3 * nz['root_rad'], 2)
        N['NoiseRootRateMradS'] = num(1e3 * nz['root_rate_rad_s'], 0)
        N['TorqueSpeedCorner'] = num(100 * pi['actuator']['corner'], 0)
        N['LagMs'] = num(1e3 * pi['actuator']['lag_s'], 0)
        N['GainLam'] = num(st['gains']['lam'], 0)
        N['GainKs'] = num(st['gains']['ks'], 0)
        N['FrictionFF'] = num(100 * st['friction_feedforward'], 0)
        N['DanceDurationS'] = num(ref['metrics']['simulated_s'], 0)
        N['ControlHz'] = num(1. / st['control_dt'], 0)
        N['PhysicsHz'] = num(1. / pi['physics_step_s'], 0)
    res = [r['settings']['reference']['residual'] for r in (d.run(f, 'nominal', 'mujoco', 'ip', False) for f in FORMATIONS) if r]
    res += [r['settings']['reference']['residual'] for r in steps.values() if r]
    if res:
        N['RefResidualMax'] = sci(max(res))
    for f in FORMATIONS:
        F = word(f)
        x = m[(f, 'ip')]
        if x:
            N[f'Dance{F}IpPelvisRmsMm'] = num(x['pelvis_rms_mm'], 2)
            N[f'Dance{F}IpPalmRmsMm'] = num(x['palm_rms_mm'], 2)
            N[f'Dance{F}IpCompleted'] = 'yes' if x['completed'] else 'no'
        if pin[f]:
            N[f'Pin{F}IpPelvisRmsMm'] = num(pin[f]['pelvis_rms_mm'], 3)
        if audits[f]:
            N[f'Audit{F}PerRobotInfeasiblePct'] = pct(audits[f]['per_dancer']['infeasible_fraction'])
            N[f'Audit{F}ComposedSlack'] = sci(audits[f]['composed']['max_slack'])
        if saudits[f]:
            N[f'AuditSteps{F}PerRobotInfeasiblePct'] = pct(saudits[f]['per_dancer']['infeasible_fraction'])
    ip, x = m[('circle', 'ip')], None
    if ip:
        N['DanceCircleIpSlack'] = sci(max(ip['max_slack'], 1e-12))
        N['DanceCircleIpEvalMs'] = num(ip['evaluate_ms'] + ip['allocate_ms'], 1)
        N['DanceCircleIpModelMs'] = num(ip['evaluate_ms'], 1)
        N['DanceCircleLinkErrN'] = num(ip['link_force_model_rms_error_N'], 1)
        N['DanceCircleLinkRmsN'] = num(ip['link_force_rms_N'], 1)
        N['DanceCircleSoleErrN'] = num(ip['support_force_model_rms_error_N'], 0)
        N['DanceCircleIpSaturatedPct'] = pct(ip['effort_saturated_fraction'], 1)
        N['DanceCircleIpQpPct'] = pct(ip['qp_fraction'])
        rec = m[('circle', 'recursive')]
        if rec:
            N['DanceCircleRecursivePelvisRmsMm'] = num(rec['pelvis_rms_mm'], 2)
            N['DanceCircleRecursiveEvalMs'] = num(rec['evaluate_ms'] + rec['allocate_ms'], 1)
            N['DanceCircleRecursiveModelMs'] = num(rec['evaluate_ms'], 1)
        for p in ('simplified', 'massless', 'no_curvature', 'gravity_pd', 'actuator_pd', 'decentralized'):
            y = m[('circle', p)]
            if not y:
                continue
            P = word(p)
            N[f'DanceCircle{P}PelvisRatio'] = ratio(y['pelvis_rms_mm'] / ip['pelvis_rms_mm'])
            N[f'DanceCircle{P}PalmRatio'] = ratio(y['palm_rms_mm'] / ip['palm_rms_mm'])
            N[f'DanceCircle{P}PalmPct'] = num(100 * (y['palm_rms_mm'] / ip['palm_rms_mm'] - 1), 0)
            N[f'DanceCircle{P}PelvisPct'] = num(100 * (y['pelvis_rms_mm'] / ip['pelvis_rms_mm'] - 1), 0)
        N['DanceCircleNoCurvPalmPct'] = N.values.get('DanceCircleNoCurvaturePalmPct', '--')
        N['DanceCircleGravityPalmRatio'] = N.values.get('DanceCircleGravityPdPalmRatio', '--')
        N['DanceCircleActuatorPelvisRatio'] = N.values.get('DanceCircleActuatorPdPelvisRatio', '--')
        N['DanceCircleIpMinFz'] = num(ip['min_fz_N'], 0)
        for f in ('semicircle', 'line'):
            y = m[(f, 'actuator_pd')]
            if y:
                N[f'Dance{word(f)}ActuatorPdFell'] = 'no' if y['completed'] else 'yes'
                N[f'Dance{word(f)}ActuatorPdEndS'] = num(y['simulated_s'], 1)
        dec = m[('circle', 'decentralized')]
        if dec:
            N['DanceCirclePerRobotLinkRmsN'] = num(dec['link_force_rms_N'], 1)
            N['DanceCirclePerRobotInfeasiblePct'] = pct(dec['infeasible_fraction'])
            N['DanceCirclePerRobotSlack'] = num(dec['max_slack'], 0)
            N['DanceCirclePerRobotMinFz'] = num(dec['min_fz_N'], 0)
            N['DanceCirclePerRobotPelvisRmsMm'] = num(dec['pelvis_rms_mm'], 2)
    ii = ideal['ip']
    if ii:
        N['IdealCircleIpPelvisRmsMm'] = num(ii['pelvis_rms_mm'], 3)
        N['IdealCircleIpPalmRmsMm'] = num(ii['palm_rms_mm'], 3)
        N['IdealCircleLinkErrN'] = num(ii['link_force_model_rms_error_N'], 2)
        N['IdealCircleSoleErrN'] = num(ii['support_force_model_rms_error_N'], 1)
        for p in IDEAL[1:]:
            y = ideal[p]
            if not y:
                continue
            P = word(p)
            N[f'IdealCircle{P}PelvisRatio'] = ratio(y['pelvis_rms_mm'] / ii['pelvis_rms_mm'])
            N[f'IdealCircle{P}PalmRatio'] = ratio(y['palm_rms_mm'] / ii['palm_rms_mm'])
            N[f'IdealCircle{P}PalmPct'] = num(100 * (y['palm_rms_mm'] / ii['palm_rms_mm'] - 1), 0)
        N['IdealCircleNoCurvPalmPct'] = N.values.get('IdealCircleNoCurvaturePalmPct', '--')
        if ideal['gravity_pd']:
            N['IdealCircleGravityPdMinFzN'] = num(ideal['gravity_pd']['min_fz_N'], 0)
        if ideal['decentralized']:
            N['IdealCircleDecentralizedInfeasiblePct'] = pct(ideal['decentralized']['infeasible_fraction'])
    if pay['updated'] and pay['stale']:
        N['PayloadKg'] = num(pay['updated']['settings']['payload']['mass_kg'], 0)
        N['PayloadTimeS'] = num(pay['updated']['settings']['payload']['time'], 0)
        for p in ('updated', 'stale'):
            y = pay[p]['metrics']
            N[f'Payload{word(p)}PalmRmsMm'] = num(y['palm_rms_mm'], 2)
            N[f'Payload{word(p)}PelvisRmsMm'] = num(y['pelvis_rms_mm'], 2)
    sc = steps[('circle', 'ip')]
    if sc:
        dn = sc['settings']['dance']
        N['StepCycles'] = dn['cycles']
        N['StepClearanceMm'] = num(1e3 * dn['clearance_m'], 0)
        N['StepLengthMm'] = num(1e3 * dn['step_m'], 0)
        N['StepCycleS'] = num(2 * (dn['t_shift_s'] + dn['t_swing_s']), 1)
        N['StepSwingS'] = num(dn['t_swing_s'], 2)
        N['StepDurationS'] = num(sc['settings']['reference']['duration_s'], 0)
    for f in FORMATIONS:
        F = word(f)
        s_ = steps[(f, 'ip')]
        if s_:
            x = s_['metrics']
            N[f'Steps{F}Completed'] = 'yes' if x['completed'] else 'no'
            N[f'Steps{F}PelvisRmsMm'] = num(x['pelvis_rms_mm'], 2)
            N[f'Steps{F}Switches'] = x['mode_switches']
            N[f'Steps{F}LandMm'] = num(x['landing_error_max_mm'], 1)
            N[f'Steps{F}SlipMm'] = num(x['stance_slip_max_mm'], 1)
            N[f'Steps{F}ClearMinMm'] = num(x['clearance_min_mm'], 0)
            N[f'Steps{F}SaturatedPct'] = pct(x['effort_saturated_fraction'], 1)
            N[f'Steps{F}TravelM'] = num(x['travel_mean_m'], 2)
            N[f'Steps{F}MaxFriction'] = num(x['max_friction_ratio'], 2)
            N[f'Steps{F}MaxCopMm'] = num(1e3 * x['max_cop_x_m'], 0)
            if 'cop_edge_fraction' in x:
                N[f'Steps{F}CopEdgePct'] = pct(x['cop_edge_fraction'], 1)
                N[f'Steps{F}FrictionLimitPct'] = pct(x['friction_limit_fraction'], 1)
                N[f'Steps{F}CopMaxFraction'] = num(100 * max(x['max_cop_x_fraction'], x['max_cop_y_fraction']), 0)
                N['StepsEdgeThresholdPct'] = num(100 * x['contact_edge_threshold'], 0)
            if x['turn_deg'] is not None:
                N[f'Steps{F}TurnDeg'] = num(x['turn_deg'], 1)
                N[f'Steps{F}TurnRefDeg'] = num(x['turn_reference_deg'], 1)
        dd = steps[(f, 'decentralized')]
        if dd:
            y = dd['metrics']
            N[f'Steps{F}PerRobotCompleted'] = 'yes' if y['completed'] else 'no'
            N[f'Steps{F}PerRobotFallS'] = num(y['simulated_s'], 1) if not y['completed'] else '--'
            N[f'Steps{F}PerRobotInfeasiblePct'] = pct(y['infeasible_fraction'])
    return dict(m=m, ideal=ideal, pin=pin, pay=pay, steps=steps, audits=audits)


# ---------------------------------------------------------------------- tables
def table_systems(sysj):
    c = sysj['cases']
    lines = ['\\footnotesize\\setlength{\\tabcolsep}{3.2pt}', '\\begin{tabular}{@{}lrrrrrl@{}}', '\\toprule',
             'System, mode & Bodies & Joints & Chords & Ports & $d$ & $d_1$ \\\\', '\\midrule']
    for key, label in SYSTEM_ROWS:
        if key not in c:                          # excavator rows without a private record
            continue
        x = c[key]
        d1 = x['level1_speeds']
        d1s = f"{len(d1)}$\\times${d1[0]}" if len(d1) > 1 else '--'
        lines.append(f"{label} & {x['record_bodies']} & {x['record_joints']} & {x['chords']} & {x['actuators']} & "
                     f"{x['independent_speeds']} & {d1s} \\\\")
    lines += ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(lines) + '\n'


def table_cost(tim, comp, upd):
    ms = lambda x: num(1e3 * x, 2) if 1e3 * x < 10 else num(1e3 * x, 1)
    lines = ['\\footnotesize\\setlength{\\tabcolsep}{3.0pt}', '\\begin{tabular}{@{}lrrrrr@{}}', '\\toprule',
             '& $d$ & IP & \\multicolumn{2}{c}{Projected} & Pinocchio \\\\', '\\cmidrule(lr){4-5}',
             'System & & & analytic & FD & KKT \\\\', '\\midrule']
    for key, label in COST_ROWS:
        if key not in tim:
            continue
        r = tim[key]
        lines.append(f"{label} & {r['case']['independent_speeds']} & {ms(r['ip_two_level_full_s'])} & "
                     f"{ms(r['projected_tree_analytic_s'])} & {ms(r['projected_tree_fd_s'])} & {ms(r['pinocchio_constraint_dynamics_s'])} \\\\")
    lines += ['\\midrule']
    modes = comp['modes']
    e = upd['inertial_edit']
    lines.append(f"\\multicolumn{{6}}{{@{{}}l}}{{Ring, new closure mode (build and prepare): {num(1e3 * max(m['build_s'] + m['prepare_s'] for m in modes.values()), 1)}}} \\\\")
    # the full evaluation of the ring is the IP entry of the table (same call, timing protocol of the table)
    lines.append(f"\\multicolumn{{6}}{{@{{}}l}}{{Ring, payload edit (rank 6): {num(1e3 * e['local_update_s']['median'], 3)}}} \\\\")
    lines += ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(lines) + '\n'


def table_dance(dn):
    m, ideal = dn['m'], dn['ideal']
    f2 = lambda x: num(x, 2)
    lines = ['\\footnotesize\\setlength{\\tabcolsep}{3.6pt}', '\\begin{tabular}{@{}l rrrrrr rrr rr@{}}', '\\toprule',
             '& \\multicolumn{6}{c}{Ring, realistic plant} & \\multicolumn{3}{c}{Ring, ideal plant} & \\multicolumn{2}{c}{Realistic, pelvis} \\\\',
             '\\cmidrule(lr){2-7}\\cmidrule(lr){8-10}\\cmidrule(l){11-12}',
             'Policy & pelvis & palm & link $F$ & sole $F$ & min $F_z$ & infeas. & pelvis & palm & link $F$ & semicircle & line \\\\',
             '& mm & mm & err.\\ N & err.\\ N & N & \\% & mm & mm & err.\\ N & mm & mm \\\\', '\\midrule']
    for p, label in POLICIES:
        x = m.get(('circle', p))
        if x is None:
            continue
        # the projected tree shares the allocation and wrench recovery of the proposed model: no force column of its own
        link = num(x['link_force_model_rms_error_N'], 1) if p not in ('recursive', 'decentralized', 'gravity_pd', 'actuator_pd') else '--'
        sole = num(x['support_force_model_rms_error_N'], 0) if p not in ('recursive', 'gravity_pd', 'actuator_pd') else '--'
        y = ideal.get(p)
        yl = '--'
        if y is not None and p not in ('decentralized', 'gravity_pd'):
            yl = num(y['link_force_model_rms_error_N'], 2)
        cells = [label, f2(x['pelvis_rms_mm']), f2(x['palm_rms_mm']), link, sole, num(x['min_fz_N'], 0),
                 pct(x['infeasible_fraction']) if p != 'actuator_pd' else '--',
                 num(y['pelvis_rms_mm'], 3) if y else '--', num(y['palm_rms_mm'], 3) if y else '--', yl]
        for f in ('semicircle', 'line'):
            z = m.get((f, p))
            cells.append('--' if not z else f2(z['pelvis_rms_mm']) if z['completed'] else f"fell, {z['simulated_s']:.0f}\\,s")
        if not x['completed']:
            cells[1] = f"fell, {x['simulated_s']:.0f}\\,s"
        lines.append(' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------------- evidence
def write_evidence(d, out, numbers):
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    files = {}
    for rel in sorted(d.used):
        src = d.root / rel
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        files[rel] = hashlib.sha256(dst.read_bytes()).hexdigest()
    hashes = {h for h in d.hashes.values() if h}
    manifest = dict(code_sha256=sorted(hashes)[0] if len(hashes) == 1 else sorted(hashes), files=files)
    (out / 'MANIFEST_SHA256.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    lines = ['# Evidence summary', '', f"Code hash: `{manifest['code_sha256']}`", '',
             'Every macro of `paper/generated/numbers.tex` with its value:', '', '| Macro | Value |', '|---|---|']
    for k in sorted(numbers.values):
        lines.append(f'| `{k}` | {numbers.values[k]} |')
    (out / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default=str(ROOT / 'results'))
    ap.add_argument('--out', default=str(ROOT / 'paper' / 'generated'))
    ap.add_argument('--evidence', default=str(ROOT / 'evidence'))
    ap.add_argument('--allow-mixed', action='store_true')
    ap.add_argument('--allow-missing', action='store_true')
    args = ap.parse_args(argv)
    d = Data(args.results)
    N = Numbers()
    sw = statewise_numbers(d, N)
    dn = dance_numbers(d, N)
    d.check(args.allow_mixed, args.allow_missing)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'numbers.tex').write_text(N.tex())
    if sw['systems']:
        (out / 'tab_systems.tex').write_text(table_systems(sw['systems']))
    if sw['timing'] and sw['composition'] and sw['updates']:
        (out / 'tab_cost.tex').write_text(table_cost(sw['timing'], sw['composition'], sw['updates']))
    (out / 'tab_dance.tex').write_text(table_dance(dn))
    write_evidence(d, args.evidence, N)
    print(f'make_tables: {len(N.values)} macros, {len(d.used)} result files -> {out}, {args.evidence}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
