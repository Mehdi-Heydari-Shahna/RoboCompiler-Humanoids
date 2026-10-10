"""State-wise mechanical-consistency studies of the inertia-partitioned compiled dynamics.

    consistency      constrained accelerations of random feasible states by several routes against
                     a hard-constraint reference solved with 50 digits (metrics of the RoboCompiler
                     paper, Eq. 23) and against a maximal-coordinate reference that does not pass
                     through the compiler
    structure        factorization identity Gammadot = C + C^T (finite differences, body by body), potential rate,
                     linearity in the inertial parameters
    quasi_velocity   holonomic Lagrangian bias formula against the body-wise Coriolis factor
    covariance       the same physical states through two partitions
    updates          local inertial update, Woodbury solve, constraint wrenches, connection event
    composition      reuse of member modules across formations and support modes
    conservation     energy drift of the compiled ODE without dissipation (integration test)
    reconfiguration  integration through a forced change of partition
    timing           evaluation cost of each route (run alone on an idle machine)
    systems          sizes of every case and the drive data of the humanoid

Results are written as JSON under ``--out`` (default ``results/statewise``); every file
carries the provenance stamp under the key ``_provenance``.
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import maximal as MX                                                   # noqa: E402
import methods as M                                                    # noqa: E402
from systems import CASES, available                                   # noqa: E402
from examples import excavator as excavator_record                     # noqa: E402
from ipd import partition as ip                                        # noqa: E402
from ipd import updates as up                                          # noqa: E402
from ipd.backends.pin_backend import PinBackend                        # noqa: E402
from ipd.provenance import stamp                                       # noqa: E402

ROUTES = ('ip_two_level', 'ip_one_level', 'projected_analytic', 'projected_fd', 'pinocchio_kkt', 'mujoco',
          'mujoco_rigid_limit', 'no_curvature', 'simplified', 'massless', 'decentralized')


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


def describe(case):
    """Sizes of a case.  ``record_bodies`` and ``record_joints`` count the bodies with mass and the physical joints of
    the records (tree joints and cut joints; a floating root has no joint), ``internal_loops`` the cycles of the joint graphs."""
    s, comp = case.sys, case.comp
    cmgs = [m.comp.cmg for m in s.members]
    extra = dict(record_bodies=int(sum(sum(1 for b in g['bodies'] if not b.get('massless')) for g in cmgs)),
                 record_joints=int(sum(len(g['joints']) + len(g['closures']) for g in cmgs)),
                 internal_loops=int(sum(len(g['closures']) for g in cmgs)))
    if getattr(case, 'record_sha256', None):         # a user-supplied record lies outside the code hash
        extra['record_sha256'] = case.record_sha256
    return dict(name=case.name, **extra, bodies=int(s.nbp), frames=int(s.nb), augmented_coordinates=int(s.n), tree_speeds=int(s.nT),
                independent_speeds=int(comp.d), actuators=int(s.na), chords=int(comp.mode.m),
                closure_rows=int(6 * comp.mode.m), mass_kg=float(s.mass.sum()), members=int(s.K),
                level1_speeds=[int(x) for x in comp.d1], level2_chords=int(comp.m2),
                largest_module=int(max(comp.mode.module_sizes)) if comp.mode.module_sizes else 0)


def summarize(values):
    v = np.asarray(values, float)
    return dict(max=float(v.max()), median=float(np.median(v)), min=float(v.min()), n=int(len(v)))


# ---------------------------------------------------------------------- consistency
def differential_fd(case, q, rng, h=1e-6):
    """Path-assembled closure differential against centred differences of the PACDM residual."""
    s, mode = case.sys, case.comp.mode
    kin = s.kinematics(q)
    rho0, C = mode.jacobian(kin)
    v = rng.normal(size=s.n)
    v[[i for sl in s.root_slice if sl is not None for i in range(sl.start, sl.stop)]] *= 0.3
    rp = mode.jacobian(s.kinematics(s.integrate(q, h * v)))[0]
    rm = mode.jacobian(s.kinematics(s.integrate(q, -h * v)))[0]
    return float(np.max(abs(C @ v - (rp - rm) / (2 * h))) / max(1., np.max(abs(C @ v))))


def consistency(names, states, seed, out, mujoco=True, sensitivity=6, dps=50, only=None, label='consistency',
                maximal_states=None):
    result = {}
    path = Path(out) / f'{label}.json'
    want = (lambda r: True) if only is None else (lambda r: r in only)
    for name in names:
        t0 = time.time()
        case = CASES[name]()
        s, comp = case.sys, case.comp
        rng = np.random.default_rng(seed)
        rng_fd = np.random.default_rng(seed + 1)         # directions of the difference check; the states do not depend on ``only``
        pb = PinBackend(s)
        solver = pb.solver(comp.mode)
        mj = M.MujocoStatewise(case) if mujoco else None
        is_ensemble = s.K > 1 and comp.nl > 0
        dec = None
        if is_ensemble:
            dec = case.ens.composite(stance=comp.stance, links=[])
            dec.prepare(case.q0)
        simp = None
        if case.ens is not None:
            from dancers.ensemble import simplified_inertia
            simp = simplified_inertia(s)
        rows = {r: dict(delta_a=[], delta_f=[], r_c=[]) for r in ROUTES}
        extra = dict(rank=[], radius=[], residual=[], cond_gamma=[], a_scale=[], h_scale=[], sens_a=[], sens_f=[],
                     two_vs_one=[], lift_vs_pacdm=[], mj_vs_nc=[], mj_vs_own=[], velocity_residual=[], maximal=[],
                     maximal_position=[], maximal_velocity=[], differential_fd=[])
        n_states = states[name] if isinstance(states, dict) else states
        n_max = n_states if maximal_states is None else maximal_states
        old_mujoco = mj is not None and tuple(int(x) for x in mj.version.split('.')[:2]) < (3, 7)
        for i in range(n_states):
            q, u, f = case.sample(rng)
            ev = comp.evaluate(q, u)
            vT = ev.v[:s.nT]
            ref = M.reference(case, q, vT, f, pb, dps)
            cache = M.one_level(case, q, u)
            acc = {}
            _, aux = M.ip_two_level(case, q, u, f, ev)
            aux1 = None
            if want('ip_two_level'):
                acc['ip_two_level'] = _
            if want('ip_one_level'):
                acc['ip_one_level'], aux1 = M.ip_one_level(case, q, u, f, cache)
            if want('projected_analytic'):
                acc['projected_analytic'], _ = M.projected_tree(case, q, u, f, pb, cache, 'analytic')
            if want('projected_fd'):
                acc['projected_fd'], _ = M.projected_tree(case, q, u, f, pb, cache, 'fd')
            if want('pinocchio_kkt'):
                acc['pinocchio_kkt'], _ = M.pinocchio_kkt(case, q, u, f, pb, solver, vT)
            if mj is not None and want('mujoco'):
                acc['mujoco'], _ = mj(q, vT, f)
            if mj is not None and want('mujoco_rigid_limit'):
                acc['mujoco_rigid_limit'], _ = mj.rigid_limit(q, vT, f)
            if want('no_curvature'):
                acc['no_curvature'], _ = M.no_curvature(case, q, u, f, cache)
            if simp is not None and want('simplified'):
                acc['simplified'], _ = M.simplified(case, q, u, f, *simp, gravity_full=True)
            if simp is not None and want('massless'):
                acc['massless'], _ = M.simplified(case, q, u, f, *simp, gravity_full=False)
            if dec is not None and want('decentralized'):
                acc['decentralized'], _ = M.decentralized(case, dec, q, ev.v, f)
            if 'mujoco_rigid_limit' in acc and 'no_curvature' in acc:
                extra['mj_vs_nc'].append(float(np.max(abs(acc['mujoco_rigid_limit'] - acc['no_curvature']))
                                               / max(1., np.max(abs(ref['a'])))))
            if 'mujoco_rigid_limit' in acc and old_mujoco and ref['rows']:
                a_old = M.kkt_without_velocity_product_mujoco(case, ref, vT, dps)
                extra['mj_vs_own'].append(float(np.max(abs(acc['mujoco_rigid_limit'] - a_old)) / max(1., np.max(abs(ref['a'])))))
            for r, a in acc.items():
                dsc = M.discrepancy(ref, a)
                for k in rows[r]:
                    rows[r][k].append(dsc.get(k, 0.))
            extra['rank'].append(ref['rank']); extra['radius'].append(ref['radius'])
            extra['velocity_residual'].append(ref['velocity_residual'])
            if only is None and i < n_max:
                mx = MX.compare(case, s.kinematics(q, ev.v), aux['E'] @ aux['ud'] + aux['Ed'] @ u, f)
                extra['maximal'].append(mx['delta_body']); extra['maximal_position'].append(mx['position_residual'])
                extra['maximal_velocity'].append(mx['velocity_residual'])
                extra['maximal_info'] = dict(unknowns=mx['unknowns'], rows=mx['rows'], rank=mx['rank'])
            if only is None:
                extra['differential_fd'].append(differential_fd(case, q, rng_fd))
            extra['residual'].append(comp.mode.gap(q)[0])
            extra['cond_gamma'].append(float(np.linalg.cond(ev.Gamma)))
            extra['a_scale'].append(float(np.max(abs(ref['a'])))); extra['h_scale'].append(float(np.max(abs(ref['h']))))
            if aux1 is not None:
                t = aux1['terms']
                scale = max(1., float(np.max(abs(t.C @ u + t.g))))
                extra['two_vs_one'].append(max(float(np.max(abs(ev.Gamma - t.Gamma)) / np.max(abs(t.Gamma))),
                                               float(np.max(abs(ev.C @ u + ev.g - t.C @ u - t.g)) / scale)))
            if only is None and i < sensitivity and ref['rows'] > ref['rank']:
                da, df = M.rf.row_sensitivity(ref['M'], ref['h'], ref['J'], ref['gamma'], ref['a'], trials=2, seed=seed + i)
                extra['sens_a'].append(da); extra['sens_f'].append(df)
            if i == 0 and only is None:
                N, minfo, graph = comp.mode.mapping_pacdm(q)
                extra['lift_vs_pacdm'].append(float(np.max(abs(aux['E'][graph.keep] - N))))
                extra['pacdm'] = dict(success=bool(minfo['success']), rank=int(minfo['rank_full']),
                                      expected_rank=int(minfo['expected_rank']), rcond=float(minfo['rcond']))
        entry = dict(case=describe(case), states=len(extra['rank']), seed=seed,
                     reference=dict(digits=dps, rank=int(max(extra['rank'])), rank_min=int(min(extra['rank'])),
                                    rank_expected=int(s.nT - comp.d), physical_rows=int(ref['rows']),
                                    enclosure_radius_max=float(max(extra['radius'])),
                                    velocity_residual_max=float(max(extra['velocity_residual'])),
                                    row_subset_sensitivity_states=len(extra['sens_a']),
                                    row_subset_sensitivity_a=float(max(extra['sens_a'])) if extra['sens_a'] else None,
                                    row_subset_sensitivity_f=float(max(extra['sens_f'])) if extra['sens_f'] else None),
                     maximal_coordinates=dict(extra.get('maximal_info', {}), states=len(extra['maximal']),
                                              delta_body=summarize(extra['maximal']),
                                              record_position_residual_max=float(max(extra['maximal_position'])),
                                              record_velocity_residual_max=float(max(extra['maximal_velocity'])))
                     if extra['maximal'] else None,
                     closure_differential_vs_fd=float(max(extra['differential_fd'])) if extra['differential_fd'] else None,
                     closure_residual_max=float(max(extra['residual'])), cond_gamma_max=float(max(extra['cond_gamma'])),
                     acceleration_scale=summarize(extra['a_scale']), force_scale=summarize(extra['h_scale']),
                     two_level_vs_one_level_max=float(max(extra['two_vs_one'])) if extra['two_vs_one'] else None,
                     lift_vs_vendored_pacdm=float(extra['lift_vs_pacdm'][0]) if extra['lift_vs_pacdm'] else None,
                     pacdm=extra.get('pacdm'), mujoco_version=mj.version if mj is not None else None,
                     mujoco_rigid_limit_vs_no_curvature=summarize(extra['mj_vs_nc']) if extra['mj_vs_nc'] else None,
                     mujoco_rigid_limit_vs_kkt_without_velocity_product=summarize(extra['mj_vs_own']) if extra['mj_vs_own'] else None,
                     routes={r: {k: summarize(v) for k, v in rows[r].items()} for r in ROUTES if rows[r]['delta_a']},
                     wall_s=time.time() - t0)
        result[name] = entry
        line = '  '.join('%s %.1e' % (r, entry['routes'][r]['delta_a']['max']) for r in entry['routes'])
        if entry['maximal_coordinates']:
            line += '  maximal(bodies) %.1e' % entry['maximal_coordinates']['delta_body']['max']
        print('[consistency] %-18s d=%3d rank %3d/%3d  %s  (%.0f s)' % (name, comp.d, entry['reference']['rank'],
                                                                      entry['reference']['physical_rows'], line, entry['wall_s']), flush=True)
        if entry['reference']['rank'] != entry['reference']['rank_expected'] or entry['reference']['rank_min'] != entry['reference']['rank']:
            raise RuntimeError(f'{name}: constraint rank {extra["rank"]} differs from nT - d = {s.nT - comp.d}')
        write_json(path, result)
    return result


# ---------------------------------------------------------------------- structure
def bodywise_gamma_rate_fd(case, q, v, h0=1e-5):
    """Centred-difference rate of every member ``Gamma_i = J_i^T M_i J_i`` of the partition along ``v``."""
    s, mode = case.sys, case.comp.mode
    h = h0 / max(1., float(np.max(abs(v))))
    G = []
    for sign in (1., -1.):
        kin = s.kinematics(s.integrate(q, sign * h * v))
        J, _ = ip.body_jacobians(s, kin, mode.lift(kin))
        G.append(np.einsum('bia,bij,bjc->bac', J, s.M, J))
    return (G[0] - G[1]) / (2 * h)


def structure(names, states, seed, out):
    """Structural properties that are not identities of the implementation.

    * ``skew_fd``: ``Gammadot - C - C^T`` with ``Gammadot`` by centred differences of the
      composed inertia along the motion;
    * ``skew_bodywise_fd``: the same test for every single body of the partition
      (``Gammadot_i`` by differences of ``J_i^T M_i J_i``), largest defect over the bodies;
    * ``potential_rate``: ``u^T g`` against the differenced gravitational potential;
    * ``regressor``: linearity of ``Gamma a + C u_r + g`` in the inertial parameters.
    """
    result = {}
    for name in names:
        t0 = time.time()
        case = CASES[name]()
        s, comp = case.sys, case.comp
        rng = np.random.default_rng(seed + 1)
        rec = dict(skew_fd=[], skew_bodywise_fd=[], potential_rate=[], regressor=[], gamma_sym=[], gamma_min_eig=[])
        for i in range(states):
            q, u, f = case.sample(rng)
            ev = comp.evaluate(q, u)
            kin, E, Ed = M.one_level(case, q, u)
            t = ip.assemble(s, kin, E, Ed, per_body=True)
            Gd = M.gamma_rate_fd(case, q, u)
            gs = float(np.max(abs(ev.Gamma)))
            rec['skew_fd'].append(float(np.max(abs(Gd - ev.C - ev.C.T)) / gs))
            Gdi = bodywise_gamma_rate_fd(case, q, E @ u)
            rec['skew_bodywise_fd'].append(float(np.max(abs(Gdi - t.C_i - np.swapaxes(t.C_i, 1, 2))) / gs))
            dU = M.potential_rate_fd(case, q, u)
            rec['potential_rate'].append(float(abs(dU - u @ ev.g) / max(1., abs(dU))))
            # linearity in the inertial parameters of the rigid bodies (rotor terms added separately)
            a = rng.normal(size=comp.d); ur = rng.normal(size=comp.d)
            s.velocity(kin, E @ u)
            Y = ip.regressor(s, kin, E, Ed, a, ur)
            phi = ip.inertia_parameters(s)
            ET = E[:s.nT]
            rotor = ET.T @ (s.armature * (ET @ a + Ed[:s.nT] @ ur))
            lhs = ev.Gamma @ a + ev.C @ ur + ev.g
            rec['regressor'].append(float(np.max(abs(np.einsum('bak,bk->a', Y, phi) + rotor - lhs)) / max(1., np.max(abs(lhs)))))
            rec['gamma_sym'].append(float(np.max(abs(ev.Gamma - ev.Gamma.T)) / gs))
            rec['gamma_min_eig'].append(float(np.linalg.eigvalsh((ev.Gamma + ev.Gamma.T) / 2).min()))
        entry = dict(case=describe(case), states=states, seed=seed + 1,
                     **{k: summarize(v) for k, v in rec.items()}, wall_s=time.time() - t0)
        result[name] = entry
        print('[structure] %-18s skew(fd) %.1e  skew(per body, fd) %.1e  dU %.1e  regressor %.1e (%.0f s)' % (
            name, entry['skew_fd']['max'], entry['skew_bodywise_fd']['max'], entry['potential_rate']['max'],
            entry['regressor']['max'], entry['wall_s']), flush=True)
        write_json(Path(out) / 'structure.json', result)
    return result


def quasi_velocity(out, seed=3, states=4):
    """Bias force of the holonomic Lagrangian formula against the body-wise Coriolis factor."""
    result = {}
    for name in ('manipulator', 'manipulator_joint', 'dancer_floating', 'dancer_double'):
        case = CASES[name]()
        comp = case.comp
        rng = np.random.default_rng(seed)
        err = []
        for _ in range(states):
            q, u, f = case.sample(rng)
            ev = comp.evaluate(q, u)
            h_l = M.lagrangian_bias_fd(case, q, u)
            err.append(float(np.max(abs(h_l - ev.C @ u)) / max(1e-12, np.max(abs(ev.C @ u)))))
        result[name] = dict(independent_speeds=int(comp.d), floating=bool(case.sys.floating[0]),
                            supports=int(comp.ns), relative_difference=summarize(err))
        print('[quasi-velocity] %-18s holonomic formula vs C u: max relative difference %.2e' % (name, max(err)), flush=True)
    write_json(Path(out) / 'quasi_velocity.json', result)
    return result


def covariance(out, seed=5, states=6):
    """The same physical state through different partitions of the hand links (same compiled tree)."""
    case = CASES['circle']()
    ens, s, comp = case.ens, case.sys, case.comp
    anchors = {sid: comp.mode.anchors[i] for i, sid in enumerate(comp.mode.support_ids)}
    alt = {l: next(t for t in ((0, 2, 3), (1, 2, 3), (0, 1, 2), (0, 1, 3)) if t != comp.triples[l]) for l in comp.links}
    other = ens.composite(comp.stance, comp.links, anchors, triples=alt)
    other.prepare(case.q0)
    rng = np.random.default_rng(seed)
    da, dp, dk, cond = [], [], [], []
    for _ in range(states):
        q, u, f = case.sample(rng)
        ev = comp.evaluate(q, u)
        a1, _ = M.ip_two_level(case, q, u, f, ev)
        u2 = ev.v[other.mode.independent]
        ev2 = other.evaluate(q, u2)
        ud2 = np.linalg.solve(ev2.Gamma, ev2.L.T @ f - ev2.C @ u2 - ev2.g - ev2.D @ u2)
        E2, Ed2 = other.lift(ev2, True)
        a2 = E2[:s.nT] @ ud2 + Ed2[:s.nT] @ u2
        da.append(float(np.max(abs(a1 - a2)) / max(1., np.max(abs(a1)))))
        dk.append(float(abs(u @ ev.Gamma @ u - u2 @ ev2.Gamma @ u2) / (u @ ev.Gamma @ u)))
        # u = F u2 with F the rows of the other lift at this mode's independent coordinates
        F = E2[comp.mode.independent]
        dp.append(float(np.max(abs(F.T @ ev.Gamma @ F - ev2.Gamma)) / np.max(abs(ev2.Gamma))))
        cond.append(float(np.linalg.cond(F)))
    result = dict(case=describe(case), partitions=dict(first={k: list(v) for k, v in comp.triples.items()},
                                                       second={k: list(v) for k, v in alt.items()}),
                  acceleration_difference=summarize(da), kinetic_energy_difference=summarize(dk),
                  congruence_defect=summarize(dp), change_of_speeds_condition=summarize(cond))
    print('[covariance] circle: acceleration %.1e  kinetic energy %.1e  Gamma congruence %.1e' % (
        max(da), max(dk), max(dp)), flush=True)
    write_json(Path(out) / 'covariance.json', result)
    return result


# ---------------------------------------------------------------------- local updates and events
def updates(out, seed=5, states=6):
    import pinocchio as pin
    import scipy.linalg as sla
    case = CASES['circle']()
    ens, s, comp = case.ens, case.sys, case.comp
    pb = PinBackend(s)
    rng = np.random.default_rng(seed)
    rec = dict(update_gamma=[], update_c=[], update_g=[], rank=[], woodbury=[], t_local=[], t_full=[], t_woodbury=[],
               t_refactor=[], wrench_force=[], wrench_moment=[], link_force=[], weight=[], closing_velocity=[],
               closing_energy=[], closing_loss_J=[], closing_body_gain=[])
    body = s.body_index['d2/r_forearm']
    dM = up.point_mass_inertia(3.0, [0.06, 0.0, -0.20], np.diag([2e-3, 2e-3, 1e-3]))
    links_open = [l for l in comp.links if l != 'hand_0_4']
    anchors = {sid: comp.mode.anchors[i] for i, sid in enumerate(comp.mode.support_ids)}
    opened = ens.composite(comp.stance, links_open, anchors, triples={l: comp.triples[l] for l in links_open})
    opened.prepare(case.q0)
    for i in range(states):
        q, u, f = case.sample(rng)
        ev = comp.evaluate(q, u)
        # ---- inertial edit of one body
        t0 = time.perf_counter()
        for _ in range(50):
            dG, dC, dg, J = up.inertial_update(comp, ev, body, dM)
        rec['t_local'].append((time.perf_counter() - t0) / 50)
        M0 = s.M
        try:
            s.M = M0.copy(); s.M[body] += dM
            t0 = time.perf_counter()
            ev2 = comp.evaluate(q, u)
            rec['t_full'].append(time.perf_counter() - t0)
        finally:
            s.M = M0
        rec['update_gamma'].append(float(np.max(abs(ev.Gamma + dG - ev2.Gamma)) / np.max(abs(ev2.Gamma))))
        rec['update_c'].append(float(np.max(abs(ev.C + dC - ev2.C)) / max(1e-12, np.max(abs(ev2.C)))))
        rec['update_g'].append(float(np.max(abs(ev.g + dg - ev2.g)) / np.max(abs(ev2.g))))
        rec['rank'].append(int(np.linalg.matrix_rank(dG, 1e-10)))
        cho = sla.cho_factor(ev.Gamma)
        rhs = rng.normal(size=comp.d)
        t0 = time.perf_counter()
        for _ in range(50):
            xw = up.woodbury_solve(lambda B: sla.cho_solve(cho, B), J, dM, rhs)
        rec['t_woodbury'].append((time.perf_counter() - t0) / 50)
        t0 = time.perf_counter()
        for _ in range(50):
            xd = sla.cho_solve(sla.cho_factor(ev2.Gamma), rhs)
        rec['t_refactor'].append((time.perf_counter() - t0) / 50)
        rec['woodbury'].append(float(np.max(abs(xw - xd)) / np.max(abs(xd))))
        # ---- constraint wrenches against Pinocchio's multipliers
        ud = np.linalg.solve(ev.Gamma, ev.L.T @ f - ev.C @ u - ev.g - ev.D @ u)
        lam = up.constraint_wrenches(comp, ev, f, ud)
        vT = ev.v[:s.nT]
        models, layout = pb.constraint_models(comp.mode)
        datas = [m.createData() for m in models]
        data = pb.model.createData()
        pb.model.armature = pb.P @ pb.armature
        pin.initConstraintDynamics(pb.model, data, models)
        pin.constraintDynamics(pb.model, data, pb.q_pin(q), pb.v_pin(vT), pb.P @ (s.B @ f - s.damping * vT), models, datas,
                               pin.ProximalSettings(1e-13, 1e-9, 300))
        pb.model.armature = np.zeros(pb.model.nv)
        wf = wm = wl = 0.
        for (e, nrow), cd in zip(layout, datas):
            cid = comp.mode.ids[e]
            if cid not in lam:
                continue
            lp = np.array(cd.contact_force.vector)
            if nrow == 6:                       # Pinocchio reports [force; moment] with the opposite sign convention
                wf = max(wf, float(np.max(abs(lam[cid][3:] + lp[:3]))))
                wm = max(wm, float(np.max(abs(lam[cid][:3] + lp[3:]))))
            else:
                wl = max(wl, float(abs(np.linalg.norm(lam[cid][3:]) - np.linalg.norm(lp[:3]))))
        rec['wrench_force'].append(wf); rec['wrench_moment'].append(wm); rec['link_force'].append(wl)
        # ---- closing the ring: inelastic transition from the open mode
        uA = rng.normal(size=opened.d) * .5
        evA = opened.evaluate(q, uA)
        w_minus = up.level1_speeds(comp, ev, evA.v)
        evB = comp.evaluate(q)
        u_plus, loss, loss_k = up.closing_transition(comp, evB, w_minus)
        v_plus = comp.lift(evB) @ u_plus
        MT = pb.mass(q)
        Jc, _, _ = pb.physical_rows(comp.mode, q, np.zeros(s.nT))
        Vt = np.linalg.svd(Jc, full_matrices=False)
        r = int(np.sum(Vt[1] > 1e-9 * Vt[1][0]))
        Jr = Vt[2][:r]
        vm = evA.v[:s.nT]
        MinvJt = np.linalg.solve(MT, Jr.T)
        vp = vm - MinvJt @ np.linalg.solve(Jr @ MinvJt, Jr @ vm)
        Tm, Tp = .5 * vm @ MT @ vm, .5 * vp @ MT @ vp
        rec['closing_velocity'].append(float(np.max(abs(v_plus[:s.nT] - vp)) / max(1., np.max(abs(vp)))))
        rec['closing_energy'].append(float(abs((Tm - Tp) - loss) / max(1e-12, Tm - Tp)))
        rec['closing_loss_J'].append(float(loss))
        # kinetic energy of single bodies before and after: individual bodies may gain energy
        kb = s.kinematics(q, evA.v); ka = s.kinematics(q, v_plus)
        Tb_before, _, _ = ip.energies(s, kb, per_body=True)
        Tb_after, _, _ = ip.energies(s, ka, per_body=True)
        rec['closing_body_gain'].append(int(np.sum(Tb_after > Tb_before + 1e-12)))
    result = dict(case=describe(case), states=states, seed=seed,
                  inertial_edit=dict(body='d2/r_forearm', payload_kg=3.0,
                                     gamma=summarize(rec['update_gamma']), coriolis=summarize(rec['update_c']),
                                     gravity=summarize(rec['update_g']), rank=int(max(rec['rank'])),
                                     local_update_s=summarize(rec['t_local']), full_evaluation_s=summarize(rec['t_full']),
                                     woodbury=summarize(rec['woodbury']), woodbury_s=summarize(rec['t_woodbury']),
                                     refactor_s=summarize(rec['t_refactor'])),
                  constraint_wrenches=dict(support_force_N=summarize(rec['wrench_force']),
                                           support_moment_Nm=summarize(rec['wrench_moment']),
                                           link_force_N=summarize(rec['link_force'])),
                  closing_transition=dict(opened_link='hand_0_4', velocity=summarize(rec['closing_velocity']),
                                          energy=summarize(rec['closing_energy']), loss_J=summarize(rec['closing_loss_J']),
                                          bodies_gaining_energy=summarize(rec['closing_body_gain'])))
    print('[updates] inertial edit: Gamma %.1e C %.1e g %.1e rank %d ; local %.0f us vs full %.2f ms ; Woodbury %.1e' % (
        max(rec['update_gamma']), max(rec['update_c']), max(rec['update_g']), max(rec['rank']),
        1e6 * np.median(rec['t_local']), 1e3 * np.median(rec['t_full']), max(rec['woodbury'])), flush=True)
    print('[updates] wrenches: support force %.1e N moment %.1e N m, link force %.1e N ; closing: velocity %.1e energy %.1e '
          '(bodies gaining energy: up to %d)' % (
              max(rec['wrench_force']), max(rec['wrench_moment']), max(rec['link_force']), max(rec['closing_velocity']),
              max(rec['closing_energy']), max(rec['closing_body_gain'])), flush=True)
    write_json(Path(out) / 'updates.json', result)
    return result


# ---------------------------------------------------------------------- systems
def systems(names, out):
    """Sizes of every case and the drive data of the humanoid (joint-level reflected inertia of each drive kind)."""
    from dancers.humanoid import DRIVES, reflected_inertia
    result = dict(cases={name: describe(CASES[name]()) for name in names},
                  humanoid_drives={k: dict(spec, reflected_inertia=float(reflected_inertia(spec))) for k, spec in DRIVES.items()})
    write_json(Path(out) / 'systems.json', result)
    for name, d in result['cases'].items():
        print('[systems] %-18s bodies %3d joints %3d loops %2d chords %2d ports %3d d %3d' % (
            name, d['record_bodies'], d['record_joints'], d['internal_loops'], d['chords'], d['actuators'], d['independent_speeds']))
    return result


# ---------------------------------------------------------------------- composition
def composition(out, seed=9):
    """Composition of the same member modules into several closure modes.

    The member modules do not depend on the mode by construction (they are evaluated
    before any support or link is considered); what is checked is that the composed model
    of every mode equals the one-level reduction of that mode over all its chords, which is
    assembled by different code (``ipd.mode.Mode.lift`` and ``ipd.partition.assemble``).
    """
    from dancers.ensemble import Ensemble, PartitionManager
    from ipd.system import Config
    rng = np.random.default_rng(seed)
    result = {}
    ens = Ensemble('circle', 'joint_space')
    s = ens.system
    man = PartitionManager(ens, s.neutral(), acquire=True)
    q, comp = man.q, man.comp
    dv = np.zeros(s.n)
    dv[comp.mode.independent] = rng.uniform(-1, 1, comp.d) * 0.03
    q, ci = comp.mode.correct(s.integrate(q, dv), maxiter=40)
    assert ci['success']
    ev = comp.evaluate(q)
    modes = {
        'ring, double support': dict(),
        'ring opened at one link': dict(links=[l for l in comp.links if l != 'hand_0_4']),
        'no hand links': dict(links=[]),
        'ring, left feet only': dict(stance=['left'] * ens.n),
        'ring, alternating feet': dict(stance=['left', 'right', 'left', 'right', 'both']),
        'unsupported, no links': dict(stance=['none'] * ens.n, links=[]),
    }
    table = {}
    for label, kw in modes.items():
        links = kw.get('links', comp.links)
        t0 = time.perf_counter()
        other = ens.composite(kw.get('stance'), links, triples={l: comp.triples[l] for l in links})
        t_build = time.perf_counter() - t0
        t0 = time.perf_counter()
        other.prepare(q)
        t_prepare = time.perf_counter() - t0
        t0 = time.perf_counter()
        ev2 = other.evaluate(q)
        t_eval = time.perf_counter() - t0
        u2 = rng.uniform(-1., 1., other.d) * 0.4
        ev2u = other.evaluate(q, u2)
        kin = s.kinematics(q)
        E = other.mode.lift(kin)
        s.velocity(kin, E @ u2)
        E, Ed = other.mode.lift(kin, derivative=True)
        t = ip.assemble(s, kin, E, Ed)
        scale = max(1., float(np.max(abs(t.C @ u2 + t.g))))
        d12 = max(float(np.max(abs(ev2u.Gamma - t.Gamma)) / np.max(abs(t.Gamma))),
                  float(np.max(abs(ev2u.C @ u2 + ev2u.g - t.C @ u2 - t.g)) / scale))
        table[label] = dict(independent_speeds=int(other.d), level2_chords=int(other.m2),
                            level2_rows=int(6 * other.m2), two_level_vs_one_level=d12,
                            build_s=t_build, prepare_s=t_prepare, evaluate_position_s=t_eval,
                            cond_gamma=float(np.linalg.cond(ev2.Gamma)))
        print('[composition] %-26s d=%3d  level-2 chords %2d  two-level vs one-level %.1e  prepare %.1f ms' % (
            label, other.d, other.m2, d12, 1e3 * t_prepare), flush=True)
    result['modes'] = table
    # the same member in another formation: its level-1 terms depend on its own state only
    line = Ensemble('line', 'joint_space')
    sl = line.system
    ql = sl.neutral()
    k = 2
    ql.theta[sl.tree_slice[k]] = q.theta[s.tree_slice[k]]
    ql.theta[sl.cut_slice[k]] = q.theta[s.cut_slice[k]]
    ql = sl.lift(ql)
    compl = line.composite(stance=['none'] * line.n, links=[])
    compl.prepare(ql)
    evl = compl.evaluate(ql)
    n1 = comp.d1[k]
    # gravity enters through the root orientation; compare with the same root orientation
    ql.T_root[k] = ql.T_root[k].copy()
    ql.T_root[k][:3, :3] = q.T_root[k][:3, :3]
    evl2 = compl.evaluate(ql)
    result['member_reuse'] = dict(member=k, gamma1_difference=float(np.max(abs(evl.Gamma1[k][:n1, :n1] - ev.Gamma1[k][:n1, :n1]))),
                                  gravity1_difference_same_orientation=float(np.max(abs(evl2.g1[k][:n1] - ev.g1[k][:n1]))),
                                  gamma1_scale=float(np.max(abs(ev.Gamma1[k]))))
    print('[composition] member %d in another formation: Gamma1 difference %.1e, g1 difference %.1e' % (
        k, result['member_reuse']['gamma1_difference'], result['member_reuse']['gravity1_difference_same_orientation']), flush=True)
    write_json(Path(out) / 'composition.json', result)
    return result


# ---------------------------------------------------------------------- integration tests
def _hamiltonian(case, q, u, f):
    """``T + U - f . l`` with the port coordinates ``l``: constant along the undamped ODE with constant efforts."""
    s, comp = case.sys, case.comp
    ev = comp.evaluate(q)
    return float(.5 * u @ ev.Gamma @ u + M.potential(s, q) - f @ q.theta[s.port_dof]), float(.5 * u @ ev.Gamma @ u)


def conservation(out, duration=0.256, h=2e-3, seed=11):
    """Drift of the energy integral of the compiled ODE (no damping, constant efforts) under RK4.

    The integral ``T + U - f . l`` is conserved by the exact flow.  Its drift over the run
    is reported for two step sizes; a ratio near 16 shows that it is the truncation error
    of the fourth-order scheme and not an inconsistency of ``Gamma``, ``C`` and ``g``.
    """
    from ipd.integrate import MinimalModel
    result = {}
    # The supported humanoid starts slowly and runs for 0.1 s: with constant efforts its posture is
    # an unstable equilibrium, and the legs cannot follow a pelvis that has moved far (straight knees).
    # Its integral is a small difference of large terms (weight times height, effort times stroke), so
    # larger steps are used there to keep the drift above round-off.
    for name, speed, span, h in (('manipulator', 1., duration, 4. * h), ('manipulator_joint', 1., duration, 4. * h),
                                 ('dancer_floating', 1., duration, h), ('dancer_double', 0.25, 0.096, 4. * h)):
        case = CASES[name]()
        s, comp = case.sys, case.comp
        model = MinimalModel(comp, correct_tol=1e-13)          # closure error below the drift that is measured
        damping = s.damping
        s.damping = np.zeros_like(damping)
        try:
            rng = np.random.default_rng(seed)
            q0, u0, _ = case.sample(rng)
            u0 = speed * u0
            ev = comp.evaluate(q0)
            f = np.linalg.lstsq(ev.L.T, ev.g, rcond=None)[0]             # holds the posture where it can
            f = np.clip(f, s.effort_lower, s.effort_upper)
            drifts, scales = [], []
            for hh in (h, h / 2.):
                q, u = q0.copy(), u0.copy()
                H0, T0 = _hamiltonian(case, q, u, f)
                terms = abs(M.potential(s, q)) + abs(f @ q.theta[s.port_dof])
                drift, scale = 0., T0
                for _ in range(int(round(span / hh))):
                    q, u = model.step(q, u, f, hh)
                    H, T = _hamiltonian(case, q, u, f)
                    drift, scale = max(drift, abs(H - H0)), max(scale, T)
                drifts.append(drift); scales.append(scale)
        finally:
            s.damping = damping
        result[name] = dict(independent_speeds=int(comp.d), duration_s=span, initial_speed_scale=speed, steps=[h, h / 2.],
                            drift_J=drifts, kinetic_energy_scale_J=scales[0], round_off_floor_J=float(1e-13 * terms),
                            relative_drift=[d / max(sc, 1e-12) for d, sc in zip(drifts, scales)],
                            ratio=float(drifts[0] / max(drifts[1], 1e-300)))
        print('[conservation] %-18s drift %.2e J (h) %.2e J (h/2), ratio %.1f, kinetic energy up to %.1f J' % (
            name, drifts[0], drifts[1], result[name]['ratio'], scales[0]), flush=True)
    write_json(Path(out) / 'conservation.json', result)
    return result


def reconfiguration(out, duration=0.096, h=4e-3, seed=7, speed=0.25):
    """Integration of the ring through a forced change of the hand-link partition.

    The same initial state is integrated (a) in the partition of the mode, (b) in that
    partition for the first half and in an alternative partition for the second half,
    and (c) in the alternative partition throughout.  The physical end states must agree
    up to the truncation error of the integrator, which is exposed by halving the step.
    The efforts hold the initial posture; the run is short and starts slowly because that
    posture is an unstable equilibrium under constant efforts.
    """
    from ipd.integrate import MinimalModel
    case = CASES['circle']()
    ens, s, comp = case.ens, case.sys, case.comp
    anchors = {sid: comp.mode.anchors[i] for i, sid in enumerate(comp.mode.support_ids)}
    alt = {l: next(t for t in ((0, 2, 3), (1, 2, 3), (0, 1, 2), (0, 1, 3)) if t != comp.triples[l]) for l in comp.links}
    other = ens.composite(comp.stance, comp.links, anchors, triples=alt)
    other.prepare(case.q0)
    models = {'first': MinimalModel(comp), 'second': MinimalModel(other)}
    comps = {'first': comp, 'second': other}
    rng = np.random.default_rng(seed)
    q0, u0, _ = case.sample(rng)
    u0 = speed * u0
    ev = comp.evaluate(q0, u0)
    v0 = comp.lift(ev) @ u0
    f = np.clip(np.linalg.lstsq(ev.L.T, ev.g, rcond=None)[0], s.effort_lower, s.effort_upper)

    def integrate(plan, hh):
        """``plan`` lists the partition used in each half of the run; returns the tree state at the end."""
        q, v = q0.copy(), v0.copy()
        n = int(round(duration / hh))
        for part, steps in zip(plan, (n // 2, n - n // 2)):
            c, model = comps[part], models[part]
            u = v[c.mode.independent]
            for _ in range(steps):
                q, u = model.step(q, u, f, hh)
            v = c.lift(c.evaluate(q)) @ u
        return q, v

    def distance(a, b):
        dq = s.difference(a[0], b[0])[:s.nT]
        return float(np.max(abs(dq))), float(np.max(abs(a[1][:s.nT] - b[1][:s.nT])))

    table = {}
    for hh in (h, h / 2.):
        ra, rb, rc = integrate(('first', 'first'), hh), integrate(('first', 'second'), hh), integrate(('second', 'second'), hh)
        table[f'{hh:g}'] = dict(switched_vs_first=distance(rb, ra), second_vs_first=distance(rc, ra),
                                motion=distance(ra, (q0, v0)))
    keys = list(table)
    result = dict(case=describe(case), duration_s=duration, steps=[h, h / 2.], initial_speed_scale=speed,
                  partitions=dict(first={k: list(v) for k, v in comp.triples.items()}, second={k: list(v) for k, v in alt.items()}),
                  end_state_differences=table,
                  ratio_switched=float(table[keys[0]]['switched_vs_first'][0] / max(table[keys[1]]['switched_vs_first'][0], 1e-300)),
                  ratio_second=float(table[keys[0]]['second_vs_first'][0] / max(table[keys[1]]['second_vs_first'][0], 1e-300)))
    print('[reconfiguration] circle, %.2f s: configuration difference switched/first %.2e (h) %.2e (h/2); '
          'second/first %.2e %.2e; motion %.2e' % (duration, table[keys[0]]['switched_vs_first'][0], table[keys[1]]['switched_vs_first'][0],
                                                   table[keys[0]]['second_vs_first'][0], table[keys[1]]['second_vs_first'][0],
                                                   table[keys[0]]['motion'][0]), flush=True)
    write_json(Path(out) / 'reconfiguration.json', result)
    return result


# ---------------------------------------------------------------------- timing
def timing(names, out, repeats=30, seed=2):
    import mujoco
    import pinocchio as pin
    result = {}
    for name in names:
        case = CASES[name]()
        s, comp = case.sys, case.comp
        rng = np.random.default_rng(seed)
        q, u, f = case.sample(rng)
        pb = PinBackend(s)
        solver = pb.solver(comp.mode)
        mj = M.MujocoStatewise(case)
        ev = comp.evaluate(q, u)
        vT = ev.v[:s.nT]
        qp, vp = pb.q_pin(q), pb.v_pin(vT)
        tau = pb.P @ (s.B @ f - s.damping * vT)

        def clock(fn, n=repeats):
            fn()
            ts = []
            for _ in range(n):
                t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
            return float(np.median(ts))

        def pin_kkt():
            pb.model.armature = pb.P @ pb.armature
            pin.constraintDynamics(pb.model, solver['data'], qp, vp, tau, solver['models'], solver['datas'], solver['settings'])
            pb.model.armature = np.zeros(pb.model.nv)

        def pin_tree():
            pin.crba(pb.model, pb.data, qp); pin.nonLinearEffects(pb.model, pb.data, qp, vp)

        mj.plant.set_effort(f)
        mj.plant.set_state(q, vT)                       # time mj_forward at the sampled state

        def mj_forward():
            mujoco.mj_forward(mj.plant.model, mj.plant.data)

        def solve_reduced():
            np.linalg.solve(ev.Gamma, ev.L.T @ f - ev.C @ u - ev.g - ev.D @ u)

        cache = M.one_level(case, q, u)
        entry = dict(case=describe(case),
                     ip_two_level_position_s=clock(lambda: comp.evaluate(q)),
                     ip_two_level_full_s=clock(lambda: comp.evaluate(q, u)),
                     ip_one_level_full_s=clock(lambda: M.ip_one_level(case, q, u, f), max(3, repeats // 6)),
                     one_level_lift_s=clock(lambda: M.one_level(case, q, u), max(3, repeats // 6)),
                     projected_tree_analytic_s=clock(lambda: M.projected_tree(case, q, u, f, pb, None, 'analytic'), max(3, repeats // 6)),
                     projected_tree_fd_s=clock(lambda: M.projected_tree(case, q, u, f, pb, None, 'fd'), max(3, repeats // 6)),
                     curvature_fd_s=clock(lambda: M.curvature_fd(comp.mode, q, cache[1] @ u), max(3, repeats // 6)),
                     reduced_solve_s=clock(solve_reduced),
                     pinocchio_constraint_dynamics_s=clock(pin_kkt),
                     pinocchio_crba_rnea_s=clock(pin_tree),
                     mujoco_forward_s=clock(mj_forward),
                     closure_correction_s=clock(lambda: comp.mode.correct(q), max(3, repeats // 6)))
        result[name] = entry
        print('[timing] %-18s IP two-level %.2f ms (position only %.2f) one-level %.2f  projected tree %.2f (FD curvature %.2f)  '
              'pin KKT %.3f  pin CRBA+RNEA %.3f  MuJoCo forward %.3f ms' % (
                  name, 1e3 * entry['ip_two_level_full_s'], 1e3 * entry['ip_two_level_position_s'], 1e3 * entry['ip_one_level_full_s'],
                  1e3 * entry['projected_tree_analytic_s'], 1e3 * entry['projected_tree_fd_s'], 1e3 * entry['pinocchio_constraint_dynamics_s'],
                  1e3 * entry['pinocchio_crba_rnea_s'], 1e3 * entry['mujoco_forward_s']), flush=True)
        write_json(Path(out) / 'timing.json', result)
    return result


SMALL = ('manipulator', 'manipulator_joint', 'excavator', 'excavator_cylinder', 'dancer_floating', 'dancer_single', 'dancer_double')
LARGE = ('circle', 'semicircle', 'line', 'circle_single', 'circle_actuator')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('study', choices=('systems', 'consistency', 'structure', 'quasi_velocity', 'covariance', 'updates', 'composition',
                                      'conservation', 'reconfiguration', 'timing', 'all'))
    ap.add_argument('--cases', nargs='+', default=None)
    ap.add_argument('--states', type=int, default=None)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--out', default=str(ROOT / 'results' / 'statewise'))
    ap.add_argument('--no-mujoco', action='store_true')
    ap.add_argument('--routes', nargs='+', default=None, help='restrict the consistency study to these routes')
    ap.add_argument('--label', default='consistency', help='file name of the consistency result')
    ap.add_argument('--excavator-record', default=None,
                    help=f'private excavator record (default: ${excavator_record.ENV}); without one the excavator cases are skipped')
    args = ap.parse_args(argv)
    if args.excavator_record:
        os.environ[excavator_record.ENV] = str(Path(args.excavator_record).expanduser().resolve())
    cases = [c for c in (args.cases or SMALL + LARGE) if available(c)]
    skipped = [c for c in (args.cases or SMALL + LARGE) if not available(c)]
    if skipped:
        print(f'[statewise] skipped {", ".join(skipped)}: no excavator record (${excavator_record.ENV} or '
              f'--excavator-record) or no such partition request in it', flush=True)
    if args.study in ('systems', 'all'):
        systems(cases, args.out)
    if args.study in ('consistency', 'all'):
        states = {c: (args.states or (48 if c in SMALL else 24)) for c in cases}
        consistency(cases, states, args.seed, args.out, mujoco=not args.no_mujoco, only=args.routes, label=args.label)
    if args.study in ('structure', 'all'):
        structure(cases, args.states or 6, args.seed, args.out)
    if args.study in ('quasi_velocity', 'all'):
        quasi_velocity(args.out)
    if args.study in ('covariance', 'all'):
        covariance(args.out)
    if args.study in ('updates', 'all'):
        updates(args.out)
    if args.study in ('composition', 'all'):
        composition(args.out)
    if args.study in ('conservation', 'all'):
        conservation(args.out)
    if args.study in ('reconfiguration', 'all'):
        reconfiguration(args.out)
    if args.study == 'timing':
        timing(cases, args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
