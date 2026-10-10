"""Three plants for one compiled ensemble.

``MujocoWorld``   the realistic plant: native MuJoCo with regularised equality constraints for
                  the loops and hand links and unilateral frictional contact under the soles
                  (rubber on wood, ``dancers.stage``), plus what the compiled model leaves out:
                  Coulomb friction in every joint with drive data, actuator dynamics (a
                  first-order lag of the applied effort and the torque-speed limit of every
                  motor) and noise on the measured state.
``PinocchioKKT``  Pinocchio constrained dynamics (hard KKT with a proximal solver) on
                  rigid constraint models for loops, links and ideal sole welds; an ideal
                  rigid plant without friction, actuator dynamics or noise.
``MinimalODE``    the compiled inertia-partitioned model itself, integrated as an
                  ordinary differential equation in the independent speeds, with the
                  PACDM closure correction keeping every chord closed.

All plants expose the same interface: ``state()`` (the true state), ``measured(q, v)`` (what
the controller sees), ``advance(f, dt)``, ``measure()``, ``health()`` and
``apply_wrench(body, force, torque)``.  ``measure()`` reports the largest translational gap
of the internal loops (``gap``), of every hand link (``link_gap``) and of the ideal supports
(``support_gap``; not defined for contact).  An external wrench is a force at the centre of
mass of the body plus a torque, in world axes, on every plant.
"""
from __future__ import annotations

import numpy as np

from ipd import se3
from ipd.control import Allocator


def chord_gaps(comp, q):
    """Translational gaps (m) of the chords of a mode at ``q``: internal loops, hand links, supports."""
    mode, s = comp.mode, comp.sys
    _, D = mode.residual(q)
    gap = np.linalg.norm(D[:, :3, 3], axis=1)
    ni, nl = s.n_internal, len(mode.link_ids)
    return dict(loops=float(gap[:ni].max()) if ni else 0.,
                links={l: float(gap[ni + i]) for i, l in enumerate(mode.link_ids)},
                supports=float(gap[ni + nl:].max()) if len(gap) > ni + nl else 0.)


# realistic plant: actuator bandwidth and measurement noise (standard deviations per sample)
ACTUATOR = dict(lag_s=0.002, corner=0.5)         # first-order lag; the peak effort is available up to corner * no-load speed
NOISE = dict(joint_rad=1e-4, joint_m=5e-6, joint_rate_rad_s=0.01, joint_rate_m_s=0.001,
             root_rad=2.5e-4, root_m=2.5e-4, root_rate_rad_s=0.005, root_rate_m_s=0.005)


class MujocoWorld:
    name = 'mujoco'

    def __init__(self, ens, comp, q0, timestep=1e-3, friction=None, loop_solref=0.004, actuator=True, noise=True, seed=0,
                 coulomb=True, scene=None, **kw):
        from ipd.backends.mujoco_backend import MujocoPlant
        from .stage import Studio
        self.ens, self.comp = ens, comp
        s = self.sys = ens.system
        self.q0 = q0
        scene = scene or Studio(render=False)
        self.plant = MujocoPlant(s, q=q0, timestep=timestep, friction=friction, loop_solref=loop_solref, scene=scene, **kw)
        if not coulomb:
            self.plant.model.dof_frictionloss[:] = 0.
        for l in ens.links:
            self.plant.set_link(l['id'], l['id'] in comp.links)
        self.plant.set_state(q0, np.zeros(s.nT))
        self.timestep = timestep
        self.soles = {f'd{k}/{side}_sole': s.site_frame(f'd{k}', f'{side}_sole') for k in range(ens.n) for side in ('left', 'right')}
        self._time = float(self.plant.data.time)
        # motor data of every port, in the order of the system's ports
        motors = {}
        for k, rec in enumerate(ens.records):
            for a in rec['actuators']:
                motors[f'd{k}/{a["id"]}'] = a.get('motor', {})
        self.peak = np.array([motors.get(p, {}).get('peak', s.effort_upper[i]) for i, p in enumerate(s.port_ids)])
        self.no_load = np.array([motors.get(p, {}).get('no_load', np.inf) for i, p in enumerate(s.port_ids)])
        self.actuator = dict(ACTUATOR) if actuator is True else (dict(actuator) if actuator else None)
        self.noise = dict(NOISE) if noise is True else (dict(noise) if noise else None)
        self.rng = np.random.default_rng(seed)
        self.f_applied = np.zeros(s.na)
        self.port_v = np.array([self.plant.vadr[j] for j in s.port_dof], int)
        fl = scene.floor
        self._info = dict(engine='MuJoCo', physics_step_s=float(timestep), loop_solref_s=float(loop_solref),
                          floor_friction=float(fl['friction'] if friction is None else friction),
                          floor_contact=dict(torsional=fl['torsional'], rolling=fl['rolling'], solref=list(fl['solref']),
                                             solimp=list(fl['solimp']), condim=fl['condim']),
                          constraints='soft equalities, unilateral frictional contact',
                          joint_coulomb_friction=bool(coulomb and np.any(s.frictionloss > 0)), actuator=self.actuator, noise=self.noise,
                          noise_seed=int(seed))

    def info(self):
        """Settings of the plant for the result files."""
        return dict(self._info)

    def state(self):
        q, v = self.plant.get_state(self.q0)
        return self.sys.lift(q), v

    def measured(self, q, v):
        """The state the controller sees: the true state with sensor and estimator noise."""
        if not self.noise:
            return q, v
        s, n, rng = self.sys, self.noise, self.rng
        qm, vm = q.copy(), v.copy()
        for k, sl in enumerate(s.root_slice):
            if sl is None:
                continue
            xi = np.r_[rng.normal(0., n['root_rad'], 3), rng.normal(0., n['root_m'], 3)]
            qm.T_root[k] = q.T_root[k] @ se3.exp(xi)
            vm[sl] = v[sl] + np.r_[rng.normal(0., n['root_rate_rad_s'], 3), rng.normal(0., n['root_rate_m_s'], 3)]
        idx = self.plant.scalar
        lin = self._prismatic[idx]
        qm.theta[idx] = q.theta[idx] + rng.normal(0., 1., len(idx)) * np.where(lin, n['joint_m'], n['joint_rad'])
        vm[idx] = v[idx] + rng.normal(0., 1., len(idx)) * np.where(lin, n['joint_rate_m_s'], n['joint_rate_rad_s'])
        return s.lift(qm), vm

    @property
    def _prismatic(self):
        from ipd.system import PRISMATIC
        if not hasattr(self, '_prism'):
            s = self.sys
            self._prism = np.zeros(s.n, bool)
            for b in range(s.nbp):
                if s.jtype[b] == PRISMATIC:
                    self._prism[s.dof[b]] = True
        return self._prism

    def advance(self, f, dt):
        n = int(round(dt / self.timestep))
        if n < 1 or abs(n * self.timestep - dt) > 1e-12:
            raise ValueError(f'control period {dt} s is not a multiple of the physics step {self.timestep} s')
        if not self.actuator:
            self.plant.set_effort(f)
            self.plant.step(n)
            return
        a = 1. - np.exp(-self.timestep / self.actuator['lag_s']) if self.actuator['lag_s'] > 0 else 1.
        c = self.actuator['corner']
        for _ in range(n):
            self.f_applied += a * (f - self.f_applied)
            speed = np.abs(self.plant.data.qvel[self.port_v])
            finite = np.isfinite(self.no_load)
            ratio = np.ones_like(speed)
            ratio[finite] = np.clip((self.no_load[finite] - speed[finite]) / ((1. - c) * self.no_load[finite]), 0., 1.)
            avail = self.peak * ratio
            self.plant.set_effort(np.clip(self.f_applied, -avail, avail))
            self.plant.step(1)

    def health(self):
        """Empty string when the simulation is sound; otherwise what MuJoCo reported.

        MuJoCo resets the state after an unstable step (bad ``qacc``) and only counts a
        warning, so the warning counters and the monotony of time are checked.
        """
        import mujoco
        d = self.plant.data
        names = [mujoco.mjtWarning(i).name for i in range(len(d.warning)) if d.warning[i].number]
        if names:
            return 'MuJoCo warning: ' + ', '.join(names)
        if d.time < self._time - 1e-12 or not np.all(np.isfinite(d.qpos)):
            return 'MuJoCo state reset or not finite'
        self._time = float(d.time)
        return ''

    def measure(self):
        p = self.plant
        fw = p.foot_wrenches(self.soles)
        fz, cop, fr = [], [], []
        for key, (b, P) in self.soles.items():
            F, Nm, cnt = fw[key]
            Rw = p.data.xmat[p.body_id[b]].reshape(3, 3) @ P[:3, :3]
            Fl, Nl = Rw.T @ F, Rw.T @ Nm
            fz.append(Fl[2])
            cop.append([-Nl[1] / Fl[2], Nl[0] / Fl[2]] if Fl[2] > 5. else [0., 0.])
            # friction use in the axes of the (horizontal) floor: it cannot exceed the friction coefficient
            fr.append(np.hypot(F[0], F[1]) / F[2] if F[2] > 5. else 0.)
        gaps = p.equality_gaps()
        forces = p.equality_forces()
        links = [l['id'] for l in self.ens.links]
        loop = [v for k, v in gaps.items() if k.split('__')[0] not in links]
        return dict(fz=fz, cop=cop, friction_ratio=fr, gap=max(loop) if loop else 0.,
                    link_force=[float(np.linalg.norm(forces[l])) if l in forces else 0. for l in links],
                    link_gap=[gaps.get(l, np.nan) for l in links], support_gap=np.nan)

    def apply_wrench(self, body, force, torque=(0., 0., 0.)):
        """External force at the centre of mass of a physical body and torque, in world axes
        (MuJoCo applies ``xfrc_applied`` at the body's centre of mass)."""
        self.plant.data.xfrc_applied[self.plant.body_id[body]] = np.r_[force, torque]

    def set_body_inertia(self, body, mass, com, inertia_com):
        """Change the inertial parameters of a body of the plant at run time."""
        import mujoco
        m = self.plant.model
        bid = self.plant.body_id[body]
        w, R = np.linalg.eigh(np.asarray(inertia_com, float))
        if np.linalg.det(R) < 0:
            R[:, 0] = -R[:, 0]
        m.body_mass[bid] = mass
        m.body_ipos[bid] = com
        m.body_inertia[bid] = w
        quat = se3.quat_xyzw_from_matrix(R)
        m.body_iquat[bid] = np.r_[quat[3], quat[:3]]
        d = self.plant.data
        keep = {k: getattr(d, k).copy() for k in ('qpos', 'qvel', 'qacc_warmstart', 'ctrl', 'xfrc_applied', 'eq_active')}
        time = d.time
        mujoco.mj_setConst(m, d)                    # recomputes model constants; uses the data as scratch
        for k, v in keep.items():
            getattr(d, k)[:] = v
        d.time = time
        mujoco.mj_forward(m, d)


class MinimalODE:
    """Explicit midpoint integration of ``Gamma udot + C u + g + D u = L^T f + tau_ext``."""
    name = 'minimal_ode'

    def __init__(self, ens, comp, q0, substep=0.002, **kw):
        self.ens, self.comp = ens, comp
        self.sys = ens.system
        self.q = q0.copy()
        self.u = np.zeros(comp.d)
        self.h = substep
        self.alloc = Allocator(comp, {})
        self.ext = {}                      # body -> world wrench [n; f] with the force at the centre of mass
        self.last = None
        self.worst_residual = 0.

    def info(self):
        """Settings of the plant for the result files."""
        return dict(engine='compiled ODE', integrator='explicit midpoint with closure correction', step_s=float(self.h),
                    constraints='minimal coordinates of the closure mode')

    def state(self):
        v = np.zeros(self.sys.n)
        ev = self.comp.evaluate(self.q)
        return self.q.copy(), self.comp.lift(ev) @ self.u

    def _tau_ext(self, ev):
        tau = np.zeros(self.comp.d)
        for body, w in self.ext.items():
            P = np.eye(4)
            P[:3, 3] = self.sys.com[body]
            J = self.comp.frame_jacobian(ev, body, P)
            tau += J.T @ w
        return tau

    def _rate(self, q, u, f):
        ev = self.comp.evaluate(q, u)
        ud = np.linalg.solve(ev.Gamma, ev.L.T @ f + self._tau_ext(ev) - ev.C @ u - ev.g - ev.D @ u)
        return ev, ud

    def _move(self, q, ev, u, h):
        """Advance the configuration along ``v = E u`` for ``h`` and close the chords again."""
        E = self.comp.lift(ev)
        q1 = self.sys.integrate(q, E @ u * h)
        q1, ci = self.comp.mode.correct(q1, tol=1e-11, maxiter=20)
        if not ci['success']:
            raise RuntimeError(f'closure correction failed: {ci}')
        self.worst_residual = max(self.worst_residual, ci['residual_inf'])
        return q1

    def advance(self, f, dt):
        n = max(1, int(round(dt / self.h)))
        h = dt / n
        for _ in range(n):
            ev, ud1 = self._rate(self.q, self.u, f)
            um = self.u + .5 * h * ud1
            qm = self._move(self.q, ev, self.u, .5 * h)
            evm, ud2 = self._rate(qm, um, f)
            self.q = self._move(self.q, evm, um, h)
            self.u = self.u + h * ud2
            self.last = (evm, ud2, f)

    def measure(self):
        ens, comp = self.ens, self.comp
        soles = [f'd{k}/{side}_sole' for k in range(ens.n) for side in ('left', 'right')]
        fz, cop, fr = [np.nan] * len(soles), [[np.nan, np.nan]] * len(soles), [np.nan] * len(soles)
        if self.last is not None:
            ev, ud, f = self.last
            wdot = ev.G @ ud + ev.Gd @ ev.u
            maps = self.alloc.support_maps(ev, wdot)
            for e, (lam0, Lam) in maps.items():
                lam = lam0 - Lam @ f
                i = soles.index(comp.c_ids[e])
                fz[i] = lam[5]
                cop[i] = [-lam[1] / lam[5], lam[0] / lam[5]] if lam[5] > 5. else [0., 0.]
                fr[i] = np.hypot(lam[3], lam[4]) / lam[5] if lam[5] > 5. else 0.
        gaps = chord_gaps(comp, self.q)
        return dict(fz=fz, cop=cop, friction_ratio=fr, gap=gaps['loops'], link_force=[np.nan] * len(ens.links),
                    link_gap=[gaps['links'].get(l['id'], np.nan) for l in ens.links], support_gap=gaps['supports'])

    def health(self):
        return ''

    def apply_wrench(self, body, force, torque=(0., 0., 0.)):
        if np.any(force) or np.any(torque):
            self.ext[body] = np.r_[torque, force]
        else:
            self.ext.pop(body, None)


class PinocchioKKT:
    """Pinocchio's hard-constraint forward dynamics integrated in tree coordinates.

    The acceleration is ``pinocchio.constraintDynamics`` on rigid constraint models of all
    chords (loops, hand links, ideal sole welds).  The default integrator is the classical
    fourth-order Runge-Kutta scheme (Munthe-Kaas form on the root poses) with one step per
    control period; ``integrator='euler'``
    gives semi-implicit Euler with ``substep``.  The constraints are enforced at
    acceleration level, so the configuration drifts from the closure manifold at the order
    of the integrator; ``baumgarte = (Kp, Kd)`` adds Pinocchio's constraint corrector.

    The chords are redundant (a pin is two point pairs, planar loops repeat rows).  The
    proximal parameter must then not be too small: with ``mu = 1e-10`` the solve loses
    accuracy as soon as the drift makes the redundant rows inconsistent.
    """
    name = 'pinocchio_kkt'

    def __init__(self, ens, comp, q0, substep=0.001, baumgarte=(0., 0.), mu=1e-8, iterations=50, integrator='rk4', **kw):
        import pinocchio as pin
        from ipd.backends.pin_backend import PinBackend
        self.pin = pin
        self.ens, self.comp = ens, comp
        s = self.sys = ens.system
        self.pb = PinBackend(s)
        self.h = substep
        self.integrator = integrator
        self.q0 = q0
        self.qp = self.pb.q_pin(q0)
        self.vp = np.zeros(self.pb.model.nv)
        self.models, self.layout = self.pb.constraint_models(comp.mode)
        for cm in self.models:
            cm.corrector.Kp[:] = baumgarte[0]
            cm.corrector.Kd[:] = baumgarte[1]
        self.datas = [m.createData() for m in self.models]
        self.pb.model.armature = self.pb.P @ self.pb.armature
        self.damping = self.pb.P @ s.damping
        self.data = self.pb.model.createData()
        pin.initConstraintDynamics(self.pb.model, self.data, self.models)
        self.settings = pin.ProximalSettings(1e-12, mu, iterations)
        self.ext = {}
        self.calls = 0
        self._info = dict(engine='Pinocchio constraintDynamics', integrator=integrator,
                          step_s='control period' if integrator != 'euler' else float(substep),
                          proximal_mu=float(mu), proximal_iterations=int(iterations), baumgarte=[float(b) for b in baumgarte],
                          constraints='rigid loops, hand links and sole welds')

    def info(self):
        """Settings of the plant for the result files."""
        return dict(self._info)

    def _config(self, qp=None):
        s, pb = self.sys, self.pb
        qp = self.qp if qp is None else qp
        q = self.q0.copy()
        for k, jid in pb.root_joint.items():
            iq = pb.model.joints[jid].idx_q
            q.T_root[k] = se3.transform(se3.matrix_from_quat_xyzw(qp[iq + 3:iq + 7]), qp[iq:iq + 3])
        sc = pb.idx_q >= 0
        q.theta[:s.nT][sc] = qp[pb.idx_q[sc]]
        return q

    def state(self):
        s = self.sys
        v = np.zeros(s.n)
        v[:s.nT] = self.pb.P.T @ self.vp
        return s.lift(self._config()), v

    def _acc(self, qp, vp, tau0):
        pin, pb = self.pin, self.pb
        tau = tau0 - self.damping * vp
        if self.ext:
            tau = tau + self._tau_ext(qp)
        self.calls += 1
        return pin.constraintDynamics(pb.model, self.data, qp, vp, tau, self.models, self.datas, self.settings)

    def advance(self, f, dt):
        pin, model = self.pin, self.pb.model
        tau0 = self.pb.P @ (self.sys.B @ f)
        if self.integrator == 'euler':
            n = max(1, int(round(dt / self.h)))
            h = dt / n
            for _ in range(n):
                self.vp = self.vp + h * self._acc(self.qp, self.vp, tau0)
                self.qp = pin.integrate(model, self.qp, h * self.vp)
            self._tau_last = tau0
            return
        h, q, v = dt, self.qp, self.vp
        k1 = v
        a1 = self._acc(q, v, tau0)
        v2 = v + .5 * h * a1
        a2 = self._acc(pin.integrate(model, q, .5 * h * k1), v2, tau0)
        k2 = self._pull_back(.5 * h * k1, v2)
        v3 = v + .5 * h * a2
        a3 = self._acc(pin.integrate(model, q, .5 * h * k2), v3, tau0)
        k3 = self._pull_back(.5 * h * k2, v3)
        v4 = v + h * a3
        a4 = self._acc(pin.integrate(model, q, h * k3), v4, tau0)
        k4 = self._pull_back(h * k3, v4)
        self.qp = pin.integrate(model, q, h / 6. * (k1 + 2. * k2 + 2. * k3 + k4))
        self.vp = v + h / 6. * (a1 + 2. * a2 + 2. * a3 + a4)
        self._tau_last = tau0

    def _pull_back(self, theta, v):
        """Stage velocity of the free-flyer joints seen from the start of the step (``J_r^-1(theta) v``),
        which keeps the Runge-Kutta scheme fourth order on the root poses."""
        P, s = self.pb.P, self.sys
        th, out = P.T @ theta, P.T @ v
        for sl in s.root_slice:
            if sl is not None:
                out[sl] = se3.inv_left(-th[sl]) @ out[sl]
        return P @ out

    def _tau_ext(self, qp):
        pin, pb = self.pin, self.pb
        tau = np.zeros(pb.model.nv)
        data = pb.data
        pin.computeJointJacobians(pb.model, data, qp)
        for body, w in self.ext.items():
            jid = int(pb.joint_of[body])
            J = pin.getJointJacobian(pb.model, data, jid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            p_joint = data.oMi[jid].translation
            p_body = (data.oMi[jid].homogeneous @ pb.Mjb[body] @ np.r_[self.sys.com[body], 1.])[:3]   # centre of mass
            force, torque = w[3:], w[:3] + np.cross(p_body - p_joint, w[3:])
            tau += J.T @ np.r_[force, torque]
        return tau

    def measure(self):
        ens, comp, s = self.ens, self.comp, self.sys
        soles = [f'd{k}/{side}_sole' for k in range(ens.n) for side in ('left', 'right')]
        fz, cop, fr = [np.nan] * len(soles), [[np.nan, np.nan]] * len(soles), [np.nan] * len(soles)
        link_force = {l['id']: np.nan for l in ens.links}
        if getattr(self, '_tau_last', None) is not None:
            self._acc(self.qp, self.vp, self._tau_last)          # multipliers of the current state
        for (e, rows), cd in zip(self.layout, self.datas):
            cid = comp.mode.ids[e]
            lam = np.array(cd.contact_force.vector)
            if cid in soles:
                i = soles.index(cid)
                # LOCAL frame of the anchor: force then moment acting on the sole
                F, Nm = -lam[:3], -lam[3:]
                fz[i] = F[2]
                cop[i] = [-Nm[1] / F[2], Nm[0] / F[2]] if abs(F[2]) > 5. else [0., 0.]
                fr[i] = np.hypot(F[0], F[1]) / abs(F[2]) if abs(F[2]) > 5. else 0.
            elif cid in link_force:
                link_force[cid] = float(np.linalg.norm(lam[:3]))
        gaps = chord_gaps(comp, s.lift(self._config()))
        return dict(fz=fz, cop=cop, friction_ratio=fr, gap=gaps['loops'], link_force=[link_force[l['id']] for l in ens.links],
                    link_gap=[gaps['links'].get(l['id'], np.nan) for l in ens.links], support_gap=gaps['supports'])

    def health(self):
        return '' if np.all(np.isfinite(self.qp)) and np.all(np.isfinite(self.vp)) else 'Pinocchio state not finite'

    def apply_wrench(self, body, force, torque=(0., 0., 0.)):
        if np.any(force) or np.any(torque):
            self.ext[body] = np.r_[torque, force]
        else:
            self.ext.pop(body, None)
