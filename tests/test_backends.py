"""MuJoCo and Pinocchio views of the same compiled records."""
import unittest

import numpy as np

import conftest  # noqa: F401
from ipd import partition as ip
from ipd.backends.mujoco_backend import MujocoPlant, export_mjcf
from ipd.backends.pin_backend import PinBackend
from systems import CASES


class BackendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = CASES['dancer_floating']()

    def test_pinocchio_tree_terms_equal_the_body_sums(self):
        case, s = self.case, self.case.sys
        q, u, f = case.sample(np.random.default_rng(0))
        ev = case.comp.evaluate(q, u)
        kin = s.kinematics(q, ev.v)
        t = ip.tree_terms(s, kin)
        pb = PinBackend(s)
        vT = ev.v[:s.nT]
        self.assertLess(np.max(abs(t.Gamma - pb.mass(q))) / np.max(abs(t.Gamma)), 1e-11)
        self.assertLess(np.max(abs(t.C @ vT + t.g - pb.bias(q, vT))) / np.max(abs(pb.bias(q, vT))), 1e-10)
        b = s.body_index['d0/l_foot']
        self.assertLess(np.max(abs(pb.body_pose(q, b) - kin.T[b])), 1e-12)

    def test_mujoco_state_round_trip_and_mass(self):
        import mujoco
        case, s = self.case, self.case.sys
        q, u, f = case.sample(np.random.default_rng(1))
        ev = case.comp.evaluate(q, u)
        plant = MujocoPlant(s, q=case.q0, contact=False)
        plant.set_state(q, ev.v[:s.nT])
        q2, v2 = plant.get_state(q)
        self.assertLess(np.max(abs(q2.T_root - q.T_root)), 1e-12)
        self.assertLess(np.max(abs(q2.theta[:s.nT] - q.theta[:s.nT])[plant.scalar]), 1e-14)
        self.assertLess(np.max(abs(v2[:s.nT] - ev.v[:s.nT])), 1e-12)
        self.assertAlmostEqual(float(np.sum(plant.model.body_mass)), float(s.mass.sum()), places=9)
        self.assertEqual(plant.model.nv, s.nT)
        self.assertEqual(plant.model.nu, s.na)

    def test_external_force_acts_at_the_centre_of_mass_on_every_plant(self):
        """The response to a push of the pelvis: compiled ODE, Pinocchio and MuJoCo."""
        from dancers import plants
        case, s, comp = self.case, self.case.sys, self.case.comp
        ens, q0 = case.ens, case.q0
        body = s.body_index['d0/pelvis']
        force = np.array([40., -25., 10.])
        ode = plants.MinimalODE(ens, comp, q0)
        f0 = np.zeros(s.na)
        _, ud0 = ode._rate(q0, np.zeros(comp.d), f0)
        ode.apply_wrench(body, force)
        ev, ud1 = ode._rate(q0, np.zeros(comp.d), f0)
        da_ode = comp.lift(ev)[:s.nT] @ (ud1 - ud0)
        pk = plants.PinocchioKKT(ens, comp, q0)
        tau0 = np.zeros(pk.pb.model.nv)
        a0 = pk.pb.P.T @ np.array(pk._acc(pk.qp, pk.vp, tau0))
        pk.apply_wrench(body, force)
        a1 = pk.pb.P.T @ np.array(pk._acc(pk.qp, pk.vp, tau0))
        self.assertLess(np.max(abs((a1 - a0) - da_ode)) / np.max(abs(da_ode)), 1e-6)
        mj = MujocoPlant(s, q=q0, contact=False)
        mj.model.dof_frictionloss[:] = 0.          # the compiled model carries no Coulomb friction
        mj.set_state(q0, np.zeros(s.nT))
        b0 = mj.acceleration(q0)
        mj.data.xfrc_applied[mj.body_id[body]] = np.r_[force, 0., 0., 0.]           # what MujocoWorld.apply_wrench does
        mj.mj.mj_forward(mj.model, mj.data)
        b1 = mj.acceleration(q0)
        self.assertLess(np.max(abs((b1 - b0) - da_ode)) / np.max(abs(da_ode)), 2e-2)  # soft equality constraints
        # the same force at the body origin (2 cm from the centre of mass of the pelvis) would be told apart
        P = np.eye(4)
        tau = comp.frame_jacobian(ev, body, P).T @ np.r_[0., 0., 0., force]
        da_origin = comp.lift(ev)[:s.nT] @ np.linalg.solve(ev.Gamma, tau)
        self.assertGreater(np.max(abs(da_origin - da_ode)) / np.max(abs(da_ode)), 5e-2)

    def test_mjcf_has_one_equality_per_point_pair(self):
        s = self.case.sys
        xml = export_mjcf(s, q=self.case.q0)
        pins = sum(1 for k in s.chord_kind if k == 'revolute')
        balls = sum(1 for k in s.chord_kind if k == 'spherical')
        self.assertEqual(xml.count('<connect '), 2 * pins + balls)


if __name__ == '__main__':
    unittest.main()
