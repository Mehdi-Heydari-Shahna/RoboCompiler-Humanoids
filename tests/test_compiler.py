"""Compiler rules and a reference that does not pass through the compiler."""
import unittest

import numpy as np

import conftest  # noqa: F401
import maximal as MX
import methods as M
from dancers.humanoid import humanoid
from examples import series_parallel as sp
from ipd import se3
from ipd.compiler import ModelError, compile_graph
from ipd.shapes import body, joint, tf
from ipd.system import Member, System
from systems import CASES


class DriveDataTest(unittest.TestCase):
    def test_drive_data_never_lies_on_a_chord(self):
        """Ports and drive data act on tree coordinates; every loop is closed by an ideal bearing."""
        for partition in ('joint_space', 'actuator_space'):
            record = humanoid()
            comp = compile_graph(record, partition)
            tree = set(comp.cmg['coordinate_ids'])
            self.assertTrue(set(record['drive']) <= tree)
            self.assertTrue({a['joint_id'] for a in record['actuators']} <= tree)
            self.assertEqual(len(comp.plan['cut_joint_ids']), 8)
            for jid, data in record['drive'].items():
                self.assertEqual(comp.cmg['damping'][jid], data['damping'])
                self.assertEqual(comp.cmg['armature'][jid], data['armature'])

    def test_a_loop_of_driven_joints_is_refused(self):
        record = humanoid()
        record['drive']['l_knee_rod_pin'] = {'armature': 0., 'damping': 0.02, 'frictionloss': 0.}
        with self.assertRaises(ModelError):
            compile_graph(record, 'joint_space')

    def test_bad_drive_records_are_refused(self):
        for bad in ({'no_such_joint': {'damping': 0.1}}, {'phi': {'damping': -1.}}, {'phi': {'stiffness': 3.}}):
            record = sp.physical()
            record['drive'] = bad
            with self.assertRaises(ModelError):
                compile_graph(record, 'actuator_space')


class ReversedJointTest(unittest.TestCase):
    def test_a_joint_traversed_against_its_record_keeps_coordinate_and_limits(self):
        """The record names the moving body as ``body_a``; the tree reaches it through ``body_b``."""
        T_AJ, T_BJ = tf((0.1, 0., 0.05)), tf((0., 0.2, 0.))
        record = {
            'schema': 'cmg.physical-mechanism-graph/1.0', 'name': 'reversed', 'ground': 'g',
            'bodies': [{'id': 'g', 'massless': True, 'mass_kg': 0.}, body('b', 2., [0.03, 0., 0.], np.diag([.01, .02, .03]))],
            'joints': [joint('j', 'revolute', 'b', 'g', T_AJ=T_AJ, T_BJ=T_BJ, axis=(0., 0., 1.), lower=0.2, upper=1.0)],
            'actuators': [{'id': 'm', 'joint_id': 'j', 'gear': 1., 'effort_bounds': [-5., 5.]}],
            'partition_requests': {'joint_space': ['j']}, 'seed': {'joints': {'j': 0.5}},
        }
        comp = compile_graph(record, 'joint_space')
        edge = comp.cmg['joints'][0]
        self.assertTrue(edge['reversed'])
        self.assertEqual((edge['limits']['lower'], edge['limits']['upper']), (0.2, 1.0))
        s = System([Member('x', comp, np.eye(4))])
        self.assertEqual((s.lower[0], s.upper[0]), (0.2, 1.0))
        q = s.neutral()
        theta = 0.5
        # record: T_b T_AJ Q(theta) = T_g T_BJ with T_g = I
        Q = np.eye(4)
        Q[:3, :3] = se3.rot_axis(np.array([0., 0., 1.]), np.array(theta))
        expected = np.asarray(T_BJ) @ se3.inv(np.asarray(T_AJ) @ Q)
        self.assertLess(np.max(abs(s.poses(q)[s.body_index['x/b']] - expected)), 1e-13)


class MaximalCoordinateTest(unittest.TestCase):
    """Free bodies and the joints of the record against the compiled minimal-coordinate model."""

    def test_body_accelerations_agree(self):
        for name, tol in (('manipulator', 1e-10), ('dancer_floating', 1e-10), ('dancer_double', 1e-9)):
            case = CASES[name]()
            s, comp = case.sys, case.comp
            q, u, f = case.sample(np.random.default_rng(5))
            ev = comp.evaluate(q, u)
            _, aux = M.ip_two_level(case, q, u, f, ev)
            a_full = aux['E'] @ aux['ud'] + aux['Ed'] @ u
            report = MX.compare(case, s.kinematics(q, ev.v), a_full, f)
            self.assertEqual(report['unknowns'] - report['rank'], comp.d, name)      # mobility from the record alone
            self.assertLess(report['position_residual'], 1e-9, name)
            self.assertLess(report['velocity_residual'], 1e-9, name)
            self.assertLess(report['delta_body'], tol, name)

    def test_it_sees_a_wrong_model(self):
        """The comparison is not vacuous: without the curvature term the body accelerations differ."""
        case = CASES['dancer_double']()
        s, comp = case.sys, case.comp
        q, u, f = case.sample(np.random.default_rng(6))
        ev = comp.evaluate(q, u)
        _, aux = M.ip_two_level(case, q, u, f, ev)
        a_wrong = aux['E'] @ aux['ud']                                                # Edot u left out
        report = MX.compare(case, s.kinematics(q, ev.v), a_wrong, f)
        self.assertGreater(report['delta_body'], 1e-3)


if __name__ == '__main__':
    unittest.main()
