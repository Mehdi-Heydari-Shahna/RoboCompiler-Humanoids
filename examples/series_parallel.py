"""Three-degree-of-freedom series-parallel manipulator with a triangular closed chain.

The mechanism class and the numerical parameters follow Section IV of M. Dastranj and
J. Mattila, "Inertia Partitioning Modular Robust Control Framework for Reconfigurable
Multibody Systems" (arXiv:2510.23226v2): a base rotation ``phi``, a first-link rotation
``theta`` and a prismatic actuator ``delta`` that drives the relative rotation ``zeta``
of the second link through a triangle with two sides ``L0`` and the actuator length
``l0 + delta``.  Body frames are chosen here (x along each link, y along the hinge
axes); the published frame drawings are not reproduced.

The record lists six physical joints on six bodies, so the joint graph has one cycle.
No joint is marked as a cut and no map ``zeta(delta)`` is supplied.
"""
import numpy as np

from ipd.shapes import body, joint, tf

L, L0, l0 = 2.0, 0.35, 0.425
# Published inertia tensors are given with z along the hinge axis; here y is the hinge axis.
_R = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
_I1 = np.diag([0.804, 0.831, 1.18])
_I2 = np.array([[0.311, -0.065, 0.098], [-0.065, 22.7, -0.003], [0.098, -0.003, 22.8]])
_I3 = np.array([[0.311, 0.065, -0.098], [0.065, 22.7, -0.003], [-0.098, -0.003, 22.8]])


def zeta_of_delta(delta):
    """Hand-derived relation of the reference: zeta = -acos((delta + l0)^2 / (2 L0^2) - 1)."""
    return -np.arccos((delta + l0) ** 2 / (2 * L0 ** 2) - 1)


def dzeta_ddelta(delta):
    """Hand-derived intermediate Jacobian entry, Eq. (28) of the reference."""
    c = (delta + l0) ** 2 / (2 * L0 ** 2) - 1
    return (delta + l0) / (L0 ** 2 * np.sqrt(1 - c * c))


def physical(payload_kg=0.):
    rod = lambda m, length: np.diag([1e-4 * m, m * length ** 2 / 12, m * length ** 2 / 12])
    bodies = [
        {'id': 'world', 'massless': True, 'mass_kg': 0.},
        body('base', 30., [0., 0., 0.103], _I1),
        body('link1', 60., [0.959, -0.077, -0.001], _R @ _I2 @ _R.T),
        body('link2', 60. + payload_kg, [1.041, -0.077, -0.001], _R @ _I3 @ _R.T),
        body('cylinder', 1.5, [0.2, 0., 0.], rod(1.5, 0.4)),
        body('piston', 1.0, [-0.2, 0., 0.], rod(1.0, 0.4)),
    ]
    y = (0., 1., 0.)
    joints = [
        joint('phi', 'revolute', 'world', 'base', axis=(0., 0., 1.), lower=-3.1, upper=3.1),
        joint('theta', 'revolute', 'base', 'link1', T_AJ=tf((0.075, 0., 0.206)), axis=y, lower=-1.5, upper=1.5),
        joint('zeta', 'revolute', 'link1', 'link2', T_AJ=tf((L, 0., 0.)), axis=y, lower=-3.0, upper=-0.05),
        joint('cylinder_pivot', 'revolute', 'link1', 'cylinder', T_AJ=tf((L - L0, 0., 0.)), axis=y, lower=-3.1, upper=3.1),
        joint('delta', 'prismatic', 'cylinder', 'piston', T_AJ=tf((l0, 0., 0.)), axis=(1., 0., 0.), lower=0., upper=0.2),
        joint('piston_pin', 'revolute', 'piston', 'link2', T_BJ=tf((L0, 0., 0.)), axis=y, lower=-3.1, upper=3.1),
    ]
    actuators = [
        {'id': 'phi_drive', 'joint_id': 'phi', 'gear': 1., 'effort_bounds': [-4000., 4000.], 'effort_unit': 'Nm'},
        {'id': 'theta_drive', 'joint_id': 'theta', 'gear': 1., 'effort_bounds': [-8000., 8000.], 'effort_unit': 'Nm'},
        {'id': 'delta_cylinder', 'joint_id': 'delta', 'gear': 1., 'effort_bounds': [-60000., 60000.], 'effort_unit': 'N'},
    ]
    return {
        'schema': 'cmg.physical-mechanism-graph/1.0', 'name': 'series_parallel_3dof', 'ground': 'world',
        'gravity_m_s2': [0., 0., -9.81], 'bodies': bodies, 'joints': joints, 'actuators': actuators,
        'partition_requests': {'actuator_space': ['phi', 'theta', 'delta'], 'joint_space': ['phi', 'theta', 'zeta']},
        'seed': {'joints': {'phi': 0.2, 'theta': -0.4, 'zeta': -1.3, 'cylinder_pivot': -0.6, 'delta': 0.1}},
        'provenance': {'reference': 'arXiv:2510.23226v2, Section IV, Tables I-II',
                       'note': 'Mechanism class and parameters after the reference; frames re-expressed.'},
    }
