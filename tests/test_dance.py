"""Formations, choreography helpers and a short closed-loop execution."""
import tempfile
import unittest

import numpy as np

import conftest  # noqa: F401
from dancers.choreography import Dance, smoothstep, window
from dancers.ensemble import link_records, placement


class FormationTest(unittest.TestCase):
    def test_placements(self):
        for formation in ('circle', 'semicircle', 'line'):
            xy, yaw = placement(formation, 5, 1.05)
            gaps = np.linalg.norm(xy[1:] - xy[:-1], axis=1)
            self.assertLess(np.max(abs(gaps - 1.05)), 1e-9, formation)
        self.assertEqual(len(link_records('circle', 5)), 5)
        self.assertEqual(len(link_records('line', 5)), 4)
        self.assertEqual(len(link_records('semicircle', 5)), 4)

    def test_choreography_is_smooth_and_starts_at_rest(self):
        self.assertEqual(smoothstep(0.), 0.)
        self.assertEqual(smoothstep(1.), 1.)
        self.assertEqual(window(1., 2., 8.), 0.)
        d = Dance(5, 'circle')
        for f in (d.pelvis, d.hands, d.free_arm):
            self.assertEqual(np.max(abs(f(0.))), 0.)
            self.assertLess(np.max(abs(f(d.duration))), 1e-9)
        t = np.linspace(0., d.duration, 3001)
        p = np.array([d.pelvis(x) for x in t])
        acc = np.diff(p, 2, axis=0) / (t[1] - t[0]) ** 2
        self.assertLess(np.max(abs(acc[:, :, :3])), 5.)        # m/s^2


class StepMetricsTest(unittest.TestCase):
    def test_turn_is_measured_about_the_arc_centre(self):
        """A rigid rotation of a semicircle about its arc centre (which is not the centroid of the pelvises)."""
        from types import SimpleNamespace
        from dancers.stepping import step_metrics
        xy, _ = placement('semicircle', 5, 1.05)
        centre = np.zeros(2)                                   # placement puts the arc centre at the origin
        self.assertGreater(np.linalg.norm(xy.mean(axis=0) - centre), 0.5)
        t = np.arange(0., 6.0001, 0.01)
        alpha = np.radians(11.) * smoothstep(t / t[-1])
        c, s_ = np.cos(alpha), np.sin(alpha)
        pel = np.zeros((len(t), 5, 3))
        pel[:, :, 0] = c[:, None] * xy[:, 0] - s_[:, None] * xy[:, 1]
        pel[:, :, 1] = s_[:, None] * xy[:, 0] + c[:, None] * xy[:, 1]
        dance = SimpleNamespace(cycles=1, t_settle=1.0, cycle=4.0, t_shift=1.2, t_swing=0.8, unload=0.35, reload=0.30,
                                clearance=0.03, centre=centre)
        sole = np.zeros((len(t), 10, 3))
        for j in range(10):                                    # every sole lifts 30 mm in its swing window, no slip
            lift = 1.0 + 1.2 if j % 2 else 1.0 + 2 * 1.2 + 0.8
            x = np.clip((t - lift) / 0.8, 0., 1.)
            sole[:, j, 2] = 0.03 * 16. * x ** 2 * (1. - x) ** 2
        log = dict(t=t, sole=sole, sole_ref=sole.copy(), pelvis=pel, pelvis_ref=pel.copy())
        m = step_metrics(log, dance)
        self.assertAlmostEqual(m['turn_deg'], 11., places=9)
        self.assertAlmostEqual(m['turn_reference_deg'], 11., places=9)
        self.assertEqual(m['steps'], 10)
        self.assertAlmostEqual(m['clearance_min_mm'], 30., places=3)
        self.assertAlmostEqual(m['stance_slip_max_mm'], 0., places=12)
        self.assertAlmostEqual(m['landing_error_max_mm'], 0., places=12)
        # sole 0 lands at 5.0 s and is fully loaded from 5.34 s: a creep after that is slip,
        # the same creep while the load is still being raised is settling
        for t_creep, slip, settle in ((5.6, 2., 0.), (5.1, 0., 2.)):
            sole2 = sole.copy()
            sole2[t >= t_creep, 0, 0] += 0.002
            m2 = step_metrics(dict(log, sole=sole2), dance)
            self.assertAlmostEqual(m2['stance_slip_max_mm'], slip, places=9)
            self.assertAlmostEqual(m2['landing_settle_max_mm'], settle, places=9)
        line = SimpleNamespace(**{**vars(dance), 'centre': None})
        self.assertIsNone(step_metrics(log, line)['turn_deg'])

    def test_contact_limits_count_loaded_samples_only(self):
        """Edge and friction-limit fractions use the loaded soles and the half extent of each sole."""
        from dancers.stepping import contact_metrics
        sites = lambda hx, hy: {'sole_sites': [{'id': f'{s}_sole', 'half_extent_m': [hx, hy]} for s in ('left', 'right')]}
        records = [sites(0.10, 0.05), sites(0.12, 0.06)]
        t = np.arange(0., 2.0001, 0.01)
        n = len(t)
        fz = np.full((n, 4), 200.)
        cop = np.zeros((n, 4, 2))
        fr = np.zeros((n, 4))
        late = t >= 1.0
        cop[late, 3, 0] = 0.118                    # 98% of the half length of sole 3 (0.12 m)
        cop[late, 0, 1] = 0.047                    # 94% of the half width of sole 0: not at the edge
        fr[late, 1] = 0.79                         # friction limit of a 0.8 floor
        fz[late, 2] = 20.                          # an unloaded sole at its edge is not counted
        cop[late, 2, 0] = 0.12
        m = contact_metrics(dict(t=t, fz=fz, cop=cop, friction_ratio=fr), records, friction=0.8)
        loaded = 3 * np.sum(t >= 0.5) + np.sum((t >= 0.5) & ~late)
        self.assertAlmostEqual(m['cop_edge_fraction'], np.sum(late) / loaded, places=12)
        self.assertAlmostEqual(m['friction_limit_fraction'], np.sum(late) / loaded, places=12)
        self.assertAlmostEqual(m['max_cop_x_fraction'], 0.118 / 0.12, places=12)
        self.assertAlmostEqual(m['max_cop_y_fraction'], 0.047 / 0.05, places=12)


class ExecutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dancers.session import Session
        cls.tmp = tempfile.TemporaryDirectory()
        cls.se = Session('line', duration=1.0, cache=cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_short_run_on_the_ideal_mujoco_plant(self):
        """Without friction, actuator dynamics and noise: tracking, allocation and the timing of the logged predictions."""
        from dancers import plants
        from dancers.execute import build_policy, metrics, run
        se = self.se
        self.assertEqual(se.comp.d, 63)
        self.assertLess(se.ref.info['residual'], 1e-9)
        plant = plants.MujocoWorld(se.ens, se.comp, se.q0, actuator=False, noise=False, coulomb=False)
        policy = build_policy(se.ens, se.comp, se.ref, se.q0, 'ip', friction_feedforward=0.)
        log = run(se.ens, se.comp, se.ref, policy, plant, duration=0.3)
        m = metrics(log)
        self.assertTrue(m['completed'])
        self.assertEqual(m['failure'], '')
        self.assertEqual(m['window_s'][0], 0.)
        self.assertLess(m['pelvis_peak_mm'], 0.5)
        self.assertLess(m['loop_gap_peak_um'], 50.)
        self.assertEqual(m['qp_not_solved'], 0)
        self.assertEqual(sum(m['qp_status'].values()), log['control_steps'])
        weight = se.sys.mass.sum() * 9.81
        self.assertAlmostEqual(float(np.sum(log['fz'][-1])) / weight, 1., places=2)
        # the logged model forces are the predictions for the period in which the plant forces were measured
        self.assertTrue(np.all(np.isnan(log['fz_model'][0])))
        err = np.abs(log['fz_model'][5:] - log['fz'][5:])
        self.assertLess(np.nanmax(err), 2.0)
        link = np.abs(log['link_force_model'][5:] - log['link_force'][5:])
        self.assertLess(np.nanmax(link), 0.2)

    def test_short_run_on_the_realistic_plant(self):
        """With joint friction, actuator lag and torque-speed limits and measurement noise."""
        from dancers import plants
        from dancers.execute import build_policy, metrics, run
        se = self.se
        plant = plants.MujocoWorld(se.ens, se.comp, se.q0, seed=1)
        info = plant.info()
        self.assertTrue(info['joint_coulomb_friction'])
        self.assertIsNotNone(info['actuator'])
        self.assertIsNotNone(info['noise'])
        policy = build_policy(se.ens, se.comp, se.ref, se.q0, 'ip')
        log = run(se.ens, se.comp, se.ref, policy, plant, duration=0.3)
        m = metrics(log)
        self.assertTrue(m['completed'])
        self.assertLess(m['pelvis_peak_mm'], 0.5)
        # noise activates the support limits in most steps; the programme is always solved, and only
        # occasionally the commanded force lies outside the support limits (a positive slack)
        self.assertEqual(m['qp_not_solved'], 0)
        self.assertTrue(np.isfinite(m['max_slack']))
        self.assertLess(m['infeasible_fraction'], 0.1)

    def test_a_fall_is_reported(self):
        from dancers import plants
        from dancers.execute import build_policy, metrics, run
        se = self.se
        plant = plants.MujocoWorld(se.ens, se.comp, se.q0)
        policy = build_policy(se.ens, se.comp, se.ref, se.q0, 'actuator_pd', pd_scale=1e-4)   # far too soft to stand
        log = run(se.ens, se.comp, se.ref, policy, plant, duration=1.0)
        m = metrics(log)
        self.assertFalse(m['completed'])
        self.assertEqual(m['failure'], 'fell')
        self.assertLess(m['simulated_s'], 1.0)


if __name__ == '__main__':
    unittest.main()
