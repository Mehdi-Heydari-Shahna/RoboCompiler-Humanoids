"""Run the studies of the paper and regenerate its tables, numbers, figures and videos.

    python reproduce.py --profile smoke             a few minutes: every workflow except the audit, short runs
    python reproduce.py --profile full              every run reported in the paper
    python reproduce.py --report                    tables, numbers and figures from results/ (needs OpenGL)
    python reproduce.py --report --videos           also the videos (needs OpenGL: MUJOCO_GL=egl)

The full profile needs about an hour on two cores (``--workers 2``, the default).  The
studies that report times (updates, composition, timing) run last, one after the other,
with nothing else running.  Results are written to ``results/`` (or ``--out``).
``--report`` rewrites ``paper/generated`` (tables and number macros), ``paper/figures``,
the evidence folder and, with ``--videos``, ``videos/``.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DANCE, STEPS, AUDIT = 'workflows/dance/run_dance.py', 'workflows/dance/run_steps.py', 'workflows/dance/run_audit.py'
STATE, MANIP = 'workflows/statewise/run_statewise.py', 'workflows/manipulator/run_manipulator.py'
FORMATIONS = ('circle', 'semicircle', 'line')
MODEL = ['ip', 'decentralized', 'simplified', 'massless']          # policies that differ in the model they are given
BASE = ['no_curvature', 'gravity_pd', 'actuator_pd']
IDEAL = ['ip', 'decentralized', 'simplified', 'massless', 'no_curvature', 'gravity_pd']


def full():
    """(name, script, arguments) of every reported run, in four phases.

    The audits create every cached reference once (phase A, beside the state-wise studies,
    which do not use the cache); the closed-loop runs share a queue (phase B); the studies
    that report times run alone (phase C).
    """
    first_a = [('audit', AUDIT, [])]
    first_b = [('statewise_systems', STATE, ['systems']),
               ('statewise_consistency', STATE, ['consistency']),
               ('statewise_structure', STATE, ['structure']),
               ('statewise_quasi', STATE, ['quasi_velocity']),
               ('statewise_covariance', STATE, ['covariance']),
               ('statewise_conservation', STATE, ['conservation']),
               ('statewise_reconfiguration', STATE, ['reconfiguration']),
               ('manipulator_compile', MANIP, [])]
    queue = []
    for f in FORMATIONS:
        queue += [(f'nominal_{f}_a', DANCE, ['--formation', f, '--policies'] + MODEL),
                  (f'nominal_{f}_b', DANCE, ['--formation', f, '--policies'] + BASE)]
    queue += [('recursive_circle', DANCE, ['--formation', 'circle', '--policies', 'recursive']),
              ('ideal_circle_a', DANCE, ['--formation', 'circle', '--scenario', 'ideal', '--policies'] + IDEAL[:3]),
              ('ideal_circle_b', DANCE, ['--formation', 'circle', '--scenario', 'ideal', '--policies'] + IDEAL[3:]),
              ('payload_circle', DANCE, ['--formation', 'circle', '--scenario', 'payload', '--policies', 'updated', 'stale'])]
    queue += [('pinocchio_circle', DANCE, ['--formation', 'circle', '--plant', 'pinocchio', '--policies', 'ip'])]
    for f in FORMATIONS:
        queue += [(f'steps_{f}', STEPS, ['--formation', f, '--policies', 'ip', 'decentralized'])]
    # sensitivity to the measurement noise: the stepping semicircle with twice the noise
    queue += [('steps_semicircle_noise2', STEPS, ['--formation', 'semicircle', '--policies', 'ip', '--noise-scale', '2',
                                                  '--tag', 'steps_noise2'])]
    last = [('statewise_updates', STATE, ['updates']),
            ('statewise_composition', STATE, ['composition']),
            ('statewise_timing', STATE, ['timing'])]
    return first_a, first_b, queue, last


def smoke():
    """Every workflow except the audit, with short runs (no claims are drawn from these)."""
    first_a = [('smoke_nominal', DANCE, ['--formation', 'line', '--duration', '1', '--policies', 'ip', 'gravity_pd'])]
    first_b = [('smoke_consistency', STATE, ['consistency', '--cases', 'manipulator', 'excavator', 'dancer_double', '--states', '3']),
               ('smoke_structure', STATE, ['structure', '--cases', 'manipulator', 'dancer_double', '--states', '2']),
               ('smoke_conservation', STATE, ['conservation']),
               ('smoke_manipulator', MANIP, [])]
    queue = [('smoke_pinocchio', DANCE, ['--formation', 'line', '--plant', 'pinocchio', '--duration', '0.2', '--policies', 'ip']),
             ('smoke_ode', DANCE, ['--formation', 'line', '--plant', 'minimal_ode', '--duration', '0.2', '--policies', 'ip']),
             ('smoke_steps', STEPS, ['--formation', 'line', '--duration', '2', '--policies', 'ip'])]
    return first_a, first_b, queue, []


class Runner:
    def __init__(self, out, log):
        self.out, self.log_dir = Path(out), Path(log)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.failed = []

    def note(self, text):
        with self.lock:
            line = f'{time.strftime("%H:%M:%S", time.gmtime())} {text}'
            print(line, flush=True)
            with open(self.log_dir / 'reproduce.log', 'a') as f:
                f.write(line + '\n')

    def run(self, job):
        name, script, args = job
        args = list(args)
        if '--out' not in args:
            sub = 'statewise' if script == STATE else 'manipulator' if script == MANIP else 'dance'
            args += ['--out', str(self.out / sub)]
        if script in (DANCE, STEPS, AUDIT):
            args += ['--cache', str(self.out / 'cache')]
        env = dict(os.environ)
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
            env.setdefault(key, '1')
        self.note(f'START {name}')
        with open(self.log_dir / f'{name}.log', 'w') as f:
            rc = subprocess.run([sys.executable, script] + args, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        self.note(f'END   {name} rc={rc}')
        if rc:
            self.failed.append(name)

    def drain(self, jobs):
        while True:
            with self.lock:
                if not jobs:
                    return
                job = jobs.pop(0)
            self.run(job)


def report(out, videos):
    rc = 0
    jobs = [('workflows/report/make_tables.py', ['--results', str(out)]),
            ('workflows/report/make_figures.py', ['--results', str(out)]),
            ('workflows/report/render_media.py', ['figure'] + (['videos'] if videos else []) + ['--results', str(out)])]
    for script, extra in jobs:
        rc |= subprocess.run([sys.executable, script] + extra, cwd=ROOT).returncode
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--profile', choices=('smoke', 'full'), default=None)
    ap.add_argument('--report', action='store_true', help='regenerate tables, numbers and figures')
    ap.add_argument('--videos', action='store_true', help='with --report: also render the videos')
    ap.add_argument('--out', default=None, help='results directory (default: results, or results_smoke for the smoke profile)')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--excavator-record', default=None,
                    help='private excavator record (default: $ROBOCOMPILER_EXCAVATOR_RECORD); without one the excavator cases are skipped')
    args = ap.parse_args(argv)
    if args.excavator_record:                  # inherited by every workflow (Runner.run copies os.environ)
        os.environ['ROBOCOMPILER_EXCAVATOR_RECORD'] = str(Path(args.excavator_record).expanduser().resolve())
    if not args.profile and not args.report:
        ap.error('give --profile and/or --report')
    # Resolve a relative --out against the current directory: the workflows run with cwd=ROOT.
    out = Path(args.out).expanduser().resolve() if args.out else ROOT / ('results_smoke' if args.profile == 'smoke' else 'results')
    rc = 0
    if args.profile:
        first_a, first_b, queue, last = full() if args.profile == 'full' else smoke()
        runner = Runner(out, out / 'logs')
        t0 = time.time()
        chains = [first_a, first_b] if args.workers > 1 else [first_a + first_b]
        threads = [threading.Thread(target=runner.drain, args=(c,)) for c in chains]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        threads = [threading.Thread(target=runner.drain, args=(queue,)) for _ in range(max(1, args.workers))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        runner.drain(last)
        runner.note(f'DONE in {time.time() - t0:.0f} s; failed: {runner.failed or "none"}')
        rc = 1 if runner.failed else 0
    if args.report:
        rc |= report(out, args.videos)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
