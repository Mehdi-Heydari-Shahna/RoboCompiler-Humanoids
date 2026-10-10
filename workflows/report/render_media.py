"""Renderings of executed runs: the teaser and snapshot figures of the paper and the videos.

Run after the dances:  python workflows/report/render_media.py [figure] [videos]
The recorded configurations of the logs are replayed through the MuJoCo model of the same
compiled records on the studio stage (``dancers.stage.Studio``); nothing is simulated here
(also for the runs that were executed by the Pinocchio plant).  Rendering needs an OpenGL
context (``MUJOCO_GL=egl`` is set when the variable is unset).  ``videos/manifest.json``
lists every video with the log files it was rendered from and the hashes of their frames.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dancers.choreography import SEGMENTS                # noqa: E402
from dancers.ensemble import Ensemble                    # noqa: E402
from dancers.render import CAMERAS, render               # noqa: E402
from dancers.stage import Studio                         # noqa: E402
from ipd.system import Config                            # noqa: E402

KEYS = ('frames_t', 'frames_T_root', 'frames_theta', 't', 'pelvis_err', 'sole', 'sole_ref', 'stance')
SEGMENT_NAME = {'settle': 'settle', 'bounce canon': 'knee-bounce canon', 'arm wave': 'hand wave', 'breathing': 'common lean',
                'sway and twist': 'sway and twist', 'bow': 'bow'}
ACTUATOR_PART = ('_barrel', '_rod', '_cross')


def segment(t):
    return SEGMENT_NAME[next(name for name, t1 in SEGMENTS if t < t1)]


def stem(results, formation, scenario, policy, plant='mujoco'):
    return Path(results) / 'dance' / formation / f'{scenario}__{plant}__{policy}'


def load(results, formation, scenario, policy, plant='mujoco'):
    with np.load(stem(results, formation, scenario, policy, plant).with_suffix('.npz'), allow_pickle=False) as z:
        return {k: z[k] for k in KEYS if k in z.files}


def result(results, formation, scenario, policy, plant='mujoco'):
    p = stem(results, formation, scenario, policy, plant).with_suffix('.json')
    return json.loads(p.read_text()) if p.exists() else None


class Stage:
    """One MuJoCo model of a formation on the studio stage, reused for all stills."""

    def __init__(self, formation, width=1600, height=900, highlight_actuators=False):
        import mujoco
        from ipd.backends.mujoco_backend import MujocoPlant
        self.mj = mujoco
        self.ens = Ensemble(formation)
        self.sys = self.ens.system
        self.plant = MujocoPlant(self.sys, q=self.sys.neutral(), offscreen=(width, height), scene=Studio(render=True))
        if highlight_actuators:
            self.highlight()
        self.renderer = mujoco.Renderer(self.plant.model, height, width)

    def highlight(self):
        """Colour the visual geoms of the moving actuator parts (members of the partition) orange and turn the
        accent colours of the shells grey, so that only the actuator members stand out."""
        m, mj = self.plant.model, self.mj
        orange = np.array([0.95, 0.45, 0.08, 1.])
        for k in range(m.nmat):
            name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_MATERIAL, k) or ''
            if name.startswith(('accent_', 'led_')):
                m.mat_rgba[k] = [0.55, 0.57, 0.60, 1.]
                m.mat_emission[k] = 0.
        for g in range(m.ngeom):
            body = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ''
            if body.endswith(ACTUATOR_PART):
                m.geom_matid[g] = -1
                m.geom_rgba[g] = orange

    def shot(self, T_root, theta_T, camera, reflection=True):
        s = self.sys
        theta = np.zeros(s.n)
        theta[:s.nT] = theta_T
        self.plant.set_state(Config(T_root, theta))
        cam = self.mj.MjvCamera()
        cam.lookat[:] = camera['lookat']; cam.distance = camera['distance']
        cam.azimuth = camera['azimuth']; cam.elevation = camera['elevation']
        self.renderer.update_scene(self.plant.data, cam)
        self.renderer.scene.flags[self.mj.mjtRndFlag.mjRND_REFLECTION] = int(bool(reflection))
        self.renderer.scene.flags[self.mj.mjtRndFlag.mjRND_FOG] = 1
        return self.renderer.render().copy()

    def close(self):
        self.renderer.close()


def _frame(log, t):
    i = int(np.argmin(abs(log['frames_t'] - t)))
    return log['frames_T_root'][i], log['frames_theta'][i], float(log['frames_t'][i])


def _save(fig, out, name):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'{name}.{ext}', dpi=300, bbox_inches='tight', pad_inches=0.01)


def figure(results, out):
    """``fig_teaser``: the ring on stage and one humanoid with its actuator members highlighted;
    ``fig_snapshots``: the three formations, the travelling ring and the sole paths seen from above."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7})
    ink = '#1F2D3A'
    # ---------------------------------------------------------------- teaser
    log = load(results, 'circle', 'nominal', 'ip')
    T, th, t_ring = _frame(log, 11.0)
    stage = Stage('circle', 1600, 1000)
    ring = stage.shot(T, th, dict(lookat=(0., 0., 0.85), distance=4.0, azimuth=58., elevation=-20.))
    stage.close()
    stage = Stage('circle', 800, 1200, highlight_actuators=True)
    k = 3                                                         # seen from behind: the screws sit behind thighs, shanks and upper arms
    p = T[k, :3, 3]
    yaw = np.degrees(np.arctan2(T[k, 1, 0], T[k, 0, 0]))
    close = stage.shot(T, th, dict(lookat=(p[0], p[1], 0.82), distance=2.5, azimuth=yaw + 25., elevation=-6.))
    stage.close()
    h, w = ring.shape[:2]
    a_img = ring[int(.06 * h):int(.94 * h), int(.10 * w):int(.92 * w)]
    h, w = close.shape[:2]
    b_img = close[int(.04 * h):int(.99 * h), int(.10 * w):int(.90 * w)]
    ra, rb = a_img.shape[1] / a_img.shape[0], b_img.shape[1] / b_img.shape[0]
    gap, W = 0.03, 3.45                                           # inches; both panels get the same height
    H = (W - gap) / (ra + rb)
    fig = plt.figure(figsize=(W, H))
    ax1 = fig.add_axes([0.0, 0.0, ra * H / W, 1.0]); ax2 = fig.add_axes([1. - rb * H / W, 0.0, rb * H / W, 1.0])
    ax1.imshow(a_img); ax1.axis('off')
    ax2.imshow(b_img); ax2.axis('off')
    ax1.text(0.02, 0.97, '(a)', transform=ax1.transAxes, color='white', fontsize=7, fontweight='bold', va='top')
    ax2.text(0.05, 0.97, '(b)', transform=ax2.transAxes, color='white', fontsize=7, fontweight='bold', va='top')
    ax2.text(0.5, 0.025, 'actuator members', transform=ax2.transAxes, color='#FF9A3C', fontsize=5.4, ha='center', va='bottom',
             fontweight='bold', bbox=dict(facecolor='black', alpha=0.55, lw=0, boxstyle='round,pad=0.2'))
    _save(fig, out, 'fig_teaser')
    plt.close(fig)
    print('wrote fig_teaser', flush=True)
    # ---------------------------------------------------------------- snapshots
    picks = (('circle', 'nominal', 11.0, 'Ring: hand wave', dict(lookat=(0., 0., 0.8), distance=4.2, azimuth=58., elevation=-22.)),
             ('semicircle', 'nominal', 22.8, 'Semicircle: sway and twist', dict(CAMERAS['semicircle'], distance=4.8)),
             ('line', 'nominal', 28.0, 'Line: bow', dict(CAMERAS['line'], distance=6.1, elevation=-12.)))
    top = []
    for formation, scenario, t, title, cam in picks:
        log = load(results, formation, scenario, 'ip')
        T, th, tt = _frame(log, t)
        stage = Stage(formation, 1500, 1000)
        top.append((stage.shot(T, th, cam), f'{title} ($t={tt:.1f}$ s)'))
        stage.close()
    bottom = []
    steps = stem(results, 'circle', 'steps', 'ip').with_suffix('.npz')
    if steps.exists():
        log = load(results, 'circle', 'steps', 'ip')
        res = result(results, 'circle', 'steps', 'ip')
        dn = res['settings']['dance']
        t0 = 1.5 + 2 * (dn['t_shift_s'] + dn['t_swing_s'])         # second cycle
        when = ((t0 + 0.6 * dn['t_shift_s'], 'weight shift'), (t0 + dn['t_shift_s'] + 0.5 * dn['t_swing_s'], 'right feet swing'),
                (t0 + 2 * dn['t_shift_s'] + 1.5 * dn['t_swing_s'], 'left feet follow'))
        stage = Stage('circle', 1500, 1000)
        low = dict(lookat=(0., 0., 0.62), distance=4.4, azimuth=58., elevation=-15.)
        for t, title in when:
            T, th, tt = _frame(log, t)
            bottom.append((stage.shot(T, th, low), f'Steps: {title} ($t={tt:.1f}$ s)'))
        stage.close()
    ncol = 4
    fig = plt.figure(figsize=(7.16, 3.05))
    wcol = 1. / ncol
    for k, (img, title) in enumerate(top):
        ax = fig.add_axes([k * wcol * 4 / 3 + 0.002, 0.5, wcol * 4 / 3 - 0.004, 0.46])
        h, w = img.shape[:2]
        ax.imshow(img[int(.08 * h):int(.95 * h), int(.08 * w):int(.92 * w)]); ax.axis('off')
        ax.set_title(title, fontsize=6.2, pad=1.5, color=ink)
    for k, (img, title) in enumerate(bottom):
        ax = fig.add_axes([k * wcol + 0.002, 0.0, wcol - 0.004, 0.44])
        h, w = img.shape[:2]
        ax.imshow(img[int(.06 * h):int(.97 * h), int(.10 * w):int(.90 * w)]); ax.axis('off')
        ax.set_title(title, fontsize=6.0, pad=1.5, color=ink)
    if steps.exists():
        ax = fig.add_axes([3 * wcol + 0.035, 0.035, wcol - 0.045, 0.38])
        sole, ref = log['sole'], log['sole_ref']
        for j in range(sole.shape[1]):
            c = '#1764AB' if j % 2 else '#D9822B'
            ax.plot(ref[:, j, 0], ref[:, j, 1], color='#B8C4CE', lw=1.8, zorder=1, label='reference' if j == 0 else None)
            ax.plot(sole[:, j, 0], sole[:, j, 1], color=c, lw=0.7, zorder=2,
                    label=('right soles' if j % 2 else 'left soles') if j < 2 else None)
        ax.legend(loc='center', fontsize=4.8, frameon=False, handlelength=1.2)
        ax.set_aspect('equal'); ax.tick_params(labelsize=5, length=2)
        ax.set_xlabel('x (m)', fontsize=5.5, labelpad=1); ax.set_ylabel('y (m)', fontsize=5.5, labelpad=1)
        ax.set_title('Sole paths from above', fontsize=5.8, pad=1.5, color=ink)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    _save(fig, out, 'fig_snapshots')
    plt.close(fig)
    print('wrote fig_snapshots', flush=True)


def frames_hash(path):
    """SHA-256 of the recorded configurations of a log: what a video of that run shows."""
    h = hashlib.sha256()
    with np.load(path, allow_pickle=False) as z:
        for k in ('frames_t', 'frames_T_root', 'frames_theta'):
            h.update(np.ascontiguousarray(z[k]).tobytes())
    return h.hexdigest()


class Videos:
    """Renders videos and keeps ``manifest.json``: every video with its source logs and the hashes of their frames."""

    def __init__(self, results, out, only=None, skip_existing=False):
        self.results, self.out = Path(results), Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.only, self.skip_existing = only, skip_existing
        path = self.out / 'manifest.json'
        self.manifest = json.loads(path.read_text()) if path.exists() else []

    def wanted(self, name, sources):
        if self.only and name not in self.only:
            return False
        if not self.skip_existing or not (self.out / f'{name}.mp4').exists():
            return True
        old = next((e for e in self.manifest if e['file'] == f'{name}.mp4'), None)
        now = {str(Path(p).relative_to(self.results)): frames_hash(p) for p in sources}
        return old is None or old.get('frames_sha256') != now

    def clip(self, formation, scenario, policy, title, orbit=0., labels=None, name=None, plant='mujoco', length=None,
             size=(1280, 720), camera=None, record=True):
        path = stem(self.results, formation, scenario, policy, plant).with_suffix('.npz')
        name = name or f'{formation}_{scenario}_{policy}'
        if not path.exists():
            print('skipped', name, f'({path.name} missing)')
            return None
        if record and not self.wanted(name, [path]):
            return None
        m = (result(self.results, formation, scenario, policy, plant) or {}).get('metrics', {})
        log = load(self.results, formation, scenario, policy, plant)
        ens = Ensemble(formation)
        err = lambda t: 1e3 * float(np.max(log['pelvis_err'][int(np.argmin(abs(log['t'] - t)))]))
        lab = labels or (lambda t: f'{segment(t)}   |   largest pelvis error {err(t):.2f} mm')
        hold, banner = 0., None
        end = float(log['frames_t'][-1]) if len(log['frames_t']) else 0.
        if m and not m.get('completed', True):
            banner = f"fell at t = {end:.2f} s"
            hold = max(1.5, (length or end) - end)
        target = self.out / f'{name}.mp4'
        render(ens, log, target, title=title, orbit=orbit, labels=lab, hold=hold, banner=banner, width=size[0], height=size[1],
               camera=camera)
        if record:
            self.note(target, title, [path], m)
        print('wrote', target.name, flush=True)
        return target, path, m

    def note(self, target, title, sources, metrics=None, what=None, **extra):
        rel = [str(Path(p).relative_to(self.results)) for p in sources]
        entry = dict(file=target.name, title=title, rendered_from=rel, frames_sha256={r: frames_hash(p) for r, p in zip(rel, sources)})
        if what:
            entry['shows'] = what
        if metrics:
            entry['completed'] = metrics.get('completed')
            entry['failure'] = metrics.get('failure')
        entry.update(extra)
        path = self.out / 'manifest.json'
        disk = json.loads(path.read_text()) if path.exists() else []
        self.manifest = [e for e in disk if e['file'] != target.name] + [entry]
        self.write()

    def pair(self, formation, scenario, left, right, target, what, labels=None, orbit=0., camera=None):
        """Two runs next to each other; ``left`` and ``right`` are (policy, title)."""
        paths = [stem(self.results, formation, scenario, policy).with_suffix('.npz') for policy, _ in (left, right)]
        if not all(p.exists() for p in paths):
            print('skipped', target, '(logs missing)')
            return
        if not self.wanted(Path(target).stem, paths):
            return
        ends = []
        for p in paths:
            with np.load(p, allow_pickle=False) as z:
                ends.append(float(z['frames_t'][-1]))
        half = (960, 720)
        tmp = [f'_{side}_{Path(target).stem}' for side in ('left', 'right')]
        a = self.clip(formation, scenario, left[0], left[1], name=tmp[0], labels=labels, orbit=orbit, length=max(ends), size=half,
                      camera=camera, record=False)
        b = self.clip(formation, scenario, right[0], right[1], name=tmp[1], labels=labels, orbit=orbit, length=max(ends), size=half,
                      camera=camera, record=False)
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([exe, '-y', '-loglevel', 'error', '-i', str(a[0]), '-i', str(b[0]), '-filter_complex',
                        '[0:v][1:v]hstack=inputs=2', '-c:v', 'libx264', '-crf', '25', '-preset', 'medium',
                        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(self.out / target)], check=True)
        self.note(self.out / target, f'{left[1]}  |  {right[1]}', [a[1], b[1]], what=what,
                  left=dict(policy=left[0], completed=a[2].get('completed'), failure=a[2].get('failure')),
                  right=dict(policy=right[0], completed=b[2].get('completed'), failure=b[2].get('failure')))
        for name in tmp:
            (self.out / f'{name}.mp4').unlink(missing_ok=True)
        print('wrote', target, flush=True)

    def write(self):
        keep = sorted((e for e in self.manifest if (self.out / e['file']).exists()), key=lambda e: e['file'])
        tmp = self.out / f'manifest.json.{os.getpid()}.tmp'
        tmp.write_text(json.dumps(keep, indent=2) + '\n')
        os.replace(tmp, self.out / 'manifest.json')


def videos(results, out, only=None, skip_existing=False):
    v = Videos(results, out, only, skip_existing)
    names = {'circle': 'ring', 'semicircle': 'semicircle', 'line': 'line'}
    for formation in ('circle', 'semicircle', 'line'):
        v.clip(formation, 'nominal', 'ip', f'Five closed-chain humanoids, {names[formation]} dance: partitioned model, realistic MuJoCo plant',
               orbit=2.0 if formation == 'circle' else 0., name=f'{names[formation]}_dance')
    step_label = lambda t: 'every lift-off and landing switches the composer; the five robot modules are reused'
    for formation in ('circle', 'semicircle', 'line'):
        v.clip(formation, 'steps', 'ip', f'Travelling {names[formation]}: side steps together, realistic MuJoCo plant', labels=step_label,
               orbit=1.0 if formation == 'circle' else 0., name=f'{names[formation]}_steps')
    v.pair('circle', 'steps', ('decentralized', 'Per-robot models (links not modelled)'), ('ip', 'Composed partitioned model'),
           'ring_steps_per_robot_vs_composed.mp4', 'the travelling ring with five separate robot models and with the composed model',
           labels=step_label)
    v.write()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('what', nargs='*', default=['figure', 'videos'], choices=['figure', 'videos'])
    ap.add_argument('--results', default=str(ROOT / 'results'))
    ap.add_argument('--figures', default=str(ROOT / 'paper' / 'figures'))
    ap.add_argument('--videos', default=str(ROOT / 'videos'))
    ap.add_argument('--only', nargs='+', default=None, help='video names (file names without .mp4) to render')
    ap.add_argument('--skip-existing', action='store_true', help='keep a video rendered from the same frames')
    args = ap.parse_args(argv)
    if 'figure' in args.what:
        figure(args.results, args.figures)
    if 'videos' in args.what:
        videos(args.results, args.videos, args.only, args.skip_existing)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
