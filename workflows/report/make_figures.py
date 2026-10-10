"""Figures of the paper from saved run data (state-wise consistency, closed-loop dances, method diagram).

Run after the studies:  python workflows/report/make_figures.py [--results results]
Inputs are the JSON summaries and compressed logs under ``results/``; nothing is simulated
here.  Every figure is written as PDF and PNG to ``--out`` (default ``paper/figures``).
The renderings (teaser, snapshots) are made by ``render_media.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
INK = '#1F2D3A'; BLUE = '#1764AB'; ORANGE = '#D9822B'; GREEN = '#2A8C6F'; RED = '#B8383B'; PURPLE = '#7B5BA6'
GREY = '#8A99A5'; GOLD = '#B99A2E'; TEAL = '#2E8FA3'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.titlesize': 8, 'axes.titleweight': 'bold',
                     'axes.labelsize': 7.5, 'xtick.labelsize': 6.8, 'ytick.labelsize': 6.8, 'legend.fontsize': 6.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#9AA8B3',
                     'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': INK, 'ytick.color': INK,
                     'grid.color': '#E3E8EC', 'grid.linewidth': .5, 'axes.linewidth': .6, 'pdf.fonttype': 42,
                     'ps.fonttype': 42, 'savefig.facecolor': 'white'})

LABEL = {'ip': 'Proposed', 'recursive': 'Projected tree', 'decentralized': 'Per-robot models', 'no_curvature': 'No curvature',
         'simplified': 'No actuator inertia', 'massless': 'Massless actuators', 'gravity_pd': 'Gravity + PD',
         'actuator_pd': 'Actuator PD'}
COLOR = {'ip': BLUE, 'recursive': INK, 'decentralized': GREEN, 'no_curvature': PURPLE, 'simplified': ORANGE,
         'massless': GOLD, 'gravity_pd': RED, 'actuator_pd': GREY}
SEGMENTS = (('settle', 0., 1.5), ('bounce', 1.5, 7.5), ('hand wave', 7.5, 13.5), ('lean', 13.5, 19.5),
            ('sway, twist', 19.5, 25.5), ('bow', 25.5, 30.))
CASES = (('manipulator', 'Manipulator, actuator space'), ('manipulator_joint', 'Manipulator, joint space'),   # as in Table I
         ('excavator', 'Excavator, joint space'), ('excavator_cylinder', 'Excavator, cylinder space'),
         ('dancer_floating', 'Humanoid, no support'), ('dancer_single', 'Humanoid, one sole'),
         ('dancer_double', 'Humanoid, two soles'), ('circle', 'Ring of five, all soles'), ('circle_single', 'Ring, left soles'),
         ('circle_actuator', 'Ring, actuator space'), ('semicircle', 'Semicircle of five'), ('line', 'Line of five'))
ROUTES = (('ip_two_level', 'Proposed', BLUE, 'o', 4.2), ('maximal', 'Maximal coordinates', TEAL, 'D', 3.2),
          ('pinocchio_kkt', 'Pinocchio KKT', INK, 's', 3.2), ('projected_fd', 'Projected, FD curvature', GREY, '^', 3.4),
          ('mujoco', 'MuJoCo (regularised)', GOLD, 'v', 3.4), ('no_curvature', 'No curvature', PURPLE, 'P', 3.6),
          ('simplified', 'No actuator inertia', ORANGE, 'X', 3.6), ('decentralized', 'Per-robot models', GREEN, '*', 4.4))


class Inputs:
    def __init__(self, results):
        self.root = Path(results)

    def json(self, rel):
        r = json.loads((self.root / rel).read_text())
        return {k: v for k, v in r.items() if not k.startswith('_')} if isinstance(r, dict) else r

    def npz(self, rel):
        with np.load(self.root / rel, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def has(self, rel):
        return (self.root / rel).exists()

    def run(self, formation, scenario, plant, policy):
        stem = f'dance/{formation}/{scenario}__{plant}__{policy}'
        return self.json(stem + '.json'), self.npz(stem + '.npz')


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'{name}.{ext}', dpi=300, bbox_inches='tight', pad_inches=.03)
    plt.close(fig)


def moving_rms(x, n):
    k = np.ones(n) / n
    return np.sqrt(np.convolve(x ** 2, k, mode='same'))


def moving_mean(x, n):
    k = np.ones(n) / n
    return np.convolve(x, k, mode='same')


def shade(ax, y=1.0, labels=True):
    for i, (name, a, b) in enumerate(SEGMENTS):
        if i % 2:
            ax.axvspan(a, b, color='#EEF2F5', lw=0, zorder=0)
        if labels:
            ax.text(.5 * (a + b), y, name, transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=5.6, color='#5D6D7A')


# ---------------------------------------------------------------------- method diagram
def fig_method(inp, out):
    fig, ax = plt.subplots(figsize=(7.16, 1.99))
    ax.set_xlim(0, 100); ax.set_ylim(-0.8, 27); ax.axis('off')

    def box(x, y, w, h, title, body, face, edge='#51657A', tsize=6.8, bsize=5.9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.2,rounding_size=0.9', fc=face, ec=edge, lw=.75))
        ax.text(x + w / 2, y + h - 1.0, title, ha='center', va='top', fontsize=tsize, fontweight='bold')
        ax.text(x + w / 2, y + h - 4.0, body, ha='center', va='top', fontsize=bsize, linespacing=1.32)

    def arrow(a, b, color=INK):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=7, lw=.8, color=color, shrinkA=0, shrinkB=0))

    # the layer added by this paper: everything between the compiled lift and the generated views
    ax.add_patch(FancyBboxPatch((34.0, -0.3), 45.2, 24.9, boxstyle='round,pad=0.25,rounding_size=1.2', fc='none',
                                ec=GREEN, lw=.8, ls=(0, (3, 2))))
    ax.text(56.6, 25.0, 'inertia-partitioned layer (this paper)', ha='center', va='bottom', fontsize=6.4, color=GREEN,
            fontweight='bold')
    box(0.4, 3.5, 14.2, 17, 'Record', 'bodies, joints,\nactuator ports,\ndrive data,\nsole and palm\nframes; no cuts,\nno closure maps', '#F3F6F9')
    box(17.0, 3.5, 15.6, 17, 'RoboCompiler', 'tree and chords,\npath-assembled\ndifferential $\\mathcal{C}_e$,\nrank-checked lift\n$v=E\\,u$; projected\ntree model with\nFD curvature', '#E4EEF6')
    box(35.0, 12.6, 22.6, 11.2, 'Robot modules', 'once per robot, analytic $\\dot E$:\nbodies $J_i^{\\!\\top}\\!M_iJ_i$, actuator parts,\nrotors $E_T^{\\!\\top}\\mathcal{A}E_T$  $\\Rightarrow\\ \\Gamma^1_k,\\ C^1_k,\\ g^1_k$', '#E2F2EC', bsize=5.7)
    box(35.0, 0.2, 22.6, 11.2, 'Composer', 'one per mode (supports and\nhand links): lift $G$, rate $\\dot G$,\n$\\Gamma=\\Sigma_k\\,G_k^{\\!\\top}\\Gamma^1_kG_k$', '#E2F2EC', bsize=5.7)
    box(60.2, 12.8, 18.3, 10.8, 'Local edits', 'payload: rank-6 update\nof one member;\nmode switch: new $G$', '#FBF0E1')
    box(60.2, 0.4, 18.3, 10.8, 'Wrenches', '$\\lambda=\\lambda_0-\\Pi f$ for\nsupports and links,\naffine in the efforts', '#FBF0E1')
    box(81.0, 3.5, 18.6, 17, 'Generated views', 'partitioned model\n$\\Gamma\\dot u+Cu+g+Du$\n$=L^{\\!\\top}\\!f$,\nMuJoCo model,\nPinocchio model', '#F3F6F9')
    arrow((14.8, 12.0), (16.8, 12.0)); arrow((32.8, 14.5), (34.8, 17.0)); arrow((32.8, 9.5), (34.8, 6.5))
    arrow((46.3, 12.6), (46.3, 11.4), GREEN)
    arrow((57.8, 18.2), (60.0, 18.2), ORANGE); arrow((57.8, 5.8), (60.0, 5.8), ORANGE)
    arrow((78.7, 18.2), (80.8, 15.0)); arrow((78.7, 5.8), (80.8, 9.0))
    save(fig, out, 'fig_method')


# ---------------------------------------------------------------------- state-wise consistency
def fig_consistency(inp, out):
    cons = inp.json('statewise/consistency.json')
    cases = [(k, l) for k, l in CASES if k in cons]
    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    y = np.arange(len(cases))[::-1]
    for route, label, color, marker, size in ROUTES:
        xs, ys = [], []
        for yy, (k, _) in zip(y, cases):
            c = cons[k]
            if route == 'maximal':
                v = c['maximal_coordinates']['delta_body']['max'] if c.get('maximal_coordinates') else None
            else:
                v = c['routes'].get(route, {}).get('delta_a', {}).get('max')
            if v is not None:
                xs.append(max(v, 1e-17)); ys.append(yy)
        if route == 'ip_two_level':                  # open circle on top: the routes that agree with it stay visible
            ax.scatter(xs, ys, s=(size + 1.6) ** 2, marker='o', facecolors='none', edgecolors=color, lw=1.1, label=label, zorder=5)
        else:
            ax.scatter(xs, ys, s=size ** 2, marker=marker, color=color, label=label, zorder=3, lw=0.3, edgecolors='white')
    ax.set_xscale('log')
    ax.set_xlim(1e-15, 2e2)
    ax.set_yticks(y)
    ax.set_yticklabels([l for _, l in cases])
    ax.axvspan(1e-15, 1e-7, color='#EAF3FA', lw=0, zorder=0)
    ax.text(3e-15, len(cases) - 0.45, 'numerical agreement', fontsize=5.8, color='#4F6F8F', va='bottom')
    ax.set_xlabel('normalised acceleration discrepancy $\\delta_a$ (max over states)')
    ax.grid(axis='x', which='major')
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(-0.6, len(cases) - 0.1)
    ax.legend(loc='upper center', bbox_to_anchor=(0.42, -0.2), ncol=3, frameon=False, handletextpad=.2, columnspacing=.8)
    save(fig, out, 'fig_consistency')


# ---------------------------------------------------------------------- closed-loop dances
def fig_dance(inp, out, link_window=(8.0, 13.0)):
    fig = plt.figure(figsize=(7.16, 2.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.3, 1.2, 1.5], wspace=0.36)
    ax = fig.add_subplot(gs[0])
    for i, (name, a, b) in enumerate(SEGMENTS):
        if i % 2:
            ax.axvspan(a, b, color='#EEF2F5', lw=0, zorder=0)
        if name != 'settle':
            ax.text(.5 * (a + b), 1.0, name, transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=5.4,
                    color='#5D6D7A')
    order = ('actuator_pd', 'gravity_pd', 'simplified', 'no_curvature', 'ip')
    for p in order:
        rel = f'dance/circle/nominal__mujoco__{p}.npz'
        if not inp.has(rel):
            continue
        z = inp.npz(rel)
        e = 1e3 * np.linalg.norm(z['palm'] - z['palm_ref'], axis=2).max(axis=1)
        ax.plot(z['t'], moving_rms(e, 20), color=COLOR[p], lw=.8 if p != 'ip' else 1.05, label=LABEL[p], zorder=3 if p == 'ip' else 2)
    ax.set_yscale('log')
    ax.set_xlim(0, 30)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('palm position error (mm)')
    ax.set_title('(a) Ring, realistic plant', loc='left', pad=9)
    ax.legend(loc='upper left', ncol=3, frameon=False, fontsize=5.3, handlelength=1.2, columnspacing=.6, bbox_to_anchor=(0., 0.99))
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 6)
    # (b) RMS pelvis error per policy and formation
    bx = fig.add_subplot(gs[1])
    pols = ('ip', 'decentralized', 'no_curvature', 'simplified', 'massless', 'gravity_pd', 'actuator_pd')
    groups = (('circle', 'nominal', 'ring'), ('semicircle', 'nominal', 'semi-\ncircle'), ('line', 'nominal', 'line'),
              ('circle', 'ideal', 'ring,\nideal'))
    w = 0.11
    fell = []
    for j, p in enumerate(pols):
        vals = []
        for gi, (f, sc, _) in enumerate(groups):
            rel = f'dance/{f}/{sc}__mujoco__{p}.json'
            m = inp.json(rel)['metrics'] if inp.has(rel) else None
            if m is not None and not m['completed']:
                fell.append((gi + (j - (len(pols) - 1) / 2) * w, COLOR[p]))
                m = None
            vals.append(m['pelvis_rms_mm'] if m else np.nan)
        x = np.arange(len(groups)) + (j - (len(pols) - 1) / 2) * w
        bx.bar(x, vals, width=w, color=COLOR[p], label=LABEL[p], lw=0)
    bx.set_yscale('log')
    lo, hi = bx.get_ylim()
    for x, c in fell:                                      # a robot fell: no RMS value, a cross at the top
        bx.plot([x], [hi * 0.75], marker='x', color=c, ms=4, mew=1.2, clip_on=False)
    bx.set_ylim(lo, hi)
    bx.set_xticks(np.arange(len(groups)))
    bx.set_xticklabels([g[2] for g in groups], fontsize=6.2)
    bx.set_ylabel('RMS pelvis error (mm)')
    bx.set_title('(b) All formations', loc='left', pad=9)
    bx.grid(axis='y', which='major')
    # (c) predicted and measured force in one hand link
    cx = fig.add_subplot(gs[2])
    rel = 'dance/circle/nominal__mujoco__ip.npz'
    if inp.has(rel):
        z = inp.npz(rel)
        t = z['t']
        m = (t >= link_window[0]) & (t <= link_window[1])
        k = int(np.nanargmax(np.nanmean(z['link_force'][m], axis=0)))
        meas, pred = z['link_force'][:, k], z['link_force_model'][:, k]
        cx.plot(t[m], meas[m], color='#E7B4B5', lw=.5, zorder=1)
        cx.plot(t[m], pred[m], color='#A9C6E3', lw=.5, zorder=1)
        cx.plot(t[m], moving_mean(meas, 10)[m], color=RED, lw=1.0, ls='--', label='MuJoCo constraint force', zorder=3)
        cx.plot(t[m], moving_mean(pred, 10)[m], color=BLUE, lw=1.0, label='composed model (Prop. 4)', zorder=2)
        cx.set_xlim(*link_window)
        cx.set_xlabel('time (s)')
        cx.set_ylabel('hand-link force (N)')
        cx.legend(loc='upper left', frameon=False, fontsize=5.6, handlelength=1.6)
        cx.set_ylim(0, np.nanmax(meas[m]) * 1.35)
    cx.set_title('(c) Force between two robots', loc='left', pad=9)
    save(fig, out, 'fig_dance')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default=str(ROOT / 'results'))
    ap.add_argument('--out', default=str(ROOT / 'paper' / 'figures'))
    ap.add_argument('figures', nargs='*', default=['method', 'consistency', 'dance'])
    args = ap.parse_args(argv)
    inp, out = Inputs(args.results), Path(args.out)
    makers = dict(method=fig_method, consistency=fig_consistency, dance=fig_dance)
    rc = 0
    for name in args.figures:
        try:
            makers[name](inp, out)
            print('figure', name, 'written')
        except FileNotFoundError as exc:
            print('figure', name, 'skipped: missing', exc.filename)
            rc = 1
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
