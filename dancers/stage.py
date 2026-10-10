"""The stage: floor contact data for the simulator and the look of the scene for rendering.

Physics.  Rubber soles on a wooden floor: Coulomb friction 0.8 and torsional friction
0.01 m (contact dimension 4; the rolling coefficient of the triple is inactive), a contact
time constant of 5 ms with critical damping and MuJoCo's default impedance profile,
elliptic friction cones.  The controller assumes the same friction coefficient and keeps
the support wrenches inside a tightened cone.

Look (rendering only; no contact, no mass).  A polished oak dance floor (procedural plank
texture) on a stage enclosed by pleated velvet curtains, a key spotlight that casts
shadows, fill, top and rim lights, and fog in the depth of the hall.
"""
from __future__ import annotations

import io

import numpy as np

FLOOR = dict(friction=0.8, torsional=0.01, rolling=1e-4, solref=(0.005, 1.0), solimp=(0.9, 0.95, 0.001, 0.5, 2.0), condim=4)


def wood_texture(px=2048, extent_m=2.4, plank_m=0.14, seed=7):
    """PNG bytes of a square oak-plank texture that tiles every ``extent_m`` metres."""
    from PIL import Image
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:px, 0:px].astype(np.float32) / px * extent_m       # metres; planks run along x
    n_planks = int(round(extent_m / plank_m))
    plank_m = extent_m / n_planks
    row = np.minimum((y / plank_m).astype(int), n_planks - 1)
    img = np.zeros((px, px, 3), np.float32)
    base = np.array([0.58, 0.40, 0.25], np.float32)
    for r in range(n_planks):
        m = row == r
        # two joints per row at random places: planks of random length, staggered between rows
        cuts = np.sort(rng.uniform(0, extent_m, 2))
        edges = np.r_[0., cuts, extent_m]
        for a, b in zip(edges[:-1], edges[1:]):
            piece = m & (x >= a) & (x < b)
            if not piece.any():
                continue
            tone = base * rng.uniform(0.94, 1.06) * np.array([1., rng.uniform(0.97, 1.03), rng.uniform(0.94, 1.06)], np.float32)
            yy, xx = y[piece] - r * plank_m, x[piece] - a
            ph = rng.uniform(0, 2 * np.pi, 3)
            wob = 0.006 * np.sin(2 * np.pi * xx / rng.uniform(0.4, 1.2) + ph[0]) + 0.002 * np.sin(2 * np.pi * xx / 0.13 + ph[1])
            fine = 0.5 + 0.5 * np.sin(2 * np.pi * (yy + wob) / rng.uniform(0.0025, 0.0045) + ph[2])
            figure = 0.5 + 0.5 * np.sin(2 * np.pi * (yy + 2.5 * wob) / rng.uniform(0.02, 0.05))
            shade = 0.88 + 0.06 * fine ** 2 + 0.06 * figure
            img[piece] = tone * shade[:, None]
            img[piece & (np.abs(x - a) < 0.0012)] *= 0.6          # butt joint between two pieces
    img[(y % plank_m) < 0.0010] *= 0.5                         # seams between rows
    img *= (1. + 0.02 * rng.standard_normal(img.shape[:2]).astype(np.float32))[:, :, None]
    out = io.BytesIO()
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(out, format='PNG')
    return out.getvalue()


class Studio:
    """Scene description consumed by ``ipd.backends.mujoco_backend.export_mjcf``."""

    floor = FLOOR
    floor_size = 9.0               # half extent of the visible floor (m)
    texture_extent = 2.4
    curtain_radius, curtain_height, curtain_pleats = 6.4, 4.2, 72

    def __init__(self, render=True):
        self.render = render
        self.assets = {'oak.png': wood_texture(extent_m=self.texture_extent)} if render else {}

    def visual(self, vis, offscreen):
        import xml.etree.ElementTree as ET
        ET.SubElement(vis, 'global', offwidth=str(offscreen[0]), offheight=str(offscreen[1]), fovy='38')
        ET.SubElement(vis, 'quality', shadowsize='4096', offsamples='8')
        ET.SubElement(vis, 'headlight', ambient='0.16 0.16 0.18', diffuse='0.22 0.22 0.24', specular='0.04 0.04 0.04')
        ET.SubElement(vis, 'map', znear='0.02', zfar='60', fogstart='6', fogend='16', shadowclip='3.5', shadowscale='1.0')
        ET.SubElement(vis, 'rgba', fog='0.035 0.038 0.048 1', haze='0.06 0.065 0.08 1')

    def assets_xml(self, asset):
        import xml.etree.ElementTree as ET
        if self.render:
            ET.SubElement(asset, 'texture', name='oak', type='2d', file='oak.png')
            ET.SubElement(asset, 'material', name='floor', texture='oak', texrepeat=f'{1. / self.texture_extent:.6g} {1. / self.texture_extent:.6g}',
                          texuniform='true', specular='0.45', shininess='0.5', reflectance='0.12')
            ET.SubElement(asset, 'material', name='curtain', rgba='0.36 0.045 0.06 1', specular='0.08', shininess='0.15')
        else:
            ET.SubElement(asset, 'material', name='floor', rgba='0.62 0.46 0.30 1')
        ET.SubElement(asset, 'texture', name='sky', type='skybox', builtin='gradient', rgb1='0.10 0.11 0.14',
                      rgb2='0.015 0.016 0.02', width='256', height='256')

    def world(self, world):
        import xml.etree.ElementTree as ET
        f = self.floor
        ET.SubElement(world, 'geom', name='floor', type='plane', size=f'{self.floor_size} {self.floor_size} 0.1', material='floor',
                      contype='1', conaffinity='1', condim=str(f['condim']),
                      friction=f"{f['friction']} {f['torsional']} {f['rolling']}",
                      solref=' '.join(map(str, f['solref'])), solimp=' '.join(map(str, f['solimp'])))
        # key light (shadows), two fills and a rim light behind the dancers
        ET.SubElement(world, 'light', name='key', pos='2.6 -3.4 5.8', dir='-0.40 0.52 -0.75', diffuse='0.80 0.76 0.70',
                      specular='0.35 0.33 0.30', cutoff='34', exponent='4', castshadow='true')
        ET.SubElement(world, 'light', name='fill', pos='-3.6 2.8 4.6', dir='0.55 -0.43 -0.70', diffuse='0.28 0.31 0.38',
                      specular='0.05 0.05 0.06', cutoff='60', exponent='3', castshadow='false')
        ET.SubElement(world, 'light', name='top', pos='0 0 6.5', dir='0 0 -1', diffuse='0.30 0.30 0.30', specular='0 0 0',
                      cutoff='55', exponent='2', castshadow='false')
        ET.SubElement(world, 'light', name='rim', pos='-1.0 4.5 3.0', dir='0.15 -0.85 -0.45', diffuse='0.22 0.22 0.26',
                      specular='0.20 0.20 0.22', cutoff='50', exponent='4', castshadow='false')
        if self.render:
            self.curtains(world)

    def curtains(self, world):
        """A ring of pleated curtains around the stage: thin panels that alternate in angle."""
        import xml.etree.ElementTree as ET
        n, R, H = self.curtain_pleats, self.curtain_radius, self.curtain_height
        w = 2. * np.pi * R / n
        for i in range(n):
            a = 2. * np.pi * i / n
            yaw = a + np.pi / 2 + (0.32 if i % 2 else -0.32)          # panel along the circle, folded in and out
            c, s = np.cos(yaw / 2), np.sin(yaw / 2)
            ET.SubElement(world, 'geom', name=f'curtain{i}', type='box', size=f'{0.56 * w:.4f} 0.01 {H / 2:.3f}',
                          pos=f'{R * np.cos(a):.4f} {R * np.sin(a):.4f} {H / 2:.3f}', quat=f'{c:.6f} 0 0 {s:.6f}',
                          material='curtain', contype='0', conaffinity='0', group='2')
