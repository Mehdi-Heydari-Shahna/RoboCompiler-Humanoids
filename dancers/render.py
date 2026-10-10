"""Replay recorded configurations of a run through the MuJoCo model and encode a video."""
from __future__ import annotations

import os

import numpy as np

CAMERAS = {
    'circle': dict(lookat=(0., 0., 0.78), distance=4.3, azimuth=58., elevation=-24.),
    'semicircle': dict(lookat=(0.9, 0., 0.78), distance=5.0, azimuth=180., elevation=-20.),
    'line': dict(lookat=(0., 0., 0.78), distance=5.4, azimuth=180., elevation=-14.),
}


def render(ens, log, path, title='', fps=30, width=1280, height=800, camera=None, orbit=0., labels=None, q_template=None,
           hold=0., banner=None, crf=27, reflection=False):
    """Write an H.264 video of the frames stored in ``log`` (``frames_T_root``, ``frames_theta``).

    ``hold`` seconds of the last frame are appended with ``banner`` drawn over it (used for
    runs that stopped early, so that the two halves of a comparison have the same length).
    The floor reflection is off by default: with software rendering it costs more than the
    rest of the frame.
    """
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import imageio
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    from ipd.backends.mujoco_backend import MujocoPlant
    from ipd.system import Config
    from .stage import Studio
    s = ens.system
    q = q_template.copy() if q_template is not None else s.neutral()
    plant = MujocoPlant(s, q=q, offscreen=(width, height), scene=Studio(render=True))
    renderer = mujoco.Renderer(plant.model, height, width)
    cam = mujoco.MjvCamera()
    c = dict(CAMERAS[ens.formation], **(camera or {}))
    cam.lookat[:] = c['lookat']; cam.distance = c['distance']; cam.azimuth = c['azimuth']; cam.elevation = c['elevation']
    opt = mujoco.MjvOption()
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 22)
        small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
    except OSError:
        font = small = ImageFont.load_default()
    writer = imageio.get_writer(str(path), fps=fps, codec='libx264', quality=None, macro_block_size=None, pixelformat='yuv420p',
                                ffmpeg_params=['-crf', str(crf), '-preset', 'medium', '-movflags', '+faststart'])
    theta = np.zeros(s.n)
    T, TH, tt = log['frames_T_root'], log['frames_theta'], log['frames_t']
    for i in range(len(tt)):
        theta[:s.nT] = TH[i]
        plant.set_state(Config(T[i], theta))
        cam.azimuth = c['azimuth'] + orbit * tt[i]
        renderer.update_scene(plant.data, cam, opt)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = int(bool(reflection))
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
        img = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, width, 44], fill=(24, 40, 56))
        draw.text((16, 9), title, fill=(255, 255, 255), font=font)
        draw.text((width - 150, 12), f't = {tt[i]:5.2f} s', fill=(255, 255, 255), font=small)
        if labels:
            lab = labels(tt[i])
            if lab:
                draw.rectangle([0, height - 36, width, height], fill=(244, 247, 250))
                draw.text((16, height - 29), lab, fill=(32, 59, 80), font=small)
        writer.append_data(np.asarray(img))
    if hold > 0. and len(tt):
        if banner:
            draw = ImageDraw.Draw(img)
            y0 = height - 36 - 52                      # just above the label bar, so that the scene stays visible
            draw.rectangle([0, y0, width, y0 + 44], fill=(150, 30, 30))
            draw.text((16, y0 + 9), banner, fill=(255, 255, 255), font=font)
        last = np.asarray(img)
        for _ in range(int(round(hold * fps))):
            writer.append_data(last)
    writer.close()
    renderer.close()
    return str(path)


def snapshot(ens, q, path, width=1280, height=800, camera=None, reflection=True):
    """Render one configuration to a PNG."""
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import mujoco
    from PIL import Image
    from ipd.backends.mujoco_backend import MujocoPlant
    from .stage import Studio
    plant = MujocoPlant(ens.system, q=q, offscreen=(width, height), scene=Studio(render=True))
    plant.set_state(q)
    renderer = mujoco.Renderer(plant.model, height, width)
    cam = mujoco.MjvCamera()
    c = dict(CAMERAS[ens.formation], **(camera or {}))
    cam.lookat[:] = c['lookat']; cam.distance = c['distance']; cam.azimuth = c['azimuth']; cam.elevation = c['elevation']
    renderer.update_scene(plant.data, cam)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = int(bool(reflection))
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
    Image.fromarray(renderer.render()).save(str(path))
    renderer.close()
    return str(path)
