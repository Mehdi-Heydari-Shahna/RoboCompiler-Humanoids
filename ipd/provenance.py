"""Provenance of generated results.

Every result file of the workflows carries ``stamp()``: one SHA-256 over all source
files that can influence a number (``code_sha256``), the versions of the numerical
packages and the time of writing.  The report generators refuse to mix result files
with different code hashes (``workflows/report/make_tables.py``).
"""
from __future__ import annotations

import hashlib
import os
import platform
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Report generators, the renderer and the tests are left out: they format, draw or check numbers,
# they do not produce them.
CODE_DIRS = ('ipd', 'dancers', 'examples', 'vendor', 'workflows/dance', 'workflows/statewise', 'workflows/manipulator')
EXCLUDED = ('dancers/render.py',)


def code_files():
    files = []
    for d in CODE_DIRS:
        files += [p for p in sorted((ROOT / d).rglob('*.py'))
                  if '__pycache__' not in p.parts and p.relative_to(ROOT).as_posix() not in EXCLUDED]
    return files


def code_hash():
    """SHA-256 over the relative paths and contents of all result-producing sources."""
    h = hashlib.sha256()
    for p in code_files():
        h.update(str(p.relative_to(ROOT)).replace('\\', '/').encode())
        h.update(b'\0')
        h.update(p.read_bytes())
        h.update(b'\0')
    return h.hexdigest()


def versions():
    out = {'python': platform.python_version(), 'platform': platform.platform(),
           'blas_threads': os.environ.get('OMP_NUM_THREADS', 'default')}
    for name, module in (('numpy', 'numpy'), ('scipy', 'scipy'), ('mujoco', 'mujoco'), ('pinocchio', 'pinocchio'),
                         ('daqp', 'daqp'), ('python_flint', 'flint')):
        try:
            mod = __import__(module)
            out[name] = getattr(mod, '__version__', None)
            if out[name] is None:
                from importlib.metadata import version
                out[name] = version(name.replace('_', '-'))
        except Exception:                      # the package is optional for the calling workflow
            out[name] = None
    return out


def stamp():
    return dict(code_sha256=code_hash(), versions=versions(), written_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
