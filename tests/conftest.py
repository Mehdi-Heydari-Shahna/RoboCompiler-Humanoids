"""Make the package importable when the tests are run from the package directory."""
import os
import sys
from pathlib import Path

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')         # small dense problems: BLAS threads only add contention

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'workflows' / 'statewise', ROOT / 'workflows' / 'dance', ROOT / 'workflows' / 'manipulator'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
