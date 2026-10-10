"""Check the shipped evidence against the shipped code (standard library only).

    python verify_evidence.py [--evidence DIR]

1. every file listed in ``evidence/MANIFEST_SHA256.json`` exists and has the recorded
   SHA-256, and no other JSON file lies in the folder;
2. every evidence file carries the code hash recorded in the manifest;
3. that hash equals the hash of the result-producing sources of this checkout
   (``ipd/provenance.py``): the evidence was produced by exactly this code.

The exit status is non-zero when a check fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ipd.provenance import code_files, code_hash          # noqa: E402  (standard library only)


def verify(evidence):
    evidence = Path(evidence)
    problems = []
    manifest_path = evidence / 'MANIFEST_SHA256.json'
    if not manifest_path.exists():
        return [f'missing {manifest_path}'], {}
    manifest = json.loads(manifest_path.read_text())
    recorded, files = manifest.get('code_sha256'), manifest.get('files', {})
    present = {p.relative_to(evidence).as_posix() for p in evidence.rglob('*.json')} - {'MANIFEST_SHA256.json'}
    for rel in sorted(set(files) - present):
        problems.append(f'listed but missing: {rel}')
    for rel in sorted(present - set(files)):
        problems.append(f'not listed in the manifest: {rel}')
    for rel in sorted(set(files) & present):
        data = (evidence / rel).read_bytes()
        if hashlib.sha256(data).hexdigest() != files[rel]:
            problems.append(f'content differs from the manifest: {rel}')
            continue
        r = json.loads(data)
        prov = r.get('provenance') or r.get('_provenance') or {}
        if prov.get('code_sha256') != recorded:
            problems.append(f'produced by other code ({str(prov.get("code_sha256"))[:12]}): {rel}')
    current = code_hash()
    if current != recorded:
        problems.append(f'the sources of this checkout hash to {current[:12]}, the evidence was produced by {str(recorded)[:12]}')
    return problems, dict(files=len(files), code_sha256=recorded, source_files=len(code_files()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--evidence', default=str(ROOT / 'evidence'))
    args = ap.parse_args(argv)
    problems, info = verify(args.evidence)
    for line in problems:
        print('PROBLEM', line)
    if not problems:
        print(f"evidence verified: {info['files']} files, produced by code {info['code_sha256'][:12]} "
              f"({info['source_files']} source files of this checkout)")
    return 1 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main())
