"""Vendor the canonical contract, recording its full Git commit for verification."""
import argparse
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--check', action='store_true')
args = parser.parse_args()
commit = subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip()
schema = json.loads((args.source / 'schema/run-timing.schema.json').read_text())
schema['x-source'] = {'repository': 'https://github.com/jackspooner/router-contracts', 'commit': commit}
data = json.dumps(schema, indent=2) + '\n'
target = Path(__file__).resolve().parents[1] / 'app/run_timing.schema.json'
if args.check:
    assert target.read_text() == data, 'Stale timing contract snapshot'
else:
    target.write_text(data)
