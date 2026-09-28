"""Require identical complete input/output bytes across block dimensions."""
import argparse
import json
from pathlib import Path

from ref.verification import compare_runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare_runs(json.loads(args.left.read_text()), json.loads(args.right.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    if not result['all_byte_identical']:
        raise SystemExit('Block dimensions produced different bytes; report retained')


if __name__ == '__main__':
    main()
