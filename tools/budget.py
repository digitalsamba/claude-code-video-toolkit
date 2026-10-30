#!/usr/bin/env python3
"""Inspect project cost reservations and reconcile them with provider charges."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from production import BuildError, atomic_write, build_lock, cost_forecast, cost_history, load_state, project_path, valid_money


def main(argv=None) -> int:
    from win_encoding import ensure_utf8_output
    ensure_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--attempt', help='Attempt ID to reconcile')
    parser.add_argument('--actual-usd', type=float, help='Provider-reported total for that attempt, including zero')
    parser.add_argument('--note', default='', help='Invoice reference or explanation, without credentials')
    args = parser.parse_args(argv)
    try:
        if not args.project.is_dir():
            raise BuildError('Project directory does not exist')
        if (args.attempt is None) != (args.actual_usd is None):
            raise BuildError('--attempt and --actual-usd must be provided together')
        if args.attempt is not None:
            amount = valid_money(args.actual_usd, 'actual-usd')
            with build_lock(args.project):
                state = load_state(args.project)
                history = cost_history(state)
                entry = next((item for item in history if item['id'] == args.attempt), None)
                if entry is None:
                    raise BuildError('Unknown cost attempt ID')
                entry['actualCostUsd'] = amount
                entry['reconciliationNote'] = args.note
                atomic_write(project_path(args.project, '.video-toolkit/build-state.json'), json.dumps(state, indent=2) + '\n')
        state = load_state(args.project)
        print(json.dumps({'costs': cost_history(state), 'summary': cost_forecast([], state, None)}, indent=2))
        return 0
    except (BuildError, OSError) as error:
        print(json.dumps({'error': str(error)}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
