#!/usr/bin/env python3
"""Validate and pool all 10,030 LIBERO-Plus instances and seven official categories."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "libero"))
from summarize_results import collect_results, full_task_set, identity_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--classification', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    classification = json.loads(args.classification.read_text())
    expected = {}
    for suite, entries in classification.items():
        for row in entries:
            key = (suite, int(row['id']) - 1)
            if key in expected:
                raise ValueError(f'Duplicate classification instance: {key}')
            expected[key] = row['category']
    if set(expected) != full_task_set('libero_plus'):
        raise ValueError('Classification must cover exactly the 10,030 benchmark instances')
    rows, missing = collect_results(args.results, benchmark='libero_plus', expected=set(expected),
                                   num_trials=1, allow_partial=args.allow_partial)
    categories = defaultdict(lambda: {'successes': 0, 'trials': 0})
    for key, row in rows.items():
        category = categories[expected[key]]
        category['successes'] += row['successes']
        category['trials'] += 1
    for category in categories.values():
        category['success_rate_percent'] = 100 * category['successes'] / category['trials']
    successes = sum(row['successes'] for row in rows.values())
    payload = {'complete': not missing, 'formal': not missing, 'full_benchmark': True,
               'expected_instances': len(expected), 'completed_instances': len(rows),
               'aggregation': 'instance_pooled',
               'instruction_protocol': next(iter(rows.values())).get('instruction_protocol', 'official'),
               **identity_summary(rows),
               'overall': {'successes': successes, 'trials': len(rows),
                           'success_rate_percent': 100 * successes / len(rows)},
               'categories': dict(sorted(categories.items()))}
    output = args.output or args.results / 'plus_summary.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name('.' + output.name + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(output)
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
