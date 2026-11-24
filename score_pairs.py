"""Recompute Core100 pair scores from frozen endpoint labels."""
import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def score(records):
    scenes = {f'T{i:02}': i - 1 for i in range(1, 21)}
    groups = defaultdict(list)
    for row in records:
        groups[row['model'], row['prompt']].append(row)
    expected_roster = None
    for (model, prompt), rows in sorted(groups.items()):
        roster = Counter()
        counts = np.zeros((20, 3))
        for row in rows:
            pair = tuple(row['pair'])
            weight = row['weight']
            if weight <= 0 or len(row['gt']) != 2 or len(row['labels']) != 2:
                raise ValueError(f'Malformed pair event: {row}')
            roster[pair] += weight
            decided = all(label in (0, 1) for label in row['labels'])
            correct = decided and row['labels'] == row['gt']
            counts[scenes[row['scene']]] += weight * np.array([correct, decided, 1])
        if len(roster) != 51 or not np.allclose(list(roster.values()), 1):
            raise ValueError(f'Invalid fixed-51 roster: {(model, prompt)}')
        if expected_roster is None:
            expected_roster = set(roster)
        elif set(roster) != expected_roster:
            raise ValueError(f'Video pairs differ across groups: {(model, prompt)}')
        total = counts.sum(axis=0)
        for metric, i, j in [('PairAcc', 0, 1), ('Coverage', 1, 2), ('VPSR', 0, 2)]:
            yield dict(model=model, prompt=prompt, metric=metric, numerator=total[i],
                       denominator=total[j], roster_pairs=51,
                       score_pct=100 * total[i] / total[j] if total[j] else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected', type=Path)
    args = parser.parse_args()
    with gzip.open(args.input, 'rt', encoding='utf-8') as stream:
        results = list(score(map(json.loads, stream)))
    if args.expected:
        with args.expected.open(encoding='utf-8-sig', newline='') as stream:
            expected = {(r['model'], r['prompt'], r['metric']): r for r in csv.DictReader(stream)}
        if len(expected) != len(results):
            raise ValueError('Unexpected number of result rows')
        for r in results:
            row = expected[r['model'], r['prompt'], r['metric']]
            for key, value in list(r.items())[3:]:
                fits = not row[key] if value is None else np.isclose(value, float(row[key]), atol=1e-10, rtol=0)
                if not fits:
                    raise ValueError(f'Mismatch for {key}: {r} versus {row}')
        print(f'Verified {len(results)} metric rows')
    with args.output.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


if __name__ == '__main__':
    main()
