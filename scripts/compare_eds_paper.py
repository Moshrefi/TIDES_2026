#!/usr/bin/env python
"""Compare measured EDS metrics with repository-transcribed TIDES paper references.

A numerical difference is diagnostic; this tool never certifies paper reproduction.
"""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import runpy

PAPER = runpy.run_path(str(Path(__file__).resolve().parents[1]/'tides'/'paper_numbers.py'))


def comparison_rows(results):
    if results.get('dataset') not in (None, 'eds'):
        raise ValueError('EDS comparison requires EDS results')
    rows = []
    methods = results.get('methods', {})
    def add(table, method, metric, reference):
        value = methods.get(method, {}).get(metric)
        valid = isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)
        rows.append(dict(table=table, method=method, metric=metric,
            ours=float(value) if valid else None, paper_eds_mean=reference,
            difference=float(value)-reference if valid else None,
            difference_percent=100*(float(value)-reference)/abs(reference) if valid and reference else None,
            status='measured' if valid else ('not run' if method not in methods else 'missing/nonfinite metric')))
    for method in PAPER['METHODS']:
        for metric, reference in zip(['ig_nll','chamfer'], PAPER['TABLE1'][method]['eds']):
            add('Table 1', method, metric, reference)
        for metric, reference in zip(['same_ts','fano','isi_spike'], PAPER['TABLE2'][method]['eds']):
            add('Table 2', method, metric, reference)
    for key, (_, ig, ch) in PAPER['TABLE4'].items():
        method = 'tides_adaptive' if key == 'full' else 'ablation_'+key
        for metric, reference in [('ig_nll',ig),('chamfer',ch)]:
            add('Table 4',method,metric,reference)
    return rows


def save_comparison(results, out):
    rows = comparison_rows(results)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (out/'paper_comparison.csv').write_text(stream.getvalue(), encoding='utf-8')
    lines = ['# EDS measured results versus paper reference', '',
        f"Sequence: {results.get('sequence', 'unknown')}", '',
        'Scope: available single-sequence metrics versus paper EDS dataset means.',
        'Paper reproduction: NOT ESTABLISHED. Sequence coverage and metric assumptions differ or remain unverified.',
        'References are transcribed in tides/paper_numbers.py. Missing values are not zeros.',
        'Difference = ours - paper; negative means numerically lower, not proven methodological superiority.', '',
        '| Table | Method | Metric | Ours | Paper EDS mean | Difference | Difference % | Status |',
        '|---|---|---|---|---|---|---|---|']
    def fmt(value):
        return '-' if value is None else f'{value:.8g}'
    for row in rows:
        lines.append('| '+' | '.join([row['table'],row['method'],row['metric'],fmt(row['ours']),
            fmt(row['paper_eds_mean']),fmt(row['difference']),fmt(row['difference_percent']),row['status']])+' |')
    (out/'paper_comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--metrics', required=True)
    ap.add_argument('--out', required=True, help='report output directory')
    args = ap.parse_args()
    results = json.loads(Path(args.metrics).read_text(encoding='utf-8'))
    save_comparison(results, args.out)
    print(f'Report saved in {args.out}; paper reproduction remains unvalidated.')


if __name__ == '__main__':
    main()
