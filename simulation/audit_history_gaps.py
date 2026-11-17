"""Triage unknown exposure without converting a short gap to known rest."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path


def classify(gap, segments):
    a, b = (segments[gap[k]] for k in ('from_segment', 'to_segment'))
    duration = float(gap['duration_s'])
    actual = (datetime.fromisoformat(gap['end_naive']) -
              datetime.fromisoformat(gap['start_naive'])).total_seconds()
    if not math.isfinite(duration) or duration <= 0 or abs(actual-duration) > 2e-6:
        raise ValueError('Invalid gap duration or clock mismatch')
    if a['cell'] != gap['cell'] or b['cell'] != gap['cell']:
        raise ValueError('Cross-cell interval')
    same = a['source_file'] == b['source_file']
    category = ('within_file_record_loss' if same else
                'subsecond_file_transition' if duration <= 1 else
                'long_unobserved_exposure' if duration > 86400 else
                'interfile_unobserved_exposure')
    return {**gap, 'category': category, 'from_source': a['source_file'],
            'to_source': b['source_file'], 'from_role': a['kind'], 'to_role': b['kind'],
            'duration_s': duration, 'operation_resolved': False,
            'temperature_resolved': False, 'admitted_as_rest': False,
            'next_evidence': ('raw rows surrounding anomaly' if same else
                'linked-technique timestamps and sampling cadence' if duration <= 1 else
                'acquisition logs, control revisions, storage and temperature records')}


def audit(intervals_path, segments_path, output):
    paths = [Path(intervals_path), Path(segments_path)]
    before = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with paths[1].open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    segments = {r['segment_id']: r for r in rows}
    if len(segments) != len(rows):
        raise ValueError('Duplicate segment identity')
    with paths[0].open(encoding='utf-8-sig', newline='') as f:
        gaps = [classify(r, segments) for r in csv.DictReader(f) if r['kind'] == 'unobserved_gap']
    counts = Counter(r['category'] for r in gaps)
    report = {'status': 'gap_triage_completed', 'gap_count': len(gaps),
              'counts': dict(counts), 'duration_s_by_category': {
                  k: sum(r['duration_s'] for r in gaps if r['category'] == k) for k in counts},
              'resolved_operation_count': 0, 'full_history_replay_admissible': False,
              'source_sha256': before, 'scope': 'Evidence routing only; no exposure imputation',
              'AWS_started': False}
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in before.items()):
        raise ValueError('Source drift during audit')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    with (output/'gap_evidence_queue.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(gaps[0]) if gaps else ['category'])
        w.writeheader(); w.writerows(sorted(gaps, key=lambda r: -r['duration_s']))
    (output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--intervals', required=True); p.add_argument('--segments', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    print(json.dumps(audit(a.intervals, a.segments, a.output), indent=2))
