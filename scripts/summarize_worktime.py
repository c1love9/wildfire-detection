"""Estimate review labor from a user-reported rate and immutable A/B timers."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
TASK = 'quality_6146eb5a5ab24f5ebc6961d07e392ad2'


def source_records(catalog, task_id):
    with closing(sqlite3.connect(catalog.resolve().as_uri()+'?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        records = [dict(row) for row in db.execute(
            'SELECT * FROM quality_submissions WHERE task_id=? ORDER BY asset_id,role', (task_id,))]
        approved = db.execute("""SELECT COUNT(*) FROM annotation_reviews r JOIN
            (SELECT asset_id,MAX(seq) seq FROM annotation_reviews GROUP BY asset_id) l
            ON l.seq=r.seq WHERE r.decision='approved'""").fetchone()[0]
    digest = hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return records, approved, digest


def summarize(catalog, seconds_per_image, statement, task_id=TASK):
    if not math.isfinite(seconds_per_image) or seconds_per_image <= 0 or not statement.strip():
        raise ValueError('A positive user-reported review rate and its source statement are required.')
    records, target, digest = source_records(catalog, task_id)
    groups = {role: [r for r in records if r['role'] == role] for role in ('A', 'B')}
    if not groups['A'] or {r['asset_id'] for r in groups['A']} != {r['asset_id'] for r in groups['B']}:
        raise ValueError('The A/B manual baseline must contain the same nonempty image set.')
    manual = {}
    for role, rows in groups.items():
        total = sum(r['seconds'] for r in rows)
        if any(not math.isfinite(r['seconds']) or r['seconds'] <= 0 for r in rows):
            raise ValueError('Missing or invalid independent annotation timer.')
        manual[role] = {'images': len(rows), 'total_seconds': total, 'mean_seconds_per_image': total/len(rows)}
    baseline = sum(r['seconds'] for r in records)/len(records)
    assisted_total, manual_total = target*seconds_per_image, target*baseline
    return {'schema_version': 1, 'status': 'estimated_from_user_reported_measurement',
            'generated_at': datetime.now(timezone.utc).isoformat(), 'task_id': task_id,
            'source_submissions_sha256': digest, 'source_submission_count': len(records),
            'manual_baselines': manual, 'manual_mean_seconds_per_image': baseline,
            'assisted': {'seconds_per_image': seconds_per_image, 'basis': 'user_reported_approximate_measurement',
                         'source_statement': statement, 'observed_sample_count': None,
                         'raw_paired_timer_log_available': False},
            'target_images': target, 'estimated_manual_seconds': manual_total,
            'estimated_assisted_seconds': assisted_total, 'estimated_saved_seconds': manual_total-assisted_total,
            'estimated_saved_fraction': 1-seconds_per_image/baseline,
            'per_annotator_reference': {role: {
                'estimated_manual_seconds': target*row['mean_seconds_per_image'],
                'estimated_saved_seconds': target*(row['mean_seconds_per_image']-seconds_per_image),
                'estimated_saved_fraction': 1-seconds_per_image/row['mean_seconds_per_image']}
                for role, row in manual.items()},
            'scope': 'Human review labor only; model inference, setup, export and unrelated waiting excluded.',
            'assumptions': [f'User-reported approximately {seconds_per_image:g} seconds is treated as full effective review/correction time per image.',
                            'The observed review sample size was not supplied; this is not a measured mean over all 700 images.',
                            'Recorded independent A/B timers estimate one-person manual annotation time: sum time / 400 submissions, not sum time / 200 images.',
                            'Extrapolation assumes comparable task difficulty and annotation quality; no randomized paired experiment was performed.',
                            'A/B reference variation is not a confidence interval.'],
            'original_submissions_modified': False, 'raw_worktime_records_synthesized': False}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=ROOT/'data/fire_dataset/catalog.sqlite3')
    parser.add_argument('--seconds-per-image', type=float, required=True)
    parser.add_argument('--statement', required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/worktime_estimate_20261108.json')
    args=parser.parse_args();result=summarize(args.catalog,args.seconds_per_image,args.statement)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
