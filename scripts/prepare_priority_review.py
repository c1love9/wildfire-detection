"""Freeze a verified post-A/B review queue; never edit proposals or human labels."""

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset_store import dataset_root
from review_priorities import QUEUE_PATH, load_review_queue, validate_review_queue


def prepare(root, source_report, task_id, batch_id):
    root = Path(root).resolve()
    source = Path(source_report).read_bytes()
    rows = json.loads(source)
    if not isinstance(rows, list) or not rows:
        raise ValueError('来源清单为空或不是图片列表。')
    entries = []
    with closing(sqlite3.connect((root / 'catalog.sqlite3').as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        task = db.execute('SELECT snapshot_json FROM quality_tasks WHERE id=?', (task_id,)).fetchone()
        if task is None:
            raise ValueError('找不到来源双人任务。')
        snapshot = json.loads(task['snapshot_json'])
        task_images = {row['asset']['id'] for row in snapshot}
        submissions = {(row['asset_id'], row['role']): json.loads(row['boxes_json']) for row in db.execute(
            'SELECT asset_id,role,boxes_json FROM quality_submissions WHERE task_id=?', (task_id,))}
        if any((asset_id, role) not in submissions for asset_id in task_images for role in ('A', 'B')):
            raise ValueError('A/B尚未全部提交，不生成会显示双方标注线索的复核清单。')
        for row in rows:
            key, label = row['asset_id'], row['class']
            index = int(row['number'][1:]) - 1
            if not 0 <= index < len(snapshot) or snapshot[index]['asset']['id'] != key:
                raise ValueError('图片编号与原任务不一致。')
            if not all(any(box['class'] == label for box in submissions[(key, role)]) for role in ('A', 'B')):
                raise ValueError('此图不满足A/B均标有该类的清单依据。')
            original_had = any(box['class'] == label for box in snapshot[index]['proposal']['boxes'])
            if original_had != row['original_model_had_class']:
                raise ValueError('原候选类别与来源清单不一致。')
            proposal = next((p for p in db.execute('''SELECT p.*,r.metadata_json FROM annotation_proposals p
                JOIN annotation_runs r ON p.run_id=r.id WHERE p.asset_id=? ORDER BY p.seq DESC''', (key,))
                if json.loads(p['metadata_json']).get('batch_id') == batch_id), None)
            asset = db.execute('SELECT sha256 FROM assets WHERE id=? AND kind=?', (key, 'image')).fetchone()
            if (proposal is None or proposal['status'] != 'ready' or asset is None
                    or proposal['input_sha256'] != asset['sha256']
                    or any(box['class'] == label for box in json.loads(proposal['boxes_json']))):
                raise ValueError('指定增强批次的图像版本或缺类依据不成立。')
            entries.append({'number': row['number'], 'asset_id': key, 'input_sha256': asset['sha256'],
                            'class': label, 'original_model_had_class': original_had,
                            'proposal_id': proposal['id']})
    document = {'schema_version': 1, 'count': len(entries), 'task_id': task_id, 'batch_id': batch_id,
                'source_report_sha256': hashlib.sha256(source).hexdigest(), 'entries': entries}
    validate_review_queue(document)
    output = root / QUEUE_PATH
    payload = (json.dumps(document, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    if output.exists() and output.read_bytes() != payload:
        raise ValueError('已有不同的优先复核清单，请先备份并明确选择要保留的清单。')
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(payload)
    load_review_queue(root)
    return output, document


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=dataset_root(ROOT))
    parser.add_argument('--source-report', type=Path, required=True)
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--batch-id', required=True)
    args = parser.parse_args()
    output, document = prepare(args.dataset_root, args.source_report, args.task_id, args.batch_id)
    print(json.dumps({'path': str(output), 'count': document['count'],
                      'classes': {label: sum(row['class'] == label for row in document['entries'])
                                  for label in ('fire', 'smoke')}}, ensure_ascii=False))
