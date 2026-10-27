"""Append enhanced candidates; preserve all prior proposals and human records."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import argparse as _argparse
_parser = _argparse.ArgumentParser(description=__doc__)
_parser.add_argument('--run', action='store_true', help='Append enhanced candidates after the recorded verification checks.')
if not _parser.parse_args().run:
    print('Read-only: add --run to append candidates after reviewing the saved experiment and adapter checks.')
    raise SystemExit(0)
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/prelabel_optimization_20261027';DATA=ROOT/'data/fire_dataset';BATCH=OUT/'batch';BATCH.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT))
import torch
from annotations import AnnotationStore
from dataset_store import DatasetStore
from optimized_prelabel import OptimizedPrelabelConfig,OptimizedPrelabeler
from comparison import iou

config=OptimizedPrelabelConfig.load();validation=json.loads((OUT/'base_validation.json').read_text(encoding='utf-8'))['candidates']['f1_priority']
assert json.loads((OUT/'adapter_verification.json').read_text(encoding='utf-8'))['status']=='passed'
for role in ('A','B'):
    for subset in ('all200','remaining93'):
        row=validation['results'][subset];old,new=row['baseline'][role]['overall'],row['candidate'][role]['overall']
        assert new['precision']>old['precision'] and new['recall']>=old['recall'] and new['f1']>old['f1'] and new['fp']<old['fp']
    a,b=validation['results']['all200']['baseline'][role],validation['results']['all200']['candidate'][role]
    assert all(y['tp']>=x['tp'] and y['fp']<=x['fp'] for x,y in zip(a['per_class'],b['per_class']))

def connect():
    db=sqlite3.connect((DATA/'catalog.sqlite3').as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON');return db

def digest(rows):
    return hashlib.sha256('\n'.join(sorted(json.dumps(tuple(r),ensure_ascii=False) for r in rows)).encode()).hexdigest()

manifest=BATCH/'selection.json'
if not manifest.exists():
    with connect() as db:
        with sqlite3.connect(BATCH/'catalog_before.sqlite3') as backup:db.backup(backup)
        db.execute('BEGIN');images=[dict(r) for r in db.execute("SELECT * FROM assets WHERE kind='image' ORDER BY created_at,id")];assert len(images)==700
        protected={}
        for table in ('annotation_reviews','annotation_proposals'):
            maxseq=db.execute(f'SELECT COALESCE(MAX(seq),0) FROM {table}').fetchone()[0]
            protected[table]={'maxseq':maxseq,'sha256':digest(db.execute(f'SELECT * FROM {table} WHERE seq<=?',(maxseq,)).fetchall())}
        for table in ('quality_tasks','quality_submissions','quality_references'):
            protected[table]={'sha256':digest(db.execute(f'SELECT * FROM {table}').fetchall())}
        old_proposals={r['asset_id']:dict(r) for r in db.execute('''SELECT p.* FROM annotation_proposals p WHERE p.seq=(SELECT MAX(q.seq) FROM annotation_proposals q WHERE q.asset_id=p.asset_id)''')}
        sources={r['asset_id']:r['source_name'] for r in db.execute('SELECT asset_id,source_name FROM occurrences')};db.rollback()
    plan={'batch_id':'enhanced_base_20261027','created_at_utc':datetime.now(timezone.utc).isoformat(),'config':asdict(config),'images':images,'protected':protected,'original_proposals':old_proposals,'sources':sources,
          'decision':'Append experimental candidates after overall P/R/F1 and per-class totals improve on200 and overall reserved93. Strict every-subgroup gate did NOT fully pass; thin-smoke local regressions are retained for human review. No automatic approval or replacement of human labels.'}
    manifest.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
else:plan=json.loads(manifest.read_text(encoding='utf-8'));assert plan['config']==asdict(config)

with connect() as db:
    done={r['asset_id'] for r in db.execute('''SELECT p.asset_id FROM annotation_proposals p JOIN annotation_runs r ON p.run_id=r.id
        WHERE p.status='ready' AND json_extract(r.metadata_json,'$.batch_id')=?''',(plan['batch_id'],))}
pending=[r['id'] for r in plan['images'] if r['id'] not in done];print(json.dumps({'stage':'batch_start','already_ready':len(done),'remaining':len(pending)}),flush=True)
torch.set_num_threads(min(8,torch.get_num_threads()));model=OptimizedPrelabeler('cuda:0' if torch.cuda.is_available() else 'cpu');store=AnnotationStore(DatasetStore(DATA));started=time.perf_counter()
class TaggedModel:
    def run_metadata(self,cfg):return {**model.run_metadata(cfg),'batch_id':plan['batch_id'],'tuning_evidence':'prelabel_optimization_20261027','strict_every_subgroup_gate_passed':False}
    def predict(self,image,cfg):return model.predict(image,cfg)
results=[]
for attempt in range(2):
    if not pending:break
    retry=[]
    for offset in range(0,len(pending),100):
        batch=pending[offset:offset+100]
        def progress(n,total):
            if n%20==0 or n==total:
                p={'stage':'enhanced_batch','attempt':attempt+1,'completed_this_attempt':offset+n,'total_this_attempt':len(pending),'elapsed_seconds':round(time.perf_counter()-started,1)}
                (BATCH/'progress.json').write_text(json.dumps(p),encoding='utf-8');print(json.dumps(p),flush=True)
        rows=store.prelabel(batch,TaggedModel(),config,'自动增强候选 Base+YOLO（待人工复核）',progress=progress);results.extend(rows);retry.extend(r['asset_id'] for r in rows if r['status']!='ready')
        (BATCH/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    pending=retry
assert not pending,f'Failed candidates remain: {len(pending)}'
with connect() as db:
    db.execute('BEGIN')
    for table,old in plan['protected'].items():
        rows=db.execute(f'SELECT * FROM {table} WHERE seq<=?',(old['maxseq'],)).fetchall() if 'maxseq' in old else db.execute(f'SELECT * FROM {table}').fetchall()
        assert digest(rows)==old['sha256'],f'Existing records changed: {table}'
    latest={r['asset_id']:dict(r) for r in db.execute('''SELECT p.*,r.metadata_json FROM annotation_proposals p JOIN annotation_runs r ON r.id=p.run_id
       WHERE p.seq=(SELECT MAX(q.seq) FROM annotation_proposals q WHERE q.asset_id=p.asset_id)''')}
    expected={r['id'] for r in plan['images']};assert all(latest[k]['status']=='ready' and json.loads(latest[k]['metadata_json']).get('batch_id')==plan['batch_id'] for k in expected)
    normal_ids={r['asset_id'] for r in db.execute("SELECT p.asset_id FROM annotation_proposals p JOIN annotation_runs r ON r.id=p.run_id WHERE json_extract(r.metadata_json,'$.method')='provided_DFire_empty_label'")}
    review_count=db.execute('SELECT COUNT(*) FROM annotation_reviews').fetchone()[0];db.rollback()
new={k:json.loads(latest[k]['boxes_json']) for k in expected};priorities=[]
for key in expected:
    old=json.loads(plan['original_proposals'][key]['boxes_json']);dropped=[label for label in ('fire','smoke') if any(b['class']==label for b in old) and not any(b['class']==label for b in new[key])]
    if dropped:priorities.append({'asset_id':key,'source_name':plan['sources'].get(key,key),'reasons':['新旧模型分歧：新版本未保留'+','.join(dropped)+'候选，需看原图判断，不自动判为正常'],'old_proposal_id':plan['original_proposals'][key]['id'],'new_proposal_id':latest[key]['id']})
summary={'status':'completed','completed_at_utc':datetime.now(timezone.utc).isoformat(),'batch_id':plan['batch_id'],'new_ready_images':len(new),'failed_images':0,
    'nonempty_images':sum(bool(v) for v in new.values()),'empty_images':sum(not v for v in new.values()),'source_normal_candidates':len(normal_ids),'source_normal_with_boxes':sum(bool(new[k]) for k in normal_ids),
    'box_counts':{label:sum(sum(b['class']==label for b in v) for v in new.values()) for label in ('fire','smoke')},
    'previous_proposals_unchanged':True,'frozen_quality_task_and_AB_unchanged':True,'previous_human_reviews_unchanged':True,'automatic_human_approvals':0,'review_record_count':review_count,
    'disagreement_priority_images':len(priorities),'decision':plan['decision'],'config':asdict(config),'elapsed_seconds':round(time.perf_counter()-started,2)}
(BATCH/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');(BATCH/'review_priorities.json').write_text(json.dumps(priorities,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 增强候选重点复核清单','','新结果是待复核候选，未自动批准。以下只是新旧模型分歧，不等于已确认漏标。选择图片时可搜索ID前12位，查看原图并切换历史预标注版本。已通过人工标签保持原版本。','','|原文件名|图像ID前12位|建议检查|','|---|---|---|']
for row in sorted(priorities,key=lambda r:r['source_name']):lines.append('|'+row['source_name'].replace('|','\\|')+'|'+row['asset_id'][:12]+'|'+'；'.join(row['reasons'])+'|')
(BATCH/'重点复核清单.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(summary,ensure_ascii=False),flush=True)
