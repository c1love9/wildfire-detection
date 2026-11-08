"""Read-only box localization report; never label box IoU as mask area error."""
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from comparison import iou
from dataset_store import now
from summarize_worktime import TASK


def summarize(catalog):
    db=sqlite3.connect(catalog.resolve().as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    snapshot_text=db.execute('SELECT snapshot_json FROM quality_tasks WHERE id=?',(TASK,)).fetchone()[0]
    snapshot=json.loads(snapshot_text)
    submitted=[dict(row) for row in db.execute('SELECT * FROM quality_submissions WHERE task_id=? ORDER BY asset_id,role',(TASK,))]
    by_key={(row['asset_id'],row['role']):json.loads(row['boxes_json']) for row in submitted}
    predictions={'original_frozen':{item['asset']['id']:item['proposal']['boxes'] for item in snapshot},'enhanced_latest':{}}
    proposal_versions={}
    for item in snapshot:
        key=item['asset']['id']
        row=db.execute('SELECT id,input_sha256,boxes_json,status FROM annotation_proposals WHERE asset_id=? ORDER BY seq DESC LIMIT 1',(key,)).fetchone()
        assert row['input_sha256']==item['asset']['sha256'] and row['status']=='ready'
        predictions['enhanced_latest'][key]=json.loads(row['boxes_json']);proposal_versions[key]=row['id']
    db.close();results={}
    for variant,images in predictions.items():
        results[variant]={}
        for role in ('A','B'):
            overlaps=[];fp=fn=0
            for key,boxes in images.items():
                reference=by_key[key,role];used=set()
                for box in sorted(boxes,key=lambda row:row['confidence'],reverse=True):
                    choices=[(iou(box['xyxy'],target['xyxy']),index) for index,target in enumerate(reference) if index not in used and target['class']==box['class']]
                    overlap,index=max(choices,default=(0,None))
                    if overlap>=.5:overlaps.append(overlap);used.add(index)
                    else:fp+=1
                fn+=len(reference)-len(used)
            results[variant][role]={'matched_pairs':len(overlaps),'matched_mean_box_iou':sum(overlaps)/len(overlaps) if overlaps else None,
                                    'unmatched_prediction_boxes':fp,'unmatched_reference_boxes':fn}
    return {'status':'completed','generated_at':now(),'task_id':TASK,'images':len(snapshot),
            'protocol':'Same class, prediction confidence descending, one-to-one best available match at IoU>=0.5; mean uses matched boxes only.',
            'results':results,'latest_proposal_versions':proposal_versions,
            'raw_AB_sha256':hashlib.sha256(json.dumps(submitted,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
            'original_snapshot_sha256':hashlib.sha256(snapshot_text.encode()).hexdigest(),
            'interpretation':'Detection-box localization consistency, not pixel segmentation accuracy or area-ratio error.',
            'mask_iou':None,'area_ratio_error':None,'independent_mask_reference_available':False,
            'reference_policy':'A and B reported separately; neither is a consensus ground truth; no adjudication added.',
            'database_modified':False}


if __name__=='__main__':
    result=summarize(ROOT/'data/fire_dataset/catalog.sqlite3')
    (ROOT/'reports/annotation_iou_20261108.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('images','results','interpretation','mask_iou')},ensure_ascii=False,indent=2))
