"""Read-only tuning on 40 train images; confirmation hidden until selection."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import random
import re
import sqlite3
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/prelabel_optimization_20261027'
OUT.mkdir(parents=True, exist_ok=True)
DATA=ROOT/'data/fire_dataset'
sys.path.insert(0,str(ROOT))
import torch
from PIL import Image
from torchvision.ops import nms
from grounding_prelabel import GroundingPrelabeler,PrelabelConfig
from quality import preannotation_metrics
from comparison import iou
from dataset_store import file_sha

TASK='quality_6146eb5a5ab24f5ebc6961d07e392ad2'

def load_data():
    db=sqlite3.connect((DATA/'catalog.sqlite3').as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    db.execute('PRAGMA query_only=ON');db.execute('BEGIN')
    snap=json.loads(db.execute('SELECT snapshot_json FROM quality_tasks WHERE id=?',(TASK,)).fetchone()[0])
    refs={role:{r['asset_id']:json.loads(r['boxes_json']) for r in db.execute('SELECT asset_id,boxes_json FROM quality_submissions WHERE task_id=? AND role=?',(TASK,role))} for role in ('A','B')}
    groups={r['asset_id']:r['group_key'] for r in db.execute('SELECT o.asset_id,b.group_key FROM occurrences o JOIN batches b ON b.id=o.batch_id')}
    db.rollback();db.close()
    split=json.loads((DATA/'splits/split_20261022_091321_5a8d09dd/manifest.json').read_text(encoding='utf-8'))
    assigned={r['id']:r['split'] for r in split['assets']}
    assert len(snap)==200 and all(assigned[r['asset']['id']]=='train' for r in snap)
    return snap,refs,groups

def label(key,refs):return '+'.join(sorted({b['class'] for role in ('A','B') for b in refs[role][key]})) or 'empty'

def freeze_selection(snap,refs,groups):
    path=OUT/'protocol.json'
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    rng=random.Random(271027);keys=[r['asset']['id'] for r in snap];classes=['empty','fire','fire+smoke','smoke']
    group_names=sorted({groups[k] for k in keys});largest=max(group_names,key=lambda g:sum(groups[k]==g for k in keys))
    for attempt in range(20000):
        tuning_groups={g for g in group_names if rng.random()<.45}|{largest}
        pools={phase:{c:[] for c in classes} for phase in ('tuning','confirmation')}
        for key in keys:pools['tuning' if groups[key] in tuning_groups else 'confirmation'][label(key,refs)].append(key)
        if all(len(pools[p][c])>=10 for p in pools for c in classes):break
    else:raise ValueError('Cannot prepare disjoint grouped batches with all strata')
    batches={}
    for phase in pools:
        selected=[]
        for c in classes:
            per_group=defaultdict(list)
            for k in sorted(pools[phase][c]):per_group[groups[k]].append(k)
            ordered=list(per_group);rng.shuffle(ordered)
            for g in ordered:rng.shuffle(per_group[g])
            taken=[]
            while len(taken)<10:
                for g in ordered:
                    if per_group[g]:taken.append(per_group[g].pop())
                    if len(taken)==10:break
            selected.extend(taken)
        batches[phase]=selected
    assert not {groups[k] for k in batches['tuning']}&{groups[k] for k in batches['confirmation']}
    protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'task_id':TASK,'seed':271027,'batches':batches,
       'groups':{k:groups[k] for p in batches.values() for k in p},'strata':{k:label(k,refs) for p in batches.values() for k in p},
       'scope':'Train only. Previously explored data; this is a grouped reserved confirmation batch, not a never-used independent test.',
       'gate':{'both_human_references':True,'precision_gain_min':.05,'recall_not_lower':True,'f1_gain_min':.05,
               'fp_at_most_baseline_ratio':.80,'per_class_tp_not_lower':True,'per_class_fp_not_higher':True},
       'selection_rule':'Eligible on tuning first; rank by mean F1. Lock one choice, then run confirmation once. No bulk proposal writes before confirmation passes.'}
    path.write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8');return protocol

def evaluate(refs,preds,keys):return {role:preannotation_metrics({k:refs[role][k] for k in keys},{k:preds[k] for k in keys}) for role in ('A','B')}

def gate(old,new):
    checks={}
    for role in ('A','B'):
        a,b=old[role]['overall'],new[role]['overall'];per=[]
        for x,y in zip(old[role]['per_class'],new[role]['per_class']):per.append(y['tp']>=x['tp'] and y['fp']<=x['fp'])
        checks[role]={'precision':b['precision']>=a['precision']+.05,'recall':b['recall']>=a['recall'],
           'f1':b['f1']>=a['f1']+.05,'fp':b['fp']<=a['fp']*.8,'per_class':all(per)}
    return {'passed':all(all(v.values()) for v in checks.values()),'checks':checks}

SPECS={
 'joint_basic':{'terms':{'fire':['fire'],'smoke':['smoke']},'negative':[]},
 'joint_flame_background':{'terms':{'fire':['flame'],'smoke':['smoke']},'negative':['cloud','sun','lamp','fog','reflection']},
 'joint_synonyms_background':{'terms':{'fire':['fire','flame'],'smoke':['smoke']},'negative':['cloud','sun','light','fog','reflection']},
 'joint_plume_background':{'terms':{'fire':['flame'],'smoke':['smoke plume']},'negative':['cloud','sun','lamp','fog']},
}

def raw_joint(model,image,spec,short=640):
    terms=spec['terms'];phrases=[p for names in terms.values() for p in names]+spec['negative'];text=''.join(p+'. ' for p in phrases)
    tokenizer=model.processor.tokenizer;indices=[];token_ids=[tokenizer.cls_token_id]
    for phrase in phrases:
        ids=tokenizer.encode(phrase,add_special_tokens=False);start=len(token_ids);indices.append(list(range(start,start+len(ids))))
        token_ids.extend(ids);token_ids.extend(tokenizer.encode('.',add_special_tokens=False))
    token_ids.append(tokenizer.sep_token_id)
    assert token_ids==tokenizer.encode(text,add_special_tokens=True),'Phrase token positions do not match full query'
    assert all(indices)
    with model._lock,torch.inference_mode():
        inputs=model.processor(images=image,text=text,return_tensors='pt',size={'shortest_edge':short,'longest_edge':round(short*1.66625)}).to(model.device)
        logits=model.model(**inputs)
        probs=logits.logits[0].sigmoid();boxes=logits.pred_boxes[0]
        phrase_scores=torch.stack([probs[:,ix].mean(-1) for ix in indices],1)
        class_scores=[];start=0
        for names in terms.values():class_scores.append(phrase_scores[:,start:start+len(names)].max(1).values);start+=len(names)
        scores=torch.stack(class_scores,1);best,classes=scores.max(1)
        neg=phrase_scores[:,start:].max(1).values if spec['negative'] else torch.zeros_like(best)
        cx,cy,w,h=boxes.unbind(-1);xyxy=torch.stack([cx-w/2,cy-h/2,cx+w/2,cy+h/2],1)
        xyxy*=torch.tensor([image.width,image.height,image.width,image.height],device=xyxy.device)
        xyxy[:,[0,2]]=xyxy[:,[0,2]].clamp(0,image.width);xyxy[:,[1,3]]=xyxy[:,[1,3]].clamp(0,image.height)
        rows=[]
        for i in torch.where(best>=.10)[0].tolist():
            bounds=xyxy[i].tolist()
            if bounds[2]-bounds[0]<1 or bounds[3]-bounds[1]<1:continue
            rows.append({'id':uuid.uuid4().hex,'class':list(terms)[int(classes[i])],'xyxy':bounds,'confidence':float(best[i]),
                         'negative_score':float(neg[i]),'class_scores':scores[i].tolist(),'query_index':i,'phrase':phrases[int(phrase_scores[i].argmax())]})
    return rows

def filtered(rows,fire_threshold=.3,smoke_threshold=.3,margin=0,nms_iou=.5):
    out=[]
    for label,t in [('fire',fire_threshold),('smoke',smoke_threshold)]:
        candidates=[b for b in rows if b['class']==label and b['confidence']>=t and b['confidence']-b.get('negative_score',0)>=margin]
        if not candidates:continue
        keep=nms(torch.tensor([b['xyxy'] for b in candidates]),torch.tensor([b['confidence'] for b in candidates]),nms_iou)[:50].tolist()
        out.extend(candidates[i] for i in keep)
    return out

def main():
    snap,refs,groups=load_data();items={r['asset']['id']:r for r in snap};protocol=freeze_selection(snap,refs,groups)
    phase='confirmation' if '--confirm' in sys.argv else 'tuning';keys=protocol['batches'][phase]
    baseline={k:items[k]['proposal']['boxes'] for k in keys};old=evaluate(refs,baseline,keys)
    print(json.dumps({'stage':'selection_ready','phase':phase,'images':len(keys),'baseline':{r:old[r]['overall'] for r in old}}),flush=True)
    torch.set_num_threads(min(8,torch.get_num_threads()));model=GroundingPrelabeler('cuda:0' if torch.cuda.is_available() else 'cpu')
    if phase=='confirmation':
        selected=json.loads((OUT/'selected.json').read_text(encoding='utf-8'));assert selected['tuning_gate']['passed'];specs={selected['spec_name']:SPECS[selected['spec_name']]}
    else:specs=SPECS
    raw={}
    for name,spec in specs.items():
        path=OUT/(phase+'_'+name+'_raw.json')
        values=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        for i,key in enumerate(keys,1):
            if key in values:continue
            asset=items[key]['asset'];p=DATA/asset['path'];assert file_sha(p)==asset['sha256']
            with Image.open(p) as image:values[key]=raw_joint(model,image.convert('RGB'),spec)
            path.write_text(json.dumps(values,ensure_ascii=False),encoding='utf-8')
            if i%10==0:print(json.dumps({'stage':'inference','phase':phase,'spec':name,'done':i,'total':len(keys)}),flush=True)
        raw[name]=values
    if phase=='confirmation':
        params=selected['parameters'];preds={k:filtered(raw[selected['spec_name']][k],**params) for k in keys};new=evaluate(refs,preds,keys);check=gate(old,new)
        report={'phase':phase,'status':'passed' if check['passed'] else 'not_improved','selection':selected,'baseline':old,'candidate':new,'gate':check,'predictions':preds}
        (OUT/'confirmation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({'stage':'confirmation_complete','gate':check,'baseline':{r:old[r]['overall'] for r in old},'candidate':{r:new[r]['overall'] for r in new}}),flush=True);return
    trials=[]
    for name,values in raw.items():
        for ft in [.15,.20,.25,.30,.35,.40]:
            for st in [.15,.20,.25,.30,.35,.40]:
                for margin in [0,.03,.06]:
                    params={'fire_threshold':ft,'smoke_threshold':st,'margin':margin,'nms_iou':.5};preds={k:filtered(values[k],**params) for k in keys};new=evaluate(refs,preds,keys);check=gate(old,new)
                    trials.append({'spec_name':name,'parameters':params,'metrics':new,'tuning_gate':check,'mean_f1':sum(new[r]['overall']['f1'] for r in new)/2})
    trials.sort(key=lambda r:(r['tuning_gate']['passed'],r['mean_f1']),reverse=True)
    (OUT/'tuning_results.json').write_text(json.dumps({'baseline':old,'trials':trials},ensure_ascii=False,indent=2),encoding='utf-8')
    winner=trials[0];(OUT/'selected.json').write_text(json.dumps(winner,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'stage':'tuning_complete','eligible':sum(t['tuning_gate']['passed'] for t in trials),'selected':{k:winner[k] for k in ('spec_name','parameters','mean_f1','tuning_gate')},'metrics':{r:winner['metrics'][r]['overall'] for r in ('A','B')}}),flush=True)

if __name__=='__main__':main()
