"""Permit fire and smoke scores on one query; retain contextual background check."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import json
import hashlib
from trial import OUT,load_data,evaluate,gate,filtered,SPECS
from fusion_trial import combine

def expand(rows):
    expanded=[]
    for row in rows:
        for index,label in enumerate(('fire','smoke')):
            score=row['class_scores'][index]
            if score>=.10:
                expanded.append({**row,'id':hashlib.md5((row['id']+label).encode()).hexdigest(),'class':label,'confidence':score,'phrase':label,'source':'joint_multilabel_dino'})
    return expanded

def main():
    snap,refs,groups=load_data();items={r['asset']['id']:r for r in snap};keys=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))['batches']['tuning'];old=evaluate(refs,{k:items[k]['proposal']['boxes'] for k in keys},keys);yolo=json.loads((OUT/'tuning_specialist_raw.json').read_text(encoding='utf-8'))
    trials=[]
    for name in list(SPECS)+['joint_basic_800','joint_plume_background_800']:
        values=json.loads((OUT/('tuning_'+name+'_raw.json')).read_text(encoding='utf-8'));values={k:expand(v) for k,v in values.items()}
        for ft in [.15,.20,.25,.30,.35,.40]:
            for st in [.15,.20,.25,.30,.35,.40]:
                for margin in [-.05,0,.05]:
                    dp={'fire_threshold':ft,'smoke_threshold':st,'margin':margin,'nms_iou':.5};dino={k:filtered(values[k],**dp) for k in keys}
                    for yp in [None,{'fire_threshold':.15,'smoke_threshold':.5,'order':'dino_first'}, {'fire_threshold':.25,'smoke_threshold':.5,'order':'yolo_first'}]:
                        pred=dino if yp is None else {k:combine(dino[k],yolo[k],yp['fire_threshold'],yp['smoke_threshold'],yp['order']) for k in keys};m=evaluate(refs,pred,keys);g=gate(old,m)
                        trials.append({'spec_name':name,'multilabel':True,'dino_parameters':dp,'yolo_parameters':yp,'metrics':m,'gate':g,'mean_f1':sum(m[r]['overall']['f1'] for r in m)/2})
    trials.sort(key=lambda t:(t['gate']['passed'],t['mean_f1']),reverse=True);best=trials[0]
    (OUT/'multilabel_tuning_results.json').write_text(json.dumps({'baseline':old,'trials':trials,'hypothesis':'Overlapping fire and smoke are not mutually exclusive; each class passes its own score and background margin.'},indent=2),encoding='utf-8')
    (OUT/'multilabel_selected.json').write_text(json.dumps(best,indent=2),encoding='utf-8')
    print(json.dumps({'stage':'multilabel_selected','eligible':sum(t['gate']['passed'] for t in trials),'selected':{k:best[k] for k in ('spec_name','multilabel','dino_parameters','yolo_parameters','gate','mean_f1')},'metrics':{r:best['metrics'][r]['overall'] for r in best['metrics']}}),flush=True)

if __name__=='__main__':main()
