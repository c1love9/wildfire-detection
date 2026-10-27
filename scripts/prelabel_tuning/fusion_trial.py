"""Second predeclared family: explicit DINO + upstream YOLO candidate union."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import json
from pathlib import Path
import sys
from trial import OUT,load_data,evaluate,gate,filtered,SPECS
from comparison import iou

def combine(dino,yolo,ft,st,order):
    yolo=[{**b,'source':'upstream_yolo'} for b in yolo if b['confidence']>=(ft if b['class']=='fire' else st)]
    dino=[{**b,'source':'grounding_dino'} for b in dino]
    batches=[dino,yolo] if order=='dino_first' else [yolo,dino]
    kept=[]
    for batch in batches:
        for b in sorted(batch,key=lambda b:-b['confidence']):
            if not any(k['class']==b['class'] and iou(k['xyxy'],b['xyxy'])>=.5 for k in kept):kept.append(b)
    return kept

def main():
    snap,refs,groups=load_data();items={r['asset']['id']:r for r in snap};protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));keys=protocol['batches']['tuning']
    baseline=evaluate(refs,{k:items[k]['proposal']['boxes'] for k in keys},keys)
    yolo=json.loads((OUT/'tuning_specialist_raw.json').read_text(encoding='utf-8'));trials=[]
    for spec_name in SPECS:
        raw=json.loads((OUT/('tuning_'+spec_name+'_raw.json')).read_text(encoding='utf-8'))
        for dft in [.20,.25,.30,.35]:
            for dst in [.15,.20,.25,.30,.35]:
                dino={k:filtered(raw[k],dft,dst,0,.5) for k in keys}
                for yft in [.15,.25,.40]:
                    for yst in [.15,.30,.50]:
                        for order in ['dino_first','yolo_first']:
                            pred={k:combine(dino[k],yolo[k],yft,yst,order) for k in keys};metric=evaluate(refs,pred,keys);g=gate(baseline,metric)
                            sufficient_recall=all(metric[r]['overall']['tp']>=baseline[r]['overall']['tp']+3 for r in ('A','B'))
                            trials.append({'spec_name':spec_name,'dino_parameters':{'fire_threshold':dft,'smoke_threshold':dst,'margin':0,'nms_iou':.5},'yolo_parameters':{'fire_threshold':yft,'smoke_threshold':yst,'order':order},
                               'metrics':metric,'gate':g,'recall_gain_at_least_3_boxes':sufficient_recall,'eligible':g['passed'] and sufficient_recall,'mean_f1':sum(metric[r]['overall']['f1'] for r in metric)/2})
    trials.sort(key=lambda r:(r['eligible'],r['mean_f1']),reverse=True)
    (OUT/'fusion_tuning_results.json').write_text(json.dumps({'protocol':'Same tuning 40; require >=3 extra TP for both A/B in addition to original gate. Fusion explicitly discloses specialist weights.', 'baseline':baseline,'trials':trials},indent=2),encoding='utf-8')
    (OUT/'fusion_selected.json').write_text(json.dumps(trials[0],indent=2),encoding='utf-8');best=trials[0]
    print(json.dumps({'eligible':sum(t['eligible'] for t in trials),'selected':{k:best[k] for k in ('spec_name','dino_parameters','yolo_parameters','gate','recall_gain_at_least_3_boxes','mean_f1')},'metrics':{r:best['metrics'][r]['overall'] for r in best['metrics']}}),flush=True)

if __name__=='__main__':main()
