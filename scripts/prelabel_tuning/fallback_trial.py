"""Retain baseline smoke boxes only with independent regional model support."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import json
from trial import OUT,load_data,evaluate,gate,filtered
from fusion_trial import combine
from comparison import iou

def supplement(merged,baseline,raw,params):
    kept=[dict(b) for b in merged]
    for box in sorted(baseline,key=lambda b:-b['confidence']):
        if box['class']!='smoke' or box['confidence']<params['baseline_threshold']:continue
        if any(b['class']=='smoke' and iou(b['xyxy'],box['xyxy'])>=.5 for b in kept):continue
        supported=any(b['class']=='smoke' and b['confidence']>=params['support_threshold'] and
            b['confidence']-b.get('negative_score',0)>=params['background_margin'] and
            iou(b['xyxy'],box['xyxy'])>=params['support_iou'] for b in raw)
        if supported:kept.append({**box,'source':'independent_dino_smoke_with_joint_support'})
    return kept

def main():
    snap,refs,groups=load_data();items={r['asset']['id']:r for r in snap};p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));keys=p['batches']['tuning'];baseline={k:items[k]['proposal']['boxes'] for k in keys};old=evaluate(refs,baseline,keys)
    selected=json.loads((OUT/'round2_selected.json').read_text(encoding='utf-8'));raw=json.loads((OUT/('tuning_'+selected['spec_name']+'_raw.json')).read_text(encoding='utf-8'));yolo=json.loads((OUT/'tuning_specialist_raw.json').read_text(encoding='utf-8'));yp=selected['yolo_parameters']
    merged={k:combine(filtered(raw[k],**selected['dino_parameters']),yolo[k],yp['fire_threshold'],yp['smoke_threshold'],yp['order']) for k in keys}
    trials=[]
    for bt in [.30,.35,.40]:
        for st in [.10,.15,.20,.25,.30]:
            for si in [.10,.20,.30,.40,.50]:
                for margin in [-.10,-.05,0,.05]:
                    params={'baseline_threshold':bt,'support_threshold':st,'support_iou':si,'background_margin':margin};pred={k:supplement(merged[k],baseline[k],raw[k],params) for k in keys};m=evaluate(refs,pred,keys);g=gate(old,m)
                    trials.append({'parameters':params,'metrics':m,'gate':g,'mean_f1':sum(m[r]['overall']['f1'] for r in m)/2})
    trials.sort(key=lambda t:(t['gate']['passed'],t['mean_f1']),reverse=True)
    best=trials[0];report={'family':'Joint DINO+specialist, with original independent DINO smoke boxes retained only when supported by joint regional scores. No human labels used at inference.','frozen_base':selected,'baseline':old,'trials':trials}
    (OUT/'fallback_tuning_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');(OUT/'fallback_selected.json').write_text(json.dumps({'base':selected,**best},indent=2),encoding='utf-8')
    # Freeze all remaining previously unprocessed images before validation, regardless of their difficulty.
    seen=set(p['batches']['tuning']+p['batches']['confirmation']+json.loads((OUT/'round2_confirmation_protocol.json').read_text(encoding='utf-8'))['asset_ids'])
    remaining=[k for k in items if k not in seen];(OUT/'remaining93_protocol.json').write_text(json.dumps({'asset_ids':remaining,'sample_size':len(remaining),'scope':'Unused images; some share tuning source groups. Original grouped confirmation is also repeated; not independent final test.'},indent=2),encoding='utf-8')
    print(json.dumps({'stage':'fallback_selected','eligible':sum(t['gate']['passed'] for t in trials),'selected':{k:best[k] for k in ('parameters','gate','mean_f1')},'metrics':{r:best['metrics'][r]['overall'] for r in best['metrics']},'remaining_images':len(remaining)}),flush=True)

if __name__=='__main__':main()
