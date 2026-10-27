"""Lock two Pareto candidates from tuning, verify without changing thresholds."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import json
from datetime import datetime,timezone
from pathlib import Path
import torch
from PIL import Image
from base_trial import OfficialBase,BASE_SPECS,raw_independent
from dataset_store import file_sha
from trial import ROOT,OUT,DATA,load_data,raw_joint,filtered,evaluate,gate
from multilabel_trial import expand
from fusion_trial import combine

def main():
    snap,refs,groups=load_data();items={r['asset']['id']:r for r in snap};keys=list(items);r=json.loads((OUT/'base_tuning_results.json').read_text(encoding='utf-8'));eligible=[t for t in r['trials'] if t['gate']['passed']];assert eligible
    primary=eligible[0];recall=max(eligible,key=lambda t:(sum(t['metrics'][role]['overall']['recall'] for role in ('A','B')),t['mean_f1']))
    selected={'f1_priority':primary,'recall_priority':recall};lock={'created_at_utc':datetime.now(timezone.utc).isoformat(),'candidates':selected,'preference':'Both satisfy tuning FP/TP gate. Check both without modifying settings; if both validate, prefer original tuning F1 winner.','gate':'Original strict gate on all four verification subsets and full200; otherwise no adoption'}
    lock_path=OUT/'base_candidates_locked.json'
    if lock_path.exists():assert json.loads(lock_path.read_text(encoding='utf-8'))['candidates']==selected
    else:lock_path.write_text(json.dumps(lock,indent=2),encoding='utf-8')
    torch.set_num_threads(min(8,torch.get_num_threads()));model=OfficialBase();raw={}
    for choice in selected.values():
        name=choice['spec_name'];p=OUT/('full200_'+name+'_raw.json');values=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
        values.update(json.loads((OUT/('tuning_'+name+'_raw.json')).read_text(encoding='utf-8')))
        for i,key in enumerate(keys,1):
            if key in values:continue
            asset=items[key]['asset'];source=DATA/asset['path'];assert file_sha(source)==asset['sha256']
            with Image.open(source) as image:
                rgb=image.convert('RGB');values[key]=raw_independent(model,rgb) if name=='base_independent' else raw_joint(model,rgb,BASE_SPECS[name],640)
            p.write_text(json.dumps(values),encoding='utf-8')
            if i%20==0 or i==len(keys):print(json.dumps({'stage':'base_verification','profile':name,'position':i,'total':len(keys)}),flush=True)
        raw[name]=values
    yr=json.loads((OUT/'final_yolo_raw.json').read_text(encoding='utf-8'));proto=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));p27=json.loads((OUT/'round2_confirmation_protocol.json').read_text(encoding='utf-8'));p93=json.loads((OUT/'remaining93_protocol.json').read_text(encoding='utf-8'))
    batches={'tuning40':proto['batches']['tuning'],'grouped40':proto['batches']['confirmation'],'recheck27':p27['asset_ids'],'remaining93':p93['asset_ids'],'all200':keys};reports={}
    for candidate,choice in selected.items():
        yp=choice['yolo_parameters'];pred={}
        for key in keys:
            values=raw[choice['spec_name']][key]
            if choice['multilabel']:values=expand(values)
            ds=filtered(values,**choice['dino_parameters']);pred[key]=ds if yp is None else combine(ds,yr[key],yp['fire_threshold'],yp['smoke_threshold'],yp['order'])
        sets={}
        for phase,ids in batches.items():
            old=evaluate(refs,{k:items[k]['proposal']['boxes'] for k in ids},ids);new=evaluate(refs,pred,ids);g=gate(old,new);sets[phase]={'images':len(ids),'baseline':old,'candidate':new,'gate':g}
        passed=all(sets[k]['gate']['passed'] for k in ('grouped40','recheck27','remaining93','all200'))
        reports[candidate]={'selection':choice,'adoption_passed':passed,'results':sets,'predictions':pred}
    report={'status':'passed' if any(r['adoption_passed'] for r in reports.values()) else 'not_adopted','scope':'Previously explored training data with locked candidate settings and original per-class checks; independent final test untouched.','models':{'base':model.metadata,'auxiliary_yolo':{'path':'fire-models/fire_n.pt','sha256':file_sha(ROOT/'fire-models/fire_n.pt'),'origin':'upstream; not self-trained or zero-shot'}},'candidates':reports}
    (OUT/'base_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({'stage':'base_verification_complete','status':report['status'],'candidates':{c:{'adoption_passed':v['adoption_passed'],'subsets':{phase:{'gate':s['gate'],'metrics':{h:s['candidate'][h]['overall'] for h in ('A','B')}} for phase,s in v['results'].items()}} for c,v in reports.items()}}),flush=True)

if __name__=='__main__':main()
