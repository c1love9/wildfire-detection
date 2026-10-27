"""Locked second-family candidate on newly reserved train images."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import json
import sys
from pathlib import Path
from PIL import Image
import torch
from ultralytics import YOLO
from grounding_prelabel import GroundingPrelabeler
from dataset_store import file_sha
from trial import OUT,ROOT,DATA,load_data,raw_joint,SPECS,filtered,evaluate,gate
from fusion_trial import combine

def main():
    snap,refs,groups=load_data();items={r['asset']['id']:r for r in snap}
    selection=json.loads((OUT/'round2_selected.json').read_text(encoding='utf-8'));assert selection['gate']['passed']
    phase='full200' if '--full200' in sys.argv else 'round2_recheck'
    if phase=='full200':
        prior=json.loads((OUT/'round2_recheck.json').read_text(encoding='utf-8'));assert prior['gate']['passed'],'Do not scale a failed reserved recheck'
        keys=list(items)
    else:keys=json.loads((OUT/'round2_confirmation_protocol.json').read_text(encoding='utf-8'))['asset_ids']
    name=selection['spec_name'];base=name.removesuffix('_800');short=800 if name.endswith('_800') else 640;spec=SPECS[base]
    torch.set_num_threads(min(8,torch.get_num_threads()));model=GroundingPrelabeler('cuda:0' if torch.cuda.is_available() else 'cpu')
    yolo_model=YOLO(str(ROOT/'fire-models/fire_n.pt')) if selection['yolo_parameters'] else None
    if yolo_model:
        names={int(k):v.lower().strip() for k,v in yolo_model.names.items()};assert set(names.values())=={'fire','smoke'}
    dpath=OUT/(phase+'_dino_raw.json');ypath=OUT/(phase+'_yolo_raw.json')
    dino=json.loads(dpath.read_text(encoding='utf-8')) if dpath.exists() else {};yolo=json.loads(ypath.read_text(encoding='utf-8')) if ypath.exists() else {}
    # Reuse exact-image outputs for this same frozen specification, never other prompts.
    if phase=='full200':
        for p in [OUT/('tuning_'+name+'_raw.json'),OUT/'round2_recheck_dino_raw.json']:
            if p.exists():dino.update(json.loads(p.read_text(encoding='utf-8')))
        for p in [OUT/'tuning_specialist_raw.json',OUT/'round2_recheck_yolo_raw.json']:
            if p.exists():yolo.update(json.loads(p.read_text(encoding='utf-8')))
    for i,key in enumerate(keys,1):
        if key in dino and (not yolo_model or key in yolo):continue
        asset=items[key]['asset'];path=DATA/asset['path'];assert file_sha(path)==asset['sha256']
        with Image.open(path) as image:
            rgb=image.convert('RGB')
            if key not in dino:dino[key]=raw_joint(model,rgb,spec,short)
            if yolo_model and key not in yolo:
                result=yolo_model.predict(rgb,imgsz=640,conf=.05,iou=.7,max_det=300,device=0 if torch.cuda.is_available() else 'cpu',save=False,verbose=False)[0]
                yolo[key]=[{'id':f'{i:08x}{j:024x}','class':names[int(b.cls.item())],'xyxy':b.xyxy[0].cpu().tolist(),'confidence':float(b.conf.item()),'phrase':names[int(b.cls.item())]} for j,b in enumerate(result.boxes)]
        dpath.write_text(json.dumps(dino),encoding='utf-8');ypath.write_text(json.dumps(yolo),encoding='utf-8')
        if i%10==0 or i==len(keys):print(json.dumps({'stage':phase,'done':i,'total':len(keys)}),flush=True)
    preds={}
    for key in keys:
        ds=filtered(dino[key],**selection['dino_parameters']);yp=selection['yolo_parameters']
        preds[key]=ds if yp is None else combine(ds,yolo[key],yp['fire_threshold'],yp['smoke_threshold'],yp['order'])
    old=evaluate(refs,{k:items[k]['proposal']['boxes'] for k in keys},keys);new=evaluate(refs,preds,keys);g=gate(old,new)
    report={'phase':phase,'status':'passed' if g['passed'] else 'not_improved','scope':'Reserved/grouped training diagnosis; not independent final test; previous round failure disclosed','selection':selection,'sample_size':len(keys),'baseline':old,'candidate':new,'gate':g,'predictions':preds}
    (OUT/(phase+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'stage':phase+'_complete','sample_size':len(keys),'gate':g,'baseline':{r:old[r]['overall'] for r in old},'candidate':{r:new[r]['overall'] for r in new}}),flush=True)

if __name__=='__main__':main()
