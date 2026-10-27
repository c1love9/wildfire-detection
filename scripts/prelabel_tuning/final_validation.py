"""Evaluate the locked multilabel candidate, never write production proposals."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import json
from pathlib import Path
import torch
from PIL import Image
from ultralytics import YOLO
from grounding_prelabel import GroundingPrelabeler
from dataset_store import file_sha
from trial import ROOT,OUT,DATA,load_data,raw_joint,SPECS,filtered,evaluate,gate
from fusion_trial import combine
from multilabel_trial import expand

def main():
    snap,refs,groups=load_data();items={r['asset']['id']:r for r in snap};selection=json.loads((OUT/'multilabel_selected.json').read_text(encoding='utf-8'));assert selection['gate']['passed']
    name=selection['spec_name'];spec=SPECS[name.removesuffix('_800')];short=800 if name.endswith('_800') else 640;keys=list(items)
    dpath=OUT/'final_dino_raw.json';ypath=OUT/'final_yolo_raw.json';dino=json.loads(dpath.read_text(encoding='utf-8')) if dpath.exists() else {};yolo=json.loads(ypath.read_text(encoding='utf-8')) if ypath.exists() else {}
    for file in [OUT/('tuning_'+name+'_raw.json'),OUT/'round2_recheck_dino_raw.json']:
        if file.exists():dino.update(json.loads(file.read_text(encoding='utf-8')))
    for file in [OUT/'tuning_specialist_raw.json',OUT/'round2_recheck_yolo_raw.json']:
        if file.exists():yolo.update(json.loads(file.read_text(encoding='utf-8')))
    torch.set_num_threads(min(8,torch.get_num_threads()));grounder=GroundingPrelabeler('cuda:0' if torch.cuda.is_available() else 'cpu');ymodel=YOLO(str(ROOT/'fire-models/fire_n.pt'));names={int(k):v.lower() for k,v in ymodel.names.items()};assert set(names.values())=={'fire','smoke'}
    for i,key in enumerate(keys,1):
        if key in dino and key in yolo:continue
        asset=items[key]['asset'];source=DATA/asset['path'];assert file_sha(source)==asset['sha256']
        with Image.open(source) as image:
            rgb=image.convert('RGB')
            if key not in dino:dino[key]=raw_joint(grounder,rgb,spec,short)
            if key not in yolo:
                result=ymodel.predict(rgb,imgsz=640,conf=.05,iou=.7,max_det=300,device=0 if torch.cuda.is_available() else 'cpu',save=False,verbose=False)[0]
                yolo[key]=[{'id':f'{i:08x}{j:024x}','class':names[int(b.cls.item())],'xyxy':b.xyxy[0].cpu().tolist(),'confidence':float(b.conf.item()),'phrase':names[int(b.cls.item())]} for j,b in enumerate(result.boxes)]
        dpath.write_text(json.dumps(dino),encoding='utf-8');ypath.write_text(json.dumps(yolo),encoding='utf-8')
        if i%20==0 or i==len(keys):print(json.dumps({'stage':'final_evaluation_only','processed_position':i,'total':len(keys)}),flush=True)
    yp=selection['yolo_parameters'];pred={k:combine(filtered(expand(dino[k]),**selection['dino_parameters']),yolo[k],yp['fire_threshold'],yp['smoke_threshold'],yp['order']) for k in keys};p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));r27=json.loads((OUT/'round2_confirmation_protocol.json').read_text(encoding='utf-8'));r93=json.loads((OUT/'remaining93_protocol.json').read_text(encoding='utf-8'))
    sets={'tuning40':p['batches']['tuning'],'first_grouped40':p['batches']['confirmation'],'reused27':r27['asset_ids'],'unused93':r93['asset_ids'],'all200':keys};reports={}
    for phase,ids in sets.items():
        old=evaluate(refs,{k:items[k]['proposal']['boxes'] for k in ids},ids);new=evaluate(refs,pred,ids);g=gate(old,new);reports[phase]={'images':len(ids),'baseline':old,'candidate':new,'gate':g}
    report={'selection':selection,'scope':'All are existing training images; reserve groups/history disclosed. Final test remains untouched. No database changes.',
        'adoption_gate':'All reserved checks must pass original strict per-class TP/FP criteria. Keep failures even when aggregate scores rise.',
        'adoption_passed':all(reports[k]['gate']['passed'] for k in ('first_grouped40','reused27','unused93','all200')),
        'results':reports,'predictions':pred,'models':{'grounding_dino':grounder.metadata,'auxiliary_yolo':{'path':'fire-models/fire_n.pt','sha256':file_sha(ROOT/'fire-models/fire_n.pt'),'class_mapping':names,'origin':'upstream specialist; not self-trained or zero-shot'}}}
    (OUT/'final_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'stage':'final_validation_complete','adoption_passed':report['adoption_passed'],'metrics':{k:{'gate':r['gate'],'baseline':{h:r['baseline'][h]['overall'] for h in ('A','B')},'candidate':{h:r['candidate'][h]['overall'] for h in ('A','B')}} for k,r in reports.items()}}),flush=True)

if __name__=='__main__':main()
