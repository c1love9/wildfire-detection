"""Evaluate existing specialist as an explicitly disclosed auxiliary component."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import json
from pathlib import Path
import sys
import time
from PIL import Image
import torch
from ultralytics import YOLO
from trial import ROOT,OUT,DATA,load_data,freeze_selection,evaluate,gate
from dataset_store import file_sha

def main():
    snap,refs,groups=load_data();items={r['asset']['id']:r for r in snap};p=freeze_selection(snap,refs,groups)
    phase='confirmation' if '--confirm' in sys.argv else 'tuning';keys=p['batches'][phase]
    checkpoint=ROOT/'fire-models/fire_n.pt';model=YOLO(str(checkpoint));print(json.dumps({'model_names':model.names,'phase':phase}),flush=True)
    names={int(k):v.strip().lower() for k,v in model.names.items()};assert set(names.values())=={'fire','smoke'}
    path=OUT/(phase+'_specialist_raw.json');raw=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    for i,key in enumerate(keys,1):
        if key in raw:continue
        asset=items[key]['asset'];source=DATA/asset['path'];assert file_sha(source)==asset['sha256']
        with Image.open(source) as image:
            result=model.predict(image.convert('RGB'),imgsz=640,conf=.05,iou=.7,max_det=300,device=0 if torch.cuda.is_available() else 'cpu',verbose=False,save=False)[0]
        raw[key]=[{'id':f'{i:08x}{j:024x}','class':names[int(b.cls.item())],'xyxy':b.xyxy[0].cpu().tolist(),'confidence':float(b.conf.item()),'phrase':names[int(b.cls.item())]} for j,b in enumerate(result.boxes)]
        path.write_text(json.dumps(raw),encoding='utf-8')
        if i%10==0:print(json.dumps({'stage':'specialist_inference','done':i,'total':len(keys),'phase':phase}),flush=True)
    old=evaluate(refs,{k:items[k]['proposal']['boxes'] for k in keys},keys);trials=[]
    for ft in [.10,.15,.20,.25,.30,.40,.50,.60]:
        for st in [.10,.15,.20,.25,.30,.40,.50,.60]:
            preds={k:[b for b in raw[k] if b['confidence']>=(ft if b['class']=='fire' else st)] for k in keys};metrics=evaluate(refs,preds,keys);g=gate(old,metrics)
            trials.append({'fire_threshold':ft,'smoke_threshold':st,'metrics':metrics,'gate':g,'mean_f1':sum(metrics[r]['overall']['f1'] for r in metrics)/2})
    trials.sort(key=lambda v:(v['gate']['passed'],v['mean_f1']),reverse=True)
    report={'scope':'Specialist-only diagnostic component, not eligible for pretending to be GroundingDINO or zero-shot','sha256':file_sha(checkpoint),'model_names':names,'baseline':old,'trials':trials}
    (OUT/(phase+'_specialist_results.json')).write_text(json.dumps(report,indent=2),encoding='utf-8');best=trials[0]
    print(json.dumps({'stage':'specialist_done','eligible':sum(t['gate']['passed'] for t in trials),'parameters':{k:best[k] for k in ('fire_threshold','smoke_threshold')},'metrics':{r:best['metrics'][r]['overall'] for r in best['metrics']}}),flush=True)

if __name__=='__main__':main()
