"""Higher-resolution candidates, still scored only on frozen tuning images."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import json
from pathlib import Path
from PIL import Image
import torch
from grounding_prelabel import GroundingPrelabeler
from dataset_store import file_sha
from trial import OUT,DATA,load_data,raw_joint,SPECS,filtered,evaluate,gate
from fusion_trial import combine

def main():
    snap,refs,groups=load_data();items={r['asset']['id']:r for r in snap};keys=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))['batches']['tuning']
    torch.set_num_threads(min(8,torch.get_num_threads()));model=GroundingPrelabeler('cuda:0' if torch.cuda.is_available() else 'cpu');raw={}
    for base in ['joint_basic','joint_plume_background']:
        name=base+'_800';p=OUT/('tuning_'+name+'_raw.json');values=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
        for i,key in enumerate(keys,1):
            if key in values:continue
            asset=items[key]['asset'];source=DATA/asset['path'];assert file_sha(source)==asset['sha256']
            with Image.open(source) as image:values[key]=raw_joint(model,image.convert('RGB'),SPECS[base],800)
            p.write_text(json.dumps(values),encoding='utf-8')
            if i%10==0:print(json.dumps({'stage':'high_resolution','spec':name,'done':i,'total':len(keys)}),flush=True)
        raw[name]=values
    baseline=evaluate(refs,{k:items[k]['proposal']['boxes'] for k in keys},keys);yolo=json.loads((OUT/'tuning_specialist_raw.json').read_text(encoding='utf-8'));trials=[]
    for name,values in raw.items():
        for ft in [.15,.20,.25,.30,.35,.40]:
            for st in [.15,.20,.25,.30,.35,.40]:
                dino={k:filtered(values[k],ft,st,0,.5) for k in keys}
                for yp in [None,{'fire_threshold':.15,'smoke_threshold':.3,'order':'dino_first'}, {'fire_threshold':.25,'smoke_threshold':.5,'order':'dino_first'}, {'fire_threshold':.15,'smoke_threshold':.5,'order':'yolo_first'}]:
                    pred=dino if yp is None else {k:combine(dino[k],yolo[k],yp['fire_threshold'],yp['smoke_threshold'],yp['order']) for k in keys}
                    metrics=evaluate(refs,pred,keys);g=gate(baseline,metrics)
                    trials.append({'spec_name':name,'dino_parameters':{'fire_threshold':ft,'smoke_threshold':st,'margin':0,'nms_iou':.5},'yolo_parameters':yp,'metrics':metrics,'gate':g,'mean_f1':sum(metrics[r]['overall']['f1'] for r in metrics)/2})
    prior=json.loads((OUT/'fusion_tuning_results.json').read_text(encoding='utf-8'))['trials']
    trials.extend(prior);trials.sort(key=lambda t:(t['gate']['passed'],t['mean_f1']),reverse=True)
    report={'scope':'Second family locked before second confirmation; retains original gate, not the exploratory +3 TP ranking filter','baseline':baseline,'trials':trials}
    (OUT/'round2_tuning_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');(OUT/'round2_selected.json').write_text(json.dumps(trials[0],indent=2),encoding='utf-8')
    best=trials[0];print(json.dumps({'stage':'round2_tuning_complete','eligible':sum(t['gate']['passed'] for t in trials),'selected':{k:best[k] for k in ('spec_name','dino_parameters','yolo_parameters','mean_f1','gate')},'metrics':{r:best['metrics'][r]['overall'] for r in best['metrics']}}),flush=True)

if __name__=='__main__':main()
