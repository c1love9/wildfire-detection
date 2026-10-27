"""Pinned official Base model, selected on the original tuning batch only."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace
import threading
import json
import sys
import torch
from PIL import Image
from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
from grounding_prelabel import GroundingPrelabeler,PrelabelConfig
from dataset_store import file_sha
from trial import ROOT,OUT,DATA,load_data,raw_joint,filtered,evaluate,gate,SPECS
from multilabel_trial import expand
from fusion_trial import combine

BASE_ID='IDEA-Research/grounding-dino-base'
BASE_REV='12bdfa3120f3e7ec7b434d90674b3396eccf88eb'
BASE_SHA='5548f844c928c4b6f411fa8cbcc2bfa8dbbba437cb1d513975519f93c2a9ed21'

class OfficialBase:
    def __init__(self):
        folder=ROOT/'artifacts/models/grounding-dino-base'/BASE_REV;receipt=json.loads((folder/'model_receipt.json').read_text(encoding='utf-8'))
        assert (receipt['model_id'],receipt['revision'],receipt['files']['model.safetensors'])==(BASE_ID,BASE_REV,BASE_SHA)
        for name,value in receipt['files'].items():assert file_sha(folder/name)==value
        self.processor=AutoProcessor.from_pretrained(str(folder),local_files_only=True,use_fast=False)
        self.device='cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model=AutoModelForZeroShotObjectDetection.from_pretrained(str(folder),local_files_only=True,use_safetensors=True,disable_custom_kernels=True).to(self.device).eval();self._lock=threading.Lock()
        self.metadata={'model_id':BASE_ID,'revision':BASE_REV,'files_sha256':receipt['files'],'device':self.device,'dtype':'float32','fine_tuned':False,'disable_custom_kernels':True,'hardware':torch.cuda.get_device_name(0) if self.device.startswith('cuda') else 'CPU'}

BASE_SPECS={'base_joint_basic':SPECS['joint_basic'],'base_joint_plume_background':SPECS['joint_plume_background'],
 'base_joint_descriptive_background':{'terms':{'fire':['fire','flame'],'smoke':['smoke','smoke plume']},'negative':['cloud','sun','lamp','fog']}}

def raw_independent(model,image):
    default=PrelabelConfig();params=SimpleNamespace(**asdict(default));params.model_id=BASE_ID;params.revision=BASE_REV;params.weights_sha256=BASE_SHA;params.box_threshold=.10;params.prompts=default.prompts
    return GroundingPrelabeler.predict(model,image,params)['boxes']

def main():
    snap,refs,groups=load_data();items={r['asset']['id']:r for r in snap};keys=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))['batches']['tuning']
    torch.set_num_threads(min(8,torch.get_num_threads()));model=OfficialBase();raw={}
    for name in ['base_independent']+list(BASE_SPECS):
        path=OUT/('tuning_'+name+'_raw.json');values=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        for i,key in enumerate(keys,1):
            if key in values:continue
            asset=items[key]['asset'];p=DATA/asset['path'];assert file_sha(p)==asset['sha256']
            with Image.open(p) as image:
                rgb=image.convert('RGB');values[key]=raw_independent(model,rgb) if name=='base_independent' else raw_joint(model,rgb,BASE_SPECS[name],640)
            path.write_text(json.dumps(values),encoding='utf-8')
            if i%10==0:print(json.dumps({'stage':'base_tuning','spec':name,'done':i,'total':len(keys)}),flush=True)
        raw[name]=values
    old=evaluate(refs,{k:items[k]['proposal']['boxes'] for k in keys},keys);yr=json.loads((OUT/'tuning_specialist_raw.json').read_text(encoding='utf-8'));trials=[]
    for name,values in raw.items():
        if name!='base_independent':values={k:expand(v) for k,v in values.items()}
        for ft in [.15,.20,.25,.30,.35,.40,.45,.50]:
            for st in [.15,.20,.25,.30,.35,.40,.45,.50]:
                for margin in ([0] if name in ('base_independent','base_joint_basic') else [0,.05,.10]):
                    dp={'fire_threshold':ft,'smoke_threshold':st,'margin':margin,'nms_iou':.5};dino={k:filtered(values[k],**dp) for k in keys}
                    for yp in [None,{'fire_threshold':.15,'smoke_threshold':.5,'order':'dino_first'}]:
                        pred=dino if yp is None else {k:combine(dino[k],yr[k],yp['fire_threshold'],yp['smoke_threshold'],yp['order']) for k in keys};m=evaluate(refs,pred,keys);g=gate(old,m)
                        trials.append({'spec_name':name,'model':model.metadata,'multilabel':name!='base_independent','dino_parameters':dp,'yolo_parameters':yp,'metrics':m,'gate':g,'mean_f1':sum(m[r]['overall']['f1'] for r in m)/2})
    trials.sort(key=lambda t:(t['gate']['passed'],t['mean_f1']),reverse=True);best=trials[0]
    (OUT/'base_tuning_results.json').write_text(json.dumps({'model':model.metadata,'baseline':old,'trials':trials},indent=2),encoding='utf-8');(OUT/'base_selected.json').write_text(json.dumps(best,indent=2),encoding='utf-8')
    print(json.dumps({'stage':'base_tuning_complete','eligible':sum(t['gate']['passed'] for t in trials),'selected':{k:best[k] for k in ('spec_name','dino_parameters','yolo_parameters','gate','mean_f1')},'metrics':{r:best['metrics'][r]['overall'] for r in best['metrics']}}),flush=True)

if __name__=='__main__':main()
