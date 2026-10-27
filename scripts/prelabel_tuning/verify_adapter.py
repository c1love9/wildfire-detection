"""Real adapter must reproduce the frozen experimental candidate."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from pathlib import Path
import json
import sys
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/prelabel_optimization_20261027';sys.path.insert(0,str(ROOT))
from optimized_prelabel import OptimizedPrelabeler,OptimizedPrelabelConfig
from comparison import iou
from trial import load_data
from annotations import validate_boxes

snap,refs,groups=load_data();items={r['asset']['id']:r for r in snap};r=json.loads((OUT/'base_validation.json').read_text(encoding='utf-8'));expected=r['candidates']['f1_priority']['predictions'];p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));q27=json.loads((OUT/'round2_confirmation_protocol.json').read_text(encoding='utf-8'));q93=json.loads((OUT/'remaining93_protocol.json').read_text(encoding='utf-8'))
from trial import label
keys=[]
for pool in [p['batches']['tuning'],p['batches']['confirmation'],q27['asset_ids'],q93['asset_ids']]:
    for kind in ('empty','fire','fire+smoke','smoke'):
        match=next((key for key in pool if label(key,refs)==kind),None)
        if match is not None:keys.append(match)
critical=snap[102]['asset']['id']
if critical not in keys:keys.append(critical)
torch.set_num_threads(min(8,torch.get_num_threads()));config=OptimizedPrelabelConfig.load();model=OptimizedPrelabeler('cuda:0' if torch.cuda.is_available() else 'cpu');checks=[]
for key in keys:
    asset=items[key]['asset']
    with Image.open(ROOT/'data/fire_dataset'/asset['path']) as image:new=model.predict(image,config)['boxes']
    validate_boxes(new,asset['width'],asset['height']);old=expected[key];unused=set(range(len(new)));pairs=[]
    for box in old:
        overlap,index=max(((iou(box['xyxy'],b['xyxy']),i) for i,b in enumerate(new) if i in unused and b['class']==box['class']),default=(0,None))
        if index is not None and overlap>=.999:
            unused.remove(index);pairs.append(abs(box['confidence']-new[index]['confidence']))
    assert len(old)==len(new)==len(pairs) and all(v<1e-4 for v in pairs),(key,len(old),len(new),pairs)
    checks.append({'asset_id':key,'box_count':len(new),'matches_frozen_experiment':True})
    print(json.dumps({'stage':'adapter_check','done':len(checks),'total':len(keys)}),flush=True)
report={'status':'passed','actual_model':model.run_metadata(config),'checks':checks,'scope':'Real CUDA adapter vs frozen experiment; no formal review or proposal writes.'};(OUT/'adapter_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({'status':'passed','images':len(keys)}),flush=True)
