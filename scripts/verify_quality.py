"""Prepare a fresh isolated synthetic task; never add formal human records."""
import sys,json,uuid
from io import BytesIO
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dataset_store import DatasetStore
from quality import QualityStore
from annotations import AnnotationStore
from grounding_prelabel import PrelabelConfig
import numpy as np
from PIL import Image


def main():
    dataset=DatasetStore(ROOT/'artifacts/dual_annotation_checks'/('browser_'+uuid.uuid4().hex))
    annotations=AnnotationStore(dataset);keys=[]
    for index in range(2):
        image=Image.new('RGB',(96,64),(50+index,60,70));buffer=BytesIO();image.save(buffer,format='PNG')
        batch=dataset.create_batch(f'Synthetic browser {index}',f'isolated{index}','demo','Synthetic automated UI fixture, not actual internship data or human work',['unknown'])
        keys.append(dataset.import_image(buffer.getvalue(),batch)['asset_id'])
    class Model:
        def run_metadata(self,config):return {'model':'synthetic_known_output','parameters':{'purpose':'Automated fixture, no real model accuracy claim'}}
        def predict(self,image,config):return {'boxes':[{'class':'fire','xyxy':[20,10,70,50],'confidence':.9}] if image.getpixel((0,0))[0]==50 else []}
    annotations.prelabel(keys,Model(),PrelabelConfig(),'Synthetic automated candidate')
    store=QualityStore(dataset)
    task,tokens=store.create(keys,'Browser simulated A','Browser simulated B','Browser simulated manager')
    key=keys[0]
    output=ROOT/'artifacts/completion_checks/quality_access.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    asset=store.annotations.image(key)
    mask=np.zeros((asset['height'],asset['width']),np.uint8);mask[:asset['height']//2]=255
    Image.fromarray(mask).save(output.parent/'reference.png')
    mask[asset['height']//2:asset['height']//2+1]=255
    Image.fromarray(mask).save(output.parent/'prediction.png')
    value={'task':task,'tokens':tokens,'asset_id':key,'asset_ids':keys,'dataset_root':str(dataset.root)}
    output.write_text(json.dumps(value),encoding='utf-8')
    (ROOT/'artifacts/dual_annotation_checks/browser_fixture.json').write_text(json.dumps(value),encoding='utf-8')
    print('Synthetic browser task prepared; codes kept only in ignored local output.')


if __name__=='__main__':main()
