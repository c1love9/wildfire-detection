"""Prepare only an existing synthetic image task for browser workflow checks."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dataset_store import DatasetStore
from quality import QualityStore
import numpy as np
from PIL import Image


def main():
    fixture=json.loads((ROOT/'reports/model_integration.json').read_text())
    dataset=DatasetStore(ROOT/fixture['dataset_root']);store=QualityStore(dataset)
    key=store.annotations.catalog()[0]['id']
    task,tokens=store.create([key],'Browser simulated A','Browser simulated B','Browser simulated adjudicator')
    output=ROOT/'artifacts/completion_checks/quality_access.json'
    asset=store.annotations.image(key)
    mask=np.zeros((asset['height'],asset['width']),np.uint8);mask[:asset['height']//2]=255
    Image.fromarray(mask).save(output.parent/'reference.png')
    mask[asset['height']//2:asset['height']//2+1]=255
    Image.fromarray(mask).save(output.parent/'prediction.png')
    output.write_text(json.dumps({'task':task,'tokens':tokens,'asset_id':key,'dataset_root':str(dataset.root)}),encoding='utf-8')
    print('Synthetic browser task prepared; codes kept only in ignored local output.')


if __name__=='__main__':main()
