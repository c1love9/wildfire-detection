import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from dataset_store import DatasetStore
from model_catalog import fire_models
from inference_ui import load_detector
from grounding_prelabel import GroundingPrelabeler
from comparison import ComparisonStore,DinoDetector


def main():
    torch.set_num_threads(8)
    fixture=json.loads((ROOT/'reports/model_integration.json').read_text())
    dataset=DatasetStore(ROOT/fixture['dataset_root']);device='cuda:0' if torch.cuda.is_available() else 'cpu'
    choice=fire_models(ROOT,dataset.root)[fixture['model_key']]
    yolo=load_detector(choice,device,640);dino=DinoDetector(GroundingPrelabeler(device))
    store=ComparisonStore(dataset)
    identifier=store.run(fixture['training_id'],yolo,dino,'Automated functional verification','Existing synthetic fixture, not a formal fire evaluation')
    result=store.result(identifier)
    assert set(result['models'])=={'YOLO','GroundingDINO'}
    assert all(row['mean_ms_per_image']>0 for row in result['models'].values())
    assert result['protocol']['operating_conf']==.25
    report={'status':'passed','scope':'Existing synthetic model/dataset only; no formal accuracy claim','comparison_id':identifier,
            'same_test_images':len(result['references']),'device':device,'request_and_result_verified':True}
    (ROOT/'reports/comparison_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
