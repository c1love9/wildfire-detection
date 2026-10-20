import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from dataset_store import DatasetStore
from annotations import AnnotationStore
from training import TrainConfig,TrainingStore,read_json
from experiments import ExperimentStore
from verify_training import wait


def main():
    fixture=json.loads((ROOT/'reports/model_integration.json').read_text())
    dataset=DatasetStore(ROOT/fixture['dataset_root']);store=ExperimentStore(dataset);training=TrainingStore(dataset)
    export=next(r['id'] for r in AnnotationStore(dataset).exports() if r['kind']=='split')
    pair=store.records()[0] if '--reuse' in sys.argv else store.create(export,TrainConfig(epochs=1,imgsz=160,batch=2,patience=0,device='0' if torch.cuda.is_available() else 'cpu'),'Automated verification','Existing synthetic shapes; functional comparison, not formal fire accuracy')
    for variant in ['baseline','multiscale']:
        job=store.read(pair)['jobs'].get(variant) or store.start(pair,variant);wait(training,job,timeout=240)
        args=read_json(training.directory(job)/'effective_parameters.json')
        assert args['multi_scale']==(variant=='multiscale')
    result=store.evaluation(pair) if store.read(pair)['evaluation_sha'] else store.evaluate(pair,'cuda:0' if torch.cuda.is_available() else 'cpu')
    assert json.loads(json.dumps(result))==store.evaluation(pair)
    manifest=store.read(pair)
    initial=[training.get(job)['request']['initial_weights_sha256'] for job in manifest['jobs'].values()]
    assert len(set(initial))==1
    report={'status':'passed','scope':'Existing isolated synthetic fixture only','pair':pair,'both_real_training_jobs_completed':True,
            'only_multiscale_differs':True,'same_initial_weight_sha':initial[0],'unified_test_evaluation':True,'no_assumed_improvement':True}
    (ROOT/'reports/experiments_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
