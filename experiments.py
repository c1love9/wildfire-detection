"""Single-factor training pairs and replayed alarm-rule ablation."""
import torch
import cv2
import uuid
import time
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
from PIL import Image
from training import TrainConfig,TrainingStore,inspect_export,write_json,read_json,verify_inventory,bounded
from dataset_store import file_sha,now
from comparison import PROTOCOL,letterbox,transform_boxes,evaluate_predictions,synchronize
from alarm import AlarmConfig,AlarmEngine
from model_catalog import fire_models
from inference_ui import load_detector

ROOT=Path(__file__).resolve().parent


class ExperimentStore:
    def __init__(self,dataset):
        self.dataset=dataset;self.root=dataset.root/'experiments';self.root.mkdir(exist_ok=True)
        with dataset.connection() as db:db.execute('CREATE TABLE IF NOT EXISTS experiments(id TEXT PRIMARY KEY,manifest_sha TEXT NOT NULL)')

    def records(self):
        with self.dataset.connection() as db:return [r[0] for r in db.execute('SELECT id FROM experiments ORDER BY rowid DESC')]

    def read(self,identifier):
        with self.dataset.connection() as db:row=db.execute('SELECT manifest_sha FROM experiments WHERE id=?',(identifier,)).fetchone()
        path=bounded(self.root,identifier)/'manifest.json'
        if row is None or file_sha(path)!=row[0]:raise OSError('实验清单校验失败。')
        return read_json(path)

    def save(self,manifest):
        path=bounded(self.root,manifest['id'])/'manifest.json';path.parent.mkdir(exist_ok=True)
        write_json(path,manifest)
        with self.dataset.connection() as db:db.execute('INSERT OR REPLACE INTO experiments VALUES(?,?)',(manifest['id'],file_sha(path)))

    def create(self,export_id,config,actor,note):
        if not actor.strip() or not note.strip():raise ValueError('请填写执行者和说明。')
        _,receipt=inspect_export(self.dataset,export_id)
        manifest={'id':'pair_'+uuid.uuid4().hex,'created_at':now(),'export_id':export_id,'dataset_fingerprint':receipt['fingerprint'],
                  'actor':actor,'note':note,'initial_sha256':file_sha(ROOT/'general-models/yolov8n.pt'),
                  'configs':{name:asdict(replace(config,multi_scale=value)) for name,value in [('baseline',False),('multiscale',True)]},'jobs':{},'evaluation_sha':None}
        self.save(manifest);return manifest['id']

    def start(self,identifier,variant,launch=True):
        manifest=self.read(identifier)
        if variant not in manifest['configs'] or variant in manifest['jobs']:raise ValueError('请选择尚未启动的实验组；不覆盖旧任务。')
        _,receipt=inspect_export(self.dataset,manifest['export_id'])
        if receipt['fingerprint']!=manifest['dataset_fingerprint'] or file_sha(ROOT/'general-models/yolov8n.pt')!=manifest['initial_sha256']:
            raise OSError('数据或初始权重已改变，请创建新实验。')
        job=TrainingStore(self.dataset).start_train(manifest['export_id'],TrainConfig(**manifest['configs'][variant]),manifest['actor'],manifest['id']+' '+variant+' '+manifest['note'],launch=launch)
        manifest['jobs'][variant]=job;self.save(manifest);return job

    def evaluate(self,identifier,device='cpu'):
        manifest=self.read(identifier)
        if manifest['evaluation_sha']:raise ValueError('此实验已经评估，保留原结果。')
        training=TrainingStore(self.dataset);choices=fire_models(ROOT,self.dataset.root)
        results={};fingerprint=None;reference_ids=None
        for variant in ('baseline','multiscale'):
            job_id=manifest['jobs'].get(variant)
            if not job_id:raise ValueError('请先完成两组训练。')
            training.result(job_id);directory=training.directory(job_id)
            receipt=read_json(directory/'dataset_receipt.json');verify_inventory(directory/'input',receipt['inventory'])
            if fingerprint is not None and fingerprint!=receipt['fingerprint']:raise OSError('两组训练数据不同。')
            fingerprint=receipt['fingerprint']
            detector=load_detector(choices['trained:'+job_id+':best'],device,640)
            references={};predictions={};times=[]
            images=sorted((directory/'input/images/test').glob('*.png'))
            if {'images/test/'+p.name for p in images}!={p for p in receipt['inventory'] if p.startswith('images/test/')}:
                raise OSError('测试图像清单被修改。')
            for index,path in enumerate(images):
                image=Image.open(path).convert('RGB');_,transform=letterbox(image)
                boxes=[]
                for line in (directory/'input/labels/test'/(path.stem+'.txt')).read_text().splitlines():
                    cls,x,y,w,h=map(float,line.split());boxes.append({'class':['fire','smoke'][int(cls)],'xyxy':[(x-w/2)*image.width,(y-h/2)*image.height,(x+w/2)*image.width,(y+h/2)*image.height]})
                references[path.stem]=transform_boxes(boxes,transform)
                def predict():
                    prepared,_=letterbox(image)
                    return detector.predict_frame(cv2.cvtColor(np.array(prepared),cv2.COLOR_RGB2BGR),.001,.5).detections
                if index==0:predict()
                synchronize(device);started=time.perf_counter();predictions[path.stem]=predict();synchronize(device);times.append((time.perf_counter()-started)*1000)
            if reference_ids is not None and reference_ids!=references:raise OSError('参考标签或图像不一致。')
            reference_ids=references
            verify_inventory(directory/'input',receipt['inventory'])
            results[variant]={**evaluate_predictions(references,predictions),'mean_ms':float(np.mean(times)),'predictions':predictions,'model':detector.metadata}
        delta={k:results['multiscale']['overall'][k]-results['baseline']['overall'][k] if results['baseline']['overall'][k] is not None else None for k in results['baseline']['overall']}
        output={'protocol':PROTOCOL,'dataset_fingerprint':fingerprint,'results':results,'difference_multiscale_minus_baseline':delta,'reference':reference_ids,
                'warning':'An optimization experiment; improvement is not assumed. Formal conclusions require actual independent data.'}
        path=bounded(self.root,identifier)/'evaluation.json';write_json(path,output);manifest['evaluation_sha']=file_sha(path);self.save(manifest)
        return output

    def evaluation(self,identifier):
        manifest=self.read(identifier);path=bounded(self.root,identifier)/'evaluation.json'
        if not manifest['evaluation_sha'] or file_sha(path)!=manifest['evaluation_sha']:raise OSError('评估结果缺失或校验失败。')
        return read_json(path)


def alarm_ablation(frames,config=None):
    config=config or AlarmConfig();engine=AlarmEngine(config)
    single_events=confirmed_events=0;previous=False;first_single=first_confirmed=None
    for number,(seconds,boxes) in enumerate(frames,1):
        detected=any(b['class'] in ('fire','smoke') and b['confidence']>=(config.fire_confidence if b['class']=='fire' else config.smoke_confidence) for b in boxes)
        single_events+=int(detected and not previous);previous=detected
        if detected and first_single is None:first_single=seconds
        state=engine.update(number,seconds,boxes)
        confirmed_events+=int(state.level>0 and state.previous_level==0)
        if state.level and first_confirmed is None:first_confirmed=seconds
    return {'single_frame_trigger_episodes':single_events,'continuous_confirmed_events':confirmed_events,
            'first_single_seconds':first_single,'first_confirmed_seconds':first_confirmed,
            'note':'Same detection sequence; trigger episode counts are not false-alarm rates without event reference labels.'}
