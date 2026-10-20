"""Shared detector evaluation over a verified, immutable training test snapshot."""
import torch
import cv2
import json
import time
import uuid
import shutil
import platform
import sys
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
from PIL import Image
from dataset_store import file_sha, now
from training import TrainingStore, read_json, write_json, verify_inventory, bounded
from grounding_prelabel import PrelabelConfig

PROTOCOL={'version':'common_detection_v1','imgsz':640,'batch':1,'precision':'FP32','candidate_conf':.001,
          'operating_conf':.25,'match_iou':.5,'ap_ious':[round(.5+i*.05,2) for i in range(10)],'nms_iou':.5,'max_det':300,
          'timing':'warmed synchronous preprocessing+model+postprocessing; excludes loading, disk, UI, network',
          'pr_point':'fixed confidence .25; macro over classes with reference objects; absent class=null'}


def iou(first, second):
    x1,y1=max(first[0],second[0]),max(first[1],second[1])
    x2,y2=min(first[2],second[2]),min(first[3],second[3])
    intersection=max(0,x2-x1)*max(0,y2-y1)
    union=max(0,first[2]-first[0])*max(0,first[3]-first[1])+max(0,second[2]-second[0])*max(0,second[3]-second[1])-intersection
    return intersection/union if union else 0.


def letterbox(image, size=640):
    width,height=image.size; scale=min(size/width,size/height)
    resized=(round(width*scale),round(height*scale));left,top=(size-resized[0])//2,(size-resized[1])//2
    output=Image.new('RGB',(size,size),(114,114,114));output.paste(image.resize(resized),(left,top))
    return output, (resized[0]/width,resized[1]/height,left,top)


def transform_boxes(boxes, transform):
    sx,sy,left,top=transform
    return [{**box,'xyxy':[box['xyxy'][0]*sx+left,box['xyxy'][1]*sy+top,box['xyxy'][2]*sx+left,box['xyxy'][3]*sy+top]} for box in boxes]


def evaluate_predictions(references, predictions):
    from ultralytics.utils.metrics import compute_ap
    rows=[]
    for label in ('fire','smoke'):
        targets={key:[b['xyxy'] for b in boxes if b['class']==label] for key,boxes in references.items()}
        total=sum(map(len,targets.values()))
        candidates=sorted([(float(b['confidence']),key,b['xyxy']) for key,boxes in predictions.items() for b in boxes if b['class']==label],key=lambda b:-b[0])
        scores=np.array([b[0] for b in candidates]);tp=np.zeros((len(candidates),10))
        for column,threshold in enumerate(PROTOCOL['ap_ious']):
            used={key:set() for key in targets}
            for index,(_,key,box) in enumerate(candidates):
                available=[(iou(box,gt),j) for j,gt in enumerate(targets.get(key,[])) if j not in used.get(key,set())]
                if available:
                    overlap,j=max(available)
                    if overlap+1e-9>=threshold:tp[index,column]=1;used[key].add(j)
        selected=scores>=PROTOCOL['operating_conf'];true=int(tp[selected,0].sum());false=int(selected.sum())-true
        precision=true/(true+false) if true+false else 0.;recall=true/total if total else 0.
        aps=[]
        for column in range(10):
            cumulative=np.cumsum(tp[:,column]);aps.append(float(compute_ap(cumulative/max(total,1),cumulative/np.arange(1,len(candidates)+1))[0]) if len(candidates) and total else 0.)
        rows.append({'class':label,'instances':total,'tp':true,'fp':false,'fn':total-true,
                     'precision':precision if total else None,'recall':recall if total else None,
                     'f1':2*precision*recall/(precision+recall) if total and precision+recall else 0. if total else None,
                     'map50':aps[0] if total else None,'map50_95':float(np.mean(aps)) if total else None})
    present=[row for row in rows if row['instances']]
    overall={key:sum(row[key] for row in present)/len(present) if present else None for key in ['precision','recall','f1','map50','map50_95']}
    negative=[key for key,boxes in references.items() if not boxes]
    false_images=sum(any(b['confidence']>=.25 for b in predictions.get(key,[])) for key in negative)
    return {'overall':overall,'per_class':rows,'negative_images':len(negative),'negative_images_with_false_detection':false_images,
            'negative_image_false_positive_rate':false_images/len(negative) if negative else None}


class DinoDetector:
    def __init__(self, model, config=None):
        self.model=model;self.device=model.device
        self.config=replace(config or PrelabelConfig(),shortest_edge=640,longest_edge=640,box_threshold=.001,nms_iou=.5,max_boxes_per_class=100)
        self.metadata=model.run_metadata(self.config)

    def predict_boxes(self,image):
        return sorted(self.model.predict(image,self.config)['boxes'],key=lambda b:-b['confidence'])[:300]


def synchronize(device):
    if device.startswith('cuda'):torch.cuda.synchronize(device)


def predict_pair(image,yolo,dino,confidence=.25):
    square,transform=letterbox(image)
    yresult=yolo.predict_frame(cv2.cvtColor(np.array(square),cv2.COLOR_RGB2BGR),.001,.5)
    dboxes=dino.predict_boxes(square)
    def paint(boxes):
        from PIL import ImageDraw
        output=square.copy();draw=ImageDraw.Draw(output)
        for box in boxes:
            if box['confidence']>=confidence:
                draw.rectangle(box['xyxy'],outline='red' if box['class']=='fire' else 'cyan',width=2)
                draw.text(box['xyxy'][:2],f"{box['class']} {box['confidence']:.2f}",fill='white')
        return output
    return paint(yresult.detections),paint(dboxes)


class ComparisonStore:
    def __init__(self,dataset):
        self.dataset=dataset;self.root=dataset.root/'comparisons';self.root.mkdir(exist_ok=True)
        with dataset.connection() as db:
            db.execute('CREATE TABLE IF NOT EXISTS comparisons(id TEXT PRIMARY KEY, created_at TEXT NOT NULL, request_sha TEXT NOT NULL, result_sha TEXT, status TEXT NOT NULL)')

    def records(self):
        with self.dataset.connection() as db:return [dict(r) for r in db.execute('SELECT * FROM comparisons ORDER BY created_at DESC')]

    def result(self,identifier):
        record=next(r for r in self.records() if r['id']==identifier)
        directory=bounded(self.root,identifier)
        if record['status']!='completed' or file_sha(directory/'request.json')!=record['request_sha'] or file_sha(directory/'result.json')!=record['result_sha']:
            raise OSError('对比任务未完成或文件校验失败。')
        return read_json(directory/'result.json')

    def run(self,training_id,yolo,dino,actor,note,progress=None):
        if not actor.strip() or not note.strip():raise ValueError('请填写执行者与实验说明。')
        training=TrainingStore(self.dataset);job=training.get(training_id);trained=training.result(training_id)
        if job['kind']!='train' or job['status']!='completed' or yolo.metadata.get('training_id')!=training_id:
            raise ValueError('统一评测须使用所选训练任务的已完成best模型。')
        parent=training.directory(training_id);receipt=read_json(parent/'dataset_receipt.json')
        if trained['dataset_fingerprint']!=receipt['fingerprint']:raise OSError('数据指纹不一致。')
        verify_inventory(parent/'input',receipt['inventory'])
        identifier='compare_'+uuid.uuid4().hex;directory=self.root/identifier;directory.mkdir()
        request={'id':identifier,'training_id':training_id,'actor':actor,'note':note,'protocol':PROTOCOL,
                 'dataset_fingerprint':receipt['fingerprint'],'yolo':yolo.metadata,'dino':dino.metadata,
                 'fine_tuned':{'yolo':True,'grounding_dino':False},'open_model_also_used_for_preannotation':True,
                 'environment':{'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'device':yolo.device,
                                'hardware':torch.cuda.get_device_name(0) if yolo.device.startswith('cuda') else platform.processor()},'created_at':now()}
        if yolo.device!=dino.device:raise ValueError('两个模型必须使用同一计算设备。')
        write_json(directory/'request.json',request)
        with self.dataset.connection() as db:db.execute('INSERT INTO comparisons VALUES(?,?,?,?,?)',(identifier,now(),file_sha(directory/'request.json'),None,'running'))
        try:
            references={};predictions={'YOLO':{},'GroundingDINO':{}};times={key:[] for key in predictions};inventory={}
            images=sorted((parent/'input/images/test').glob('*.png'))
            expected={name for name in receipt['inventory'] if name.startswith('images/test/')}
            if {'images/test/'+path.name for path in images} != expected:
                raise OSError('冻结测试集出现额外或缺失图像。')
            if not images:raise ValueError('独立测试集没有图片。')
            for index,path in enumerate(images):
                label_path=parent/'input/labels/test'/(path.stem+'.txt')
                target=directory/'input';target.mkdir(exist_ok=True)
                for source in (path,label_path):
                    destination=target/source.name;shutil.copy2(source,destination);inventory['input/'+source.name]=file_sha(destination)
                image=Image.open(target/path.name).convert('RGB');square,transform=letterbox(image)
                boxes=[]
                for line in (target/label_path.name).read_text().splitlines():
                    cls,cx,cy,w,h=map(float,line.split());boxes.append({'class':['fire','smoke'][int(cls)],'xyxy':[(cx-w/2)*image.width,(cy-h/2)*image.height,(cx+w/2)*image.width,(cy+h/2)*image.height]})
                references[path.stem]=transform_boxes(boxes,transform)
                for name in predictions:
                    def predict():
                        prepared,_=letterbox(image)
                        return yolo.predict_frame(cv2.cvtColor(np.array(prepared),cv2.COLOR_RGB2BGR),.001,.5).detections if name=='YOLO' else dino.predict_boxes(prepared)
                    if index==0:predict()
                    synchronize(yolo.device);started=time.perf_counter();boxes=predict();synchronize(yolo.device)
                    times[name].append((time.perf_counter()-started)*1000);predictions[name][path.stem]=boxes
                if progress:progress((index+1)/len(images))
            verify_inventory(parent/'input',receipt['inventory']);verify_inventory(directory,inventory)
            result={'id':identifier,'created_at':now(),'protocol':PROTOCOL,'dataset_fingerprint':receipt['fingerprint'],'inventory':inventory,
                    'models':{name:{**evaluate_predictions(references,values),'mean_ms_per_image':float(np.mean(times[name])),
                                    'parameters':sum(p.numel() for p in (yolo.model.model if name=='YOLO' else dino.model.model).parameters()),
                                    'fps':1000/float(np.mean(times[name])),'file_bytes':Path(yolo.metadata['path']).stat().st_size if name=='YOLO' else sum(p.stat().st_size for p in Path(dino.model.model.config._name_or_path).glob('*.safetensors'))}
                              for name,values in predictions.items()},
                    'references':references,'predictions':predictions,'timings_ms':times,
                    'note':'Tool output, not an accuracy guarantee. Same GroundingDINO also used for preannotation; reference independence must be established by actual human protocol.'}
            write_json(directory/'result.json',result)
            with self.dataset.connection() as db:db.execute('UPDATE comparisons SET result_sha=?,status=? WHERE id=?',(file_sha(directory/'result.json'),'completed',identifier))
            return identifier
        except Exception:
            with self.dataset.connection() as db:db.execute('UPDATE comparisons SET status=? WHERE id=?',('failed',identifier))
            raise
