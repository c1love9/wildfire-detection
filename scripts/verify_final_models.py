"""Check real own weights, segmentation, video controls and camera lifecycle."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import cv2
import numpy as np
from PIL import Image
from dataset_store import DatasetStore,now
from inference_ui import load_detector,with_segmentation,_load
from model_catalog import fire_models
from training import TrainingStore,read_json,write_json
from video_detection import VideoSession
from recording import RecordingConfig
from verify_camera import hardware_check
from verify_recording import inspect_media
from alarm_log import AlarmLog,alarm_media_root


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def verify(workspace,skip_camera=False):
    torch.set_num_threads(8);cv2.setNumThreads(2)
    formal=read_json(workspace/'formal_experiments.json');dataset=DatasetStore(ROOT/'data/fire_dataset')
    training=TrainingStore(dataset);choices=fire_models(ROOT,dataset.root)
    checks=[]
    for variant,identifier in formal['jobs'].items():
        result=training.result(identifier)
        assert result['completed_epochs']==50
        choice=choices['trained:'+identifier+':best'];choice.verify()
        assert choice.origin=='trained' and choice.completed_epochs==50 and not choice.error
        checks.append({'variant':variant,'training_id':identifier,'sha256':choice.sha256,'epochs':50})
    selection=choices['trained:'+formal['selection']['training_id']+':best']
    examples=sorted(p for p in (ROOT/'demo-images').glob('*.jpeg') if p.stat().st_size>1000)
    assert examples
    image_checks=[];best_example=examples[0];best_count=-1
    for device in ('cpu','cuda:0'):
        detector=load_detector(selection,device)
        for path in examples[:5]:
            frame=cv2.imread(str(path));result=detector.predict_frame(frame,.25,.5)
            assert result.width==frame.shape[1] and result.height==frame.shape[0]
            assert all(b['class'] in ('fire','smoke') and np.isfinite(b['xyxy']).all() for b in result.detections)
            image_checks.append({'image':path.name,'device':device,'counts':result.counts,'model_sha256':result.model['sha256']})
            if device=='cuda:0' and len(result.detections)>best_count:
                best_example,best_count=path,len(result.detections)
    detector=with_segmentation(load_detector(selection,'cuda:0'),True)
    bgr=cv2.imread(str(best_example));result=detector.predict_frame(bgr,.25,.5)
    assert result.segmentation_status in ('ready','no_targets'),result.segmentation_error
    assert result.masks is not None and result.masks['fire'].shape==bgr.shape[:2]
    areas=result.areas();union=np.logical_or(result.masks['fire'],result.masks['smoke'])
    assert abs(areas['combined']-float(union.mean()))<1e-9
    assert max(areas['fire'],areas['smoke'])<=areas['combined']<=areas['fire']+areas['smoke']+1e-9
    output=workspace/'model_checks';output.mkdir(exist_ok=True)
    Image.fromarray(result.rgb).save(output/'own_model_masks.png')
    for label in ('fire','smoke'):Image.fromarray(result.masks[label].astype(np.uint8)*255).save(output/(label+'_mask.png'))
    log=output/'alarms.sqlite3';videos=[]
    for name in ('fire1.mov','fire2.mov'):
        session=VideoSession(ROOT/'demo-videos'/name,detector,.25,.5,log_path=log,
                             source_name='正式自训模型验证_'+name,
                             recording_config=RecordingConfig(pre_seconds=1,post_seconds=2,output_fps=15))
        try:
            for _ in range(180):
                if session.status!='running':break
                session.step()
            before=session.processed_frames;session.pause()
            assert session.step() is None and session.processed_frames==before
            session.resume();session.step();assert session.processed_frames==before+1
            videos.append({'source':name,'frames':session.processed_frames,'pause_does_not_sample':True,'resumed':True})
        finally:session.stop();session.wait_for_media(timeout=45)
        assert not session.capture.isOpened()
    store=AlarmLog(log);events=store.events()
    media=[inspect_media(store,event,alarm_media_root(log)) for event in events if store.media(event['id']) and store.media(event['id'])['status'] in ('ready','partial')]
    assert events and media,'No real self-trained model alarm/media; inspect samples before declaring integration passed.'
    # Camera pictures never touch disk and recording is not enabled.
    camera={'status':'not_run','personal_images_saved':False}
    if not skip_camera:
        try:
            camera={'status':'passed',**hardware_check(load_detector(selection,'cuda:0'))}
        except (ValueError,RuntimeError,OSError,AssertionError) as exc:
            camera={'status':'unavailable_or_failed','reason':str(exc),'personal_images_saved':False}
    receipt=read_json(workspace/'review_receipt.json')
    db=sqlite3.connect(dataset.path.as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    reviews=[dict(r) for r in db.execute('SELECT r.* FROM annotation_reviews r JOIN (SELECT asset_id,MAX(seq) seq FROM annotation_reviews GROUP BY asset_id) l ON r.seq=l.seq ORDER BY r.asset_id')]
    independent=[dict(r) for r in db.execute('SELECT * FROM quality_submissions ORDER BY task_id,asset_id,role')];db.close()
    assert fingerprint(reviews)==receipt['latest_reviews_sha256']
    assert fingerprint(independent)==receipt['independent_submissions_sha256']
    report={'status':'passed','verified_at':now(),'models':checks,'image_checks':image_checks,'video_checks':videos,
            'sam':{'shape':list(bgr.shape[:2]),'status':result.segmentation_status,'areas':areas,'accuracy_not_measured':True},
            'real_model_confirmed_events':len(events),'media':media,'physical_camera':camera,
            'all_700_reviews_unchanged':True,'original_AB_400_submissions_unchanged':True,
            'scope':'Functional integration of real own weights. Physical camera status is reported separately; no independent mask accuracy or camera fire accuracy claim.'}
    write_json(ROOT/'reports/final_model_verification_20261108.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',type=Path,required=True);parser.add_argument('--skip-camera',action='store_true')
    args=parser.parse_args();verify(args.workspace.resolve(),args.skip_camera)
