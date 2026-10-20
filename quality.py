"""Access-code-separated independent annotations, adjudication and measured quality."""
import hashlib
import json
import math
import secrets
import time
import uuid
from pathlib import Path
import numpy as np
from PIL import Image
from annotations import AnnotationStore,validate_boxes
from dataset_store import file_sha,now
from comparison import iou,evaluate_predictions


def token_hash(token):return hashlib.sha256(token.encode()).hexdigest()


class WorkTimer:
    def __init__(self,clock=None):
        self.clock=clock or time.monotonic;self.total=0.;self.last=None
    def start(self):
        if self.last is None:self.last=self.clock()
    def activity(self):
        if self.last is not None:
            current=self.clock();delta=current-self.last
            self.total+=max(0,min(60.,delta));self.last=current if delta<=60 else None
    def pause(self):self.activity();self.last=None
    def seconds(self):return self.total+(max(0,min(60,self.clock()-self.last)) if self.last is not None else 0.)
    def expire(self):
        if self.last is not None and self.clock()-self.last>60:self.pause()


def mask_metrics(reference,prediction):
    reference,prediction=np.asarray(reference),np.asarray(prediction)
    if reference.ndim!=2 or not reference.size or reference.shape!=prediction.shape:raise ValueError('两个掩码须为与原图一致的二维尺寸。')
    for mask in [reference,prediction]:
        if not np.isin(mask,[0,1,255]).all():raise ValueError('仅接受0/1或0/255二值掩码，不自动阈值化彩色图。')
    a,b=reference>0,prediction>0;intersection=int((a&b).sum());union=int((a|b).sum());count=int(a.sum()+b.sum())
    return {'iou':intersection/union if union else 1.,'dice':2*intersection/count if count else 1.,
            'reference_ratio':float(a.mean()),'prediction_ratio':float(b.mean()),
            'absolute_area_error_percentage_points':abs(float(a.mean()-b.mean()))*100,
            'relative_area_error':abs(int(a.sum())-int(b.sum()))/int(a.sum()) if a.any() else None,
            'empty_pair_convention':'both empty IoU=Dice=1; reference empty relative area error=null'}


class QualityStore:
    def __init__(self,dataset):
        self.dataset=dataset;self.annotations=AnnotationStore(dataset)
        with dataset.connection() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS quality_tasks(id TEXT PRIMARY KEY,snapshot_json TEXT NOT NULL,created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS quality_access(token_sha TEXT PRIMARY KEY,task_id TEXT NOT NULL,role TEXT NOT NULL,actor TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS quality_submissions(task_id TEXT NOT NULL,asset_id TEXT NOT NULL,role TEXT NOT NULL,boxes_json TEXT NOT NULL,seconds REAL NOT NULL,created_at TEXT NOT NULL,PRIMARY KEY(task_id,asset_id,role));
            CREATE TABLE IF NOT EXISTS quality_references(seq INTEGER PRIMARY KEY AUTOINCREMENT,task_id TEXT NOT NULL,asset_id TEXT NOT NULL,boxes_json TEXT NOT NULL,actor TEXT NOT NULL,note TEXT NOT NULL,created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS quality_masks(id TEXT PRIMARY KEY,record_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS quality_worktime(id TEXT PRIMARY KEY,group_id TEXT NOT NULL,actor TEXT NOT NULL,condition TEXT NOT NULL,seconds REAL NOT NULL,note TEXT NOT NULL,created_at TEXT NOT NULL,UNIQUE(group_id,actor,condition));
            ''')

    def create(self,assets,actor_a,actor_b,admin):
        if not assets or not actor_a.strip() or not actor_b.strip() or actor_a.strip()==actor_b.strip() or not admin.strip():
            raise ValueError('请选择图片，填写两名不同标注者及管理者。')
        snapshot=[]
        for identifier in dict.fromkeys(assets):
            asset=self.annotations.image(identifier)
            proposals=self.annotations.proposals(identifier)
            proposal=next((p for p in proposals if p['status']=='ready' and p['input_sha256']==asset['sha256']),None)
            snapshot.append({'asset':asset,'proposal':proposal})
        task='quality_'+uuid.uuid4().hex;tokens={role:secrets.token_urlsafe(24) for role in ['A','B','admin']}
        with self.dataset.connection() as db:
            db.execute('INSERT INTO quality_tasks VALUES(?,?,?)',(task,json.dumps(snapshot),now()))
            for role,actor in [('A',actor_a.strip()),('B',actor_b.strip()),('admin',admin.strip())]:
                db.execute('INSERT INTO quality_access VALUES(?,?,?,?)',(token_hash(tokens[role]),task,role,actor))
        return task,tokens

    def access(self,token,admin=False):
        with self.dataset.connection() as db:
            row=db.execute('SELECT a.*,t.snapshot_json FROM quality_access a JOIN quality_tasks t ON a.task_id=t.id WHERE token_sha=?',(token_hash(token),)).fetchone()
        if row is None or admin and row['role']!='admin':raise ValueError('访问码无效或无此权限。')
        return {**dict(row),'snapshot':json.loads(row['snapshot_json'])}

    def own_submission(self,token,asset_id):
        access=self.access(token)
        if access['role']=='admin':raise ValueError('管理访问码不参与独立标注。')
        with self.dataset.connection() as db:
            row=db.execute('SELECT * FROM quality_submissions WHERE task_id=? AND asset_id=? AND role=?',(access['task_id'],asset_id,access['role'])).fetchone()
        return {**dict(row),'boxes':json.loads(row['boxes_json'])} if row else None

    def frozen_asset(self,access,asset_id):
        item=next((i for i in access['snapshot'] if i['asset']['id']==asset_id),None)
        if item is None:raise ValueError('此图片不在任务中。')
        asset=self.annotations.image(asset_id)
        if asset['sha256']!=item['asset']['sha256']:raise OSError('图像版本已改变。')
        return asset

    def submit(self,token,asset_id,boxes,seconds,negative_confirmed=False):
        access=self.access(token)
        if access['role'] not in ('A','B'):raise ValueError('请使用独立标注访问码。')
        asset=self.frozen_asset(access,asset_id);boxes=validate_boxes(boxes,asset['width'],asset['height'])
        if not boxes and not negative_confirmed:raise ValueError('空标签必须明确确认无火烟。')
        if not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or not 0<seconds<=86400:raise ValueError('请启动并暂停工作计时后再提交。')
        with self.dataset.connection() as db:
            if db.execute('SELECT 1 FROM quality_submissions WHERE task_id=? AND asset_id=? AND role=?',(access['task_id'],asset_id,access['role'])).fetchone():raise ValueError('该独立标注已经提交并锁定。')
            db.execute('INSERT INTO quality_submissions VALUES(?,?,?,?,?,?)',(access['task_id'],asset_id,access['role'],json.dumps(boxes),seconds,now()))

    def completed(self,access):
        with self.dataset.connection() as db:
            count=db.execute('SELECT count(*) FROM quality_submissions WHERE task_id=?',(access['task_id'],)).fetchone()[0]
        return count==len(access['snapshot'])*2

    def submissions(self,admin_token):
        access=self.access(admin_token,True)
        if not self.completed(access):raise ValueError('两名标注者全部完成前，不展示对方标注或预标注。')
        with self.dataset.connection() as db:
            return [{**dict(r),'boxes':json.loads(r['boxes_json'])} for r in db.execute('SELECT * FROM quality_submissions WHERE task_id=?',(access['task_id'],))]

    def arbitrate(self,admin_token,asset_id,boxes,note):
        access=self.access(admin_token,True);self.submissions(admin_token)
        asset=self.frozen_asset(access,asset_id)
        if not note.strip():raise ValueError('请填写人工仲裁说明。')
        boxes=validate_boxes(boxes,asset['width'],asset['height'])
        with self.dataset.connection() as db:db.execute('INSERT INTO quality_references(task_id,asset_id,boxes_json,actor,note,created_at) VALUES(?,?,?,?,?,?)',(access['task_id'],asset_id,json.dumps(boxes),access['actor'],note,now()))

    def report(self,admin_token):
        access=self.access(admin_token,True);submissions=self.submissions(admin_token)
        with self.dataset.connection() as db:refs=[dict(r) for r in db.execute('SELECT * FROM quality_references WHERE task_id=? ORDER BY seq',(access['task_id'],))]
        references={r['asset_id']:json.loads(r['boxes_json']) for r in refs}
        if len(references)!=len(access['snapshot']):raise ValueError('请先完成人工仲裁参考版本。')
        agreement=[];predictions={};provenance={}
        for item in access['snapshot']:
            key=item['asset']['id'];self.frozen_asset(access,key)
            a,b=[next(r for r in submissions if r['asset_id']==key and r['role']==role) for role in ['A','B']]
            pairs=sorted([(iou(x['xyxy'],y['xyxy']),i,j) for i,x in enumerate(a['boxes']) for j,y in enumerate(b['boxes']) if x['class']==y['class']],reverse=True)
            used_a,used_b,matched=set(),set(),[]
            for overlap,i,j in pairs:
                if overlap>=.5 and i not in used_a and j not in used_b:used_a.add(i);used_b.add(j);matched.append(overlap)
            agreement.append({'asset_id':key,'matched_boxes':len(matched),'unmatched_A':len(a['boxes'])-len(matched),'unmatched_B':len(b['boxes'])-len(matched),
                              'mean_iou_of_matched':sum(matched)/len(matched) if matched else None,'manual_A_seconds':a['seconds'],'manual_B_seconds':b['seconds']})
            if item['proposal'] is not None:
                predictions[key]=item['proposal']['boxes'];provenance[key]=item['proposal']['id']
        reference_with_proposal={key:references[key] for key in predictions}
        return {'task_id':access['task_id'],'images':len(references),'at_least_200_images':len(references)>=200,
                'agreement':agreement,'preannotation_quality':evaluate_predictions(reference_with_proposal,predictions) if predictions else None,
                'images_with_frozen_proposal':len(predictions),'proposal_versions':provenance,'adjudication_history':refs,
                'independent_submissions':submissions,'image_sha256':{i['asset']['id']:i['asset']['sha256'] for i in access['snapshot']},
                'proposal_metadata':{i['asset']['id']:i['proposal']['metadata'] for i in access['snapshot'] if i['proposal'] is not None},
                'note':'Access-code separation is not system account authentication. Actual two-person independence and human reference quality require an external protocol; local fixture submissions do not prove compliance.'}

    def compare_masks(self,asset_id,reference,prediction,actor,note,independent=False,label='fire'):
        if not independent or not actor.strip() or not note.strip() or label not in ('fire','smoke','combined'):raise ValueError('请声明独立人工参考来源、类别、执行者和说明。')
        asset=self.annotations.image(asset_id)
        if np.asarray(reference).shape!=(asset['height'],asset['width']):raise ValueError('参考掩码必须与原图尺寸相同。')
        metrics=mask_metrics(reference,prediction);identifier='mask_'+uuid.uuid4().hex
        directory=self.dataset.root/'quality_masks'/identifier;directory.mkdir(parents=True)
        for name,array in [('reference',reference),('prediction',prediction)]:Image.fromarray((np.asarray(array)>0).astype(np.uint8)*255).save(directory/(name+'.png'))
        record={'id':identifier,'asset_id':asset_id,'input_sha256':asset['sha256'],'actor':actor,'note':note,'label':label,'reference_source':'declared independent human mask',
                'metrics':metrics,'reference_sha256':file_sha(directory/'reference.png'),'prediction_sha256':file_sha(directory/'prediction.png'),'created_at':now()}
        with self.dataset.connection() as db:db.execute('INSERT INTO quality_masks VALUES(?,?)',(identifier,json.dumps(record)))
        return record

    def worktime(self,group,actor,condition,seconds,note):
        if condition not in ('manual','assisted') or not all(s.strip() for s in [group,actor,note]) or not math.isfinite(seconds) or not 0<seconds<=86400:raise ValueError('请填写完整的人工实测工时记录。')
        with self.dataset.connection() as db:db.execute('INSERT INTO quality_worktime VALUES(?,?,?,?,?,?,?)',(uuid.uuid4().hex,group,actor,condition,seconds,note,now()))

    def worktime_report(self):
        with self.dataset.connection() as db:rows=[dict(r) for r in db.execute('SELECT * FROM quality_worktime ORDER BY created_at')]
        groups={}
        for row in rows:groups.setdefault((row['group_id'],row['actor']),{})[row['condition']]=row['seconds']
        paired=[{'group':group,'actor':actor,'manual_seconds':r['manual'],'assisted_seconds':r['assisted'],
                 'estimated_saved_fraction':1-r['assisted']/r['manual']} for (group,actor),r in groups.items() if set(r)=={'manual','assisted'}]
        return {'records':rows,'paired_estimates':paired,'note':'Self-recorded controlled-work measurements; equal task difficulty/quality and counterbalanced order must be established externally. Estimates may be negative.'}
