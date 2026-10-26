"""Locked independent annotations, separate human-reference reports and worktime."""
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
from comparison import iou


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


def pause_work_timers(workspaces):
    """End the current work interval before changing image, role or workspace."""
    for workspace in workspaces:
        if isinstance(workspace, dict) and isinstance(workspace.get('timer'), WorkTimer):
            workspace['timer'].pause()


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


def preannotation_metrics(references,predictions):
    """P/R of the actual frozen candidates, without inventing low-score AP inputs."""
    def counts(tp,fp,fn):
        p=tp/(tp+fp) if tp+fp else (0. if fn else None)
        r=tp/(tp+fn) if tp+fn else None
        f=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
        return {'tp':tp,'fp':fp,'fn':fn,'precision':p,'recall':r,'f1':f}
    rows=[]
    for label in ('fire','smoke'):
        tp=fp=fn=0
        for key,boxes in references.items():
            targets=[b for b in boxes if b['class']==label];used=set()
            candidates=sorted((b for b in predictions[key] if b['class']==label),key=lambda b:-b['confidence'])
            for candidate in candidates:
                overlap,index=max(((iou(candidate['xyxy'],target['xyxy']),index) for index,target in enumerate(targets) if index not in used),default=(0.,None))
                if index is not None and overlap>=.5:used.add(index);tp+=1
                else:fp+=1
            fn+=len(targets)-len(used)
        rows.append({'class':label,'instances':tp+fn,**counts(tp,fp,fn)})
    negatives=[key for key,boxes in references.items() if not boxes]
    false_images=sum(bool(predictions[key]) for key in negatives)
    return {'overall':counts(*(sum(row[key] for row in rows) for key in ('tp','fp','fn'))),
            'per_class':rows,'negative_images':len(negatives),'negative_images_with_false_detection':false_images,
            'negative_image_false_positive_rate':false_images/len(negatives) if negatives else None}


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
            CREATE TABLE IF NOT EXISTS quality_task_preparations(seq INTEGER PRIMARY KEY AUTOINCREMENT,task_id TEXT NOT NULL,before_snapshot_json TEXT NOT NULL,after_snapshot_json TEXT NOT NULL,actor TEXT NOT NULL,reason TEXT NOT NULL,created_at TEXT NOT NULL);
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

    def progress(self,token):
        access=self.access(token)
        keys=[item['asset']['id'] for item in access['snapshot']]
        with self.dataset.connection() as db:
            if access['role']=='admin':
                counts=dict(db.execute('SELECT role,COUNT(*) FROM quality_submissions WHERE task_id=? GROUP BY role',(access['task_id'],)).fetchall())
                return {'total':len(keys),'A':counts.get('A',0),'B':counts.get('B',0)}
            done={row[0] for row in db.execute('SELECT asset_id FROM quality_submissions WHERE task_id=? AND role=?',(access['task_id'],access['role']))}
        return {'total':len(keys),'submitted':len(done),'submitted_asset_ids':[key for key in keys if key in done],
                'remaining_asset_ids':[key for key in keys if key not in done]}

    def refresh_proposals_before_start(self,admin_token,reason):
        """Explicit preparation revision, forbidden once either person submitted."""
        access=self.access(admin_token,True)
        if not reason.strip():raise ValueError('请记录准备阶段重冻结的实际原因。')
        refreshed=[]
        for item in access['snapshot']:
            asset=self.frozen_asset(access,item['asset']['id'])
            proposal=next((row for row in self.annotations.proposals(asset['id']) if row['status']=='ready' and row['input_sha256']==asset['sha256']),None)
            refreshed.append({'asset':item['asset'],'proposal':proposal})
        after=json.dumps(refreshed)
        with self.dataset.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            submitted=db.execute('SELECT COUNT(*) FROM quality_submissions WHERE task_id=?',(access['task_id'],)).fetchone()[0]
            references=db.execute('SELECT COUNT(*) FROM quality_references WHERE task_id=?',(access['task_id'],)).fetchone()[0]
            if submitted or references:raise ValueError('已有独立提交或人工参考，不能改写冻结候选；须保留原任务版本。')
            before=db.execute('SELECT snapshot_json FROM quality_tasks WHERE id=?',(access['task_id'],)).fetchone()[0]
            if before!=access['snapshot_json']:raise ValueError('任务准备版本已变化，请重新载入。')
            if before==after:return 0
            db.execute('INSERT INTO quality_task_preparations(task_id,before_snapshot_json,after_snapshot_json,actor,reason,created_at) VALUES(?,?,?,?,?,?)',
                       (access['task_id'],before,after,access['actor'],reason,now()))
            db.execute('UPDATE quality_tasks SET snapshot_json=? WHERE id=?',(after,access['task_id']))
        return sum((old['proposal']['id'] if old['proposal'] else None)!=(new['proposal']['id'] if new['proposal'] else None) for old,new in zip(access['snapshot'],refreshed))

    def frozen_asset(self,access,asset_id):
        item=next((i for i in access['snapshot'] if i['asset']['id']==asset_id),None)
        if item is None:raise ValueError('此图片不在任务中。')
        asset=self.annotations.image(asset_id)
        if any(asset[field]!=item['asset'][field] for field in ('sha256','width','height')):raise OSError('图像版本已改变。')
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

    def submit_timed(self,token,asset_id,boxes,timer,negative_confirmed=False):
        """Finish the active work interval when explicitly locking an image."""
        timer.pause()
        return self.submit(token,asset_id,boxes,timer.seconds(),negative_confirmed)

    def completed(self,access):
        with self.dataset.connection() as db:
            actual={(row[0],row[1]) for row in db.execute('SELECT asset_id,role FROM quality_submissions WHERE task_id=?',(access['task_id'],))}
        return actual=={(item['asset']['id'],role) for item in access['snapshot'] for role in ('A','B')}

    def submissions(self,admin_token):
        access=self.access(admin_token,True)
        if not self.completed(access):raise ValueError('两名标注者全部完成前，不展示对方标注或预标注。')
        with self.dataset.connection() as db:
            rows=[{**dict(r),'boxes':json.loads(r['boxes_json'])} for r in db.execute('SELECT * FROM quality_submissions WHERE task_id=? ORDER BY role,asset_id',(access['task_id'],))]
        assets={item['asset']['id']:self.frozen_asset(access,item['asset']['id']) for item in access['snapshot']}
        for row in rows:
            asset=assets[row['asset_id']];validate_boxes(row['boxes'],asset['width'],asset['height'])
            if not math.isfinite(row['seconds']) or not 0<row['seconds']<=86400:raise ValueError('独立提交计时记录无效。')
        return rows

    def report(self,admin_token):
        access=self.access(admin_token,True);submissions=self.submissions(admin_token)
        agreement=[];all_overlaps=[];predictions={};provenance={};references={'A':{},'B':{}}
        by_key={(row['asset_id'],row['role']):row for row in submissions}
        for number,item in enumerate(access['snapshot'],1):
            key=item['asset']['id']
            a,b=[by_key[key,role] for role in ('A','B')]
            pairs=sorted([(iou(x['xyxy'],y['xyxy']),i,j) for i,x in enumerate(a['boxes']) for j,y in enumerate(b['boxes']) if x['class']==y['class']],reverse=True)
            used_a,used_b,matched=set(),set(),[]
            for overlap,i,j in pairs:
                if overlap>=.5 and i not in used_a and j not in used_b:used_a.add(i);used_b.add(j);matched.append(overlap)
            all_overlaps.extend(matched)
            agreement.append({'number':f'Q{number:03d}','asset_id':key,'matched_boxes':len(matched),'unmatched_A':len(a['boxes'])-len(matched),'unmatched_B':len(b['boxes'])-len(matched),
                              'both_empty':not a['boxes'] and not b['boxes'],'mean_iou_of_matched':sum(matched)/len(matched) if matched else None,
                              'manual_A_seconds':a['seconds'],'manual_B_seconds':b['seconds']})
            if item['proposal'] is not None:
                proposal=item['proposal'];asset=item['asset']
                if proposal['status']!='ready' or proposal['input_sha256']!=asset['sha256']:raise ValueError('冻结预标注与原图版本不一致。')
                validate_boxes(proposal['boxes'],asset['width'],asset['height'])
                for box in proposal['boxes']:
                    score=box.get('confidence')
                    if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=1:raise ValueError('冻结预标注置信度无效。')
                predictions[key]=proposal['boxes'];provenance[key]=proposal['id']
                for role in ('A','B'):references[role][key]=by_key[key,role]['boxes']
        with self.dataset.connection() as db:
            preparation=[dict(row) for row in db.execute('SELECT seq,actor,reason,created_at FROM quality_task_preparations WHERE task_id=? ORDER BY seq',(access['task_id'],))]
        total=len(access['snapshot'])
        return {'schema_version':2,'task_id':access['task_id'],'images':total,'at_least_200_images':total>=200,
                'reference_policy':'Report A and B separately; neither is a consensus ground truth. Do not select the more favorable result.',
                'protocol':{'match_iou':.5,'preannotation_candidates':'All actual candidates saved in the frozen proposal; no extra confidence filtering or rerun.',
                            'preannotation_matching':'Confidence descending, same class, one-to-one best available reference match.',
                            'agreement_matching':'Same class, greedy descending IoU, one-to-one match at IoU>=0.5.',
                            'overall_metrics':'Micro TP/FP/FN across fire and smoke; Precision is TP/(TP+FP), not image classification accuracy.',
                            'mean_iou':'Matched pairs only; always report unmatched counts. Empty pairs have null IoU.'},
                'task_preparation_revisions':preparation,
                'agreement':agreement,'agreement_summary':{'matched_pairs':len(all_overlaps),'mean_iou_of_matched':sum(all_overlaps)/len(all_overlaps) if all_overlaps else None,
                    'unmatched_A':sum(r['unmatched_A'] for r in agreement),'unmatched_B':sum(r['unmatched_B'] for r in agreement),
                    'images_with_unmatched_boxes':sum(bool(r['unmatched_A'] or r['unmatched_B']) for r in agreement),'both_empty_images':sum(r['both_empty'] for r in agreement)},
                'preannotation_quality':{role:preannotation_metrics(references[role],predictions) if predictions else None for role in ('A','B')},
                'manual_work_seconds':{role:sum(row['seconds'] for row in submissions if row['role']==role) for role in ('A','B')},
                'images_with_frozen_proposal':len(predictions),'proposal_versions':provenance,
                'independent_submissions':submissions,'image_sha256':{i['asset']['id']:i['asset']['sha256'] for i in access['snapshot']},
                'proposal_metadata':{i['asset']['id']:i['proposal']['metadata'] for i in access['snapshot'] if i['proposal'] is not None},
                'note':'Annotation quality analysis, not independent model test performance or proof of human independence. Raw A/B are unchanged; export/report does not approve training labels. Worktime savings require a separate controlled comparison.'}

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
