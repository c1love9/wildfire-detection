"""Freeze a real 200-image blind task; credentials remain in ignored local data.

Selection uses source declarations, scenes and visual diversity, never model
predictions. No human submissions, references or approvals are generated.
"""
import argparse
import collections
import hashlib
import json
import math
import random
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch  # Initialize native runtime before NumPy/OpenCV.
import cv2
import numpy as np
from PIL import Image
from dataset_store import DatasetStore,csv_bytes,file_sha,now
from quality import QualityStore
from grounding_prelabel import MODEL_ID,REVISION,WEIGHTS_SHA

SEED=20261024
QUOTAS={'kmu':100,'firesense':60,'dfire':40}


def content(row):return '+'.join(sorted(row['declared_content']))


def select_records(records,project=ROOT):
    pool=[row for row in records if row['kind']=='image' and row['split']=='train']
    if len({row['asset_id'] for row in pool})!=len(pool):raise ValueError('训练清单含重复图片。')
    counters={name:collections.Counter() for name in ['setting','lighting','distance','content','video']}
    targets={'setting':{'室内':35,'室外':125},'lighting':{'白天':90,'夜间':50,'逆光':20},
             'distance':{'近景':70,'中景':55,'远景':35},'content':{'fire':55,'smoke':50,'fire+smoke':55}}
    hashes={}
    for row in pool:
        with Image.open(project/row['path']) as image:
            pixels=np.asarray(image.convert('L').resize((32,32)),np.float32)
        values=cv2.dct(pixels)[:8,:8].ravel()[1:]
        hashes[row['asset_id']]=sum(int(value>np.median(values))<<i for i,value in enumerate(values))
    selected=[];counts=collections.Counter();chosen=set()
    def tie(row):return hashlib.sha256(f"{SEED}:{row['asset_id']}".encode()).hexdigest()
    for _ in range(sum(QUOTAS.values())):
        source=min((name for name in QUOTAS if counts[name]<QUOTAS[name]),key=lambda name:counts[name]/QUOTAS[name])
        candidates=[row for row in pool if row['source']==source and row['asset_id'] not in chosen]
        if not candidates:raise ValueError('训练集中没有足够的来源样本：'+source)
        def score(row):
            same=[other for other in selected if other['source']==source]
            distance=min(((hashes[row['asset_id']]^hashes[other['asset_id']]).bit_count() for other in same),default=63)/63
            value=2*distance
            if source!='dfire':
                video=row['source_video']
                value+=4/math.sqrt(1+counters['video'][video])
                for name,values in targets.items():
                    label=content(row) if name=='content' else row['scene'][name]
                    target=values.get(label,0)
                    if target:value+=3*max(0,target-counters[name][label])/target
                times=[other['source_seconds'] for other in same if other['source_video']==video]
                value+=min(1.,min((abs(row['source_seconds']-time) for time in times),default=10.)/10.)
            return value,tie(row)
        row=max(candidates,key=score);selected.append(row);chosen.add(row['asset_id']);counts[source]+=1
        if source!='dfire':
            for name in targets:counters[name][content(row) if name=='content' else row['scene'][name]]+=1
            counters['video'][row['source_video']]+=1
    random.Random(SEED).shuffle(selected)
    return selected


def summary(rows):
    positive=[row for row in rows if row['source']!='dfire']
    return {'images':len(rows),'split_counts':dict(collections.Counter(row['split'] for row in rows)),
            'source_counts':dict(collections.Counter(row['source'] for row in rows)),
            'source_declared_content_counts':dict(collections.Counter(content(row) for row in rows)),
            'positive_scene_declarations':{name:dict(collections.Counter(row['scene'][name] for row in positive)) for name in ['setting','lighting','distance','view']},
            'positive_original_videos':len({row['source_video'] for row in positive}),
            'unique_asset_ids':len({row['asset_id'] for row in rows}),
            'note':'Source declarations are not human ground truth; normal scenes remain individually unverified. Test/val not sampled; no per-image answers distributed to A/B.'}


def has_grounding_parameters(proposal):
    if proposal is None:return False
    metadata=proposal['metadata']
    for level in [metadata,metadata.get('runtime')]:
        parameters=level.get('parameters') if isinstance(level,dict) else None
        if isinstance(parameters,dict) and tuple(parameters.get(key) for key in ['model_id','revision','weights_sha256'])==(MODEL_ID,REVISION,WEIGHTS_SHA):return True
    return False


def ensure_grounding_candidates(store,keys):
    pending=[]
    for key in keys:
        asset=store.annotations.image(key)
        proposal=next((row for row in store.annotations.proposals(key) if row['status']=='ready' and row['input_sha256']==asset['sha256']),None)
        if not has_grounding_parameters(proposal):pending.append(key)
    if pending:
        from grounding_prelabel import GroundingPrelabeler,PrelabelConfig
        model=GroundingPrelabeler('cuda' if torch.cuda.is_available() else 'cpu')
        config=PrelabelConfig.load()
        for start in range(0,len(pending),100):
            results=store.annotations.prelabel(pending[start:start+100],model,config,'自动准备双人任务候选（非人工标注）')
            if any(row['status']!='ready' for row in results):raise ValueError('部分模型候选失败，不能把失败当正常，也不冻结未完成配置。')
    return len(pending)


def prepare(output,preview=False,refresh_before_start=False):
    registration=ROOT/'data/minimum_collection/registration.json'
    dataset=DatasetStore(ROOT/'data/fire_dataset');store=QualityStore(dataset)
    private=output/'task_private.json'
    if private.exists():
        saved=json.loads(private.read_text(encoding='utf-8'))
        access=store.access(saved['tokens']['admin'],True)
        if access['task_id']!=saved['task_id'] or len(access['snapshot'])!=200:raise ValueError('已有任务文件与正式库不一致，停止重复建任务。')
        for item in access['snapshot']:store.frozen_asset(access,item['asset']['id'])
        if refresh_before_start:
            progress=store.progress(saved['tokens']['admin'])
            if progress['A'] or progress['B'] or progress['references']:raise ValueError('任务已经有人工工作，不允许重新冻结。')
            generated=ensure_grounding_candidates(store,[item['asset']['id'] for item in access['snapshot']])
            changed=store.refresh_proposals_before_start(saved['tokens']['admin'],'补跑来源空标签的正常候选，使本任务200幅均有实际DINO候选；人工提交仍为0，图像与次序不变。')
            result=json.loads((ROOT/'reports/dual_annotation_setup.json').read_text(encoding='utf-8'))
            result.update(actual_grounding_candidates_frozen=200,new_grounding_inferences=generated,proposal_versions_replaced_before_human_work=changed)
            (ROOT/'reports/dual_annotation_setup.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            print(json.dumps({'status':'preparation_refrozen_without_human_work','actual_grounding_images':200,'generated':generated,'updated_proposal_versions':changed}));return
        print(json.dumps({'status':'existing_task_preserved','task_id':saved['task_id'],'images':200},ensure_ascii=True));return
    rows=select_records(json.loads(registration.read_text(encoding='utf-8'))['records'])
    info=summary(rows)
    assert len(rows)==200 and info['unique_asset_ids']==200 and info['split_counts']=={'train':200}
    for row in rows:
        asset=store.annotations.image(row['asset_id'])
        if store.annotations.latest(asset['id']) is not None:raise ValueError('选中图片已有人工版本；请先审查独立性再调整方案。')
    if preview:
        print(json.dumps(info,ensure_ascii=True,indent=2));return
    output.mkdir(parents=True,exist_ok=False)
    with closing(sqlite3.connect(dataset.path.resolve().as_uri()+'?mode=ro',uri=True)) as source:
        with closing(sqlite3.connect(output/'catalog_before_setup.sqlite3')) as target:source.backup(target)
    ensure_grounding_candidates(store,[row['asset_id'] for row in rows])
    task,tokens=store.create([row['asset_id'] for row in rows],'成员A（负责人）','成员B（队友）','成员A（仲裁管理者）')
    private.write_text(json.dumps({'task_id':task,'dataset_root':str(dataset.root),'created_at':now(),
                                  'tokens':tokens,'selection_seed':SEED,'registration_sha256':file_sha(registration)},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    access=store.access(tokens['admin'],True)
    assets={item['asset']['id']:item['asset'] for item in access['snapshot']}
    blind=[{'number':f'Q{index+1:03d}','asset_id':row['asset_id'],'width':assets[row['asset_id']]['width'],
            'height':assets[row['asset_id']]['height'],'archived_sha256':assets[row['asset_id']]['sha256']} for index,row in enumerate(rows)]
    (output/'blind_manifest.csv').write_bytes(csv_bytes(blind))
    (output/'selection_manager_only.json').write_text(json.dumps({'summary':info,'records':rows},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for role,name in [('A','成员A_仅本人打开.txt'),('B','成员B_仅本人打开.txt'),('admin','管理员_两人全部完成后打开.txt')]:
        text=(f"任务编号：{task}\n身份：{role}\n图片数：200\n访问码（不要转发给另一标注者）：\n{tokens[role]}\n\n"
              "项目启动：在项目根目录双击‘开始双人标注.cmd’。\n网址：http://127.0.0.1:8506/Quality\n")
        if role=='admin':text+="选择‘管理与仲裁’；A/B各200幅提交完成前不查看结果。\n"
        else:text+="选择‘独立标注’，粘贴本人访问码。不要进入Annotations查看本批模型候选。\n开始计时→检查并画框→暂停计时→提交并锁定；退出后再交接电脑。\n"
        text+="详细操作见项目docs/dual_annotation_200_guide.md。\n访问码只保存在本机，不提交Git。\n"
        (output/name).write_text(text,encoding='utf-8-sig')
    for role in ['A','B']:
        (output/f'疑难记录_{role}.csv').write_bytes(csv_bytes([{'number':'','asset_id':'','problem_location':'','uncertainty':'','action':'先保留原始独立判断；不与另一人讨论具体答案'}]))
    result={'status':'task_ready_no_human_work_fabricated','created_at':now(),'task_id':task,'seed':SEED,
            'selection':info,'frozen_image_hashes_verified':True,'total_expected_independent_submissions':400,
            'current_submissions':0,'current_references':0,'current_human_approved_images':0,
            'normal_data_catalog':True,'credential_files_excluded_from_git':True,
            'private_directory_relative_to_project':str(output.relative_to(ROOT)).replace('\\','/'),
            'blind_manifest_sha256':file_sha(output/'blind_manifest.csv'),
            'backup_before_setup_sha256':file_sha(output/'catalog_before_setup.sqlite3'),
            'limitations':['Two actual people must follow the blind protocol; access codes are not OS account isolation',
                           'Prior human exposure to model labels must be disclosed; source declarations do not establish truth',
                           'Training-only quality subset is not an independent test set or a population accuracy estimate']}
    (ROOT/'reports/dual_annotation_setup.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'task_id':task,'summary':info,'credential_values_printed':False},ensure_ascii=True,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true')
    parser.add_argument('--refresh-before-start',action='store_true',help='Explicit, logged candidate preparation revision; rejects any existing human submission')
    parser.add_argument('--output',type=Path,default=ROOT/'data/fire_dataset/quality_assignments/dual_200_20261024')
    args=parser.parse_args();output=args.output.resolve()
    if not output.is_relative_to(ROOT/'data/fire_dataset/quality_assignments'):raise ValueError('任务交接文件必须保存在被Git忽略的正式数据任务目录。')
    prepare(output,args.preview,args.refresh_before_start)
