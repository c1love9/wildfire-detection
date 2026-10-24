"""Export actual A/B, adjudication and approved labels; never synthesize completion."""
import argparse
import json
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from dataset_store import DatasetStore,csv_bytes,file_sha,now
from annotations import yolo_text
from quality import QualityStore

DEFAULT_TASK=ROOT/'data/fire_dataset/quality_assignments/dual_200_20261024/task_private.json'


def geometry(boxes):return sorted((box['class'],tuple(box['xyxy'])) for box in boxes)


def export_task(task_file,output_root,minimum_images=200):
    private=json.loads(Path(task_file).read_text(encoding='utf-8'))
    dataset=DatasetStore(private['dataset_root']);store=QualityStore(dataset);token=private['tokens']['admin']
    access=store.access(token,True);progress=store.progress(token)
    if access['task_id']!=private['task_id']:raise ValueError('交接文件与任务不一致。')
    if progress['total']<minimum_images:raise ValueError(f'本提交要求至少{minimum_images}幅同批图片。')
    if progress['A']!=progress['total'] or progress['B']!=progress['total']:
        raise ValueError(f"尚未完成双人独立标注：A {progress['A']}/{progress['total']}，B {progress['B']}/{progress['total']}；未生成提交包。")
    report=store.report(token)  # Also gates every manual reference.
    keys=[item['asset']['id'] for item in access['snapshot']]
    reviewed=store.annotations._approved(keys)
    by_id={item['asset']['id']:item for item in reviewed}
    references={}
    for reference in report['adjudication_history']:references[reference['asset_id']]=reference
    for key in keys:
        if geometry(by_id[key]['review']['boxes'])!=geometry(json.loads(references[key]['boxes_json'])):
            raise ValueError('人工通过标签与最新仲裁参考不同，请重新复核：'+key[:12])
    # All gates above are read-only; an unfinished task creates no export folder.
    target=Path(output_root)/('双人独立标注提交_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    target.mkdir(parents=True,exist_ok=False)
    def write(name,value):
        (target/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    write('annotation_quality.json',report)
    write('adjudication_history.json',report['adjudication_history'])
    write('reviewed_audit.json',reviewed)
    provenance=dataset.snapshot();key_set=set(keys)
    occurrences=[row for row in provenance['occurrences'] if row['asset_id'] in key_set]
    batch_ids={row['batch_id'] for row in occurrences}
    write('provenance.json',{'occurrences':occurrences,'batches':[row for row in provenance['batches'] if row['id'] in batch_ids]})
    for role in ['A','B']:write('independent_'+role+'.json',[row for row in report['independent_submissions'] if row['role']==role])
    (target/'agreement.csv').write_bytes(csv_bytes(report['agreement']))
    rows=[]
    for folder in ['images','labels_A','labels_B','labels_adjudicated']:(target/folder).mkdir()
    for index,key in enumerate(keys):
        item=by_id[key];asset=item['asset'];path=dataset.asset_path(asset)
        copied=target/'images'/(key+path.suffix);shutil.copy2(path,copied)
        if file_sha(copied)!=asset['sha256']:raise OSError('复制的图片哈希不一致。')
        for role in ['A','B']:
            submission=next(row for row in report['independent_submissions'] if row['asset_id']==key and row['role']==role)
            (target/('labels_'+role)/(key+'.txt')).write_text(yolo_text(submission['boxes'],asset['width'],asset['height']),encoding='utf-8')
        (target/'labels_adjudicated'/(key+'.txt')).write_text(yolo_text(item['review']['boxes'],asset['width'],asset['height']),encoding='utf-8')
        rows.append({'number':f'Q{index+1:03d}','asset_id':key,'image_file':'images/'+copied.name,'width':asset['width'],'height':asset['height'],
                     'sha256':asset['sha256'],'adjudication_seq':references[key]['seq'],'approved_review_id':item['review']['id']})
    (target/'image_manifest.csv').write_bytes(csv_bytes(rows))
    with dataset.connection() as db:
        actors=[dict(row) for row in db.execute('SELECT role,actor FROM quality_access WHERE task_id=? ORDER BY role',(access['task_id'],))]
    write('task_summary.json',{'task_id':access['task_id'],'exported_at':now(),'images':len(keys),'independent_submissions':len(report['independent_submissions']),
                              'participants':actors,'at_least_200_images':len(keys)>=200,'class_ids':{'0':'fire','1':'smoke'},
                              'scope':'Human task results, not a formal model test or proof of independence; source data and protocol must accompany report'})
    (target/'README.txt').write_text('本包保留同一批原图、A/B原始独立标签、人工仲裁标签、版本记录与质量指标。\n'
        '0=fire，1=smoke；空TXT表示该版本提交为空。图片与标签按asset_id同名对应。\n'
        '本批来自训练集，用于标注质量检查；不是独立测试集。其余500幅尚需复核后才能导出完整700幅训练划分。\n'
        '不含访问码；不要另行提交task_private.json、角色访问码文件或管理员文件。\n',encoding='utf-8-sig')
    files=sorted(path for path in target.rglob('*') if path.is_file())
    write('checksums.json',{str(path.relative_to(target)).replace('\\','/'):{'bytes':path.stat().st_size,'sha256':file_sha(path)} for path in files})
    archive=target.with_suffix('.zip')
    with ZipFile(archive,'w',ZIP_DEFLATED,compresslevel=6) as package:
        for path in sorted(target.rglob('*')):
            if path.is_file():package.write(path,str(path.relative_to(target)))
    with ZipFile(archive) as package:
        if package.testzip() is not None:raise OSError('提交包CRC失败。')
    return {'folder':str(target),'zip':str(archive),'sha256':file_sha(archive),'images':len(keys),'submissions':len(report['independent_submissions'])}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task-file',type=Path,default=DEFAULT_TASK)
    parser.add_argument('--output',type=Path,default=ROOT/'data/fire_dataset/quality_deliveries')
    args=parser.parse_args()
    try:print(json.dumps(export_task(args.task_file,args.output),ensure_ascii=True))
    except (ValueError,OSError,KeyError) as error:
        print(str(error),file=sys.stderr);raise SystemExit(2)
