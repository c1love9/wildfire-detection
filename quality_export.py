"""Build independent-label archives without modifying annotations or approvals."""
import hashlib
import json
from io import BytesIO
from zipfile import ZipFile,ZIP_DEFLATED

from annotations import yolo_text
from dataset_store import csv_bytes,now


def quality_archive(store,admin_token,selection='both',minimum_images=0):
    if selection not in ('both','A','B'):raise ValueError('导出范围只能为both、A或B。')
    access=store.access(admin_token,True)
    if len(access['snapshot'])<minimum_images:raise ValueError(f'本提交要求至少{minimum_images}幅同批图片。')
    # Single-role administrator exports keep the same blindness gate.
    submissions=store.submissions(admin_token)
    roles=('A','B') if selection=='both' else (selection,)
    chosen=[row for row in submissions if row['role'] in roles]
    files={}
    def write(name,value):files[name]=(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
    for role in roles:write('independent_'+role+'.json',[row for row in chosen if row['role']==role])
    by_key={(row['asset_id'],row['role']):row for row in chosen};manifest=[]
    for number,item in enumerate(access['snapshot'],1):
        asset=item['asset'];key=asset['id'];path=store.dataset.asset_path(asset)
        content=path.read_bytes()
        if hashlib.sha256(content).hexdigest()!=asset['sha256']:raise OSError('导出原图哈希不一致。')
        image_name='images/'+key+path.suffix;files[image_name]=content
        for role in roles:
            files[f'labels_{role}/{key}.txt']=yolo_text(by_key[key,role]['boxes'],asset['width'],asset['height']).encode('utf-8')
        manifest.append({'number':f'Q{number:03d}','asset_id':key,'image_file':image_name,
                         'width':asset['width'],'height':asset['height'],'sha256':asset['sha256']})
    files['image_manifest.csv']=csv_bytes(manifest)
    provenance=store.dataset.snapshot();keys={key for key,role in by_key}
    occurrences=[row for row in provenance['occurrences'] if row['asset_id'] in keys]
    batches={row['batch_id'] for row in occurrences}
    write('provenance.json',{'occurrences':occurrences,'batches':[row for row in provenance['batches'] if row['id'] in batches]})
    with store.dataset.connection() as db:
        participants=[dict(row) for row in db.execute('SELECT role,actor FROM quality_access WHERE task_id=? ORDER BY role',(access['task_id'],)) if row['role'] in roles]
        # Existing historical references are preserved, never required or scored.
        legacy=[dict(row) for row in db.execute('SELECT * FROM quality_references WHERE task_id=? ORDER BY seq',(access['task_id'],))] if selection=='both' else []
    if legacy:write('legacy_reference_history.json',legacy)
    if selection=='both':
        report=store.report(admin_token);write('annotation_quality.json',report)
        files['agreement.csv']=csv_bytes(report['agreement'])
        metrics=[]
        for role,result in report['preannotation_quality'].items():
            if result:
                metrics.extend({'reference':role,**row} for row in result['per_class'])
                metrics.append({'reference':role,'class':'micro','instances':result['overall']['tp']+result['overall']['fn'],**result['overall']})
        files['preannotation_metrics.csv']=csv_bytes(metrics)
    summary={'schema_version':2,'task_id':access['task_id'],'exported_at':now(),'selection':selection,
             'images':len(manifest),'independent_submissions':len(chosen),'participants':participants,
             'at_least_200_images':len(manifest)>=200,'class_ids':{'0':'fire','1':'smoke'},
             'quality_report_included':selection=='both','training_labels_approved_by_export':False,
             'scope':'Original human submissions; A and B are separate references, not consensus truth or formal model test results.'}
    write('task_summary.json',summary)
    files['README.txt']=('本包保留选定角色的原图、原始独立YOLO标签、计时与来源，不改写A/B，不生成训练批准。\n'
        '0=fire，1=smoke；空TXT表示该角色提交为空，不是漏交。图片和标签按asset_id同名对应。\n'
        '完整包另含A/B一致性及预标注分别对A、B的P/R/F1。匹配框平均IoU须连同未匹配数量解释。\n'
        '无冻结候选时预标注指标为null；这些指标不是正式独立测试精度，也不能直接推算节省工时。\n'
        '最终训练数据仍通过Annotations真实复核流程导出；无须额外逐图仲裁或填写仲裁说明。\n'
        '不含访问码。历史参考记录如存在仅作为旧版历史保留，不参与本次指标或导出门槛。\n').encode('utf-8-sig')
    write('checksums.json',{name:{'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()} for name,content in sorted(files.items())})
    output=BytesIO()
    with ZipFile(output,'w',ZIP_DEFLATED,compresslevel=6) as archive:
        for name,content in sorted(files.items()):archive.writestr(name,content)
    content=output.getvalue()
    with ZipFile(BytesIO(content)) as archive:
        if archive.testzip() is not None:raise OSError('导出包CRC校验失败。')
    return content,summary
