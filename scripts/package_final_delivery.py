"""Build a versioned teacher package with real reviewed data and own weights.

Unlike the historical functional package, this includes a portable restoration
map and a sanitized catalog. It never packages annotation access credentials,
personal camera images, environment folders or unselected raw archives.
"""
import argparse
import csv
import hashlib
from io import BytesIO
import json
import re
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
from zipfile import ZipFile,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from dataset_store import DatasetStore,file_sha,now
from quality import QualityStore
from quality_export import quality_archive
from training import TrainingStore,read_json,write_json,verify_inventory
from comparison import ComparisonStore
from experiments import ExperimentStore


def package(workspace, output, report_document=None):
    if output.exists():
        raise FileExistsError('Use a new versioned output directory; never replace an earlier release.')
    output.mkdir(parents=True)
    stage=output/'智能火灾报警系统_交付材料'
    stage.mkdir()
    workflow=read_json(workspace/'workflow.json');formal=read_json(workspace/'formal_experiments.json')
    dataset=DatasetStore(ROOT/'data/fire_dataset');training=TrainingStore(dataset)
    restore=[]
    def add(path,member,target=None):
        path=Path(path);destination=stage/member
        if not destination.resolve().is_relative_to(stage.resolve()) or destination.exists():
            raise ValueError('Unsafe or duplicate member: '+member)
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
        if target:restore.append({'source':member,'target':target})
        return destination
    def text(member,value):
        path=stage/member;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value,encoding='utf-8')
    export=Path(workflow['export_path'])
    frozen=read_json(export/'source_split.json')
    for path in sorted(export.rglob('*')):
        if not path.is_file():continue
        rel=path.relative_to(export).as_posix()
        member='03_最终数据集/'+rel
        add(path,member,'data/fire_dataset/exports/'+workflow['export_id']+'/'+rel)
        if rel.startswith('images/'):
            restore.append({'source':member,'target':'data/fire_dataset/media/images/'+path.name})
        if rel in ('data.yaml','audit.json','manifest.json','source_split.json') or rel.startswith(('images/','labels/')):
            for job in formal['jobs'].values():
                restore.append({'source':member,'target':'data/fire_dataset/training/'+job+'/input/'+rel})
    sources=read_json(ROOT/'data/minimum_collection/registration.json')['records']
    for asset in frozen['assets']:
        if asset['kind']=='video':
            path=dataset.asset_path(asset)
            if file_sha(path)!=asset['sha256']:raise OSError('Video changed')
            add(path,'03_最终数据集/videos/'+asset['id']+'.mp4','data/fire_dataset/'+asset['path'])
    split_record=next(row for row in dataset.splits() if row['id']==workflow['split_id'])
    split_path=dataset.root/split_record['manifest_path']
    if file_sha(split_path)!=split_record['manifest_sha256']:
        raise OSError('Original frozen split changed')
    # The reviewed export serializes the same JSON without the original final
    # newline. Restore the original bytes, because the catalog pins their hash.
    for path in sorted(split_path.parent.glob('*')):
        if path.is_file():
            add(path,'03_最终数据集/frozen_split/'+path.name,
                'data/fire_dataset/'+path.relative_to(dataset.root).as_posix())
    for name in ('final_dataset_statistics.json','final_image_statistics.csv'):
        add(workspace/name,'03_最终数据集/'+name)
    safe_sources=[]
    for row in sources:
        safe_sources.append({k:v for k,v in row.items() if k not in ('annotation_status',)})
    write_json(stage/'03_最终数据集/sources.json',safe_sources)
    fields=['asset_id','kind','source','original_provenance','accounting_origin','collection_group','source_video','source_seconds','source_start_seconds','source_end_seconds','sha256','split']
    with (stage/'03_最终数据集/sources.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(safe_sources)
    text('03_最终数据集/README.md','最终版本是人工通过的700幅，不是旧yolo_draft。0=fire，1=smoke；193幅正常、507幅含火烟。正常还差7幅。\n'
         '图像484/140/76；test仅1幅正常，不能据此推断稳定误报率。300段视频900秒来自46个原视频。\n'
         'data.yaml采用相对路径。图像目录与最终标签、source_split、audit和复核版本一一对应。\n'
         '公开来源自主整理计自采采用教师认可课程口径，非本人实拍；scene仍为来源声明，严格阶段待确认。\n')
    text('03_最终数据集/class_mapping.json',json.dumps({'0':'fire','1':'smoke'},indent=2)+'\n')
    for name in ('collection_sources.md','annotation_protocol.md'):
        add(ROOT/'docs'/name,'03_最终数据集/licenses/'+name)
    for name in ('download_manifest.json','source_integrity.json','dfire_license.html','kmu_license_page.html','flame2_mirror_readme.txt'):
        add(ROOT/'data/public_sources'/name,'03_最终数据集/licenses/'+name)

    private=read_json(dataset.root/'quality_assignments/dual_200_20261024/task_private.json')
    content,quality=quality_archive(QualityStore(dataset),private['tokens']['admin'],'both',200)
    with ZipFile(BytesIO(content)) as archive:
        for member in archive.namelist():
            path=stage/'04_标注过程与质量/独立200幅'/member
            if not path.resolve().is_relative_to((stage/'04_标注过程与质量').resolve()):raise ValueError('Invalid quality member')
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(archive.read(member))
    add(export/'audit.json','04_标注过程与质量/human_review_history_and_prelabels.json')
    for name in ('preannotation_diagnosis.md','preannotation_optimization.md','annotation_protocol.md'):
        add(ROOT/'docs'/name,'04_标注过程与质量/'+name)
    for name in ('preannotation_diagnosis.json','preannotation_optimization.json'):
        add(ROOT/'reports'/name,'04_标注过程与质量/'+name)
    for path in (ROOT/'configs').glob('*preannot*.json'):
        add(path,'04_标注过程与质量/configs/'+path.name)
    text('04_标注过程与质量/worktime.csv','group_id,actor,condition,effective_seconds,note\n')
    worktime_path=ROOT/'reports/worktime_estimate_20261108.json'
    if worktime_path.exists():
        add(worktime_path,'04_标注过程与质量/worktime_estimate.json')
        add(ROOT/'reports/annotation_iou_20261108.json','04_标注过程与质量/annotation_iou.json')
        add(ROOT/'docs/worktime_and_iou.md','04_标注过程与质量/worktime_report.md')
    else:
        text('04_标注过程与质量/worktime_report.md','辅助复核用时缺少依据，工时估算待补。\nA/B独立标注计时随原始提交保留，不将两人总和与单人辅助时间混比，不把Annotations页面停留时间当有效工时。\n')

    for variant,identifier in formal['jobs'].items():
        result=training.result(identifier);directory=training.directory(identifier)
        receipt=read_json(directory/'dataset_receipt.json');verify_inventory(directory/'input',receipt['inventory'])
        for rel in ('request.json','dataset_receipt.json','environment.json','effective_parameters.json','result.json','metrics.json','worker.log','initial.pt'):
            add(directory/rel,'05_模型与训练/'+variant+'/'+rel,'data/fire_dataset/training/'+identifier+'/'+rel)
        for path in (directory/'output').rglob('*'):
            if not path.is_file() or path.name.endswith('.cache'):continue
            rel=path.relative_to(directory).as_posix()
            add(path,'05_模型与训练/'+variant+'/'+rel,'data/fire_dataset/training/'+identifier+'/'+rel)
    add(workspace/'model_selection.json','05_模型与训练/model_selection.json')
    add(workspace/'convergence_analysis.json','05_模型与训练/convergence_analysis.json')
    add(workspace/'formal_experiments.json','06_评测与实验/formal_experiments.json')
    for name in ('formal_metrics.csv','scene_generalization.csv','device_generalization.csv','negative_false_alarms.csv','open_model_protocol.json'):
        add(workspace/name,'06_评测与实验/'+name)
    for path in (workspace/'alarm_ablation').glob('*.json'):
        add(path,'06_评测与实验/alarm_ablation/'+path.name)
    ComparisonStore(dataset).result(formal['comparison_id'])
    directory=dataset.root/'comparisons'/formal['comparison_id']
    for path in directory.rglob('*'):
        if path.is_file():
            rel=path.relative_to(directory).as_posix()
            add(path,'06_评测与实验/comparison/'+rel,'data/fire_dataset/comparisons/'+formal['comparison_id']+'/'+rel)
    experiments=ExperimentStore(dataset);experiments.evaluation(formal['experiment_id'])
    experiment_manifest=experiments.read(formal['experiment_id'])
    for identifiers in experiment_manifest.get('previous_attempts',{}).values():
        for identifier in identifiers:
            directory=training.directory(identifier)
            for path in directory.rglob('*'):
                if path.is_file() and not path.relative_to(directory).as_posix().startswith('input/'):
                    rel=path.relative_to(directory).as_posix()
                    add(path,'09_过程附录/interrupted/'+identifier+'/'+rel,'data/fire_dataset/training/'+identifier+'/'+rel)
    for path in (dataset.root/'experiments'/formal['experiment_id']).glob('*.json'):
        add(path,'06_评测与实验/optimization/'+path.name,'data/fire_dataset/experiments/'+formal['experiment_id']+'/'+path.name)
    add(ROOT/'docs/formal_results.md','06_评测与实验/README_结果口径.md')
    text('06_评测与实验/area_error.csv','asset_id,class,reference_sha256,prediction_sha256,iou,dice,absolute_area_error_percentage_points\n')
    text('06_评测与实验/area_error_pending.md','分割IoU与面积误差待独立参考；现有检测框IoU仅作定位一致性，不替代SAM分割IoU。原题未指定必须手绘掩码或样本数，本工具可使用独立人工原图同尺寸二值参考掩码，不以SAM自身输出作真值。\n')

    for name in ('智能火灾报警系统设计报告.docx','智能火灾报警系统完整使用指南.docx'):
        document=report_document if '设计报告' in name and report_document else workspace/'documents'/name
        if '设计报告' in name and file_sha(document)!=read_json(ROOT/'reports/template_report.json')['output_sha256']:
            raise OSError('Report differs from the latest verified template build; supply --report-document for the current revision.')
        add(document,('01_设计报告/' if '设计报告' in name else '07_演示与操作/')+name)
    report_source=(ROOT/'docs/template_report.md').read_text(encoding='utf-8')
    # Keep Markdown image links usable inside the teacher package as well.
    def local_figure(match):
        source=(ROOT/'docs'/match.group(2)).resolve();name=source.name
        destination=stage/'01_设计报告/figures'/name
        if destination.exists():
            if file_sha(destination)!=file_sha(source):raise ValueError('Conflicting figure basename')
        else:add(source,'01_设计报告/figures/'+name)
        return '!['+match.group(1)+'](figures/'+name+')'
    for path in (ROOT/'docs/figures').glob('*.svg'):add(path,'01_设计报告/figures/'+path.name)
    for path in (workspace/'figures').glob('*'):
        if path.is_file():add(path,'01_设计报告/figures/'+path.name)
    text('01_设计报告/报告可编辑正文.md',re.sub(r'!\[(.*?)\]\((.*?)\)',local_figure,report_source))
    add(workspace/'demo/系统演示.mp4','07_演示与操作/系统演示.mp4')
    for name in ('quick_start.md','user_guide.md','final_delivery_plan.md','teammate_annotation_manual.md'):
        path=ROOT/'docs'/name
        if path.exists():add(path,'07_演示与操作/'+name)
    for name in ('智能火灾报警系统项目入门手册.docx','火焰烟雾数据标注操作手册.docx'):
        add(ROOT/'artifacts/release'/name,'09_过程附录/历史入门材料/'+name)
    text('09_过程附录/历史入门材料/版本说明.md','10月历史手册仅作过程材料；当前操作以07中的11月8日完整使用指南为准。逐图仲裁已取消，700幅复核和A/B双提交已完成；自采采用教师认可课程口径并保留public原始来源。')
    for path in sorted((ROOT/'demo-images').glob('*.jpeg')):
        add(path,'07_演示与操作/demo_media/images/'+path.name,'demo-images/'+path.name)
    for name in ('fire1.mov','fire2.mov'):
        add(ROOT/'demo-videos'/name,'07_演示与操作/demo_media/videos/'+name,'demo-videos/'+name)
    # These two runs were generated by verify_final_models on public demos;
    # never export the user's production alarm history or camera recordings.
    demo_db=workspace/'model_checks/alarms.sqlite3'
    connection=sqlite3.connect(demo_db.as_uri()+'?mode=ro',uri=True)
    runs=connection.execute('SELECT source,metadata_json FROM runs').fetchall()
    selected_sha=training.result(formal['selection']['training_id'])['weights']['best']['sha256']
    for source,metadata in runs:
        meta=json.loads(metadata)
        if (source not in ('正式自训模型验证_fire1.mov','正式自训模型验证_fire2.mov')
                or meta.get('input_type')!='video' or meta.get('model',{}).get('sha256')!=selected_sha):
            raise ValueError('Only verified public-video alarm examples may be packaged.')
    example_root='07_演示与操作/demo_media/alarm_examples'
    alarm_backup=stage/example_root/'alarms.sqlite3';alarm_backup.parent.mkdir(parents=True,exist_ok=True)
    backup=sqlite3.connect(alarm_backup);connection.backup(backup);backup.close()
    restore.append({'source':example_root+'/alarms.sqlite3','target':'artifacts/alarms/alarms.sqlite3'})
    media=connection.execute('SELECT snapshot_path,clip_path FROM event_media WHERE status IN (?,?)',('ready','partial')).fetchall()
    connection.close()
    if not runs or not media:raise ValueError('Missing actual public alarm examples')
    verified_media=read_json(ROOT/'reports/final_model_verification_20261108.json')['media']
    allowed_hashes={row[key] for row in verified_media for key in ('snapshot_sha256','clip_sha256')}
    for paths in media:
        for relative in paths:
            media_root=(workspace/'model_checks/media').resolve();path=(media_root/relative).resolve()
            if not path.is_relative_to(media_root) or file_sha(path) not in allowed_hashes:raise OSError('Unverified media')
            add(path,example_root+'/media/'+relative,'artifacts/alarms/media/'+relative)
    text(example_root+'/README.md','本例由本人模型在公开fire1/fire2视频上真实推理生成，验证窗口为确认前1秒、后2秒，实际范围见日志。非个人摄像头画面，不计入正式数据或精度评测。未设置通道点位的样例保留未记录；现场选命名通道A/B后生成的事件才含对应模拟位置快照。\n')
    for folder,name in (('fire-models','fire_n.pt'),('general-models','yolov8n.pt')):
        add(ROOT/folder/name,'05_模型与训练/upstream/'+name,folder+'/'+name)
    text('05_模型与训练/upstream/README.md','fire_n为上游火烟模型；yolov8n为COCO初始化模型，均不冒充本人训练成果。保留上游及Ultralytics相应许可。\nSAM与DINO不包含在本包，按源码scripts/prepare_sam.py和prepare_grounding_dino.py固定版本下载校验。\n')

    # Backup a consistent catalog; remove credential hashes and VACUUM so deleted
    # access material is not recoverable from freed database pages in the ZIP.
    catalog=stage/'08_验收与校验/catalog_sanitized.sqlite3';catalog.parent.mkdir(parents=True,exist_ok=True)
    src=sqlite3.connect(dataset.path.as_uri()+'?mode=ro',uri=True);dst=sqlite3.connect(catalog)
    src.backup(dst);src.close()
    with dst:dst.execute('DELETE FROM quality_access')
    dst.execute('VACUUM');assert dst.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    assert dst.execute('SELECT COUNT(*) FROM quality_access').fetchone()[0]==0;dst.close()
    restore.append({'source':'08_验收与校验/catalog_sanitized.sqlite3','target':'data/fire_dataset/catalog.sqlite3'})
    write_json(stage/'08_验收与校验/restore_map.json',{'schema_version':1,'files':restore,'access_codes_excluded':True})
    for name in ('requirements_acceptance.json','formal_experiments_20261108.json','formal_alarm_ablation_20261108.json','final_dataset_statistics_20261108.json','final_browser_20261108.json','final_model_verification_20261108.json','final_prelabel_device_check_20261108.json','final_demo_20261108.json','final_documents_20261108.json','final_checks_20261108.json'):
        add(ROOT/'reports'/name,'08_验收与校验/'+name)
    if worktime_path.exists():
        for name in ('worktime_estimate_20261108.json','annotation_iou_20261108.json','worktime_revision_20261108.json'):
            add(ROOT/'reports'/name,'08_验收与校验/'+name)
    add(workspace/'regression_tests.txt','08_验收与校验/regression_tests.txt')
    add(ROOT/'docs/requirements_audit.md','08_验收与校验/requirements_checklist.md')
    add(workspace/'review_receipt.json','08_验收与校验/review_freeze_receipt.json')
    pending=workspace/'pending_samples'
    if pending.exists():
        for path in pending.rglob('*'):
            if path.is_file():add(path,'09_待补素材/'+path.relative_to(pending).as_posix())
    text('08_验收与校验/复现说明.md','在全新目录解压02_源代码/source_code.zip，按quick_start安装依赖。\n'
         '在该源码目录运行：.venv\\Scripts\\python.exe scripts\\restore_final_delivery.py --package "交付材料完整目录"\n'
         '该命令逐文件校验，拒绝覆盖冲突文件；恢复数据、两组训练、比较和自训模型列表。旧独立标注访问码已移除，可直接查看04中的导出报告与A/B原始标签。\n'
         '按固定脚本准备SAM/DINO后启动网页，可查看真实结果并演示；FFmpeg需单独安装。历史参数中的旧绝对路径保留为原始证据，不用它们覆盖当前目录。\n')

    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    tracked=subprocess.check_output(['git','ls-tree','-r','-z','--name-only','HEAD'],cwd=ROOT).decode().split('\0')
    allow=[name for name in tracked if name and not name.startswith(('artifacts/','data/','fire-models/','general-models/','demo-images/','demo-videos/'))]
    archive=stage/'02_源代码/source_code.zip';archive.parent.mkdir(exist_ok=True)
    subprocess.run(['git','archive','--format=zip','-o',str(archive),'HEAD','--',*allow],cwd=ROOT,check=True)
    add(ROOT/'requirements-lock.txt','02_源代码/requirements-lock.txt')
    text('02_源代码/repository.txt',f'https://github.com/c1love9/wildfire-detection\nbranch=main\ncommit={commit}\n上游及许可见源码docs/upstream.md与LICENSE。\n')
    acceptance=read_json(ROOT/'reports/requirements_acceptance.json')
    text('README_先看这里.md','# 智能火灾报警系统交付材料\n\n'
         '700幅人工复核、A/B各200幅原始独立标注、两组50轮自训模型与统一独立test结果均已保存；报告采用教师Word模板，姓名学号暂不填写，教师签章留空。\n\n'
         '先阅读07中的quick_start与完整使用指南；01是可编辑报告；03是最终标签；05为本人训练权重（output/weights/best.pt），06为逐图评测，08含新目录恢复与校验。\n\n'
         '当前不能标记所有原题要求完成。仍待补：\n'+''.join('- '+gate+'\n' for gate in acceptance['human_and_formal_gates'])+
         '\n测试仅1幅正常图，误报率参考价值有限；已知同源组隔离不证明全部公开预训练素材独立。各数据源保留署名和学术用途限制；不含私人摄像头画面、旧访问码、token或虚拟环境。\n')
    files={path.relative_to(stage).as_posix():{'bytes':path.stat().st_size,'sha256':file_sha(path)} for path in sorted(stage.rglob('*')) if path.is_file()}
    manifest={'schema_version':1,'release_id':output.name,'created_at':now(),'source_commit':commit,
              'repository':'https://github.com/c1love9/wildfire-detection','formal_acceptance_complete':False,
              'remaining':acceptance['human_and_formal_gates'],'export_id':workflow['export_id'],
              'experiment_id':formal['experiment_id'],'selected_training_id':formal['selection']['training_id'],
              'comparison_id':formal['comparison_id'],'files':files}
    write_json(stage/'manifest.json',manifest)
    checksum={**files,'manifest.json':{'sha256':file_sha(stage/'manifest.json')}}
    text('SHA256SUMS.txt',''.join(info['sha256']+'  '+name+'\n' for name,info in sorted(checksum.items())))
    target=output/'智能火灾报警系统_交付包.zip'
    with ZipFile(target,'x',ZIP_DEFLATED,compresslevel=4) as archive:
        for path in sorted(stage.rglob('*')):
            if path.is_file():archive.write(path,path.relative_to(stage).as_posix())
    with ZipFile(target) as archive:
        assert archive.testzip() is None
        for name,info in files.items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=info['sha256']:raise OSError('ZIP member mismatch: '+name)
    result={'status':'passed','archive':str(target),'folder':str(stage),'files':len(files)+2,
            'bytes':target.stat().st_size,'sha256':file_sha(target),'source_commit':commit,
            'images':700,'videos':300,'independent_A':200,'independent_B':200,'formal_training_jobs':2,
            'old_access_codes_included':False,'crc_and_all_payload_hashes_verified':True,
            'formal_acceptance_complete':False}
    write_json(output/'verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report-document',type=Path)
    args=parser.parse_args();package(args.workspace.resolve(),args.output.resolve(),args.report_document.resolve() if args.report_document else None)
