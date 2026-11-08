"""Assemble one six-criterion submission from a verified data release and current source.

The frozen data and experiments are reused unchanged. Never copy a workspace
wholesale: credentials, personal media and virtual environments are excluded.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def renamed(name):
    first,_,rest=name.partition('/')
    if first=='01_设计报告':return '06_源码报告与复现/设计报告/'+rest
    if first=='02_源代码':return None
    if first=='03_最终数据集':return '01_数据与标注/最终数据集/'+rest
    if first=='04_标注过程与质量':
        if rest=='worktime_report.md':return None
        return '01_数据与标注/标注过程与质量/'+rest
    if first=='05_模型与训练':return '02_模型与训练/'+rest
    if first=='06_评测与实验':
        if rest.startswith(('optimization/','alarm_ablation/')):return '04_优化对照/'+rest
        return '03_性能评测/'+rest
    if first=='07_演示与操作':
        if rest.startswith('demo_media/'):return '05_系统演示/'+rest
        return None
    if first=='08_验收与校验':return '06_源码报告与复现/验收与复现/'+rest
    if first=='09_待补素材':return '01_数据与标注/未确认候选_不计数/'+rest
    if first=='09_过程附录' and rest.startswith('interrupted/'):
        return '04_优化对照/过程历史/'+rest
    return None


def package(base,output,guide):
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise ValueError('Commit reviewed source and documents before archiving')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    original=read(base/'manifest.json')
    oldmap=read(base/'08_验收与校验/restore_map.json')
    # Validate every base member, including files intentionally superseded.
    for name,info in original['files'].items():
        path=(base/name).resolve()
        if not path.is_relative_to(base.resolve()) or path.stat().st_size!=info['bytes'] or digest(path)!=info['sha256']:
            raise OSError('Base payload changed: '+name)
    stage=output/'智能火灾报警系统_提交材料';stage.mkdir(parents=True,exist_ok=False)
    mapping=[]

    def add(source,name,target=None):
        dest=stage/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        if target:mapping.append({'source':name,'target':target})
        return dest

    def text(name,value):
        path=stage/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value.rstrip()+'\n',encoding='utf-8')

    for name in original['files']:
        replacement=renamed(name)
        if replacement:add(base/name,replacement)
    for row in oldmap['files']:
        replacement=renamed(row['source'])
        if not replacement:raise ValueError('Required restore member excluded: '+row['source'])
        mapping.append({'source':replacement,'target':row['target']})

    checks='06_源码报告与复现/验收与复现/'
    # A consistent current backup, not the private pre-work catalog.
    catalog=stage/checks/'catalog_sanitized.sqlite3'
    catalog.unlink()
    source=sqlite3.connect((ROOT/'data/fire_dataset/catalog.sqlite3').as_uri()+'?mode=ro',uri=True)
    target=sqlite3.connect(catalog)
    try:
        source.backup(target)
        with target:target.execute('DELETE FROM quality_access')
        target.execute('VACUUM')
        assert target.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        assert target.execute('SELECT COUNT(*) FROM quality_access').fetchone()[0]==0
        target.row_factory=sqlite3.Row
        reviews=[dict(r) for r in target.execute('SELECT r.* FROM annotation_reviews r JOIN (SELECT asset_id,MAX(seq) seq FROM annotation_reviews GROUP BY asset_id) l ON r.seq=l.seq ORDER BY r.asset_id')]
        independent=[dict(r) for r in target.execute('SELECT * FROM quality_submissions ORDER BY task_id,asset_id,role')]
        freeze=read(stage/checks/'review_freeze_receipt.json')
        fingerprint=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        assert fingerprint(reviews)==freeze['latest_reviews_sha256']
        assert fingerprint(independent)==freeze['independent_submissions_sha256']
        assert len(reviews)==700 and len(independent)==400
    finally:target.close();source.close()

    snapshots=[]
    for family in ('sam-vit-base','grounding-dino-tiny','grounding-dino-base'):
        for receipt_path in sorted((ROOT/'artifacts/models'/family).glob('*/model_receipt.json')):
            receipt=read(receipt_path);folder=receipt_path.parent
            expected=set(receipt['files'])|{'model_receipt.json'}
            assert {p.name for p in folder.iterdir() if p.is_file()}==expected
            for name,sha in receipt['files'].items():
                assert digest(folder/name)==sha,(family,name)
            for path in sorted(folder.iterdir()):
                if path.is_file():
                    relative=path.relative_to(ROOT/'artifacts/models').as_posix()
                    add(path,'02_模型与训练/开放模型/'+relative,'artifacts/models/'+relative)
            snapshots.append({'model_id':receipt['model_id'],'revision':receipt['revision'],
                              'official_page':'https://huggingface.co/'+receipt['model_id'],
                              'receipt':receipt_path.relative_to(ROOT).as_posix()})
    assert len(snapshots)==3
    write(stage/'02_模型与训练/开放模型/来源版本.json',snapshots)
    text('02_模型与训练/开放模型/README.md','本目录为已按官方固定版本校验的完整推理快照，模型自身不在本项目上训练或微调。来源和版本见来源版本.json及各model_receipt.json。原始模型与项目许可分别适用，官方页面见清单。Tiny用于本次独立检测对比，Base用于增强预标注，SAM用于框提示分割；不能互换实验身份。恢复到新项目后可离线推理，依赖和驱动另装。')
    text('02_模型与训练/upstream/README.md','fire_n.pt是上游火烟模型，yolov8n.pt是COCO初始化模型，均不是本人训练成果。本人模型见baseline和multiscale/output/weights。SAM及DINO完整固定版本另见开放模型，恢复后无需重新下载。源码保留原始LICENSE及docs/upstream.md。')

    for name in ('worktime_estimate_20261108.json','annotation_iou_20261108.json'):
        add(ROOT/'reports'/name,'01_数据与标注/标注过程与质量/'+name)
    add(ROOT/'docs/worktime_and_iou.md','01_数据与标注/标注过程与质量/工时与框IoU说明.md')
    text('01_数据与标注/标注过程与质量/worktime_report.md','本目录worktime.csv是保留的原始工作计时导出，不代表700幅辅助复核全量计时。当前估算使用负责人实测约5秒/幅及A/B原始计时，结果见工时与框IoU说明.md和worktime_estimate_20261108.json；不再将总工时全部标记为待补。')
    # Old report is superseded by the latest teacher-template editable document.
    report=ROOT/'artifacts/worktime_update_20261108/智能火灾报警系统设计报告_工时修订版.docx'
    assert digest(report)==read(ROOT/'reports/worktime_revision_20261108.json')['report_sha256']
    add(report,'06_源码报告与复现/设计报告/智能火灾报警系统设计报告.docx')
    editable=(ROOT/'docs/template_report.md').read_text(encoding='utf-8')
    def figure_link(match):
        name=Path(match.group(2)).name
        assert (stage/'06_源码报告与复现/设计报告/figures'/name).is_file(),name
        return '!['+match.group(1)+'](figures/'+name+')'
    text('06_源码报告与复现/设计报告/报告可编辑正文.md',re.sub(r'!\[(.*?)\]\((.*?)\)',figure_link,editable))
    add(guide,'05_系统演示/智能火灾报警系统完整使用指南.docx')
    add(ROOT/'artifacts/live_demo_20261108/智能火灾报警系统现场演示操作指南.docx','05_系统演示/智能火灾报警系统现场演示操作指南.docx')
    for name in ('user_guide.md','live_demo_guide.md','quick_start.md','worktime_and_iou.md'):
        add(ROOT/'docs'/name,'05_系统演示/'+name)
    for name in ('channels.json','monitoring_points.json'):
        add(ROOT/'configs'/name,'05_系统演示/通道配置/'+name)
    video=ROOT/'artifacts/submission_20261108/演示视频修订.mp4'
    add(video,'05_系统演示/系统演示.mp4','artifacts/final_completion_20261108/demo/系统演示.mp4')
    text('05_系统演示/视频说明.md','视频保留原有真实界面操作，末段覆盖了已经过时的“工时待补”字幕，标明约5秒/幅为负责人实测输入、总工时为估算。不是重新采集或重训。实际操作以本包完整指南和现场指南为准。界面中的位置与视角为可操作模拟值。')

    for name in ('requirements_acceptance.json','worktime_estimate_20261108.json','annotation_iou_20261108.json','worktime_revision_20261108.json','live_demo_guide_20261108.json'):
        add(ROOT/'reports'/name,checks+name)
    for name in ('document_verification.json','video_verification.json'):
        add(ROOT/'artifacts/submission_20261108'/name,checks+name)
    add(ROOT/'docs/requirements_audit.md',checks+'requirements_checklist.md')
    add(ROOT/'artifacts/submission_20261108/regression_tests.txt',checks+'regression_tests.txt')
    write(stage/checks/'restore_map.json',{'schema_version':2,'files':mapping,'access_codes_excluded':True})
    text(checks+'复现说明.md',(ROOT/'docs/submission_guide.md').read_text(encoding='utf-8'))

    source_dir=stage/'06_源码报告与复现/源代码';source_dir.mkdir(parents=True,exist_ok=True)
    tracked=subprocess.check_output(['git','ls-tree','-r','-z','--name-only','HEAD'],cwd=ROOT).decode().split('\0')
    allowed=[name for name in tracked if name and not name.startswith(('artifacts/','data/','fire-models/','general-models/','demo-images/','demo-videos/'))]
    subprocess.run(['git','archive','--format=zip','-o',str(source_dir/'source_code.zip'),'HEAD','--',*allowed],cwd=ROOT,check=True)
    add(ROOT/'requirements-lock.txt','06_源码报告与复现/源代码/requirements-lock.txt')
    text('06_源码报告与复现/源代码/repository.txt',f'https://github.com/c1love9/wildfire-detection\nbranch=main\ncommit={commit}\n源码对应已提交版本，不包含本机未提交文件。数据和权重单独交付，不推入Git。')
    acceptance=read(ROOT/'reports/requirements_acceptance.json')
    write(stage/'六项验收清单.json',{'formal_acceptance_complete':False,'facts':acceptance['facts'],
          'remaining':acceptance['human_and_formal_gates'],'source_document_sha256':acceptance['source_sha256'],
          'criteria':[{'criterion':i,'folder':folder} for i,folder in enumerate(sorted(p.name for p in stage.iterdir() if p.is_dir()),1)]})
    text('README_先读.md',(ROOT/'docs/submission_guide.md').read_text(encoding='utf-8'))
    for folder in [p for p in stage.iterdir() if p.is_dir()]:
        if not (folder/'README.md').exists():
            text(folder.name+'/README.md','本目录对应原题第'+folder.name[:2].lstrip('0')+'项材料，具体内容、结果口径与待补项见总目录README_先读.md。')
    text('03_性能评测/README_指标口径.md',(ROOT/'docs/formal_results.md').read_text(encoding='utf-8'))
    text('06_源码报告与复现/设计报告/版本说明.md','本Word为教师模板的工时修订版，保留封面和评定表，身份及教师签章留空。报告记录的216项测试是先前回归快照，本次新增恢复映射检查后的最新结果见验收与复现/regression_tests.txt。报告所述官方固定下载脚本仍保留，本统一包另附对应完整快照，可离线恢复。')

    files={p.relative_to(stage).as_posix():{'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(stage.rglob('*')) if p.is_file()}
    manifest={'schema_version':2,'release_id':'submission_20261108','created_at':datetime.now(timezone.utc).isoformat(),
              'source_commit':commit,'repository':original['repository'],'formal_acceptance_complete':False,
              'remaining':acceptance['human_and_formal_gates'],'export_id':original['export_id'],
              'experiment_id':original['experiment_id'],'selected_training_id':original['selected_training_id'],
              'comparison_id':original['comparison_id'],'restore_map':checks+'restore_map.json',
              'review_freeze_receipt':checks+'review_freeze_receipt.json','offline_models_included':True,
              'base_release_id':original['release_id'],'base_source_commit':original['source_commit'],'files':files}
    write(stage/'manifest.json',manifest)
    text('SHA256SUMS.txt','\n'.join(record['sha256']+'  '+name for name,record in sorted({**files,'manifest.json':{'sha256':digest(stage/'manifest.json')}}.items())))
    archive=output/'智能火灾报警系统_提交包.zip'
    with ZipFile(archive,'x',ZIP_DEFLATED,compresslevel=4,allowZip64=True) as z:
        for p in sorted(stage.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(stage).as_posix())
    with ZipFile(archive) as z:
        assert z.testzip() is None
        for name,info in files.items():
            h=hashlib.sha256()
            with z.open(name) as f:
                for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
            assert h.hexdigest()==info['sha256'],name
    result={'status':'passed','archive':str(archive),'folder':str(stage),'bytes':archive.stat().st_size,'sha256':digest(archive),
            'source_commit':commit,'payload_files':len(files),'restore_rows':len(mapping),'all_zip_members_hashed':True,
            'data_and_AB_fingerprints_unchanged':True,'old_access_codes_excluded':True,'offline_models_included':True,
            'formal_acceptance_complete':False,'remaining':acceptance['human_and_formal_gates']}
    write(output/'verification.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--guide',type=Path,required=True)
    a=p.parse_args();package(a.base.resolve(),a.output.resolve(),a.guide.resolve())
