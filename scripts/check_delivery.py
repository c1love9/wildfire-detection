"""Read-only final acceptance on existing demos and isolated fixture records."""
from pathlib import Path
import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
from zipfile import ZipFile
from xml.etree import ElementTree as ET

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch  # Initialize Windows native runtime before image/Arrow modules.
from dataset_store import DatasetStore,file_sha,now
from comparison import ComparisonStore
from experiments import ExperimentStore
from segmentation import verify_model
from recording import resolve_media_asset

RELEASE=ROOT/'artifacts/release'


def database_snapshot(path):
    result={}
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
        assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        names=[r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        for name in names:
            quoted='"'+name.replace('"','""')+'"'
            rows=db.execute('SELECT * FROM '+quoted).fetchall()
            payload=json.dumps(sorted(rows,key=repr),ensure_ascii=False,default=str,sort_keys=True).encode()
            result[name]={'rows':len(rows),'content_sha256':hashlib.sha256(payload).hexdigest()}
    return result


def snapshot():
    fixture=json.loads((ROOT/'reports/model_integration.json').read_text(encoding='utf-8'))
    data=ROOT/fixture['dataset_root']
    return {'catalog':database_snapshot(data/'catalog.sqlite3'),
            'alarms':database_snapshot(ROOT/'artifacts/completion_checks/alarms.sqlite3')}


def verify():
    fixture=json.loads((ROOT/'reports/model_integration.json').read_text(encoding='utf-8'))
    dataset=DatasetStore(ROOT/fixture['dataset_root'])
    comparison_id=json.loads((ROOT/'reports/comparison_verification.json').read_text())['comparison_id']
    assert ComparisonStore(dataset).result(comparison_id)['id']==comparison_id
    pair=json.loads((ROOT/'reports/experiments_verification.json').read_text())['pair']
    experiments=ExperimentStore(dataset);experiments.read(pair);experiments.evaluation(pair)
    sam=verify_model()
    with dataset.connection() as db:
        mask_rows=db.execute('SELECT record_json FROM quality_masks').fetchall()
        assert mask_rows,'No saved synthetic mask workflow'
        for row in mask_rows:
            record=json.loads(row[0]);directory=dataset.root/'quality_masks'/record['id']
            assert file_sha(directory/'reference.png')==record['reference_sha256']
            assert file_sha(directory/'prediction.png')==record['prediction_sha256']
        assert db.execute('SELECT COUNT(*) FROM quality_submissions').fetchone()[0]>=2
        assert db.execute('SELECT COUNT(*) FROM quality_references').fetchone()[0]>=1
    alarm_path=ROOT/'artifacts/completion_checks/alarms.sqlite3'
    with sqlite3.connect(alarm_path) as db:
        rows=db.execute("SELECT snapshot_path,clip_path FROM event_media WHERE status IN ('ready','partial') ORDER BY updated_at DESC LIMIT 5").fetchall()
        assert rows
        for still,clip in rows:
            assert resolve_media_asset(alarm_path.parent/'media',still,'.jpg')
            assert resolve_media_asset(alarm_path.parent/'media',clip,'.mp4')
        samples=db.execute('SELECT COUNT(*) FROM observation_samples').fetchone()[0]
        assert samples>0
    production=ROOT/'data/fire_dataset/catalog.sqlite3'
    production_counts=database_snapshot(production)
    assert production_counts['assets']['rows']==0,'Synthetic fixture must not populate the normal dataset'
    assert production_counts.get('training_jobs',{'rows':0})['rows']==0
    report=RELEASE/'智能火灾报警系统设计报告.docx'
    with ZipFile(report) as archive:
        tree=ET.fromstring(archive.read('word/document.xml'))
        ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        text=''.join(tree.itertext())
        assert '待正式数据' in text and '160' in text
        assert len(tree.findall('.//w:tbl',ns))>=8
        assert len([p for p in archive.namelist() if p.startswith('word/media/')])>=2
        assert len(tree.findall('.//w:p',ns))>100,'Report must contain editable paragraphs'
        fidelity=json.loads((ROOT/'reports/template_report.json').read_text(encoding='utf-8'))
        assert fidelity['output_sha256']==file_sha(report) and fidelity['preserve_only_parts_identical']
        assert fidelity['teacher_signature_and_scores_blank']
        assert 'word/media/image6.png' not in archive.namelist()
        assert len(tree.findall('.//{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath'))==2
    video=RELEASE/'智能火灾报警系统演示.mp4'
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_name,width,height,pix_fmt','-of','json',str(video)]))
    duration=float(probe['format']['duration']);assert 180<=duration<=300
    stream=probe['streams'][0];assert stream['codec_name']=='h264' and stream['pix_fmt']=='yuv420p'
    tests=(ROOT/'artifacts/completion_tests.txt').read_text(encoding='utf-8-sig')
    assert 'Ran 160 tests' in tests and tests.rstrip().endswith('OK')
    before=json.loads((RELEASE/'persistence_before.json').read_text())
    after=snapshot();assert after==before,'History changed across server restart'
    result={'status':'passed','verified_at':now(),'scope':'System features and materials only; public demos and existing isolated synthetic fixture',
            'unit_tests_passed':160,'sam_revision':sam['revision'],'comparison_and_pair_inventory_verified':True,
            'quality_masks_and_submissions_persisted':True,'alarm_media_pairs_checked':len(rows),'trend_samples_persisted':samples,
            'exact_database_content_survives_restart':True,'normal_dataset_assets':0,'normal_training_jobs':0,
            'docx_editable_text_and_tables':True,'docx_sha256':file_sha(report),'video':probe,'video_sha256':file_sha(video),
            'formal_experiments':'pending real data, human independence, convergence, test, mask reference, controlled worktime and optimization'}
    (ROOT/'reports/delivery_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',action='store_true');args=parser.parse_args()
    if args.snapshot:
        (RELEASE/'persistence_before.json').write_text(json.dumps(snapshot(),sort_keys=True),encoding='utf-8');print('Read-only database content snapshot saved; no role access codes exported.')
    else:verify()
