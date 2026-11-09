"""Assemble an executable demo copy from committed source and the verified frozen release."""
import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import secrets
import shutil
import sqlite3
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.restore_final_delivery import restore
from scripts.demo_runtime import digest, read, safe_path, write


def table_fingerprints(database):
    output = {}
    with sqlite3.connect(Path(database).resolve().as_uri()+'?mode=ro', uri=True) as db:
        names = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        for name in names:
            quoted = '"'+name.replace('"', '""')+'"'
            rows = sorted(db.execute('SELECT * FROM '+quoted).fetchall(), key=repr)
            output[name] = {'rows': len(rows), 'sha256': hashlib.sha256(json.dumps(rows, ensure_ascii=False, default=str).encode()).hexdigest()}
    return output


def assemble(package, output):
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise ValueError('Commit reviewed launchers and documentation before packaging.')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if output.exists():
        raise FileExistsError('Use a new output directory; existing packages are retained.')
    stage = output/'现场运行'
    stage.mkdir(parents=True)
    tracked = subprocess.check_output(['git', 'ls-tree', '-r', '-z', '--name-only', 'HEAD'], cwd=ROOT).decode().split('\0')
    selected = [name for name in tracked if name and not name.startswith(('artifacts/', 'data/', 'fire-models/', 'general-models/', 'demo-images/', 'demo-videos/', '.runtime/'))]
    archive = subprocess.check_output(['git', 'archive', '--format=zip', 'HEAD', '--', *selected], cwd=ROOT)
    with ZipFile(BytesIO(archive)) as source:
        for name in source.namelist():
            target = safe_path(stage, name)
            if name.endswith('/'):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read(name))
    print('Committed code copied. Verifying and restoring the frozen release...', flush=True)
    restoration = restore(package, stage)
    frozen = read(package/'manifest.json')
    write(stage/'.runtime/runtime.json', {
        'schema_version': 1, 'release_id': 'onsite_20261109', 'source_commit': commit,
        'frozen_release_id': frozen['release_id'], 'frozen_source_commit': frozen['source_commit'],
        'selected_training_id': frozen['selected_training_id'], 'comparison_id': frozen['comparison_id'],
        'experiment_id': frozen['experiment_id'], 'preferred_port': 8506,
        'target': 'Current Windows computer; Python environment is reused, not copied.'})
    write(stage/'.runtime/machine.json', {'python_executable': sys.executable,
          'ffmpeg_executable': shutil.which('ffmpeg'), 'ffprobe_executable': shutil.which('ffprobe'),
          'scope': 'Current computer only. Preserve the original project .venv and installed FFmpeg.'})
    database = stage/'data/fire_dataset/catalog.sqlite3'
    before = table_fingerprints(database)
    token = secrets.token_urlsafe(24)
    with sqlite3.connect(database) as db:
        if db.execute('SELECT COUNT(*) FROM quality_access').fetchone()[0] != 0:
            raise ValueError('Use the sanitized release without historical access credentials.')
        tasks = db.execute("SELECT task_id FROM quality_submissions GROUP BY task_id HAVING SUM(role='A')=200 AND SUM(role='B')=200").fetchall()
        if len(tasks) != 1:
            raise ValueError('Expected one completed 200-image A/B task.')
        db.execute('INSERT INTO quality_access VALUES(?,?,?,?)',
                   (hashlib.sha256(token.encode()).hexdigest(), tasks[0][0], 'admin', '现场演示报告查看'))
    after = table_fingerprints(database)
    if {k: v for k, v in before.items() if k != 'quality_access'} != {k: v for k, v in after.items() if k != 'quality_access'}:
        raise ValueError('Demonstration access setup must not change any original annotation or experiment row.')
    (stage/'.runtime/标注报告管理码.txt').write_text(token+'\n', encoding='utf-8')
    # In this delivery, the familiar launcher delegates to the checked demo launcher.
    (stage/'启动项目.cmd').write_bytes(('@echo off\r\nchcp 65001 >nul\r\ncall "%~dp0启动现场演示.cmd"\r\n').encode('utf-8'))
    materials = stage/'演示资料'; materials.mkdir()
    shutil.copy2(ROOT/'docs/live_runtime.md', stage/'00_先读说明.md')
    shutil.copy2(ROOT/'docs/live_demo_guide.md', materials/'完整现场操作指南.md')
    shutil.copy2(ROOT/'docs/formal_results.md', materials/'冻结实验结果说明.md')
    shutil.copytree(package/'01_数据与标注/最终数据集/licenses', materials/'数据来源与许可')
    shutil.copy2(package/'02_模型与训练/开放模型/来源版本.json', materials/'开放模型来源版本.json')
    shutil.copy2(stage/'artifacts/final_completion_20261108/demo/系统演示.mp4', materials/'备用_系统演示.mp4')
    defense = ROOT/'artifacts/defense_20261108/交付'
    if defense.is_dir():
        for name in ('智能火灾报警系统_10分钟验收答辩_定稿.pptx', '10分钟讲稿与现场操作.md'):
            shutil.copy2(defense/name, materials/name)
    figures = ROOT/'artifacts/performance_visualizations_20261108/03_性能图表_交付版'
    if figures.is_dir():
        shutil.copy2(figures/'独立测试性能图表汇总.pdf', materials/'独立测试性能图表.pdf')
    files = {}
    for p in sorted(stage.rglob('*')):
        if not p.is_file():
            continue
        relative = p.relative_to(stage).as_posix()
        mutable = relative.endswith('.sqlite3') or relative in (
            '.runtime/machine.json', 'configs/channels.json', 'configs/monitoring_points.json',
            'configs/alarm.json', 'configs/recording.json', 'configs/camera.json')
        files[relative] = {'bytes': p.stat().st_size, 'sha256': digest(p), 'mutable': mutable}
    manifest = {'schema_version': 1, 'release_id': 'onsite_20261109', 'source_commit': commit,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'source_package_manifest_sha256': digest(package/'manifest.json'),
                'runtime_adjustments': ['Fresh local report-only usage admin access, original submissions unchanged.',
                                        'Legacy launcher delegates to demo launcher.'],
                'python_environment_included': False, 'offline_model_snapshots_included': True,
                'restoration': restoration, 'files': files}
    write(stage/'runtime_manifest.json', manifest)
    (stage/'SHA256SUMS.txt').write_text(''.join(v['sha256']+'  '+k+'\n' for k, v in sorted({**files, 'runtime_manifest.json': {'sha256': digest(stage/'runtime_manifest.json')}}.items())), encoding='utf-8')
    report = {'status': 'assembled', 'folder': str(stage), 'source_commit': commit,
              'payload_files': len(files), 'bytes_uncompressed': sum(v['bytes'] for v in files.values()),
              'restored_files': restoration['restored_files'], 'old_access_credentials_included': False,
              'new_local_report_access_created': True, 'original_annotation_and_experiment_tables_unchanged': True,
              'source_data_models_and_logs_modified': False, 'python_environment_included': False}
    write(output/'assembly.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return stage


def archive_verified(stage):
    manifest = read(stage/'runtime_manifest.json')
    archive = stage.parent/'智能火灾报警系统_现场运行包.zip'
    if archive.exists():
        raise FileExistsError('Archive already exists.')
    files = [p for p in sorted(stage.rglob('*')) if p.is_file() and '.runtime/run/' not in p.relative_to(stage).as_posix() and '__pycache__' not in p.parts]
    with ZipFile(archive, 'x', ZIP_DEFLATED, compresslevel=3, allowZip64=True) as z:
        for index, p in enumerate(files, 1):
            relative = p.relative_to(stage).as_posix()
            if relative in manifest['files'] and digest(p) != manifest['files'][relative]['sha256']:
                raise OSError('Staging payload changed before archiving: '+relative)
            z.write(p, '现场运行/'+relative)
            if index % 500 == 0:
                print(f'Archive {index}/{len(files)} files', flush=True)
    print('Checking every archived member...', flush=True)
    with ZipFile(archive) as z:
        if len(z.namelist()) != len(files):
            raise ValueError('Archive inventory differs.')
        for p in files:
            h = hashlib.sha256()
            with z.open('现场运行/'+p.relative_to(stage).as_posix()) as stream:
                for block in iter(lambda: stream.read(4*1024**2), b''):
                    h.update(block)
            if h.hexdigest() != digest(p):
                raise OSError('Archive content differs: '+p.name)
    report = {'status': 'passed', 'archive': str(archive), 'bytes': archive.stat().st_size,
              'sha256': digest(archive), 'payload_files': len(files), 'every_member_sha256_verified': True,
              'source_commit': manifest['source_commit'], 'python_environment_included': False,
              'target': 'Current Windows computer, existing Python 3.11 environment reused.'}
    write(stage.parent/'archive_verification.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--archive-stage', type=Path)
    args = parser.parse_args()
    if args.archive_stage:
        archive_verified(args.archive_stage.resolve())
    elif args.package and args.output:
        assemble(args.package.resolve(), args.output.resolve())
    else:
        parser.error('Provide --package and --output, or --archive-stage.')
