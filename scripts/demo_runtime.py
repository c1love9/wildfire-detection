"""Check and launch a separate local demonstration, without touching the source workspace."""
import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def safe_path(root, relative):
    root = Path(root).resolve()
    if Path(relative).is_absolute():
        raise ValueError('演示清单须使用包内相对路径。')
    target = (root/relative).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError('演示清单路径超出运行包。')
    return target


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def configure(root):
    root = Path(root).resolve()
    config_path = root/'.runtime/runtime.json'
    config = read(config_path) if config_path.is_file() else {}
    os.environ.update({
        'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1', 'PYTHONPATH': str(root),
        'FIRE_DATASET_ROOT': str(root/'data/fire_dataset'),
        'FIRE_ALARM_LOG_PATH': str(root/'artifacts/alarms/alarms.sqlite3'),
        'FIRE_ALARM_MEDIA_ROOT': str(root/'artifacts/alarms/media'),
        'FIRE_CHANNEL_CONFIG': str(root/'configs/channels.json'),
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
        'STREAMLIT_BROWSER_GATHER_USAGE_STATS': 'false', 'YOLO_AUTOINSTALL': 'false',
    })
    if config.get('selected_training_id'):
        os.environ['FIRE_DEMO_MODEL_KEY'] = 'trained:'+config['selected_training_id']+':best'
    else:
        os.environ.pop('FIRE_DEMO_MODEL_KEY', None)
    # Pin explicit local snapshots even if the parent shell points at a different project.
    os.environ['GROUNDING_DINO_MODEL_DIR'] = str(root/'artifacts/models/grounding-dino-tiny/a2bb814dd30d776dcf7e30523b00659f4f141c71')
    machine = root/'.runtime/machine.json'
    if machine.is_file():
        executable = read(machine).get('ffmpeg_executable')
        if executable and Path(executable).is_file():
            os.environ['FIRE_ALARM_FFMPEG'] = executable
    return config


def check(root, deep=False):
    root = Path(root).resolve()
    config = configure(root)
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError('请使用本项目已经验证的Python 3.11环境。')
    import torch  # Windows: initialize Torch before OpenCV/Arrow/Transformers.
    import cv2
    import streamlit
    import ultralytics
    import transformers
    from segmentation import verify_model
    from channels import load_channels
    manifest_path = root/'runtime_manifest.json'
    checked = 0
    if manifest_path.is_file():
        manifest = read(manifest_path)
        for relative, info in manifest['files'].items():
            path = safe_path(root, relative)
            if not path.is_file():
                raise FileNotFoundError('运行包缺少文件：'+relative)
            if info.get('mutable'):
                continue
            if path.stat().st_size != info['bytes'] or deep and digest(path) != info['sha256']:
                raise OSError('运行包文件损坏或已改变：'+relative)
            checked += 1
    for filename in ('1_🔥_Home.py', 'demo-videos/fire1.mov', 'demo-videos/fire2.mov'):
        if not (root/filename).is_file():
            raise FileNotFoundError('缺少现场演示文件：'+filename)
    sam = verify_model(root/'artifacts/models/sam-vit-base/70c1a07f894ebb5b307fd9eaaee97b9dfc16068f')
    open_models = []
    for family in ('grounding-dino-tiny', 'grounding-dino-base'):
        receipts = list((root/'artifacts/models'/family).glob('*/model_receipt.json'))
        if len(receipts) != 1:
            raise FileNotFoundError('缺少完整开放模型：'+family)
        receipt = read(receipts[0])
        for name, expected in receipt['files'].items():
            item = safe_path(receipts[0].parent, name)
            if not item.is_file() or deep and digest(item) != expected:
                raise OSError('开放模型文件不完整：'+family+'/'+name)
        open_models.append({'model_id': receipt['model_id'], 'revision': receipt['revision']})
    db_path = root/'data/fire_dataset/catalog.sqlite3'
    with closing(sqlite3.connect(db_path.as_uri()+'?mode=ro', uri=True)) as db:
        if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise OSError('演示副本数据库完整性检查未通过。')
        kinds = dict(db.execute('SELECT kind,COUNT(*) FROM assets GROUP BY kind'))
        submissions = db.execute('SELECT COUNT(*) FROM quality_submissions').fetchone()[0]
        if config.get('selected_training_id'):
            row = db.execute('SELECT status FROM training_jobs WHERE id=?', (config['selected_training_id'],)).fetchone()
            if not row or row[0] != 'completed':
                raise ValueError('演示指定的自训任务未完成或缺失。')
            folder = root/'data/fire_dataset/training'/config['selected_training_id']
            result = read(folder/'result.json')
            weights = result['weights']['best']
            if digest(safe_path(folder, weights['path'])) != weights['sha256']:
                raise OSError('演示指定的自训best权重缺失或校验不符。')
    encoder = shutil.which(os.environ.get('FIRE_ALARM_FFMPEG', 'ffmpeg'))
    if not encoder:
        raise RuntimeError('缺少FFmpeg，不能演示报警片段保存。请保留本机已安装的FFmpeg。')
    encoders = subprocess.run([encoder, '-hide_banner', '-encoders'], capture_output=True,
                             text=True, encoding='utf-8', errors='replace', timeout=15, check=True)
    if 'libx264' not in encoders.stdout:
        raise RuntimeError('当前FFmpeg不支持本系统使用的libx264编码。')
    channels = load_channels(root/'configs/channels.json')
    report = {'status': 'passed', 'project_root': str(root), 'python': sys.executable,
              'python_version': sys.version.split()[0], 'torch': torch.__version__,
              'streamlit': streamlit.__version__, 'ultralytics': ultralytics.__version__,
              'transformers': transformers.__version__, 'opencv': cv2.__version__,
              'gpu_available': torch.cuda.is_available(),
              'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              'images': kinds.get('image', 0), 'videos': kinds.get('video', 0),
              'independent_submissions': submissions, 'sam_revision': sam['revision'],
              'open_models': open_models, 'channels': len(channels), 'ffmpeg': encoder,
              'selected_training_id': config.get('selected_training_id'),
              'immutable_files_checked': checked, 'deep_sha256_check': deep,
              'scope': 'Current computer: reuses an existing Python environment; datasets, models and state are local to this copy.'}
    write(root/'.runtime/run/environment_check.json', report)
    print('环境检查通过。图像 %s 幅，视频 %s 段，双人标注 %s 份。' %
          (report['images'], report['videos'], report['independent_submissions']))
    print('GPU：'+(report['gpu'] or '未发现可用GPU，可使用CPU，速度会降低。'))
    print('自训YOLO、SAM、DINO、本地演示视频、历史和FFmpeg已检查。')
    return report


def owns_process(root, state, process):
    """A saved PID alone never authorizes stopping an unrelated process."""
    import psutil
    try:
        command = process.cmdline()
        same_entry = any(Path(arg).resolve() == Path(root).resolve()/'1_🔥_Home.py'
                         for arg in command if arg.endswith('_Home.py'))
        return ('streamlit' in command and same_entry and
                abs(process.create_time()-state['created_at']) < .01)
    except (OSError, ValueError, KeyError, psutil.Error):
        return False


def running(root):
    import psutil
    state_path = Path(root)/'.runtime/run/server.json'
    if not state_path.is_file():
        return None
    state = read(state_path)
    try:
        process = psutil.Process(state['pid'])
        return (state, process) if process.is_running() and owns_process(root, state, process) else None
    except (psutil.Error, KeyError):
        return None


def free_port(preferred):
    for port in range(preferred, preferred+20):
        with socket.socket() as probe:
            try:
                probe.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    raise RuntimeError('演示端口均被占用，请关闭重复启动的演示窗口后重试。')


def start(root, port=8506, open_browser=True):
    import psutil
    root = Path(root).resolve()
    current = running(root)
    if current:
        print('本运行包已启动：'+current[0]['url'])
        if open_browser:
            webbrowser.open(current[0]['url'])
        return current[0]
    check(root)
    chosen = free_port(port)
    url = f'http://127.0.0.1:{chosen}'
    run = root/'.runtime/run'; run.mkdir(parents=True, exist_ok=True)
    with (run/'server.log').open('a', encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', str(root/'1_🔥_Home.py'),
                                    '--server.address', '127.0.0.1', '--server.port', str(chosen),
                                    '--server.headless', 'true', '--server.fileWatcherType', 'none'],
                                   cwd=root, env=os.environ.copy(), stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    state = {'pid': process.pid, 'created_at': psutil.Process(process.pid).create_time(),
             'root': str(root), 'port': chosen, 'url': url}
    write(run/'server.json', state)
    deadline = time.monotonic()+90
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError('服务未能启动，请查看.runtime/run/server.log。')
        try:
            with urllib.request.urlopen(url+'/_stcore/health', timeout=2) as response:
                if response.status == 200 and process.poll() is None:
                    print('现场演示已启动：'+url)
                    print('请保留原项目的.venv。停止时先在网页结束检测，再双击“停止现场演示.cmd”。')
                    if open_browser:
                        webbrowser.open(url)
                    return state
        except OSError:
            pass
        time.sleep(.4)
    # Only the child created by this invocation is terminated on startup timeout.
    process.terminate()
    raise RuntimeError('服务启动超时，请查看.runtime/run/server.log。')


def stop(root):
    import psutil
    current = running(root)
    if not current:
        print('没有本运行包启动的服务，不会停止其他项目。')
        return
    state, process = current
    process.terminate()
    try:
        process.wait(timeout=10)
    except psutil.TimeoutExpired:
        if owns_process(root, state, process):
            process.kill(); process.wait(timeout=5)
    (Path(root)/'.runtime/run/server.json').unlink(missing_ok=True)
    print('本运行包服务已停止。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'start', 'stop'])
    parser.add_argument('--deep', action='store_true')
    parser.add_argument('--port', type=int, default=8506)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    try:
        if args.action == 'check':
            check(ROOT, args.deep)
        elif args.action == 'start':
            start(ROOT, args.port, not args.no_browser)
        else:
            stop(ROOT)
    except Exception as error:
        print('无法继续：'+str(error), file=sys.stderr)
        sys.exit(1)
