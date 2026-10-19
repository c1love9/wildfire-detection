"""Exercise real weights, CPU/CUDA inference, media and the upstream Streamlit UI.

This is a functional reproduction check. It does not measure model accuracy.
Run from the project environment: python scripts/verify_reproduction.py
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import threading
from collections import Counter
from datetime import datetime, timedelta, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from streamlit.testing.v1 import AppTest
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'artifacts' / 'reproduction'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_app(app: AppTest, name: str, require_prediction: bool = False) -> dict:
    exceptions = [item.message for item in app.exception]
    errors = [item.value for item in app.error]
    assert not exceptions and not errors, f'{name}: {exceptions or errors}'
    if require_prediction:
        assert len(app.success) > 0, f'{name}: prediction did not finish'
    return {'check': name, 'passed': True, 'messages': [x.value for x in app.success]}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    packages = ['torch', 'torchvision', 'ultralytics', 'streamlit', 'numpy', 'opencv-python', 'pillow']
    report = {
        'generated_at': datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
        'scope': 'Functional reproduction with upstream pretrained weights; not an accuracy evaluation or project training run.',
        'python': platform.python_version(),
        'platform': platform.platform(),
        'packages': {name: importlib.metadata.version(name) for name in packages},
        'cuda_available': torch.cuda.is_available(),
        'torch_cuda': torch.version.cuda,
        'inference': [],
        'device_checks': [],
        'ui_checks': [],
        'video_checks': [],
    }
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda:0')
        props = torch.cuda.get_device_properties(0)
        report['gpu'] = {
            'name': props.name,
            'memory_bytes': props.total_memory,
            'capability': list(torch.cuda.get_device_capability(0)),
            'compiled_architectures': torch.cuda.get_arch_list(),
        }
        matrix = torch.randn(128, 128, device='cuda:0')
        assert torch.isfinite(matrix @ matrix).all().item(), 'CUDA matrix operation failed'
        from torchvision.ops import nms
        boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device='cuda:0')
        scores = torch.tensor([0.9, 0.8], device='cuda:0')
        kept = nms(boxes, scores, 0.5).tolist()
        assert kept == [0], 'CUDA NMS failed'
        report['cuda_checks'] = {'matrix_multiply': True, 'nms_kept_indices': kept}
    weights = ROOT / 'fire-models' / 'fire_n.pt'
    general_weights = ROOT / 'general-models' / 'yolov8n.pt'
    report['weights'] = {
        'fire_n.pt': {'sha256': sha256(weights), 'source_commit': 'a44e1ecdb84b06bda145b1c3ad53854046fe14dd'},
        'yolov8n.pt': {'sha256': sha256(general_weights), 'source_commit': 'a44e1ecdb84b06bda145b1c3ad53854046fe14dd'},
    }
    fire_model = YOLO(str(weights))
    report['fire_classes'] = fire_model.names
    print(f'Fire classes: {fire_model.names}', flush=True)
    sample_paths = sorted((ROOT / 'demo-images').glob('*.jpeg'))
    assert sample_paths, 'Upstream demo images are missing'
    example_for_class = {}
    for sample in sample_paths:
        prediction = fire_model.predict(str(sample), conf=0.2, iou=0.5, device=devices[-1], verbose=False)[0]
        counts = dict(Counter(prediction.names[int(cls)] for cls in prediction.boxes.cls.tolist()))
        for name in counts:
            if name not in example_for_class:
                example_for_class[name] = sample
            if set(counts) == {name}:
                example_for_class[name] = sample
        report['inference'].append({'image': sample.relative_to(ROOT).as_posix(), 'device': devices[-1], 'counts': counts})
    assert {'smoke', 'fire'}.issubset(example_for_class), 'Demo inputs did not exercise both fire and smoke outputs'
    report['class_examples'] = {name: path.relative_to(ROOT).as_posix() for name, path in example_for_class.items()}
    for device in devices:
        for class_name in ['fire', 'smoke']:
            sample = example_for_class[class_name]
            prediction = fire_model.predict(str(sample), conf=0.2, iou=0.5, device=device, verbose=False)[0]
            assert prediction.orig_shape == cv2.imread(str(sample)).shape[:2]
            assert torch.isfinite(prediction.boxes.data).all().item()
            report['device_checks'].append({
                'device': device, 'image': sample.relative_to(ROOT).as_posix(),
                'counts': dict(Counter(prediction.names[int(cls)] for cls in prediction.boxes.cls.tolist())),
                'valid_coordinates': True,
            })
            prediction.save(str(OUTPUT / f'{class_name}_{device.replace(":", "_")}.jpg'))
    # Explicitly synthetic empty background: a negative-path check, not a dataset sample.
    blank = Image.fromarray(np.full((480, 640, 3), 128, dtype=np.uint8))
    prediction = fire_model.predict(blank, conf=0.2, device=devices[-1], verbose=False)[0]
    assert len(prediction.boxes) == 0, 'Synthetic empty background unexpectedly produced detections'
    prediction.save(str(OUTPUT / 'synthetic_empty_background.jpg'))
    report['synthetic_negative_check'] = {'kind': 'uniform_gray_image', 'counts': 0, 'included_in_accuracy_metrics': False}
    general_model = YOLO(str(general_weights))
    assert len(general_model.names) == 80
    general_model.predict(str(sample_paths[0]), device=devices[-1], verbose=False)
    report['general_model_class_count'] = len(general_model.names)
    for video in sorted((ROOT / 'demo-videos').glob('*')):
        cap = cv2.VideoCapture(str(video))
        try:
            ok, frame = cap.read()
            assert ok and frame is not None, f'Cannot decode {video.name}'
            report['video_checks'].append({
                'file': video.relative_to(ROOT).as_posix(), 'readable': True,
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            })
        finally:
            cap.release()
    entrypoint = next(ROOT.glob('1_*_Home.py'))
    app = AppTest.from_file(str(entrypoint), default_timeout=120).run()
    report['ui_checks'].append(check_app(app, 'Home: default nano model and built-in image', True))
    next(widget for widget in app.selectbox if widget.label == 'Inference Device').select('CPU').run()
    report['ui_checks'].append(check_app(app, 'Home: CPU image prediction', True))
    next(widget for widget in app.radio if widget.label == 'Select Model Type').set_value('General').run()
    report['ui_checks'].append(check_app(app, 'Home: general-model selector', True))
    next(widget for widget in app.radio if widget.label == 'Select Model Type').set_value('Fire Detection').run()
    next(widget for widget in app.radio if widget.label == 'Select image source:').set_value('Enter URL').run()
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(SimpleHTTPRequestHandler, directory=str(ROOT / 'demo-images')))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f'http://127.0.0.1:{server.server_port}/{sample_paths[0].name}'
        app.text_input[0].set_value(url).run()
        report['ui_checks'].append(check_app(app, 'Home: image URL served locally', True))
    finally:
        server.shutdown()
        server.server_close()
    next(widget for widget in app.radio if widget.label == 'Select image source:').set_value('Upload from Computer').run()
    report['ui_checks'].append(check_app(app, 'Home: upload selector with no file'))
    for page in sorted((ROOT / 'pages').glob('*.py')):
        page_app = AppTest.from_file(str(page), default_timeout=120).run()
        report['ui_checks'].append(check_app(page_app, f'Page: {page.name}'))
    report['status'] = 'passed'
    reports = ROOT / 'reports'
    reports.mkdir(exist_ok=True)
    destination = reports / 'reproduction.json'
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'cuda': report['cuda_available'], 'demo_images': len(sample_paths), 'ui_checks': len(report['ui_checks']), 'videos': len(report['video_checks']), 'report': str(destination)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
