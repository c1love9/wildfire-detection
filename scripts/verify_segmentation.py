"""Real SAM CPU/GPU inference and camera lifecycle; no formal accuracy claim."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import cv2
import json
from PIL import Image
from model_catalog import fire_models
from inference_ui import load_detector, with_segmentation
from segmentation import REVISION
from verify_camera import hardware_check


def main():
    torch.set_num_threads(8)
    output = ROOT / 'artifacts/completion_checks'
    output.mkdir(parents=True, exist_ok=True)
    image = next(p for p in (ROOT / 'demo-images').glob('*.jpeg') if p.stat().st_size > 1000)
    bgr = cv2.imread(str(image))
    choice = fire_models(ROOT, output / 'catalog')['fire_n']
    checks = []
    for device in ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else []):
        detector = with_segmentation(load_detector(choice, device), True)
        result = detector.predict_frame(bgr, .2, .5)
        assert result.masks is not None, result.segmentation_error
        assert result.masks['fire'].shape == bgr.shape[:2]
        Image.fromarray(result.rgb).save(output / ('segmentation_' + device.replace(':', '_') + '.png'))
        checks.append({'device': device, 'shape': list(bgr.shape), 'counts': result.counts,
                       'areas': result.areas(), 'timing_ms': result.timing_ms})
    hardware = hardware_check(detector)
    report = {'status': 'passed', 'sam_revision': REVISION, 'scope': 'Existing public demo; physical camera frames discarded; no formal accuracy claim',
              'checks': checks, 'physical_camera': hardware}
    (ROOT / 'reports/segmentation_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'passed', 'physical_camera': hardware}), flush=True)


if __name__ == '__main__': main()
