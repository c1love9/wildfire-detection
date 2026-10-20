"""Pinned local SAM box prompting, applied only to the current source frame."""
import torch
import cv2
import json
import threading
import time
from pathlib import Path
import numpy as np
from PIL import Image
from dataset_store import file_sha
from inference_result import merge_masks

ROOT = Path(__file__).resolve().parent
MODEL_ID = 'facebook/sam-vit-base'
REVISION = '70c1a07f894ebb5b307fd9eaaee97b9dfc16068f'
WEIGHT_SHA = '892c410e496344e527255ccdcb2cb7244a609acb5389c7c4fdba1288f861c579'
DIRECTORY = ROOT / 'artifacts/models/sam-vit-base' / REVISION


def verify_model(directory=DIRECTORY):
    directory = Path(directory)
    receipt = json.loads((directory / 'model_receipt.json').read_text(encoding='utf-8'))
    if (receipt.get('model_id'), receipt.get('revision')) != (MODEL_ID, REVISION):
        raise ValueError('SAM来源或版本不匹配。')
    files = receipt.get('files', {})
    if set(files) != {'config.json', 'preprocessor_config.json', 'model.safetensors'} or files.get('model.safetensors') != WEIGHT_SHA:
        raise ValueError('SAM文件清单无效。')
    for name, sha in files.items():
        if file_sha(directory / name) != sha:
            raise OSError('SAM文件校验失败，请重新准备模型。')
    return receipt


class SamSegmenter:
    def __init__(self, device='cpu', directory=DIRECTORY):
        from transformers import SamModel, SamProcessor
        receipt = verify_model(directory)
        self.processor = SamProcessor.from_pretrained(str(directory), local_files_only=True)
        self.model = SamModel.from_pretrained(str(directory), local_files_only=True, use_safetensors=True).to(device).eval()
        self.device = device
        self._lock = threading.Lock()
        self.metadata = {'model_id': MODEL_ID, 'revision': REVISION, 'files': receipt['files'],
                         'device': device, 'fine_tuned': False, 'precision': 'FP32', 'prompt': 'detector bounding boxes'}
        # Prewarm before opening the camera; initialization does not count as observation.
        self.predict(Image.new('RGB', (64, 64)), [{'xyxy': [8, 8, 56, 56], 'class': 'fire'}])

    def predict(self, image, detections):
        if not detections:
            return []
        with self._lock, torch.inference_mode():
            inputs = self.processor(images=image, input_boxes=[[d['xyxy'] for d in detections]], return_tensors='pt').to(self.device)
            output = self.model(**inputs, multimask_output=False)
            masks = self.processor.image_processor.post_process_masks(output.pred_masks.cpu(), inputs['original_sizes'].cpu(),
                                                                      inputs['reshaped_input_sizes'].cpu())[0]
            return [mask.squeeze(0).numpy().astype(bool) for mask in masks]


def overlay_masks(rgb, masks):
    output = rgb.copy()
    for label, color in [('fire', (255, 60, 20)), ('smoke', (30, 150, 255))]:
        mask = masks[label]
        output[mask] = (output[mask] * .65 + np.array(color) * .35).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, color, 2)
    return output


class SegmentedDetector:
    def __init__(self, detector, segmenter):
        self.detector, self.segmenter = detector, segmenter
        self.device, self.model = detector.device, detector.model

    @property
    def metadata(self):
        return {**self.detector.metadata, 'segmentation': self.segmenter.metadata}

    def predict_frame(self, bgr, conf, iou):
        result = self.detector.predict_frame(bgr, conf, iou)
        result.model = self.metadata
        started = time.perf_counter()
        try:
            masks = self.segmenter.predict(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), result.detections)
            result.masks = merge_masks(masks, result.detections, (result.height, result.width))
            result.rgb = overlay_masks(result.rgb, result.masks)
            result.segmentation_status = 'ready' if result.detections else 'no_targets'
        except Exception as exc:
            result.masks = None
            result.segmentation_status, result.segmentation_error = 'error', str(exc)
            for detection in result.detections:
                detection.pop('area_ratio', None)
        result.timing_ms['segmentation'] = (time.perf_counter() - started) * 1000
        return result

    def predict(self, bgr, conf, iou):
        result = self.predict_frame(bgr, conf, iou)
        return result.rgb, result.detections, result.counts, sum(result.timing_ms.values())

    def predict_image(self, image, conf, iou):
        return self.predict(cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR), conf, iou)
