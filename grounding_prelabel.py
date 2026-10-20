"""Pinned local GroundingDINO inference; proposals never imply human approval."""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import Image

from dataset_store import file_sha


ROOT = Path(__file__).resolve().parent
MODEL_ID = 'IDEA-Research/grounding-dino-tiny'
REVISION = 'a2bb814dd30d776dcf7e30523b00659f4f141c71'
WEIGHTS_SHA = '1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3'


@dataclass(frozen=True)
class PrelabelConfig:
    model_id: str = MODEL_ID
    revision: str = REVISION
    weights_sha256: str = WEIGHTS_SHA
    fire_prompt: str = 'fire. flame. burning. blaze.'
    smoke_prompt: str = 'smoke. smoke plume. heavy smoke.'
    box_threshold: float = 0.3
    text_threshold: float = 0.25
    nms_iou: float = 0.5
    max_boxes_per_class: int = 50
    shortest_edge: int = 640
    longest_edge: int = 1066

    def __post_init__(self):
        if (self.model_id, self.revision, self.weights_sha256) != (MODEL_ID, REVISION, WEIGHTS_SHA):
            raise ValueError('当前使用已固定并核验的GroundingDINO Tiny模型版本。')
        for name in ('fire_prompt', 'smoke_prompt'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip(' .,;\n') or len(value) > 200:
                raise ValueError('每类提示词须填写，且不超过200个字符。')
        for name in ('box_threshold', 'text_threshold', 'nms_iou'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
                raise ValueError('置信度、文本阈值与NMS IoU须在0和1之间。')
        for name, low, high in (('max_boxes_per_class', 1, 100), ('shortest_edge', 320, 800), ('longest_edge', 320, 1333)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'{name}须为{low}～{high}的整数。')
        if self.longest_edge < self.shortest_edge:
            raise ValueError('长边限制不能小于短边尺寸。')

    def prompts(self):
        return {label: '. '.join(part.strip().lower() for part in re.split(r'[.,;\n]+', getattr(self, label + '_prompt')) if part.strip()) + '.'
                for label in ('fire', 'smoke')}

    @classmethod
    def load(cls, path=ROOT / 'configs' / 'preannotation.json'):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8')))


def model_directory():
    return Path(os.environ.get('GROUNDING_DINO_MODEL_DIR', ROOT / 'artifacts' / 'models' / 'grounding-dino-tiny' / REVISION)).resolve()


class GroundingPrelabeler:
    def __init__(self, device='cpu', directory=None):
        import transformers
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        directory = Path(directory or model_directory()).resolve()
        if not (directory / 'model_receipt.json').is_file():
            raise OSError('尚未准备GroundingDINO模型，请先运行 scripts/prepare_grounding_dino.py。')
        receipt = json.loads((directory / 'model_receipt.json').read_text(encoding='utf-8'))
        if (receipt.get('model_id'), receipt.get('revision')) != (MODEL_ID, REVISION):
            raise ValueError('模型来源或版本记录不匹配。')
        for name, sha in receipt['files'].items():
            path = (directory / name).resolve()
            if not path.is_relative_to(directory) or file_sha(path) != sha:
                raise OSError('模型文件完整性校验失败，请重新准备模型。')
        if receipt['files'].get('model.safetensors') != WEIGHTS_SHA:
            raise ValueError('模型权重不是已固定的官方版本。')
        self.device = device
        self.processor = AutoProcessor.from_pretrained(str(directory), local_files_only=True, use_fast=False)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            str(directory), local_files_only=True, use_safetensors=True, disable_custom_kernels=True).to(device).eval()
        self._lock = threading.Lock()
        self.metadata = {'model_id': MODEL_ID, 'revision': REVISION, 'files_sha256': receipt['files'],
                         'transformers': transformers.__version__, 'torch': torch.__version__, 'device': device,
                         'hardware': torch.cuda.get_device_name(0) if device.startswith('cuda') else 'CPU',
                         'dtype': 'float32', 'disable_custom_kernels': True, 'fine_tuned': False}

    def predict(self, image: Image.Image, config: PrelabelConfig):
        from torchvision.ops import nms
        started = time.perf_counter()
        boxes = []
        with self._lock, torch.inference_mode():
            for label, prompt in config.prompts().items():
                inputs = self.processor(images=image.convert('RGB'), text=prompt, return_tensors='pt',
                                        size={'shortest_edge': config.shortest_edge, 'longest_edge': config.longest_edge}).to(self.device)
                outputs = self.model(**inputs)
                result = self.processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, threshold=config.box_threshold, text_threshold=config.text_threshold,
                    target_sizes=[(image.height, image.width)])[0]
                raw_boxes, scores = result['boxes'].cpu(), result['scores'].cpu()
                phrases = result['text_labels'] if 'text_labels' in result else result['labels']
                if not len(scores):
                    continue
                raw_boxes[:, (0, 2)] = raw_boxes[:, (0, 2)].clamp(0, image.width)
                raw_boxes[:, (1, 3)] = raw_boxes[:, (1, 3)].clamp(0, image.height)
                keep = nms(raw_boxes, scores, config.nms_iou)[:config.max_boxes_per_class]
                for index in keep.tolist():
                    xyxy = raw_boxes[index].tolist()
                    if xyxy[2] - xyxy[0] < 1 or xyxy[3] - xyxy[1] < 1:
                        continue
                    boxes.append({'id': uuid.uuid4().hex, 'class': label, 'xyxy': xyxy,
                                  'confidence': float(scores[index]), 'phrase': str(phrases[index])})
        return {'boxes': boxes, 'elapsed_seconds': time.perf_counter() - started}

    def run_metadata(self, config):
        return {'parameters': asdict(config), 'effective_prompts': config.prompts(), 'runtime': self.metadata,
                'class_strategy': 'independent class-conditioned prompts; within-class NMS; overlapping fire/smoke retained'}
