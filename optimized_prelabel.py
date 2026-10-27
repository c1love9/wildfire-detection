"""Versioned Base + specialist proposals; never approve human labels."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import threading
import time
import uuid

import torch
from torchvision.ops import nms

from comparison import iou
from dataset_store import file_sha

ROOT = Path(__file__).resolve().parent
BASE_ID = 'IDEA-Research/grounding-dino-base'
BASE_REVISION = '12bdfa3120f3e7ec7b434d90674b3396eccf88eb'
BASE_WEIGHTS_SHA = '5548f844c928c4b6f411fa8cbcc2bfa8dbbba437cb1d513975519f93c2a9ed21'
AUXILIARY_SHA = '22364797523398115f67f43ff379c3d80c00dde9b4bc2e92e844bb357ca777e7'


def phrases(value):
    if not isinstance(value, str) or len(value) > 200:
        raise ValueError('每类提示词须为不超过200字符的文本。')
    result = list(dict.fromkeys(p.strip().lower() for p in re.split(r'[.,;\n]+', value) if p.strip()))
    if not result:
        raise ValueError('两类提示词均须填写。')
    return result


@dataclass(frozen=True)
class OptimizedPrelabelConfig:
    model_id: str = BASE_ID
    revision: str = BASE_REVISION
    weights_sha256: str = BASE_WEIGHTS_SHA
    fire_prompt: str = 'fire.'
    smoke_prompt: str = 'smoke.'
    fire_threshold: float = .30
    smoke_threshold: float = .35
    nms_iou: float = .50
    max_boxes_per_class: int = 50
    shortest_edge: int = 640
    longest_edge: int = 1066
    auxiliary_weights: str = 'fire-models/fire_n.pt'
    auxiliary_sha256: str = AUXILIARY_SHA
    auxiliary_fire_threshold: float = .15
    auxiliary_smoke_threshold: float = .50
    auxiliary_iou: float = .70
    fusion_iou: float = .50

    def __post_init__(self):
        if (self.model_id, self.revision, self.weights_sha256) != (BASE_ID, BASE_REVISION, BASE_WEIGHTS_SHA):
            raise ValueError('增强方案须使用已固定并校验的官方GroundingDINO Base。')
        if (self.auxiliary_weights, self.auxiliary_sha256) != ('fire-models/fire_n.pt', AUXILIARY_SHA):
            raise ValueError('辅助模型须为本次验证的上游火烟权重。')
        fire, smoke = phrases(self.fire_prompt), phrases(self.smoke_prompt)
        if set(fire) & set(smoke):
            raise ValueError('火焰和烟雾提示词不能含相同短语。')
        for name in ('fire_threshold', 'smoke_threshold', 'nms_iou', 'auxiliary_fire_threshold',
                     'auxiliary_smoke_threshold', 'auxiliary_iou', 'fusion_iou'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
                raise ValueError('阈值须为0到1之间的有限数。')
        for name, lo, hi in [('max_boxes_per_class', 1, 100), ('shortest_edge', 320, 800), ('longest_edge', 320, 1333)]:
            if type(getattr(self, name)) is not int or not lo <= getattr(self, name) <= hi:
                raise ValueError('候选框数或图像尺寸超出允许范围。')
        if self.longest_edge < self.shortest_edge:
            raise ValueError('长边上限不能小于短边。')

    @classmethod
    def load(cls, path=ROOT / 'configs/preannotation_optimized.json'):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8')))

    def prompts(self):
        return {label: '. '.join(phrases(value)) + '.' for label, value in
                [('fire', self.fire_prompt), ('smoke', self.smoke_prompt)]}


def model_directory():
    return ROOT / 'artifacts/models/grounding-dino-base' / BASE_REVISION


def query_tokens(tokenizer, config):
    groups = [phrases(config.fire_prompt), phrases(config.smoke_prompt)]
    text = ''.join(phrase + '. ' for group in groups for phrase in group)
    ids, spans = [tokenizer.cls_token_id], []
    for group in groups:
        group_spans = []
        for phrase in group:
            tokens = tokenizer.encode(phrase, add_special_tokens=False)
            if not tokens:
                raise ValueError('提示词没有有效文本token。')
            start = len(ids)
            group_spans.append(list(range(start, start + len(tokens))))
            ids.extend(tokens)
            ids.extend(tokenizer.encode('.', add_special_tokens=False))
        spans.append(group_spans)
    ids.append(tokenizer.sep_token_id)
    if len(ids) > 256 or ids != tokenizer.encode(text, add_special_tokens=True):
        raise ValueError('提示词过长或分词位置不一致，请缩短提示词。')
    return text, spans, groups


def joint_boxes(logits, normalized_boxes, spans, groups, width, height, config):
    """Per-phrase mean token score; aliases max; classes are not exclusive."""
    relevant = sorted({i for group in spans for phrase in group for i in phrase})
    if (logits.ndim != 2 or normalized_boxes.ndim != 2 or normalized_boxes.shape != (len(logits), 4)
            or not relevant or min(relevant) < 0 or max(relevant) >= logits.shape[1]
            or not torch.isfinite(logits[:, relevant]).all() or not torch.isfinite(normalized_boxes).all()):
        raise ValueError('模型返回了非有限数或无效形状，不能当作空目标。')
    probs = logits.sigmoid()
    scores = [torch.stack([probs[:, indices].mean(-1) for indices in group], 1) for group in spans]
    cx, cy, w, h = normalized_boxes.unbind(-1)
    bounds = torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], 1)
    bounds = bounds * torch.tensor([width, height, width, height], device=bounds.device)
    bounds[:, [0, 2]] = bounds[:, [0, 2]].clamp(0, width)
    bounds[:, [1, 3]] = bounds[:, [1, 3]].clamp(0, height)
    result = []
    for index, label in enumerate(('fire', 'smoke')):
        confidence, alias = scores[index].max(1)
        threshold = config.fire_threshold if label == 'fire' else config.smoke_threshold
        candidates = []
        for query in torch.where(confidence >= threshold)[0].tolist():
            xyxy = bounds[query].tolist()
            if xyxy[2]-xyxy[0] < 1 or xyxy[3]-xyxy[1] < 1:
                continue
            candidates.append({'id': uuid.uuid4().hex, 'class': label, 'xyxy': xyxy,
                'confidence': float(confidence[query]), 'phrase': groups[index][int(alias[query])],
                'prelabel_source': 'grounding_dino_base_joint', 'query_index': query})
        if candidates:
            keep = nms(torch.tensor([b['xyxy'] for b in candidates]),
                       torch.tensor([b['confidence'] for b in candidates]), config.nms_iou)[:config.max_boxes_per_class]
            result.extend(candidates[i] for i in keep.tolist())
    return result


def merge_candidates(dino, auxiliary, config):
    """Prefer DINO geometry; only add same-class spatially distinct YOLO boxes."""
    result = []
    for batch in (dino, auxiliary):
        for box in sorted(batch, key=lambda b: -b['confidence']):
            if not any(old['class'] == box['class'] and iou(old['xyxy'], box['xyxy']) >= config.fusion_iou for old in result):
                result.append(box)
    if len(result) > 300:
        raise ValueError('单图候选超过300个，请提高阈值后重试，不自动截断。')
    return result


class OptimizedPrelabeler:
    def __init__(self, device='cpu', directory=None):
        import transformers
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from ultralytics import YOLO
        folder = Path(directory or model_directory()).resolve()
        receipt_path = folder / 'model_receipt.json'
        if not receipt_path.is_file():
            raise OSError('尚未准备增强方案模型，请运行 scripts/prepare_grounding_dino.py --base。')
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if (receipt.get('model_id'), receipt.get('revision'), receipt.get('files', {}).get('model.safetensors')) != (BASE_ID, BASE_REVISION, BASE_WEIGHTS_SHA):
            raise ValueError('Base模型来源、版本或权重记录不匹配。')
        for name, digest in receipt['files'].items():
            target = (folder / name).resolve()
            if not target.is_relative_to(folder) or file_sha(target) != digest:
                raise OSError('Base模型完整性校验失败。')
        auxiliary = ROOT / 'fire-models/fire_n.pt'
        if file_sha(auxiliary) != AUXILIARY_SHA:
            raise OSError('辅助火烟权重与已验证版本不一致。')
        self.processor = AutoProcessor.from_pretrained(str(folder), local_files_only=True, use_fast=False)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(str(folder), local_files_only=True,
            use_safetensors=True, disable_custom_kernels=True).to(device).eval()
        self.auxiliary = YOLO(str(auxiliary))
        self.names = {int(k): str(v).strip().lower() for k, v in self.auxiliary.names.items()}
        if set(self.names.values()) != {'fire', 'smoke'} or len(self.names) != 2:
            raise ValueError('辅助模型类别须明确为fire/smoke，不能按编号猜测。')
        self.device, self._lock = device, threading.Lock()
        self.metadata = {'model_id': BASE_ID, 'revision': BASE_REVISION, 'files_sha256': receipt['files'],
            'transformers': transformers.__version__, 'torch': torch.__version__, 'device': device, 'dtype': 'float32',
            'hardware': torch.cuda.get_device_name(0) if device.startswith('cuda') else 'CPU',
            'disable_custom_kernels': True, 'fine_tuned': False}

    def run_metadata(self, config):
        return {'parameters': asdict(config), 'effective_prompts': config.prompts(), 'runtime': self.metadata,
            'method': 'grounding_dino_base_joint_plus_upstream_yolo',
            'auxiliary_model': {'path': config.auxiliary_weights, 'sha256': config.auxiliary_sha256,
                'origin': 'upstream supervised specialist; not self-trained or zero-shot', 'classes': self.names,
                'imgsz': 640, 'inference_conf': .05, 'iou': config.auxiliary_iou, 'max_det': 300},
            'class_strategy': 'Per-phrase mean token score, aliases max, per-class thresholds; overlapping fire/smoke allowed; per-class NMS.',
            'fusion': 'DINO first; add YOLO boxes only if same-class IoU with kept boxes is below fusion_iou. Scores retain source meanings and are not calibrated together.',
            'quality_scope': 'Overall training-sample P/R improved; some thin-smoke subgroup regressions remain. Candidates require human review; no labels are automatically approved.'}

    def predict(self, image, config):
        started = time.perf_counter()
        image = image.convert('RGB')
        text, spans, groups = query_tokens(self.processor.tokenizer, config)
        with self._lock, torch.inference_mode():
            inputs = self.processor(images=image, text=text, return_tensors='pt',
                size={'shortest_edge': config.shortest_edge, 'longest_edge': config.longest_edge}).to(self.device)
            outputs = self.model(**inputs)
            dino = joint_boxes(outputs.logits[0], outputs.pred_boxes[0], spans, groups, image.width, image.height, config)
            prediction = self.auxiliary.predict(image, imgsz=640, conf=.05, iou=config.auxiliary_iou, max_det=300,
                device=self.device, verbose=False, save=False)[0]
            extra = []
            for box in prediction.boxes:
                label = self.names[int(box.cls.item())]
                score = float(box.conf.item())
                if not math.isfinite(score):
                    raise ValueError('辅助模型返回非有限置信度。')
                if score < (config.auxiliary_fire_threshold if label == 'fire' else config.auxiliary_smoke_threshold):
                    continue
                xyxy = box.xyxy[0].cpu().tolist()
                if not all(math.isfinite(value) for value in xyxy):
                    raise ValueError('辅助模型返回非有限坐标。')
                if xyxy[2]-xyxy[0] < 1 or xyxy[3]-xyxy[1] < 1:
                    continue
                extra.append({'id': uuid.uuid4().hex, 'class': label, 'xyxy': xyxy, 'confidence': score,
                              'phrase': label, 'prelabel_source': 'upstream_yolo_fire_n'})
        return {'boxes': merge_candidates(dino, extra, config), 'elapsed_seconds': time.perf_counter()-started}
