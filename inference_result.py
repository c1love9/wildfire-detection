"""Original-image inference results; absent segmentation is never a zero mask."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class FrameResult:
    rgb: np.ndarray
    detections: list
    counts: dict
    width: int
    height: int
    timing_ms: dict = field(default_factory=dict)
    model: dict = field(default_factory=dict)
    masks: dict | None = None
    segmentation_status: str = 'disabled'
    segmentation_error: str | None = None

    def areas(self):
        if self.masks is None:
            return {'fire': None, 'smoke': None, 'combined': None}
        fire, smoke = self.masks['fire'], self.masks['smoke']
        return {'fire': float(fire.mean()), 'smoke': float(smoke.mean()),
                'combined': float(np.logical_or(fire, smoke).mean())}


def merge_masks(masks, detections, shape):
    if len(masks) != len(detections):
        raise ValueError('分割数量与检测框数量不一致。')
    merged = {label: np.zeros(shape, dtype=bool) for label in ('fire', 'smoke')}
    for mask, detection in zip(masks, detections):
        mask = np.asarray(mask)
        if mask.shape != shape or not np.isfinite(mask).all():
            raise ValueError('分割掩码尺寸或像素无效。')
        mask = mask.astype(bool)
        detection['area_ratio'] = float(mask.mean())
        merged[detection['class']] |= mask
    return merged
