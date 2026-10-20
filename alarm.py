"""Temporal alarm rules. Times are source-media seconds, never UI wall time."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path


LEVEL_NAMES = {0: '正常', 1: '关注', 2: '预警', 3: '报警'}
CLASS_NAMES = {'fire': '火焰', 'smoke': '烟雾'}


@dataclass(frozen=True)
class AlarmConfig:
    confirm_frames: int = 3
    fire_confidence: float = 0.5
    smoke_confidence: float = 0.5
    attention_seconds: float = 0.5
    warning_seconds: float = 2.0
    alarm_seconds: float = 5.0
    recovery_seconds: float = 2.0
    match_iou: float = 0.1
    max_frame_gap_seconds: float = 1.0
    area_warning_ratio: float = 0.03
    area_alarm_ratio: float = 0.10
    area_warning_seconds: float = 1.0
    area_alarm_seconds: float = 2.0

    def __post_init__(self):
        if type(self.confirm_frames) is not int or self.confirm_frames < 2:
            raise ValueError('连续确认至少需要2帧。')
        values = asdict(self)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values.values()):
            raise ValueError('报警参数必须是有效数字。')
        if not all(0 < value <= 1 for value in (self.fire_confidence, self.smoke_confidence, self.match_iou)):
            raise ValueError('报警置信度和区域匹配IoU必须大于0且不超过1。')
        if not 0 <= self.attention_seconds < self.warning_seconds < self.alarm_seconds:
            raise ValueError('持续时间应满足：0 ≤ 关注 < 预警 < 报警。')
        if min(self.recovery_seconds, self.max_frame_gap_seconds) <= 0:
            raise ValueError('恢复等待和相邻帧最大间隔必须大于0。')
        if not 0 < self.area_warning_ratio < self.area_alarm_ratio <= 1 or min(self.area_warning_seconds, self.area_alarm_seconds) <= 0:
            raise ValueError('面积阈值须满足0 < 预警 < 报警 ≤ 100%，面积持续时间须大于0。')

    @classmethod
    def load(cls, path: str | Path) -> AlarmConfig:
        return cls(**json.loads(Path(path).read_text(encoding='utf-8')))


@dataclass(frozen=True)
class AlarmSnapshot:
    level: int = 0
    previous_level: int = 0
    classes: tuple[str, ...] = ()
    pending_frames: int = 0
    pending_seconds: float = 0.0
    max_confidence: float = 0.0


@dataclass
class _Track:
    label: str
    box: list[float]
    last_frame: int
    last_seen: float
    streak_start: float
    streak_frames: int = 1
    peak_confidence: float = 0.0
    supported_at: dict[int, float] = field(default_factory=dict)
    area_started: dict[int, float] = field(default_factory=dict)


def _iou(first, second):
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    union = (first[2] - first[0]) * (first[3] - first[1]) + (second[2] - second[0]) * (second[3] - second[1]) - intersection
    return intersection / union if union else 0.0


class AlarmEngine:
    """Associate boxes within each class; confirm uninterrupted evidence per region.

    Each level has a last-supported media time. A missing/lower detection cannot
    escalate; previous levels expire only after the recovery wait (hysteresis).
    """

    def __init__(self, config: AlarmConfig | None = None):
        self.config = config or AlarmConfig()
        self.snapshot = AlarmSnapshot()
        self._tracks: list[_Track] = []
        self._last_frame = 0
        self._last_time = -1.0

    def update(self, number: int, seconds: float, detections: list[dict]) -> AlarmSnapshot:
        if number != self._last_frame + 1 or not math.isfinite(seconds) or seconds < 0 or seconds <= self._last_time:
            raise ValueError('报警输入必须逐帧递增，且视频时间有效并递增。')
        cfg = self.config
        self._tracks = [track for track in self._tracks if seconds - track.last_seen < max(cfg.recovery_seconds, cfg.max_frame_gap_seconds)]
        boxes = []
        for detection in detections:
            label = detection['class']
            if label not in CLASS_NAMES:
                continue
            confidence = float(detection['confidence'])
            box = list(detection['xyxy'])
            threshold = cfg.fire_confidence if label == 'fire' else cfg.smoke_confidence
            if not math.isfinite(confidence) or not threshold <= confidence <= 1:
                continue
            if len(box) != 4 or not all(math.isfinite(value) for value in box) or box[2] <= box[0] or box[3] <= box[1]:
                continue
            area = detection.get('area_ratio')
            if not isinstance(area, (float, int)) or isinstance(area, bool) or not math.isfinite(area) or not 0 <= area <= 1:
                area = None
            boxes.append((label, confidence, box, area))

        # Greedy, one-to-one association prevents different regions or duplicate
        # boxes within a frame from accumulating one continuous streak.
        pairs = sorted(
            ((_iou(track.box, box), index, box_index)
             for index, track in enumerate(self._tracks)
             for box_index, (label, _, box, _) in enumerate(boxes) if label == track.label),
            reverse=True,
        )
        matched_tracks, assignments = set(), {}
        for overlap, index, box_index in pairs:
            if overlap >= cfg.match_iou and index not in matched_tracks and box_index not in assignments:
                matched_tracks.add(index)
                assignments[box_index] = index
        for index, track in enumerate(self._tracks):
            if index not in matched_tracks:
                track.streak_frames = 0

        current_tracks = []
        for box_index, (label, confidence, box, area) in enumerate(boxes):
            if box_index in assignments:
                track = self._tracks[assignments[box_index]]
                adjacent = track.last_frame == number - 1 and seconds - track.last_seen <= cfg.max_frame_gap_seconds
                if adjacent:
                    track.streak_frames += 1
                else:
                    track.streak_frames, track.streak_start = 1, seconds
                    track.area_started.clear()
                track.box, track.last_frame, track.last_seen = box, number, seconds
                track.peak_confidence = max(track.peak_confidence, confidence)
            else:
                track = _Track(label, box, number, seconds, seconds, peak_confidence=confidence)
                self._tracks.append(track)
            current_tracks.append(track)
            duration = seconds - track.streak_start
            if track.streak_frames >= cfg.confirm_frames:
                for level, required in ((1, cfg.attention_seconds), (2, cfg.warning_seconds), (3, cfg.alarm_seconds)):
                    if duration + 1e-9 >= required:
                        track.supported_at[level] = seconds
            for level, threshold, required in ((2, cfg.area_warning_ratio, cfg.area_warning_seconds),
                                                (3, cfg.area_alarm_ratio, cfg.area_alarm_seconds)):
                if area is None or area < threshold:
                    track.area_started.pop(level, None)
                    continue
                track.area_started.setdefault(level, seconds)
                if track.streak_frames >= cfg.confirm_frames and seconds - track.area_started[level] + 1e-9 >= required:
                    track.supported_at[level] = seconds

        held = [(track, max((level for level, supported in track.supported_at.items()
                             if seconds - supported + 1e-9 < cfg.recovery_seconds), default=0)) for track in self._tracks]
        level = max((value for _, value in held), default=0)
        self.snapshot = AlarmSnapshot(
            level=level, previous_level=self.snapshot.level,
            classes=tuple(sorted({track.label for track, value in held if value})),
            pending_frames=max((track.streak_frames for track in current_tracks), default=0),
            pending_seconds=max((seconds - track.streak_start for track in current_tracks), default=0.0),
            max_confidence=max((track.peak_confidence for track, value in held if value), default=0.0),
        )
        self._last_frame, self._last_time = number, seconds
        return self.snapshot
