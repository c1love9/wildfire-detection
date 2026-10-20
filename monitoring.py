"""Simulated monitoring-point metadata and geographic view geometry."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class MonitoringPoint:
    id: str
    name: str
    latitude: float
    longitude: float
    heading: float = 0.0
    pitch: float = -15.0
    horizontal_fov: float = 60.0
    simulated: bool = True
    coordinate_system: str = 'WGS84'

    def __post_init__(self):
        if not all(isinstance(value, str) and value.strip() and len(value) <= 100 for value in (self.id, self.name)):
            raise ValueError('监控点编号与名称不能为空，且不得超过100字。')
        for label, value, minimum, maximum in (
            ('纬度', self.latitude, -90, 90), ('经度', self.longitude, -180, 180),
            ('水平角', self.heading, 0, 360), ('俯仰角', self.pitch, -90, 90),
            ('水平视场角', self.horizontal_fov, 1, 179),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(f'{label}须在{minimum}～{maximum}之间。')
        # This adapter has no hardware GPS/pose provider. Never label it as one.
        if self.simulated is not True or self.coordinate_system != 'WGS84':
            raise ValueError('当前仅支持WGS84模拟位置与角度。')

    def metadata(self):
        return asdict(self)


def load_monitoring_points(path: str | Path):
    with Path(path).open(encoding='utf-8') as stream:
        points = [MonitoringPoint(**row) for row in json.load(stream)]
    if not points or len({point.id for point in points}) != len(points):
        raise ValueError('监控点列表不能为空，编号不得重复。')
    return points


def point_from_metadata(metadata):
    """Legacy/malformed logs have no position; never supply today's preset."""
    try:
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        return MonitoringPoint(**metadata['monitoring_point'])
    except (KeyError, TypeError, ValueError):
        return None


def destination(point: MonitoringPoint, bearing: float, distance=100.0):
    """Short spherical-geodesic offset, returned in map [longitude, latitude]."""
    lat, lon, azimuth = map(math.radians, (point.latitude, point.longitude, bearing))
    arc = distance / 6_371_008.8
    out_lat = math.asin(max(-1.0, min(1.0, math.sin(lat) * math.cos(arc) + math.cos(lat) * math.sin(arc) * math.cos(azimuth))))
    out_lon = lon + math.atan2(math.sin(azimuth) * math.sin(arc) * math.cos(lat), math.cos(arc) - math.sin(lat) * math.sin(out_lat))
    return [(math.degrees(out_lon) + 180) % 360 - 180, math.degrees(out_lat)]


def view_sector(point: MonitoringPoint):
    origin = [point.longitude, point.latitude]
    return [origin, *[destination(point, point.heading - point.horizontal_fov / 2 + point.horizontal_fov * index / 24)
                      for index in range(25)], origin]


def location_columns(metadata):
    point = point_from_metadata(metadata)
    return {
        '监控点': point.name if point else '未记录',
        '位置来源': '模拟（WGS84）' if point else '未记录',
        '纬度': point.latitude if point else None, '经度': point.longitude if point else None,
        '水平角（度）': point.heading if point else None, '俯仰角（度）': point.pitch if point else None,
        '水平视场角（度）': point.horizontal_fov if point else None,
    }
