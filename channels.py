"""Named local inputs and monitoring-point bindings, snapshotted per observation."""
import json
import os
from pathlib import Path
from monitoring import load_monitoring_points

ROOT=Path(__file__).resolve().parent


def config_path():
    return Path(os.environ.get('FIRE_CHANNEL_CONFIG', ROOT/'configs/channels.json'))


def load_channels(path=None):
    rows=json.loads(Path(path or config_path()).read_text(encoding='utf-8'))
    ids=set();points={p.id for p in load_monitoring_points(ROOT/'configs/monitoring_points.json')}
    for row in rows:
        if set(row)!={'id','name','kind','source','point_id'} or not isinstance(row['id'],str) or not row['id'].strip() or row['id'] in ids:
            raise ValueError('通道编号缺失、重复或配置字段无效。')
        ids.add(row['id'])
        if not isinstance(row['name'],str) or not row['name'].strip() or len(row['name'])>100 or row['point_id'] not in points:
            raise ValueError('通道名称或关联监控点无效。')
        if row['kind']=='video':
            path=(ROOT/row['source']).resolve()
            if not path.is_relative_to(ROOT/'demo-videos') or path.suffix.lower() not in {'.mp4','.mov','.mkv','.avi'}:
                raise ValueError('命名视频通道须绑定demo-videos中的视频。上传视频仍使用手动来源。')
        elif row['kind']=='camera':
            if type(row['source']) is not int or not 0<=row['source']<=9:raise ValueError('USB编号须为0～9。')
        else:raise ValueError('当前支持本地视频与USB通道。')
    if not rows:raise ValueError('通道列表不能为空。')
    return rows


def save_channels(rows,path=None):
    path=Path(path or config_path());path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix('.pending.json')
    pending.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    try:load_channels(pending);pending.replace(path)
    finally:pending.unlink(missing_ok=True)
