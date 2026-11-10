"""Display names and local paths for curated, unannotated demonstration videos."""
import json
from pathlib import Path

VIDEO_SUFFIXES = {'.mp4', '.mov', '.avi', '.mkv'}


def video_choices(project_root):
    folder = Path(project_root).resolve()/'demo-videos'
    manifest = folder/'curated/manifest.json'
    if manifest.is_file():
        rows = []
        for sample in json.loads(manifest.read_text(encoding='utf-8'))['samples']:
            path = (manifest.parent/sample['file']).resolve()
            poster = (manifest.parent/sample['poster']).resolve()
            if (not path.is_relative_to(manifest.parent) or path.suffix.lower() not in VIDEO_SUFFIXES or
                    not poster.is_relative_to(manifest.parent)):
                raise ValueError('演示素材须位于精选视频目录内。')
            if not path.is_file() or not poster.is_file():
                raise ValueError('缺少精选视频或预览图：'+sample['file'])
            rows.append({**sample, 'path': path, 'poster_path': poster, 'curated': True})
        return rows
    return [{'file': path.name, 'label': path.name, 'path': path, 'curated': False}
            for path in sorted(folder.glob('*')) if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES]
