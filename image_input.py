"""Decode still images and load the selected Home demonstration gallery."""

import json
from pathlib import Path
from PIL import Image, ImageOps


def demo_image_choices(project_root):
    folder = Path(project_root).resolve()/'demo-images'
    manifest = folder/'curated/manifest.json'
    if manifest.is_file():
        samples = json.loads(manifest.read_text(encoding='utf-8'))['samples']
        choices = []
        for sample in samples:
            name, label = sample['file'], sample['label']
            if not isinstance(name, str) or not isinstance(label, str) or not label.strip():
                raise ValueError('演示图片清单中的名称无效。')
            path = (manifest.parent/name).resolve()
            if (Path(name).is_absolute() or not path.is_relative_to(manifest.parent) or
                    path.suffix.lower() not in {'.png', '.jpg', '.jpeg'}):
                raise ValueError('演示图片必须位于精选图片目录内。')
            if not path.is_file():
                raise ValueError('缺少精选演示图片：'+name)
            choices.append({'path': path, 'label': label, 'curated': True})
        return choices
    return [{'path': path, 'label': path.name, 'curated': False}
            for path in sorted(folder.glob('*'))
            if path.is_file() and path.suffix.lower() in {'.png', '.jpg', '.jpeg'}]


def load_image(source, max_pixels=24_000_000):
    """Return an owned RGB image; EXIF rotation also defines mask coordinates."""
    try:
        with Image.open(source) as raw:
            if raw.width * raw.height > max_pixels:
                raise ValueError('图片超过2400万像素，请先缩小。')
            image = ImageOps.exif_transpose(raw).convert('RGB')
            image.load()
            return image
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError('图片内容无法解码，请使用有效的PNG或JPEG图片。') from exc
