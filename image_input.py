"""Decode still images in their displayed orientation before inference."""

from PIL import Image, ImageOps


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
