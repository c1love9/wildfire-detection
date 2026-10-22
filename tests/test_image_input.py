import unittest
from io import BytesIO
from unittest.mock import patch

from PIL import Image

from image_input import load_image


class ImageInputTests(unittest.TestCase):
    def test_phone_orientation_matches_dataset_and_exported_masks(self):
        raw = Image.new('RGB', (40, 20), 'red')
        exif = raw.getexif()
        exif[274] = 6  # Portrait phone photo stored as landscape pixels.
        source = BytesIO()
        raw.save(source, format='JPEG', exif=exif)
        source.seek(0)
        actual = load_image(source)
        self.assertEqual(actual.size, (20, 40))
        self.assertNotIn(274, actual.getexif())
        self.assertEqual(actual.mode, 'RGB')

    def test_decoded_image_survives_source_close_and_rgb_conversion(self):
        with BytesIO() as source:
            Image.new('L', (12, 8), 127).save(source, format='PNG')
            source.seek(0)
            actual = load_image(source)
        self.assertEqual(actual.getpixel((0, 0)), (127, 127, 127))

    def test_invalid_and_oversized_images_fail_before_inference(self):
        with self.assertRaises(ValueError):
            load_image(BytesIO(b'not an image'))
        source = BytesIO()
        Image.new('RGB', (20, 20)).save(source, format='PNG')
        source.seek(0)
        with self.assertRaisesRegex(ValueError, '2400'):
            load_image(source, max_pixels=100)
        with patch('image_input.Image.open', side_effect=Image.DecompressionBombError('too big')):
            with self.assertRaises(ValueError):
                load_image(source)
