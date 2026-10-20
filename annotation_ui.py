"""Browser canvas wrapper; original image coordinates stay independent of zoom."""

import base64
from io import BytesIO
from pathlib import Path

from PIL import Image
import streamlit.components.v1 as components


_editor = components.declare_component('box_editor', path=str(Path(__file__).parent / 'components' / 'box_editor'))


def box_editor(path, width, height, boxes, document_key):
    with Image.open(path) as original:
        preview = original.convert('RGB')
        preview.thumbnail((1280, 960))
        encoded = BytesIO()
        preview.save(encoded, format='JPEG', quality=90)
    initial = {'document_key': document_key, 'boxes': boxes}
    return _editor(image='data:image/jpeg;base64,' + base64.b64encode(encoded.getvalue()).decode('ascii'),
                   width=width, height=height, boxes=boxes, document_key=document_key,
                   default=initial, key='box_editor_' + document_key)
