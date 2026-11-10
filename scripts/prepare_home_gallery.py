"""Copy six reviewed raw training images into the Home demonstration gallery."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
TRAINING_ID = 'train_984a486d3e6f459eb8eea567da186ef0'
SAMPLES = [
    ('01_building_fire_smoke.png', '01｜建筑火灾：火焰与烟雾',
     'ceae1dcdaabb791182bdc658db4a0bf12aa99914a9272d2c2b30987d336da30a'),
    ('02_indoor_fire.png', '02｜室内明火',
     'de7c707026a12adc94aba30280fa4109f2df16cc94715cf005ffe83fedd23596'),
    ('03_night_fire.png', '03｜夜间明火',
     '908eddbdbab10eac1e98a11de26260181cacf38db0d291ed589dadbba6ce8b00'),
    ('04_outdoor_smoke.png', '04｜室外烟雾',
     '3300997705f32ee6e214e7d114aa64c5f9696e3e1da1fb9e81d9675902b5b6fd'),
    ('05_forest_smoke.png', '05｜林地烟雾',
     '571fb54bc50c4f0582abb70937df472253b539543dd7a3818439383cf995641e'),
    ('06_sunset_normal.png', '06｜正常场景：夕阳',
     'ab29b328755473a7aec2fab748f72ae9feb7d09db5291e9e550bb61ad817dd10'),
]


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(project_root):
    root = Path(project_root).resolve()
    training = root/'data/fire_dataset/training'/TRAINING_ID
    receipt = json.loads((training/'dataset_receipt.json').read_text(encoding='utf-8'))
    result = json.loads((training/'result.json').read_text(encoding='utf-8'))
    manifest = {'schema_version': 1, 'training_id': TRAINING_ID,
                'model_sha256_used_for_selection': result['weights']['best']['sha256'],
                'purpose': 'Curated function demonstration, not independent performance evaluation.',
                'source_split': 'train', 'input_images_have_detection_overlays': False,
                'selection_inference': {'imgsz': 640, 'confidence': .2, 'nms_iou': .5}, 'samples': []}
    # Validate all source files before creating or replacing any gallery file.
    for filename, label, asset_id in SAMPLES:
        relative = f'images/train/{asset_id}.png'
        source = training/'input'/relative
        expected = receipt['inventory'][relative]
        if not source.is_file() or sha256(source) != expected:
            raise OSError('原始演示候选图缺失或与冻结数据不一致：'+asset_id)
        manifest['samples'].append({'file': filename, 'label': label, 'asset_id': asset_id,
                                    'source': source.relative_to(root).as_posix(),
                                    'source_split': 'train', 'sha256': expected})
    destination = root/'demo-images/curated'
    destination.mkdir(parents=True, exist_ok=True)
    for sample in manifest['samples']:
        target = destination/sample['file']
        shutil.copy2(root/sample['source'], target)
        if sha256(target) != sample['sha256']:
            raise OSError('演示图片复制校验失败：'+sample['file'])
    temporary = destination/'manifest.json.pending'
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(destination/'manifest.json')
    print(json.dumps({'gallery': str(destination), 'images': len(manifest['samples']),
                      'raw_bytes_preserved': True, 'dataset_modified': False}, ensure_ascii=True))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    prepare(parser.parse_args().root)
