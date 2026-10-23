"""Create an isolated synthetic teaching workspace; never approve formal assets."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from dataset_store import DatasetStore
from annotations import AnnotationStore


def main():
    directory = ROOT / 'artifacts/teammate_manual/lesson_dataset'
    dataset = DatasetStore(directory)
    existing = dataset.assets()
    if existing:
        asset_id = existing[0]['id']
    else:
        batch = dataset.create_batch('合成教学练习 非正式数据', 'synthetic_teammate_lesson', 'demo',
                                     '程序绘制的教学示意 不计课程采集数量', ['fire', 'smoke'])
        result = dataset.import_image(ROOT / 'artifacts/teammate_manual/lesson_scene.png', batch,
                                      source_name='synthetic_lesson_not_formal.png')
        asset_id = result['asset_id']

    class TeachingCandidates:
        def run_metadata(self, config):
            return {'method': 'manually constructed teaching candidates; NOT measured model inference', 'formal_data': False}

        def predict(self, image, config):
            factor = image.width / 1000
            return {'boxes': [{'class': 'fire', 'xyxy': [value * factor for value in [355, 315, 505, 520]], 'confidence': .8, 'phrase': 'teaching example'},
                              {'class': 'smoke', 'xyxy': [value * factor for value in [365, 25, 750, 355]], 'confidence': .8, 'phrase': 'teaching example'}],
                    'elapsed_seconds': 0.}

    AnnotationStore(dataset).prelabel([asset_id], TeachingCandidates(), None, '教学构造候选 非模型实测')
    print('Prepared isolated demo-only canvas; no human approval created')


if __name__ == '__main__':
    main()
