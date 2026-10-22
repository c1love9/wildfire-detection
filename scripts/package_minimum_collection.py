"""Package only final registered media and draft labels, with checked hashes.

Raw archives, previous revisions and unselected candidates never enter this ZIP.
The archive is a review package, not an approved training-data export.
"""
import csv
import hashlib
import io
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/minimum_collection'
TARGET = OUT / '最低数量数据集_700图_300视频_待复核.zip'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    registration = json.loads((OUT / 'registration.json').read_text(encoding='utf-8'))
    statistics = json.loads((OUT / 'dataset_statistics.json').read_text(encoding='utf-8'))
    reports = {}
    for name in ('minimum_collection_verification', 'minimum_collection_browser'):
        reports[name] = json.loads((ROOT / 'reports' / (name + '.json')).read_text(encoding='utf-8'))
        if reports[name]['status'] != 'passed':
            raise ValueError('Verification has not passed: ' + name)
    if reports['minimum_collection_verification']['unresolved_positive_images_not_negative_ground_truth'] != statistics['unresolved_positive_images']:
        raise ValueError('Verification refers to an earlier selection')
    test_log = (ROOT / 'artifacts/collection_regression_tests.txt').read_text(encoding='utf-8')
    if 'Ran 165 tests' not in test_log or not test_log.rstrip().endswith('OK'):
        raise ValueError('Expected successful regression-test record')
    files = {}

    def add(path, member):
        path = path.resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError('Unexpected package input: ' + str(path))
        if member in files or member.startswith('/') or '..' in Path(member).parts:
            raise ValueError('Unsafe or duplicate package member: ' + member)
        files[member] = path

    packaged = []
    for row in registration['records']:
        record = {**row, 'human_reviewed': False}
        if row['kind'] == 'image':
            member = f"yolo_draft/images/{row['split']}/{row['id']}.png"
            path = OUT / member
            label = f"yolo_draft/labels/{row['split']}/{row['id']}.txt"
            record['package_label_path'] = label
            add(OUT / label, label)
            if row['source'] != 'dfire':
                proposal = 'preannotations/' + row['id'] + '.json'
                candidate = json.loads((OUT / proposal).read_text(encoding='utf-8'))
                if candidate['input_sha256'] != row['sha256'] or candidate['human_reviewed']:
                    raise ValueError('Candidate mismatch: ' + row['id'])
                record['package_proposal_path'] = proposal
                add(OUT / proposal, proposal)
        elif row['kind'] == 'video':
            member = f"videos/{row['split']}/{row['id']}.mp4"
            path = ROOT / row['path']
        else:
            raise ValueError('Unknown media kind')
        if sha256(path) != row['sha256']:
            raise ValueError('Selected file changed: ' + row['id'])
        record['package_path'] = member
        add(path, member)
        packaged.append(record)
    if sum(r['kind'] == 'image' for r in packaged) != 700 or sum(r['kind'] == 'video' for r in packaged) != 300:
        raise ValueError('Wrong final collection count')
    for name in ('data.yaml', 'DO_NOT_TRAIN_BEFORE_REVIEW.txt', 'unresolved_positive.json'):
        add(OUT / 'yolo_draft' / name, 'yolo_draft/' + name)
    for name in ('registration.json', 'dataset_statistics.json', 'dataset_manifest.csv'):
        add(OUT / name, name)
    add(OUT / 'preannotations/grounding_dino_metadata.json', 'preannotations/grounding_dino_metadata.json')
    for name in ('download_manifest.json', 'archive_checks.json', 'source_integrity.json'):
        add(ROOT / 'data/public_sources' / name, 'sources/' + name)
    for name in ('dataset_collection', 'collection_sources'):
        add(ROOT / 'docs' / (name + '.md'), 'README.md' if name == 'dataset_collection' else 'collection_sources.md')
    for name in reports:
        add(ROOT / 'reports' / (name + '.json'), 'checks/' + name + '.json')
    for stem, total in (('images', 14), ('clips', 6)):
        for number in range(1, total + 1):
            name = f'{stem}_{number:02}.jpg'
            add(OUT / 'review/final' / name, 'review/' + name)
    for name in ('image_index.json', 'clip_index.json'):
        add(OUT / 'review/final' / name, 'review/' + name)
    for name in ('collection_firesense.json', 'collection_negative_exclusions.json'):
        add(ROOT / 'configs' / name, 'sources/' + name)
    fields = ('id', 'kind', 'source', 'split', 'package_path', 'package_label_path', 'source_video',
              'source_seconds', 'source_start_seconds', 'source_end_seconds', 'sha256', 'human_reviewed')
    csv_stream = io.StringIO(newline='')
    writer = csv.DictWriter(csv_stream, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(packaged)
    assets_csv = OUT / 'package_assets.csv'
    assets_csv.write_bytes(csv_stream.getvalue().encode('utf-8-sig'))
    add(assets_csv, 'package_assets.csv')
    manifest = {
        'status': 'quantity_collected_and_file_verified; annotation_and_strict_scene_acceptance_pending',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'source_base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'statistics': statistics,
        'pending': ['All labels require student human review', 'Empty proposals in positive images require boxes, not negative approval',
                    'Literal early sparks and combustion-stage coverage need confirmation', 'Formal training and accuracy evaluation have not run'],
        'layout': 'package_path is relative to this ZIP root; path/source_video retain original project provenance',
        'records': packaged,
        'files': {name: {'bytes': path.stat().st_size, 'sha256': sha256(path)} for name, path in files.items()},
    }
    manifest_path = OUT / 'package_manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    pending = TARGET.with_suffix('.zip.pending')
    with zipfile.ZipFile(pending, 'w', allowZip64=True) as archive:
        for name, path in files.items():
            codec = zipfile.ZIP_STORED if path.suffix.lower() in ('.png', '.jpg', '.mp4') else zipfile.ZIP_DEFLATED
            archive.write(path, name, compress_type=codec)
        archive.write(manifest_path, 'package_manifest.json', compress_type=zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(pending) as archive:
        if archive.testzip() is not None:
            raise ValueError('Package CRC failure')
        if len(archive.namelist()) != len(files) + 1:
            raise ValueError('Unexpected archive entries')
        for name, record in manifest['files'].items():
            digest = hashlib.sha256()
            with archive.open(name) as stream:
                for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
                    digest.update(chunk)
            if digest.hexdigest() != record['sha256']:
                raise ValueError('Package hash mismatch: ' + name)
    pending.replace(TARGET)
    report = {
        'status': 'review_package_ready; formal_annotation_and_scene_acceptance_pending',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'package': TARGET.relative_to(ROOT).as_posix(), 'bytes': TARGET.stat().st_size, 'sha256': sha256(TARGET),
        'files': len(files) + 1, 'images': 700, 'fire_scene_images': 500, 'normal_images': 200,
        'video_clips': 300, 'video_seconds': statistics['video_seconds'],
        'unresolved_positive_images': statistics['unresolved_positive_images'], 'human_approved_images': 0,
        'source_base_commit': manifest['source_base_commit'], 'regression_tests_passed': 165,
        'all_package_members_crc_and_sha256_verified': True, 'all_videos_fully_decoded': True,
        'browser_verified': True, 'pending': manifest['pending'],
        'raw_archives_and_media_excluded_from_git': True,
    }
    (ROOT / 'reports/minimum_collection_delivery.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
