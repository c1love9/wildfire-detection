"""Resumable, audited public-source downloads for the minimum dataset collection."""
import argparse
import concurrent.futures
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data/public_sources'
SOURCES = [
    {'id': 'dfire', 'file': 'dfire_kaggle.zip', 'url': 'https://www.kaggle.com/api/v1/datasets/download/sayedgamal99/smoke-fire-detection-yolo',
     'page': 'https://github.com/gaia-solutions-on-demand/DFireDataset', 'mirror': 'https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo',
     'bytes': 3049605157, 'license': 'CC0 collection; third-party PublicDataset images retain original provider terms'},
    {'id': 'firesense', 'file': 'fire_videos.1406.zip', 'url': 'https://zenodo.org/records/836749/files/fire_videos.1406.zip?download=1',
     'page': 'https://zenodo.org/records/836749', 'bytes': 622676845, 'md5': '7cfe8286cc4b45193149186bf67ea00d', 'license': 'CC BY 4.0'},
    {'id': 'firesense', 'file': 'smoke_videos.1407.zip', 'url': 'https://zenodo.org/records/836749/files/smoke_videos.1407.zip?download=1',
     'page': 'https://zenodo.org/records/836749', 'bytes': 197589846, 'md5': '5ee6bc867c4a9332b4e75f308c26fc23', 'license': 'CC BY 4.0'},
    *[{'id': 'kmu', 'file': name, 'url': 'https://cvpr.kmu.ac.kr/Dataset/' + quote(name),
       'page': 'https://cvpr.kmu.ac.kr/Dataset/Dataset.htm', 'license': 'KMU CVPR Lab copyright; free download for academic research only'}
      for name in ['Indoor_outdoor(short_distance)flame.zip', 'Indoor_outdoor(short_distance)smoke.zip', 'Wildfire_smoke.zip']],
    {'id': 'flame2', 'file': 'flame2_pair6.zip',
     'url': 'https://huggingface.co/datasets/hanvithSai/gavin-flame2-raw-partial/resolve/main/' + quote('#6) Video Pair 6.zip'),
     'page': 'https://doi.org/10.21227/swyw-6j78', 'mirror': 'https://huggingface.co/datasets/hanvithSai/gavin-flame2-raw-partial',
     'bytes': 1054178380, 'license': 'CC BY 4.0; cite original FLAME 2 dataset and paper; third-party raw mirror'},
]


def digest(path, name='sha256'):
    value = hashlib.new(name)
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def download(source):
    directory = DEST / source['id']; directory.mkdir(parents=True, exist_ok=True)
    target = directory / source['file']; pending = target.with_suffix(target.suffix + '.part')
    receipt = target.with_suffix(target.suffix + '.receipt.json')
    if target.exists() and receipt.exists():
        previous = json.loads(receipt.read_text(encoding='utf-8'))
        if target.stat().st_size == previous['downloaded_bytes'] and digest(target) == previous['sha256']:
            print(source['file'] + ': already verified', flush=True); return previous
    error = None
    for attempt in range(4):
        try:
            offset = pending.stat().st_size if pending.exists() else 0
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            with requests.get(source['url'], stream=True, headers=headers, timeout=(30, 90)) as response:
                response.raise_for_status()
                append = bool(offset and response.status_code == 206)
                if append and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise OSError('Unexpected resume range')
                total = source.get('bytes')
                if total is None and response.headers.get('Content-Length'):
                    total = int(response.headers['Content-Length']) + (offset if append else 0)
                done = offset if append else 0
                if 'text/html' in response.headers.get('Content-Type', ''):
                    raise OSError('Download returned HTML rather than dataset')
                stamp = time.monotonic()
                with pending.open('ab' if append else 'wb') as output:
                    for chunk in response.iter_content(1024**2):
                        output.write(chunk); done += len(chunk)
                        if time.monotonic() - stamp > 20:
                            print(f"{source['file']}: {done / 1024**2:.1f} MiB / {total / 1024**2 if total else 0:.1f} MiB", flush=True)
                            stamp = time.monotonic()
                if total is not None and pending.stat().st_size != total:
                    raise OSError('Incomplete download')
            if source.get('md5') and digest(pending, 'md5') != source['md5']:
                raise OSError('Official MD5 mismatch')
            record = {**source, 'downloaded_bytes': pending.stat().st_size, 'sha256': digest(pending),
                      'retrieved_at': datetime.now(timezone.utc).isoformat()}
            pending.replace(target)
            receipt.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            print(f"{source['file']}: verified {record['downloaded_bytes']} bytes", flush=True)
            return record
        except (requests.RequestException, OSError) as exc:
            error = str(exc)
            print(f"{source['file']}: attempt {attempt + 1}: {error[:180]}", flush=True)
    return {**source, 'error': error, 'status': 'failed'}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--only', nargs='*'); args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    selected = [source for source in SOURCES if not args.only or source['id'] in args.only]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(download, selected))
    (DEST / 'download_manifest.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if any('error' in result for result in results):
        raise SystemExit('Some downloads failed; successful files and resumable parts retained.')


if __name__ == '__main__':
    main()
