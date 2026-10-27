"""Fetch a pinned official model snapshot; verify every downloaded file."""

import hashlib
import argparse
import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset_store import file_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', action='store_true', help='Prepare the pinned enhanced Base snapshot instead of the original Tiny.')
    args = parser.parse_args()
    config = json.loads((ROOT / 'configs' / ('preannotation_optimized.json' if args.base else 'preannotation.json')).read_text(encoding='utf-8'))
    model, revision = config['model_id'], config['revision']
    destination = ROOT / 'artifacts' / 'models' / ('grounding-dino-base' if args.base else 'grounding-dino-tiny') / revision
    destination.mkdir(parents=True, exist_ok=True)
    response = requests.get(f'https://huggingface.co/api/models/{model}/revision/{revision}?blobs=true', timeout=30)
    response.raise_for_status()
    metadata = response.json()
    assert metadata['sha'] == revision
    files = {row['rfilename']: row for row in metadata['siblings']}
    required = ['config.json', 'preprocessor_config.json', 'tokenizer_config.json', 'tokenizer.json',
                'special_tokens_map.json', 'vocab.txt', 'model.safetensors']
    receipt = {'model_id': model, 'revision': revision, 'files': {}}
    for name in required:
        info = files[name]
        target = destination / name
        def valid(path):
            if not path.is_file() or path.stat().st_size != info['size']:
                return False
            if info.get('lfs'):
                return file_sha(path) == info['lfs']['sha256'] == config['weights_sha256']
            content = path.read_bytes()
            return hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest() == info['blobId']
        if not valid(target):
            pending = target.with_suffix(target.suffix + '.pending')
            try:
                with requests.get(f'https://huggingface.co/{model}/resolve/{revision}/{name}', stream=True, timeout=(30, 90)) as stream:
                    stream.raise_for_status()
                    with pending.open('wb') as output:
                        size = 0
                        next_notice = 100 * 1024**2
                        for block in stream.iter_content(1024 * 1024):
                            output.write(block)
                            size += len(block)
                            if size >= next_notice:
                                print(f'{name}: {size // 1024**2} MiB', flush=True)
                                next_notice += 100 * 1024**2
                if not valid(pending):
                    raise OSError(f'{name}: official size/hash verification failed')
                pending.replace(target)
            finally:
                pending.unlink(missing_ok=True)
        receipt['files'][name] = file_sha(target)
        print(f'{name}: verified', flush=True)
    (destination / 'model_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'ready', 'path': str(destination)}), flush=True)


if __name__ == '__main__':
    main()
