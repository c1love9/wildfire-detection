"""Download only the pinned SAM snapshot, verify official blobs before publication."""
import sys
from pathlib import Path
import hashlib
import json
import requests
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from segmentation import DIRECTORY, MODEL_ID, REVISION, WEIGHT_SHA, verify_model
from dataset_store import file_sha


def main():
    response = requests.get(f'https://huggingface.co/api/models/{MODEL_ID}/revision/{REVISION}?blobs=true', timeout=30)
    response.raise_for_status()
    metadata = response.json()
    if metadata['sha'] != REVISION:
        raise ValueError('Official revision mismatch')
    files = {row['rfilename']: row for row in metadata['siblings']}
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    receipt = {'model_id': MODEL_ID, 'revision': REVISION, 'files': {}}
    for name in ['config.json', 'preprocessor_config.json', 'model.safetensors']:
        info, target = files[name], DIRECTORY / name
        def valid(path):
            if not path.is_file() or path.stat().st_size != info['size']: return False
            if info.get('lfs'): return file_sha(path) == info['lfs']['sha256'] == WEIGHT_SHA
            blob = path.read_bytes()
            return hashlib.sha1(f'blob {len(blob)}\0'.encode() + blob).hexdigest() == info['blobId']
        if not valid(target):
            pending = target.with_suffix(target.suffix + '.pending')
            with requests.get(f'https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}', stream=True, timeout=(30, 90)) as stream:
                stream.raise_for_status()
                with pending.open('wb') as output:
                    for chunk in stream.iter_content(1024 * 1024): output.write(chunk)
            if not valid(pending): raise OSError('Official file verification failed: ' + name)
            pending.replace(target)
        receipt['files'][name] = file_sha(target)
        print(name + ': verified', flush=True)
    (DIRECTORY / 'model_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    verify_model()


if __name__ == '__main__': main()
