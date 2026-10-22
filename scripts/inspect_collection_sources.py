"""Safely unpack source videos and produce visual selection sheets."""
import json
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/public_sources'
OUT = ROOT / 'data/minimum_collection'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for source in ('kmu', 'firesense', 'flame2'):
        for archive in sorted((SOURCE / source).glob('*.zip')):
            target = SOURCE / source / 'videos' / archive.stem
            with zipfile.ZipFile(archive) as package:
                for info in package.infolist():
                    if Path(info.filename).suffix.lower() not in {'.mp4', '.avi', '.mov', '.mkv', '.mpg', '.mpeg'}:
                        continue
                    if info.file_size > 8 * 1024**3:
                        raise ValueError('Unexpected member size')
                    path = (target / info.filename).resolve()
                    if not path.is_relative_to(target.resolve()):
                        raise ValueError('Unsafe archive member')
                    if path.exists() and path.stat().st_size == info.file_size:
                        continue
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with package.open(info) as stream, path.open('wb') as output:
                        import shutil
                        shutil.copyfileobj(stream, output)
    existing = json.loads((OUT / 'source_inventory.json').read_text(encoding='utf-8')) if (OUT / 'source_inventory.json').exists() else []
    previous = {row['path']: row for row in existing}
    inventory = []
    for source in ('kmu', 'firesense', 'flame2'):
        paths = sorted((SOURCE / source / 'videos').rglob('*'))
        for path in paths:
            if path.suffix.lower() not in {'.mp4', '.avi', '.mov', '.mkv', '.mpg', '.mpeg'} or not path.is_file():
                continue
            relative = path.relative_to(ROOT).as_posix()
            if relative in previous:
                inventory.append(previous[relative]); continue
            metadata = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                'stream=width,height,avg_frame_rate,nb_frames,duration:format=duration', '-of', 'json', str(path)]))
            stream = metadata['streams'][0]; fps = stream['avg_frame_rate'].split('/')
            record = {'id': source + '_' + str(len([row for row in inventory if row['source'] == source]) + 1).zfill(3),
                      'source': source, 'path': relative, 'name': path.name, 'duration': float(stream.get('duration') or metadata['format']['duration']),
                      'width': stream['width'], 'height': stream['height'], 'fps': int(fps[0]) / int(fps[1])}
            inventory.append(record)
    for row in inventory:
        row['id'] = row['source'] + '_' + hashlib.sha256(row['path'].encode()).hexdigest()[:10]
    (OUT / 'source_inventory.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    sheets = OUT / 'review'; sheets.mkdir(exist_ok=True)
    for start in range(0, len(inventory), 12):
        rows = inventory[start:start + 12]
        sheet = Image.new('RGB', (1200, ((len(rows) + 1) // 2) * 205), '#f5f5f5')
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(rows):
            x, y = index % 2 * 600, index // 2 * 205
            draw.text((x + 5, y + 3), row['id'] + ' ' + row['name'][:55], fill='black')
            capture = cv2.VideoCapture(str(ROOT / row['path']))
            for column, fraction in enumerate((.1, .5, .9)):
                stamp = row['duration'] * fraction
                capture.set(cv2.CAP_PROP_POS_MSEC, stamp * 1000)
                ok, frame = capture.read()
                if ok:
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); image.thumbnail((194, 150))
                    sheet.paste(image, (x + column * 200 + 3, y + 28))
                draw.text((x + column * 200 + 3, y + 180), f'{stamp:.1f}s', fill='black')
            capture.release()
        sheet.save(sheets / f'videos_{start // 12 + 1:02}.jpg', quality=92)
    print(json.dumps({'videos': len(inventory), 'duration': sum(row['duration'] for row in inventory), 'by_source':
                     {source: len([row for row in inventory if row['source'] == source]) for source in ('kmu', 'firesense', 'flame2')}}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
