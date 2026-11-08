"""Restore a verified delivery into a new project, refusing conflicting files."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def safe_path(root, relative):
    path = (root/relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError('Path outside selected directory')
    return path


def restore(package, project):
    manifest = json.loads((package/'manifest.json').read_text(encoding='utf-8'))
    mapping_name = manifest.get('restore_map', '08_验收与校验/restore_map.json')
    mapping_path = safe_path(package, mapping_name)
    if manifest.get('schema_version', 1) >= 2 and mapping_name not in manifest['files']:
        raise ValueError('Restore map must be included in the verified inventory')
    pending = []
    # Validate everything before copying; existing work is never overwritten.
    for relative, record in manifest['files'].items():
        source = safe_path(package, relative)
        if source.stat().st_size != record['bytes'] or sha256(source) != record['sha256']:
            raise OSError('Package verification failed: '+relative)
    mapping = json.loads(mapping_path.read_text(encoding='utf-8'))
    for row in mapping['files']:
        if row['source'] not in manifest['files']:
            raise ValueError('Restore map references a file outside the inventory')
        source, target = safe_path(package,row['source']), safe_path(project,row['target'])
        if target.exists():
            if not target.is_file() or sha256(target)!=manifest['files'][row['source']]['sha256']:
                raise FileExistsError('Choose a NEW project directory; conflicting file: '+str(target))
        else:
            pending.append((source,target))
    for source,target in pending:
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    return {'status':'passed','release_id':manifest['release_id'],'project':str(project),
            'restored_files':len(pending),'existing_identical_files':len(mapping['files'])-len(pending),
            'old_annotation_access_codes_included':False,
            'offline_models_included':manifest.get('offline_models_included',False),
            'note':('Pinned SAM/DINO snapshots restored and verified; Python dependencies, FFmpeg and drivers must be installed separately. '
                    if manifest.get('offline_models_included') else 'SAM/DINO weights require official pinned download. ')
                   +'Historical absolute paths remain provenance; restored inference uses relative state paths.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--project-root',type=Path,default=ROOT)
    args=parser.parse_args()
    print(json.dumps(restore(args.package.resolve(),args.project_root.resolve()),ensure_ascii=False,indent=2))
