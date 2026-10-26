"""Export locked A/B labels and separate-reference quality without arbitration."""
import argparse
import json
import sys
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from dataset_store import DatasetStore,file_sha
from quality import QualityStore
from quality_export import quality_archive

DEFAULT_TASK=ROOT/'data/fire_dataset/quality_assignments/dual_200_20261024/task_private.json'


def export_task(task_file,output_root,minimum_images=200,selection='both'):
    private=json.loads(Path(task_file).read_text(encoding='utf-8'))
    store=QualityStore(DatasetStore(private['dataset_root']));token=private['tokens']['admin']
    if store.access(token,True)['task_id']!=private['task_id']:raise ValueError('交接文件与任务不一致。')
    # All validation gates run before creating any delivery folder.
    content,summary=quality_archive(store,token,selection,minimum_images)
    label='双人独立标注提交' if selection=='both' else '独立标注'+selection
    target=Path(output_root)/(label+'_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    target.mkdir(parents=True,exist_ok=False)
    archive=target.with_suffix('.zip');archive.write_bytes(content)
    with ZipFile(BytesIO(content)) as package:package.extractall(target)
    return {'folder':str(target),'zip':str(archive),'sha256':file_sha(archive),
            'selection':selection,'images':summary['images'],'submissions':summary['independent_submissions']}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task-file',type=Path,default=DEFAULT_TASK)
    parser.add_argument('--output',type=Path,default=ROOT/'data/fire_dataset/quality_deliveries')
    parser.add_argument('--role',choices=['both','A','B'],default='both',help='both includes the quality report; A/B exports only that role')
    args=parser.parse_args()
    try:print(json.dumps(export_task(args.task_file,args.output,selection=args.role),ensure_ascii=True))
    except (ValueError,OSError,KeyError) as error:
        print(str(error),file=sys.stderr);raise SystemExit(2)
