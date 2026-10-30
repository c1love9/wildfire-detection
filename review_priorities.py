"""Read a frozen local review queue without changing labels or task submissions."""

import json
from pathlib import Path
import re


QUEUE_PATH = Path('review_lists') / 'priority.json'


def load_review_queue(dataset_root):
    """Missing queue is normal; reject malformed queues instead of widening scope."""
    path = Path(dataset_root) / QUEUE_PATH
    try:
        document = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return None
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('优先复核清单无法读取，请检查本机清单文件。') from exc
    return validate_review_queue(document)


def validate_review_queue(document):
    """Validate before publishing a manifest as well as when reading one."""
    if not isinstance(document, dict) or document.get('schema_version') != 1:
        raise ValueError('优先复核清单版本不支持。')
    entries = document.get('entries')
    if not isinstance(entries, list) or not entries or document.get('count') != len(entries):
        raise ValueError('优先复核清单数量不一致或为空。')
    if not all(isinstance(document.get(key), str) and document[key].strip()
               for key in ('task_id', 'batch_id', 'source_report_sha256')):
        raise ValueError('优先复核清单缺少来源记录。')
    ids, numbers = set(), set()
    for row in entries:
        if not isinstance(row, dict):
            raise ValueError('优先复核清单条目无效。')
        for key in ('asset_id', 'input_sha256'):
            if not isinstance(row.get(key), str) or not re.fullmatch(r'[0-9a-f]{64}', row[key]):
                raise ValueError('优先复核清单图像标识或哈希无效。')
        if (not isinstance(row.get('number'), str) or not re.fullmatch(r'Q\d{3,}', row['number'])
                or row.get('class') not in ('fire', 'smoke')
                or not isinstance(row.get('original_model_had_class'), bool)
                or not isinstance(row.get('proposal_id'), str) or not row['proposal_id']):
            raise ValueError('优先复核清单编号、类别或候选版本无效。')
        if row['asset_id'] in ids or row['number'] in numbers:
            raise ValueError('优先复核清单包含重复图片或编号。')
        ids.add(row['asset_id'])
        numbers.add(row['number'])
    return document


def queue_in_catalog(document, catalog):
    """Retain frozen membership/order; never substitute other images or old bytes."""
    by_id = {row['id']: row for row in catalog}
    available, unavailable = [], []
    for row in document['entries']:
        asset = by_id.get(row['asset_id'])
        target = available if asset and asset['sha256'] == row['input_sha256'] else unavailable
        target.append(row)
    return available, unavailable
