"""Built-in and completed local-training models, with verified provenance."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from dataset_store import DatasetStore, file_sha
from training import TrainingStore, bounded


@dataclass(frozen=True)
class ModelChoice:
    key: str
    label: str
    origin: str
    path: str = ''
    sha256: str = ''
    imgsz: int = 640
    training_id: str | None = None
    dataset_fingerprint: str | None = None
    actor: str | None = None
    note: str | None = None
    completed_epochs: int | None = None
    error: str | None = None

    def metadata(self):
        return asdict(self)

    def verify(self):
        if self.error:
            raise ValueError(self.error)
        if not self.path or not self.sha256 or file_sha(Path(self.path)) != self.sha256:
            raise OSError('模型权重缺失或校验失败，请恢复原权重或选择其他模型。')


def fire_models(project_root, data_root):
    """Unfinished/cancelled jobs never become detection candidates."""
    root = Path(project_root)
    choices = {}
    for path in sorted((root / 'fire-models').glob('*.pt')):
        choices[path.stem] = ModelChoice(path.stem, path.stem, 'upstream', str(path.resolve()), file_sha(path))
    store = TrainingStore(DatasetStore(data_root))
    for job in store.jobs(refresh=False):
        if job['kind'] != 'train' or job['status'] != 'completed':
            continue
        key = 'trained:' + job['id'] + ':best'
        label = f"自训 · {job['note'][:40]} · {job['id'][-8:]} · best"
        fields = dict(key=key, label=label, origin='trained', training_id=job['id'], actor=job['actor'], note=job['note'])
        try:
            result = store.result(job['id'])
            best = result['weights']['best']
            path = bounded(store.directory(job['id']), best['path'])
            if not path.is_relative_to(store.directory(job['id']) / 'output' / 'weights') or path.suffix != '.pt':
                raise ValueError('训练权重位置无效。')
            choice = ModelChoice(**fields, path=str(path), sha256=best['sha256'], imgsz=job['request']['config']['imgsz'],
                                 dataset_fingerprint=result['dataset_fingerprint'], completed_epochs=result['completed_epochs'])
            choice.verify()
            choices[key] = choice
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # Keep a failed selected ID visible, instead of silently replacing
            # it with an upstream model when the checkpoint disappears.
            choices[key] = ModelChoice(**fields, error=f'自训模型不可用：{exc}')
    return choices


def general_models(project_root):
    return {path.stem: ModelChoice(path.stem, path.stem, 'general', str(path.resolve()), file_sha(path))
            for path in sorted((Path(project_root) / 'general-models').glob('*.pt'))}


def model_columns(metadata_json):
    try:
        metadata = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        metadata = metadata if isinstance(metadata, dict) else {}
        model = metadata.get('model') or {}
        model = model if isinstance(model, dict) else {}
    except (ValueError, TypeError):
        metadata, model = {}, {}
    return {'检测模型': model.get('label') or Path(metadata.get('weights') or '').name or '旧记录未保存',
            '模型来源': {'upstream': '上游', 'trained': '自训', 'general': '通用'}.get(model.get('origin'), '旧记录未保存'),
            '训练任务': model.get('training_id'), '权重SHA-256': model.get('sha256'),
            '推理图像尺寸': model.get('imgsz'), '计算设备': model.get('device') or metadata.get('device'),
            '训练数据指纹': model.get('dataset_fingerprint')}
