"""Keep third-party device selection from changing other inference sessions."""
from contextlib import contextmanager
import os
import threading

_LOCK=threading.RLock()


@contextmanager
def preserve_cuda_visibility():
    # Ultralytics select_device('cpu') sets CUDA_VISIBLE_DEVICES=-1. Restoring
    # it is essential before another model performs its first CUDA init.
    with _LOCK:
        previous=os.environ.get('CUDA_VISIBLE_DEVICES')
        try:
            yield
        finally:
            if previous is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
            else:os.environ['CUDA_VISIBLE_DEVICES']=previous
