import random
import numpy as np
import mlx.core as mx
import threading
from queue import Queue
import re
import os

class PrefetchIterator:
    def __init__(self, generator, prefetch=2):
        self.generator = generator
        self.prefetch = prefetch
        self.queue = Queue(maxsize=prefetch)
        self.thread = threading.Thread(target=self._worker)
        self.thread.daemon = True
        self.stopped = False
        self.thread.start()

    def _worker(self):
        for item in self.generator:
            self.queue.put(item)
        self.stopped = True
        self.queue.put(None)  # Sentinel to stop

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is None:
            raise StopIteration
        return item


def train_val_test_split(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=None):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    if seed is not None:
        random.seed(seed)

    data = data.copy()
    random.shuffle(data)

    n = len(data)
    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)

    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]

    return train, val, test


def batch_iterator(dataset, batch_size, shuffle=True):
    indices = np.arange(len(dataset))
    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        batch = [dataset[i] for i in batch_indices]
        a, p, n = zip(*batch)
        yield (
            mx.array(np.stack(a)),
            mx.array(np.stack(p)),
            mx.array(np.stack(n)),
        )

def get_latest_checkpoint(models_path,filename_fn):
    if not os.path.exists(models_path):
        return 0, None

    checkpoints = [f for f in os.listdir(models_path) if f.endswith(".safetensors")]
    if not checkpoints:
        return 0, None

    def extract_epoch_batch(filename):
        filename_path = filename_fn("(\\d+)","(\\d+)")
        match = re.search(r"model_e(\d+)_b(\d+).safetensors", filename)
        if match:
            return int(match.group(1)), int(match.group(2))
        return -1, -1

    
    latest = max(checkpoints, key=lambda f: extract_epoch_batch(f))
    match = re.search(r"model_e(\d+)_b(\d+).safetensors", latest)
    return int(match.group(1)), os.path.join(models_path, latest)

def getParams(model_params):
    size = 0
    for _, module in model_params.items():
        if isinstance(module, type(mx.array([0]))):
            size += module.size
        elif isinstance(module, dict):
            size += getParams(module)
        else:
            pass
    return size
