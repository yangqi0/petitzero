"""Small serialization and reproducibility utilities for the M8 study.

Author: Yang Qi. No model construction or historical Experiment dependency.
"""
from __future__ import annotations

import os,time
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fsync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def durable_dir(path):
    path=Path(path)
    if path.exists():return
    durable_dir(path.parent);path.mkdir();fsync_dir(path.parent);fsync_dir(path)


def write_json(path: Path, value: Any) -> None:
    durable_dir(path.parent)
    temporary=path.with_name(path.name+f'.{os.getpid()}.{time.time_ns()}.tmp')
    with temporary.open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,ensure_ascii=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path);fsync_dir(path.parent)


def append_row(path: Path, value: dict) -> None:
    durable_dir(path.parent);new=not path.exists()
    with path.open('a') as stream:
        stream.write(json.dumps(value,sort_keys=True,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
    if new:fsync_dir(path.parent)


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            result.update(block)
    return result.hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def fingerprint(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {'tensor': tensor_hash(value), 'shape': list(value.shape), 'dtype': str(value.dtype)}
    if isinstance(value, np.ndarray):
        return {'array': hashlib.sha256(value.tobytes()).hexdigest(), 'shape': list(value.shape), 'dtype': str(value.dtype)}
    if isinstance(value, dict):
        return {str(key): fingerprint(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [fingerprint(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> dict:
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'cpu': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict) -> None:
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def describe(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {'n': 0}
    return {'n': len(array), 'mean': float(array.mean()), 'min': float(array.min()),
            'max': float(array.max()), **{f'p{q}': float(np.quantile(array, q / 100)) for q in [50, 90, 95, 99]}}


def alignment(records: list[dict], left: str, right: str) -> dict:
    points = []
    for record in records:
        assert len(record[left]) == len(record[right]) == len(record['output_ids'])
        for index, (a, b) in enumerate(zip(record[left], record[right])):
            delta = a - b
            points.append({'completion_id': record['completion_id'], 'action_index': index,
                           'token': record['output_ids'][index], 'left': a, 'right': b,
                           'signed': delta, 'absolute': abs(delta), 'ratio': float(np.exp(delta)),
                           'bound': 2e-4 + 1e-5 * abs(b), 'passed': abs(delta) <= 2e-4 + 1e-5 * abs(b)})
    return {'passed': all(point['passed'] for point in points),
            'failures': [point for point in points if not point['passed']],
            **{key: describe([point[key] for point in points]) for key in ['signed', 'absolute', 'ratio']},
            'tails': sorted(points, key=lambda point: -point['absolute'])[:8]}


def finalize_metrics(numeric: dict, **nested: Any) -> dict:
    """Convert a Counter before attaching dictionaries: M7 regression boundary."""
    result = dict(numeric)
    result.update(nested)
    return result


def next_request(schedule: list[dict], completed_batches: int) -> dict | None:
    if completed_batches == len(schedule):
        return None
    return schedule[completed_batches]['prompts'][0]


def verify_optimizer(optimizer: torch.optim.Optimizer, actual_steps: int) -> None:
    for state in optimizer.state.values():
        assert state['step'].item() == actual_steps
        for key in ['exp_avg', 'exp_avg_sq']:
            assert state[key].dtype == torch.float32 and bool(torch.isfinite(state[key]).all())


def verify_update_change(changed: bool, update_l2: float) -> None:
    """A zero learning signal is valid; positive rewards are never a test gate."""
    assert np.isfinite(update_l2) and update_l2 >= 0
    if update_l2 > 0:
        assert changed
