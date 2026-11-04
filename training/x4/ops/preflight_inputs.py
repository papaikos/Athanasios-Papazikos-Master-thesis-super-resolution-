"""Reject partial staging or a changed scientific/runtime input before training."""
import hashlib
import json
import os
from pathlib import Path

# Run folder on the cluster (inputs/, results/, slurm/); defaults to the folder above ops/.
PROJECT = Path(os.environ.get('PROJECT_DIR', Path(__file__).resolve().parents[1]))
# Fast scratch with the staged datasets, venv and caches.
BASE = Path(os.environ.get('SCRATCH_DIR', PROJECT / 'scratch'))
# D1, D2 or D3 (same variable as research/experiment.py).
NAME = os.environ.get('MARITIME_DATASET', 'd2').strip().upper()


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def inventory_rows(project=PROJECT):
    inventory_path = project / 'staging-inventory.json'
    inventory = json.loads(inventory_path.read_text(encoding='utf-8-sig'))
    rows = inventory['files']
    if not rows:
        raise ValueError('Empty staging inventory')
    seen = set()
    for row in rows:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts or not (row['path'].startswith(f'datasets/{NAME}/') or row['path'].startswith('datasets/control/')):
            raise ValueError(f'Unsafe or non-{NAME} inventory path')
        if row['path'] in seen or row['bytes'] <= 0 or len(row['sha256']) != 64:
            raise ValueError('Invalid staging inventory row')
        seen.add(row['path'])
    return rows


def verify_staging(project=PROJECT, base=BASE, write_ready=False):
    rows = inventory_rows(project)
    expected = {'files': len(rows), 'bytes': sum(row['bytes'] for row in rows),
                'inventory_sha256': digest(project / 'staging-inventory.json')}
    if not write_ready:
        ready = json.loads((project / 'inputs/data-ready.json').read_text())
        if ready.get('status') != 'TRANSFER_COMPLETE' or any(ready.get(key) != value for key, value in expected.items()):
            raise ValueError(f'{NAME} transfer receipt does not match pinned inventory')
    for row in rows:
        path = base / row['path']
        if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError('Changed staged input: ' + row['path'])
    # Reject stale/unselected payloads, including sealed tests, in the fresh dataset tree.
    actual = {path.relative_to(base).as_posix() for path in (base / 'datasets' / NAME).rglob('*') if path.is_file()}
    selected = {row['path'] for row in rows if row['path'].startswith(f'datasets/{NAME}/')}
    if actual != selected:
        raise ValueError(f'{NAME} payload membership differs from inventory')
    for name in ('software-payload-manifest.json', 'operational-payload-manifest.json'):
        for row in json.loads((project / name).read_text())['files']:
            relative = Path(row['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe runtime manifest path')
            path = base / relative if relative.parts[0] == 'datasets' else project / relative
            if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
                raise ValueError('Changed pinned input: ' + row['path'])
    if write_ready:
        from research.checkpointing import atomic_json
        atomic_json(project / 'inputs/data-ready.json', {'status': 'TRANSFER_COMPLETE', **expected, 'all_selected_payload_checksums': 'VERIFIED'})
    return expected


def main():
    verify_staging()
    alexnet = Path(os.environ['TORCH_HOME']) / 'hub/checkpoints/alexnet-owt-7be5be79.pth'
    if digest(alexnet) != '7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02':
        raise ValueError('AlexNet diagnostic cache pin mismatch')
    import torch
    if torch.__version__ != '2.5.1+cu124' or torch.version.cuda != '12.4':
        raise ValueError('Historical PyTorch/CUDA wheel pair required')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError('Native CUDA BF16 GPU required')
    name = torch.cuda.get_device_name()
    if 'A100' not in name:
        raise ValueError('User-selected A100 required, allocated: ' + name)
    print(json.dumps({'status': 'PASS', 'device': name, 'native_bf16': True, 'torch': torch.__version__}))


if __name__ == '__main__':
    main()
