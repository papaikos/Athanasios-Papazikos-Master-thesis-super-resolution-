"""Reject a changed runtime or the wrong GPU before training (run inside each job)."""
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def main():
    alexnet = Path(os.environ['TORCH_HOME']) / 'hub/checkpoints/alexnet-owt-7be5be79.pth'
    if digest(alexnet) != '7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02':
        raise ValueError('AlexNet diagnostic cache pin mismatch')
    import torch
    if torch.__version__ != '2.5.1+cu124' or torch.version.cuda != '12.4':
        raise ValueError('Pinned PyTorch 2.5.1+cu124 required (same as the D-series runs)')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError('Native CUDA BF16 GPU required')
    name = torch.cuda.get_device_name()
    if 'A100' not in name:
        raise ValueError('User-selected A100 required, allocated: ' + name)
    print(json.dumps({'status': 'PASS', 'device': name, 'torch': torch.__version__}))


if __name__ == '__main__':
    main()
