"""Stub osgeo so deepdespeckling imports without a native GDAL build.
We only use the in-memory denoiser classes (numpy in/out), never GDAL I/O."""
import sys, types
if 'osgeo' not in sys.modules:
    _osgeo = types.ModuleType('osgeo')
    _gdal = types.ModuleType('osgeo.gdal')
    _osgeo.gdal = _gdal
    sys.modules['osgeo'] = _osgeo
    sys.modules['osgeo.gdal'] = _gdal

# deepdespeckling ships its own trusted pretrained checkpoints, but torch>=2.6
# defaults torch.load(weights_only=True) which rejects the numpy scalar inside
# them. Force weights_only=False for these trusted bundled weights.
import torch as _torch
_orig_load = _torch.load
def _patched_load(*a, **k):
    k.setdefault('weights_only', False)
    return _orig_load(*a, **k)
_torch.load = _patched_load
