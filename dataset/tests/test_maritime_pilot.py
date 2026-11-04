import io
import zipfile

import numpy as np
import pytest
from affine import Affine

from scripts.maritime_pilot import deterministic_npz, core_window, validate_pair, core_identity
from rasterio.windows import Window


def test_identity_uses_world_native_core_not_tiff_local_offsets():
    affine = Affine(.8,0,0,0,-.8,2560)
    cropped = Affine(.8,0,1280,0,-.8,1280)
    assert core_identity('asset','EPSG:32632',affine,Window(1600,1600,1600,1600)) == core_identity(
        'asset','EPSG:32632',cropped,Window(0,0,1600,1600))
    assert core_identity('asset','EPSG:32632',affine,Window(0,0,1600,1600)) != core_identity(
        'asset','EPSG:32632',cropped,Window(0,0,1600,1600))


def test_historical_core_is_snapped_once_to_original_native_pixels():
    affine = Affine(.8, 0, 4.16, 0, -.8, 3840.24)
    win, shift = core_window(affine, [1, 1])
    assert list(win.flatten()) == [1595, 1600, 1600, 1600]
    assert shift == pytest.approx([.16, .24])


def test_npz_is_byte_reproducible_and_safe_for_training_loader():
    arrays = {'provenance': np.array('{"test":true}'),
              'hr': np.zeros((3, 512, 512), np.float32)}
    a = deterministic_npz(arrays)
    assert a == deterministic_npz(dict(reversed(list(arrays.items()))))
    with np.load(io.BytesIO(a), allow_pickle=False) as item:
        assert item['hr'].dtype == np.float32
        assert item['provenance'].item() == '{"test":true}'
        assert item['provenance'].shape == ()
    with zipfile.ZipFile(io.BytesIO(a)) as archive:
        assert all(z.date_time == (1980, 1, 1, 0, 0, 0) for z in archive.infolist())


def test_pair_validator_rejects_wrong_shapes_and_invalid_numeric_values():
    pair = dict(hr=np.ones((3, 512, 512), np.float32),
                lr=np.ones((3, 128, 128), np.float32),
                hr_mask=np.ones((512, 512), bool), lr_mask=np.ones((128, 128), bool))
    validate_pair(pair)
    pair['lr'][0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        validate_pair(pair)
