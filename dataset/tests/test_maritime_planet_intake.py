import numpy as np
import pytest
from rasterio.io import MemoryFile
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin

from scripts.maritime_planet_intake import (area_matrix, downsample, validate_source,
    validate_header, overlap, write_candidate)


def test_exact_area_and_conservation():
    weights = area_matrix(0, 5, .5, .8)
    assert np.allclose(weights.sum(axis=1), 1)
    pixels = np.arange(64, dtype=np.uint8).reshape(1, 8, 8)
    rgb = np.repeat(pixels, 3, axis=0)
    result, support = downsample(rgb, np.ones((8, 8), bool), 0, 0, 5, .5)
    assert result.dtype == np.uint8
    assert support.all()
    exact = weights @ pixels[0].astype(float) @ weights.T
    assert exact.mean() == pytest.approx(pixels.mean())
    assert np.array_equal(result[0], np.floor(exact + .5))


def test_hole_invalidates_every_contributor():
    rgb = np.full((3, 8, 8), 100, np.uint8)
    mask = np.ones((8, 8), bool)
    mask[1, 1] = False
    _, support = downsample(rgb, mask, 0, 0, 5, .5)
    assert not support[:2, :2].any()
    assert support[2:, :].all()


def test_fractional_phase_keeps_constant_and_complete_support():
    wx = area_matrix(.3, 5)
    rgb = np.full((3, wx.shape[1], wx.shape[1]), 73, np.uint8)
    result, support = downsample(rgb, np.ones(rgb.shape[1:], bool), .3, .3, 5)
    assert (result == 73).all() and support.all()


def test_same_strip_overlap_excludes_repeated_area():
    assert overlap([0, 0, 1280, 1280], [1279, 0, 2559, 1280])
    assert not overlap([0, 0, 1280, 1280], [1280, 0, 2560, 1280])


def test_candidate_has_self_contained_internal_mask(tmp_path):
    path = tmp_path / 'candidate.tif'
    rgb = np.full((3, 1700, 1700), 17, np.uint8)
    support = np.ones((1700, 1700), bool)
    support[0, 0] = False
    write_candidate(path, rgb, support, 'EPSG:32632', from_origin(0, 1360, .8, .8))
    assert not path.with_suffix('.tif.msk').exists()
    with MemoryFile(path.read_bytes()) as mem:
        with mem.open() as ds:
            assert ds.dataset_mask()[0, 0] == 0
            assert ds.dataset_mask()[1, 1] == 255


def test_native_gsd_is_distinct_from_product_spacing():
    row = dict(asset_id='one', uri='gs://open-cogs/one.tif', generation='1',
               bytes=10, crc32c='abc', license='CC-BY-4.0', native_gsd_m=.9,
               stac={'properties': {'gsd': .9, 'pl:pixel_resolution': .5}})
    with pytest.raises(ValueError, match='native GSD'):
        validate_source(row)
    row['native_gsd_m'] = .7
    row['stac']['properties']['gsd'] = .7
    validate_source(row)


def test_explicit_mask_required_and_rotated_grid_rejected():
    with MemoryFile() as mem:
        with mem.open(driver='GTiff', width=10, height=10, count=3, dtype='uint8',
                      crs='EPSG:32632', transform=from_origin(0, 10, .5, .5)) as ds:
            ds.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
            with pytest.raises(ValueError, match='validity'):
                validate_header(ds)
            ds.write_mask(np.full((10, 10), 255, np.uint8))
            assert validate_header(ds) == 'original_embedded_dataset_mask'
            ds.transform = ds.transform * ds.transform.rotation(1)
            with pytest.raises(ValueError, match='north-up'):
                validate_header(ds)
