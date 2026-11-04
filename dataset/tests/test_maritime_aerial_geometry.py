from types import SimpleNamespace

import numpy as np
import pytest
from rasterio.transform import Affine

from scripts.maritime_maxar_intake import core_geometry, area_hr, area_matrix
from src.maritime_aerial_geometry import original_contributor_geometry


def test_versioned_capture_window_includes_real_extra_row_without_changing_physical_experiment():
    parent = SimpleNamespace(transform=Affine(.3,0,335037.9,0,-.3,4700590.8),width=20000,height=6000)
    old = core_geometry(parent,265,3671)
    new = original_contributor_geometry(parent,265,3671)
    assert old['source_pixel_window'] == [13740,1302,4534,4534]
    assert new['source_pixel_window'] == [13740,1302,4534,4535]
    assert new['geometry_version'] == 'original-contributor-window-v2'
    for field in ('physical_core_bounds','physical_context_bounds','world_grid_cell',
                  'owned_core_pixel_window','useful_native_rgb_bytes',
                  'hr_context_transform','hr_core_transform'):
        assert new[field] == old[field]
    x,y,w,h = new['source_pixel_window']
    assert 0 <= x < x+w <= parent.width and 0 <= y < y+h <= parent.height
    assert w == area_matrix(new['area_phase_m'][0],544,.3,2.5).shape[1]
    assert h == area_matrix(new['area_phase_m'][1],544,.3,2.5).shape[1]


def test_already_supported_integer_grid_geometry_and_area_recipe_are_bit_exact():
    parent = SimpleNamespace(transform=Affine(.5,0,1240,0,-.5,2600),width=2720,height=2720)
    old = core_geometry(parent,1,1)
    new = original_contributor_geometry(parent,1,1)
    assert {k:v for k,v in new.items() if k != 'geometry_version'} == old
    pixels = np.full((3,2720,2720),127,np.uint8)
    mask = np.ones((2720,2720),bool)
    assert np.array_equal(area_hr(pixels,mask,old),area_hr(pixels,mask,new))


def test_extra_real_contributor_outside_parent_is_rejected_instead_of_padded():
    parent = SimpleNamespace(transform=Affine(.3,0,335037.9,0,-.3,4700590.8),width=20000,height=5836)
    with pytest.raises(ValueError,match='outside'):
        original_contributor_geometry(parent,265,3671)
