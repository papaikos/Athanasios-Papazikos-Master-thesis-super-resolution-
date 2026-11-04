"""Versioned observed source window; frozen physical SR grids stay unchanged."""
import math

from scripts.maritime_maxar_intake import core_geometry, area_matrix


def original_contributor_geometry(ds, gx, gy):
    """Capture every column/row required by the unchanged sparse area kernel.

    Legacy window endpoint snapping and kernel support use different floating
    tolerances. V2 derives the observed window from original affine coordinates
    and the frozen kernel dimensions. Extra pixels are read from the actual
    parent. No padding, phase clamping, kernel changes or output-grid shifts.
    """
    geometry = core_geometry(ds, gx, gy)
    left, _, _, top = geometry['physical_context_bounds']
    t = ds.transform
    if t.b != 0 or t.d != 0 or t.a <= 0 or t.e != -t.a:
        raise ValueError('Original north-up square aerial grid required')
    col = math.floor((left-t.c)/t.a)
    row = math.floor((top-t.f)/t.e)
    phase = [left-(t.c+col*t.a), (t.f+row*t.e)-top]
    if min(phase) < 0:
        raise ValueError('Negative original contributor phase remains unsupported')
    width = area_matrix(phase[0],544,t.a,2.5).shape[1]
    height = area_matrix(phase[1],544,t.a,2.5).shape[1]
    if min(col,row) < 0 or col+width > ds.width or row+height > ds.height:
        raise ValueError('Real original contributor window outside raster')
    geometry.update(source_pixel_window=[col,row,width,height],area_phase_m=phase,
                    geometry_version='original-contributor-window-v2')
    return geometry
