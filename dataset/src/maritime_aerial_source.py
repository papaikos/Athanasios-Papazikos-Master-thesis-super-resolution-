"""Reproduce original NOAA aerial RGB without asserting camera capture GSD.

Delivered orthomosaic spacing, original masks and upstream mosaic processing
remain distinct from satellite acquisition properties. Final releases replay
the original parent as well as the lossless captured RGB window.
"""
from contextlib import ExitStack
import io
import json
import re
from types import SimpleNamespace
from urllib.parse import urlsplit

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import ColorInterp
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from rasterio.windows import Window

from scripts.maritime_cloud_inventory import digest_bytes
from scripts.maritime_maxar_intake import core_geometry
from scripts.maritime_noaa_intake import aerial_area_hr
from scripts.maritime_release_build import pinned
from src.maritime_recipes import RECIPE_VERSIONS


def source_geometry(row):
    return row['source']['output_crs'],row['source']['physical_core_bounds']


def native_bytes(row):
    validate_source(row['source'])
    return row['source']['useful_native_rgb_bytes']


def validate_source(source):
    if (source.get('schema')!='aerial-rgb-candidate-v1' or source.get('source_kind')!='aerial_rgb'
            or source.get('source_type')!='noaa_aerial'):
        raise ValueError('Versioned NOAA aerial source required')
    original=source['source'];dataset=original['dataset']
    license_url=urlsplit(dataset.get('license_evidence_url',''))
    if (dataset.get('license')!='CC0-1.0' or license_url.scheme!='https'
            or license_url.hostname not in ('www.fisheries.noaa.gov','fisheries.noaa.gov')):
        raise ValueError('Pinned official NOAA free-use evidence required')
    for key in ('native_context','hr','source_original_object','source_evidence_pin'):
        pinned(source[key])
    http=source['source_http_identity'];href=urlsplit(source['source_href'])
    if (href.scheme!='https' or href.hostname!='coastalimagery.blob.core.windows.net'
            or source['source_href']!=original['href'] or http!=original['original_http_identity']
            or not http.get('etag') or isinstance(http.get('bytes'),bool)
            or not isinstance(http.get('bytes'),int) or http['bytes']!=source['source_original_object']['bytes']):
        raise ValueError('Original HTTP object identity mismatch')
    if (source.get('explicit_gsd_semantics')!='original_delivered_orthomosaic_grid_not_camera_capture'
            or source.get('original_camera_capture_gsd_m') is not None
            or original.get('original_camera_capture_gsd_m') is not None
            or source.get('source_product')!='original_uint8_RGB_orthorectified_aerial_mosaic'
            or original.get('original_product')!='orthorectified_aerial_mosaic'
            or source.get('source_radiometry')!='original_delivered_uint8_RGB_DN_no_new_stretch'
            or source.get('hr_radiometry')!='original_RGB_DN_divided255_then_exact_area_integration'
            or source.get('original_mask_provenance')!='original_alpha255_and_all_original_RGB_masks255'):
        raise ValueError('Truthful aerial product, radiometry and support provenance required')
    crs=CRS.from_user_input(source['source_crs']);t=Affine(*source['original_transform'])
    if (crs!=CRS.from_user_input(source['output_crs']) or crs!=CRS.from_epsg(original['epsg'])
            or not crs.is_projected or crs.linear_units_factor[1]!=1
            or not np.isfinite(list(t)).all() or t.b!=0 or t.d!=0 or not 0<t.a<=1 or t.e!=-t.a
            or source['original_transform']!=original['transform_affine']
            or source['native_gsd_m']!=t.a or source['delivered_gsd_m']!=t.a
            or original['native_gsd_m']!=t.a or original['delivered_gsd_m']!=t.a):
        raise ValueError('Original delivered metric aerial grid mismatch')
    if original['rgb_indexes']!=[1,2,3] or not isinstance(original.get('alpha_index'),int) or original['alpha_index']<=3:
        raise ValueError('Original RGB and alpha declarations required')
    if any(source.get(key)!=original.get(key) for key in ('acquisition_day','parent_tile_id','capture_identity_type')):
        raise ValueError('Aerial acquisition identity mismatch')
    if source.get('publication_datetime')!=original.get('stac_datetime'):
        raise ValueError('Publication timestamp must remain separate')
    cell=source['world_grid_cell']
    if len(cell)!=2 or any(isinstance(v,bool) or not isinstance(v,int) for v in cell):
        raise ValueError('Integer physical grid cell required')
    h,w=original['shape'];expected=core_geometry(SimpleNamespace(crs=crs,transform=t,width=w,height=h),*cell)
    for key,value in expected.items():
        if source.get(key)!=value:raise ValueError('Aerial geometry/accounting mismatch: '+key)
    c,r,_,_=source['source_pixel_window']
    if source['native_window_transform']!=list(t*Affine.translation(c,r))[:6]:
        raise ValueError('Captured original window transform mismatch')
    if source.get('canonical_hr_method') not in ('integer_aligned_box_sum_uint32_v1','canonical_sparse_exact_area_v1'):
        raise ValueError('Unsupported aerial canonical area method')
    for key in ('decoded_native_context_sha256','decoded_useful_core_sha256'):
        if not re.fullmatch('[a-f0-9]{64}',source.get(key,'')):raise ValueError('Decoded original RGB hashes required')


def validate_native_grid(ds,source):
    c,r,w,h=source['source_pixel_window']
    if (ds.count!=3 or ds.dtypes!=('uint8',)*3
            or ds.colorinterp!=(ColorInterp.red,ColorInterp.green,ColorInterp.blue)
            or ds.crs!=CRS.from_user_input(source['source_crs']) or ds.shape!=(h,w)
            or list(ds.transform)[:6]!=source['native_window_transform']):
        raise ValueError('Captured aerial original RGB grid mismatch')


def source_adapter(source):
    """Select the original product contract without borrowing another license/mask."""
    if source.get('source_type')=='noaa_aerial':
        import src.maritime_aerial_source as adapter
    elif source.get('source_type')=='naip_aerial':
        import src.maritime_naip_source as adapter
    else:
        raise ValueError('Unknown aerial original product contract')
    return adapter


def canonical_from_native(ds,source,stored=None,*,validator=validate_source,support_validator=None):
    validator(source);validate_native_grid(ds,source)
    rgb=ds.read([1,2,3]);valid=np.all(ds.read_masks([1,2,3])==255,axis=0)
    if support_validator is not None:support_validator(rgb,valid)
    context,method=aerial_area_hr(rgb,valid,source)
    if method!=source['canonical_hr_method']:raise ValueError('Declared area method mismatch')
    c,r,_,_=source['source_pixel_window'];x,y,w,h=source['owned_core_pixel_window']
    useful=rgb[:,y-r:y-r+h,x-c:x-c+w]
    context_hash=digest_bytes(rgb.tobytes());core_hash=digest_bytes(useful.tobytes())
    if context_hash!=source['decoded_native_context_sha256'] or core_hash!=source['decoded_useful_core_sha256']:
        raise ValueError('Decoded original aerial RGB hash mismatch')
    if stored is not None:
        with np.load(io.BytesIO(stored),allow_pickle=False) as arrays:
            if not np.array_equal(arrays['hr_context'],context) or not np.array_equal(arrays['hr'],context[:,16:-16,16:-16]):
                raise ValueError('Stored aerial canonical HR differs from original pixels')
    return context,w*h*3,context_hash,core_hash


def verify_original_parent(ds,source,captured):
    original=source['source'];alpha=original['alpha_index']
    if (ds.shape!=tuple(original['shape']) or ds.crs!=CRS.from_epsg(original['epsg'])
            or list(ds.transform)[:6]!=original['transform_affine'] or ds.count<alpha
            or ds.dtypes[:3]!=('uint8',)*3 or ds.dtypes[alpha-1]!='uint8'
            or ds.colorinterp[:3]!=(ColorInterp.red,ColorInterp.green,ColorInterp.blue)
            or ds.colorinterp[alpha-1]!=ColorInterp.alpha):
        raise ValueError('Pinned original parent grid/bands mismatch')
    window=Window(*source['source_pixel_window'])
    if (not np.all(ds.read_masks([1,2,3],window=window)==255)
            or not np.all(ds.read(alpha,window=window)==255)
            or not np.array_equal(ds.read([1,2,3],window=window),captured)):
        raise ValueError('Captured aerial pixels or support differ from original parent')


def process_sample(store,row,seed):
    if row.get('source_kind')!='aerial_rgb' or row.get('sample_id')!=row['source']['sample_id']:
        raise ValueError('Aerial selection/source identity mismatch')
    if row.get('split')!='train':raise ValueError('Aerial addition must preserve frozen evaluation population')
    from scripts.maritime_pilot import deterministic_npz
    from scripts.maritime_ultimate_release import common_sample,pair_payload
    source=row['source']
    # Read evidence pins even when numeric pixels have already passed intake.
    evidence=json.loads(store.get(source['source_evidence_pin']))
    if evidence.get('source')!={k:v for k,v in source['source'].items() if k!='source_evidence_pin'}:
        raise ValueError('Pinned aerial source evidence differs from candidate')
    for key in ('original_item_metadata_object','official_dataset_metadata_object'):
        store.get(pinned(evidence[key]))
    with ExitStack() as stack:
        memory=stack.enter_context(MemoryFile(store.get(source['native_context'])))
        ds=stack.enter_context(memory.open())
        context,useful,context_hash,core_hash=canonical_from_native(ds,source,store.get(source['hr']))
        original=stack.enter_context(rasterio.open(store.cached_source(source['source_original_object'])))
        verify_original_parent(original,source,ds.read([1,2,3]))
    common=common_sample(row,context,useful,context_hash,core_hash)
    common.update(delivered_gsd_m=source['delivered_gsd_m'],original_camera_capture_gsd_m=None,
        native_gsd_semantics=source['explicit_gsd_semantics'],source_platform='aerial',
        source_product=source['source_product'],original_parent_pixels_replayed=True,
        native_byte_accounting='decoded original delivered aerial orthomosaic RGB owned core; camera GSD unknown')
    shared=deterministic_npz(dict(context=context,hr=np.ascontiguousarray(context[:,16:-16,16:-16]),
        hr_mask=np.ones((512,512),bool),provenance=np.asarray(json.dumps(common,sort_keys=True))))
    children={name:pair_payload(context,common,name,seed)[0] for name in RECIPE_VERSIONS}
    return common,context,shared,children
