"""Truthful tier-B academic source validation and original-pixel reproduction.

Callable dispatcher branch for the additive uint8 builder. No producer validity
or acquisition-native resolution is inferred from a delivered product grid.
"""
from contextlib import ExitStack
import io
import json
import re
from types import SimpleNamespace

import numpy as np
from rasterio.crs import CRS
from rasterio.enums import ColorInterp
from rasterio.io import MemoryFile
from rasterio.transform import Affine

from scripts.maritime_academic_raster_intake import validate_evidence, validate_header
from scripts.maritime_cloud_inventory import digest_bytes
from scripts.maritime_maxar_intake import area_hr, core_geometry
from scripts.maritime_release_build import pinned
from src.maritime_recipes import array_hash, RECIPE_VERSIONS


def source_geometry(row):
    return row['source']['output_crs'],row['source']['physical_core_bounds']


def native_bytes(row):
    validate_source(row['source'])
    return row['source']['useful_native_rgb_bytes']


def validate_source(source):
    if source.get('schema')!='academic-raster-candidate-v1' or source.get('source_kind')!='academic_raster':
        raise ValueError('Versioned academic source required')
    evidence=source['source_provenance'];validate_evidence(evidence)
    if (source.get('validity_tier')!='B'
        or source.get('original_mask_provenance')!='inferred_rgb_extremes_not_producer_verified'
        or not isinstance(source.get('mask_inference_uncertainty'),str) or not source['mask_inference_uncertainty'].strip()):
        raise ValueError('Explicit inferred support tier B and uncertainty required')
    for key in ('native_context','hr','source_evidence'):
        pinned(source[key])
    if (source.get('license')!=evidence['license']
        or source.get('native_gsd_m')!=evidence['native_acquisition_gsd_m']
        or source.get('publisher_product_resolution_m')!=evidence['publisher_product_resolution_m']):
        raise ValueError('License/native/product resolution provenance inconsistent')
    http=source.get('source_http_pin',{})
    if not http.get('etag') or isinstance(http.get('bytes'),bool) or not isinstance(http.get('bytes'),int) or http['bytes']<=0:
        raise ValueError('Original HTTP object identity required')
    if not source.get('source_href','').startswith('https://spacenet-dataset.s3.amazonaws.com/'):
        raise ValueError('Audited public SpaceNet source required')
    crs=CRS.from_user_input(source['source_crs'])
    if crs!=CRS.from_user_input(source['output_crs']):
        raise ValueError('Unsupported reprojection in exact metric adapter')
    transform=Affine(*source['original_transform'])
    # Reconstruct source extent sufficiently to check source-relative geometry.
    window=source['source_pixel_window']
    if len(window)!=4 or any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in window) or min(window[2:])<=0:
        raise ValueError('Integer captured source window required')
    c,r,w,h=window
    ds=SimpleNamespace(crs=crs,transform=transform,width=c+w,height=r+h,count=3,
        dtypes=('uint8',)*3,colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue))
    validate_header(ds,evidence)
    cell=source['world_grid_cell']
    if len(cell)!=2 or any(isinstance(v,bool) or not isinstance(v,int) for v in cell):
        raise ValueError('Integer physical grid cell required')
    expected=core_geometry(ds,*cell)
    for key,value in expected.items():
        if source.get(key)!=value:
            raise ValueError('Academic physical geometry/accounting mismatch: '+key)
    if source.get('native_window_transform')!=list(transform*Affine.translation(c,r))[:6]:
        raise ValueError('Captured original pixel transform mismatch')
    if source.get('canonical_hr_method')!='exact_metric_fractional_area_float64_then_normalized_float32':
        raise ValueError('Unsupported canonical HR method')
    for key in ('decoded_native_context_sha256','decoded_useful_core_sha256'):
        if not isinstance(source.get(key),str) or not re.fullmatch('[a-f0-9]{64}',source[key]):
            raise ValueError('Decoded original RGB hashes required')


def canonical_from_native(ds,source,stored=None):
    """Recompute 544-square normalized context; verify captured uint8 pixels."""
    validate_source(source)
    evidence=source['source_provenance'];validate_header(ds,evidence)
    c,r,w,h=source['source_pixel_window']
    if (ds.crs!=CRS.from_user_input(source['source_crs']) or ds.shape!=(h,w)
        or list(ds.transform)[:6]!=source['native_window_transform']):
        raise ValueError('Captured academic source pixel grid mismatch')
    rgb=ds.read([1,2,3])
    # An exported internal mask is derived support, never producer evidence.
    valid=(rgb.max(axis=0)>0)&(rgb.min(axis=0)<255)
    context=area_hr(rgb,valid,source)
    x,y,cw,ch=source['owned_core_pixel_window']
    useful=rgb[:,y-r:y-r+ch,x-c:x-c+cw]
    context_hash=digest_bytes(rgb.tobytes());core_hash=digest_bytes(useful.tobytes())
    if context_hash!=source['decoded_native_context_sha256'] or core_hash!=source['decoded_useful_core_sha256']:
        raise ValueError('Decoded original academic RGB hash mismatch')
    if stored is not None:
        with np.load(io.BytesIO(stored),allow_pickle=False) as arrays:
            if not np.array_equal(arrays['hr_context'],context) or not np.array_equal(arrays['hr'],context[:,16:-16,16:-16]):
                raise ValueError('Stored academic canonical HR disagrees with source pixels')
    return context,cw*ch*3,context_hash,core_hash


def process_sample(store,row,seed):
    """Return common, context, Mother NPZ and original-float recipe children."""
    if row.get('source_kind')!='academic_raster' or row.get('sample_id')!=row['source']['sample_id']:
        raise ValueError('Academic selection/source identity mismatch')
    if row.get('split')!='train':
        raise ValueError('Academic expansion must preserve existing evaluation population')
    from scripts.maritime_pilot import deterministic_npz
    from scripts.maritime_ultimate_release import pair_payload
    source=row['source']
    with ExitStack() as stack:
        memory=stack.enter_context(MemoryFile(store.get(source['native_context'])))
        ds=stack.enter_context(memory.open())
        context,useful,context_hash,core_hash=canonical_from_native(ds,source,store.get(source['hr']))
    bounds=source['physical_core_bounds']
    common={key:value for key,value in row.items() if key!='source'}
    common.update(crs=source['output_crs'],hr_transform=source['hr_core_transform'],
        lr_transform=[10,0,bounds[0],0,-10,bounds[3]],hr_sha256=array_hash(context[:,16:-16,16:-16]),
        context_sha256=array_hash(context),useful_native_rgb_bytes=useful,
        decoded_native_context_sha256=context_hash,decoded_native_core_sha256=core_hash,
        native_gsd_m=source['native_gsd_m'],publisher_product_resolution_m=source['publisher_product_resolution_m'],
        original_full_support_verified=False,inferred_full_support_verified=True,validity_tier='B',
        mask_provenance=source['original_mask_provenance'],mask_inference_uncertainty=source['mask_inference_uncertainty'],
        native_byte_accounting='captured decoded original uint8 RGB owned core; source acquisition GSD may be unknown',
        source_provenance=source,native_context=source['native_context'])
    shared=deterministic_npz(dict(context=context,hr=np.ascontiguousarray(context[:,16:-16,16:-16]),
        hr_mask=np.ones((512,512),bool),provenance=np.asarray(json.dumps(common,sort_keys=True))))
    children={name:pair_payload(context,common,name,seed)[0] for name in RECIPE_VERSIONS}
    return common,context,shared,children
