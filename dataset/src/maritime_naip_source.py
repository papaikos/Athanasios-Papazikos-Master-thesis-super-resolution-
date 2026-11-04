"""Replay delivered NAIP RGB, retaining original mask and license evidence."""
from contextlib import ExitStack
import io
import json
import re
from types import SimpleNamespace
from urllib.parse import urlsplit

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from rasterio.windows import Window

from scripts.maritime_maxar_intake import core_geometry
from scripts.maritime_release_build import pinned
from src import maritime_aerial_source as aerial
from src.maritime_recipes import RECIPE_VERSIONS


def source_geometry(row):
    return aerial.source_geometry(row)


def native_bytes(row):
    validate_source(row['source'])
    return row['source']['useful_native_rgb_bytes']


def validate_source(source):
    if (source.get('schema')!='aerial-rgb-candidate-v1' or source.get('source_kind')!='aerial_rgb'
            or source.get('source_type')!='naip_aerial'):
        raise ValueError('Versioned NAIP aerial source required')
    original=source['source'];dataset=original['dataset']
    license_url=urlsplit(dataset.get('license_evidence_url',''))
    if (dataset.get('license')!='Public-Domain' or license_url.scheme!='https'
            or license_url.hostname not in ('www.usgs.gov','usgs.gov','www.fsa.usda.gov','fsa.usda.gov')
            or not isinstance(dataset.get('stac_license'),str)
            or dataset.get('license_conflict')!=(dataset['stac_license'] not in ('Public-Domain','public-domain'))):
        raise ValueError('Official NAIP public-domain license evidence and STAC conflict required')
    for key in ('native_context','hr','source_original_object','source_evidence_pin'):
        pinned(source[key])
    http=source['source_http_identity'];href=urlsplit(source['source_href'])
    if (href.scheme!='https' or href.hostname not in ('naipblobs.blob.core.windows.net','naipeuwest.blob.core.windows.net') or href.query
            or source['source_href']!=original['href'] or http!=original['original_http_identity']
            or not http.get('etag') or isinstance(http.get('bytes'),bool)
            or not isinstance(http.get('bytes'),int) or http['bytes']!=source['source_original_object']['bytes']):
        raise ValueError('Original anonymous NAIP HTTP object identity mismatch')
    if (source.get('explicit_gsd_semantics')!='original_delivered_orthomosaic_grid_not_camera_capture'
            or source.get('original_camera_capture_gsd_m') is not None
            or original.get('original_camera_capture_gsd_m') is not None
            or source.get('source_product')!='original_uint8_RGB_orthorectified_aerial_mosaic'
            or original.get('original_product')!='orthorectified_aerial_mosaic'
            or source.get('source_radiometry')!='original_delivered_uint8_RGB_DN_no_new_stretch'
            or source.get('hr_radiometry')!='original_RGB_DN_divided255_then_exact_area_integration'
            or source.get('original_mask_provenance')!='all_original_RGB_masks255'):
        raise ValueError('Truthful NAIP delivered product, radiometry and mask provenance required')
    header=original['original_raster_header']
    if (isinstance(header['count'],bool) or not isinstance(header['count'],int) or header['count']<3
            or header['dtypes'][:3]!=['uint8']*3 or header['colorinterp'][:3]!=['red','green','blue']
            or any(len(header[k])!=header['count'] for k in ('dtypes','colorinterp','mask_flags'))
            or original['rgb_indexes']!=[1,2,3]):
        raise ValueError('Observed NAIP original header and RGB declarations required')
    explicit_void=(header['nodata'] is not None or any(flags!=['all_valid'] for flags in header['mask_flags'][:3]))
    if (source.get('original_support_policy')!='all_original_RGB_masks255_and_no_RGB000'
            or original.get('original_support_policy')!=source['original_support_policy']
            or source.get('original_void_mask_available') is not explicit_void
            or original.get('original_void_mask_available') is not explicit_void):
        raise ValueError('NAIP original void-mask uncertainty and conservative zero guard required')
    crs=CRS.from_user_input(source['source_crs']);t=Affine(*source['original_transform'])
    if (crs!=CRS.from_user_input(source['output_crs']) or crs!=CRS.from_epsg(original['epsg'])
            or not crs.is_projected or crs.linear_units_factor[1]!=1
            or not np.isfinite(list(t)).all() or t.b!=0 or t.d!=0 or not 0<t.a<=1 or t.e!=-t.a
            or source['original_transform']!=original['transform_affine']
            or any(s.get(k)!=t.a for s in (source,original) for k in ('native_gsd_m','delivered_gsd_m'))):
        raise ValueError('Original delivered metric NAIP grid mismatch')
    for key in ('acquisition_day','parent_tile_id','capture_identity_type'):
        if source.get(key)!=original.get(key):raise ValueError('NAIP acquisition identity mismatch')
    if original.get('capture_identity_type')!='filename_acquisition_day_and_original_orthomosaic_tile':
        raise ValueError('NAIP capture identity must remain day and orthomosaic tile')
    if (source.get('publication_datetime') is not None or source.get('stac_datetime')!=original.get('stac_datetime')
            or original.get('stac_datetime_role')!='capture_calendar_day_hour_unverified'):
        raise ValueError('NAIP STAC acquisition calendar must not be called publication or exact camera exposure')
    cell=source['world_grid_cell']
    if len(cell)!=2 or any(isinstance(v,bool) or not isinstance(v,int) for v in cell):
        raise ValueError('Integer physical NAIP grid cell required')
    h,w=original['shape'];parent=SimpleNamespace(crs=crs,transform=t,width=w,height=h)
    version=source.get('geometry_version','legacy-window-v1')
    if version=='original-contributor-window-v2':
        from src.maritime_aerial_geometry import original_contributor_geometry
        geometry=original_contributor_geometry(parent,*cell)
    elif version=='legacy-window-v1':geometry=core_geometry(parent,*cell)
    else:raise ValueError('Unsupported NAIP geometry version')
    for key,value in geometry.items():
        if source.get(key)!=value:raise ValueError('NAIP geometry/accounting mismatch: '+key)
    c,r,_,_=source['source_pixel_window']
    if source['native_window_transform']!=list(t*Affine.translation(c,r))[:6]:
        raise ValueError('Captured original NAIP window transform mismatch')
    if source.get('canonical_hr_method') not in ('integer_aligned_box_sum_uint32_v1','canonical_sparse_exact_area_v1'):
        raise ValueError('Unsupported NAIP original area integration method')
    for key in ('decoded_native_context_sha256','decoded_useful_core_sha256'):
        if not re.fullmatch('[a-f0-9]{64}',source.get(key,'')):raise ValueError('Decoded original NAIP hashes required')


def canonical_from_native(ds,source,stored=None):
    return aerial.canonical_from_native(ds,source,stored,validator=validate_source,support_validator=_check_support)


def _check_support(rgb,valid):
    if not valid.all() or np.any(np.all(rgb==0,axis=0)):
        raise ValueError('Incomplete original NAIP support or conservative RGB000 guard rejection')


def verify_original_parent(ds,source,captured):
    original=source['source'];header=original['original_raster_header']
    observed=dict(count=ds.count,dtypes=list(ds.dtypes),colorinterp=[v.name for v in ds.colorinterp],
        mask_flags=[[v.name for v in flags] for flags in ds.mask_flag_enums],nodata=ds.nodata)
    if (observed!=header or ds.shape!=tuple(original['shape']) or ds.crs!=CRS.from_epsg(original['epsg'])
            or list(ds.transform)[:6]!=original['transform_affine']):
        raise ValueError('Pinned original NAIP parent header/grid mismatch')
    window=Window(*source['source_pixel_window'])
    rgb=ds.read([1,2,3],window=window)
    _check_support(rgb,np.all(ds.read_masks([1,2,3],window=window)==255,axis=0))
    if not np.array_equal(rgb,captured):
        raise ValueError('Captured NAIP RGB or support differs from original parent')


def process_sample(store,row,seed):
    if row.get('source_kind')!='aerial_rgb' or row.get('sample_id')!=row['source']['sample_id']:
        raise ValueError('NAIP selection/source identity mismatch')
    if row.get('split')!='train':raise ValueError('NAIP addition must preserve frozen evaluation population')
    from scripts.maritime_pilot import deterministic_npz
    from scripts.maritime_ultimate_release import common_sample,pair_payload
    source=row['source'];validate_source(source)
    evidence=json.loads(store.get(source['source_evidence_pin']))
    if evidence.get('source')!={k:v for k,v in source['source'].items() if k!='source_evidence_pin'}:
        raise ValueError('Pinned NAIP source evidence differs from candidate')
    for key in ('original_item_metadata_object','original_collection_metadata_object',
            'official_usda_policy_object','official_usgs_metadata_object'):
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
        original_mask_flags=source['source']['original_raster_header']['mask_flags'],
        original_void_mask_available=source['original_void_mask_available'],
        original_support_policy=source['original_support_policy'],
        original_void_mask_caveat='All-valid flags do not independently mark voids; RGB000 guard is conservative and can reject genuinely black pixels.',
        stac_datetime=source['stac_datetime'],publication_datetime=None,
        stac_datetime_role=source['source']['stac_datetime_role'],
        native_byte_accounting='decoded original delivered NAIP orthomosaic RGB owned core; camera GSD unknown')
    shared=deterministic_npz(dict(context=context,hr=np.ascontiguousarray(context[:,16:-16,16:-16]),
        hr_mask=np.ones((512,512),bool),provenance=np.asarray(json.dumps(common,sort_keys=True))))
    children={name:pair_payload(context,common,name,seed)[0] for name in RECIPE_VERSIONS}
    return common,context,shared,children
