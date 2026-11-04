"""Verified original-pixel reconstruction for provisional EarthView mosaics.

Sixteen VISUAL files are from one exact acquisition and grid. Dataset support is
the original declared nodata-band support, not an original dataset-mask claim.
Radiometry/semantic gates are independent and remain mandatory for release.
"""
from contextlib import ExitStack
import io
import json
import re

import numpy as np
from rasterio.crs import CRS
from rasterio.enums import ColorInterp
from rasterio.io import MemoryFile
from rasterio.transform import Affine

from scripts.maritime_cloud_inventory import digest_bytes
from scripts.maritime_earthview_intake import (validate_mosaic,validate_header,
    stitch_original,canonical_context,official_url,TOA_DISPLAY_MAPPING,TOA_ROBUST_RECIPE,
    apply_toa_mapping,map_toa_mosaic)
from scripts.maritime_maxar_intake import area_hr
from scripts.maritime_release_build import pinned
from src.maritime_recipes import array_hash,RECIPE_VERSIONS


def source_geometry(row):
    return row['source']['output_crs'],row['source']['physical_core_bounds']


def native_bytes(row):
    validate_source(row['source']);return row['source']['useful_native_rgb_bytes']


def validate_source(source):
    if (source.get('schema')!='earthview-candidate-v1' or source.get('source_kind')!='earthview'
        or source.get('native_gsd_m')!=1. or source.get('license')!='CC-BY-4.0'):
        raise ValueError('Documented native1m versioned EarthView source required')
    if (source.get('original_mask_provenance')!='original_declared_nodata_zero_rgb_band_masks'
        or not source.get('mask_caveat') or not source.get('upstream_visual_rendering')):
        raise ValueError('Declared nodata and upstream VISUAL uncertainty must remain explicit')
    for key in ('native_context','hr'):pinned(source[key])
    sources=source['source_items']
    product=source['input_product']
    plan=validate_mosaic([record['item'] for record in sources],source['source_provenance'],product=product)
    if product=='TOA':
        mapping=source.get('rgb_mapping')
        if not isinstance(mapping,dict):
            raise ValueError('Versioned TOA display mapping required')
        if mapping!=TOA_DISPLAY_MAPPING:
            if any(mapping.get(key)!=value for key,value in TOA_ROBUST_RECIPE.items()):
                raise ValueError('TOA whole-mosaic scientific display mapping mismatch')
            low=np.asarray(mapping.get('low_dn'),np.float64);high=np.asarray(mapping.get('high_dn'),np.float64)
            if (low.shape!=(3,) or high.shape!=(3,) or not np.isfinite(low).all() or not np.isfinite(high).all()
                or not (high>low).all() or min(low)<0 or max(high)>65535
                or not isinstance(mapping.get('statistics_valid_pixels'),int)
                or not 0<mapping['statistics_valid_pixels']<=1536**2):
                raise ValueError('Original whole-mosaic percentile parameters invalid')
            for key in ('original_mosaic_rgb_sha256','original_mosaic_support_sha256'):
                if not re.fullmatch('[a-f0-9]{64}',mapping.get(key,'')):
                    raise ValueError('Original percentile population hashes missing')
    for key,value in plan.items():
        if key!='cells' and source.get(key)!=value:
            raise ValueError('EarthView assembly/canonical physical geometry mismatch: '+key)
    cells={cell['item']['id']:cell for cell in plan['cells']}
    for record in sources:
        item=record['item'];cell=cells[item['id']]
        if record['mosaic_pixel_offset']!=[cell['col'],cell['row']]:
            raise ValueError('Original chip placement mismatch')
        for key in ('item_json','object'):pinned(record[key])
        for key,suffix in [('item_http_pin','.json'),('http_pin','_'+product+'.tif')]:
            http=record[key];official_url(http['url'],suffix)
            if not http.get('etag') or not isinstance(http.get('bytes'),int) or isinstance(http['bytes'],bool) or http['bytes']<=0:
                raise ValueError('Original per-object HTTP identity missing')
            if not isinstance(http.get('sha256'),str) or not re.fullmatch('[a-f0-9]{64}',http['sha256']):
                raise ValueError('Full original source-object SHA required')
        if record['http_pin']['url']!=item['assets']['analytic' if product=='TOA' else 'visual']['href']:
            raise ValueError('Original source asset identity mismatch')
        for key in ('decoded_rgb_sha256','declared_nodata_mask_sha256'):
            if not re.fullmatch('[a-f0-9]{64}',record.get(key,'')):
                raise ValueError('Original chip RGB/mask decoded hashes required')
    for key in ('decoded_native_context_sha256','decoded_useful_core_sha256'):
        if not re.fullmatch('[a-f0-9]{64}',source.get(key,'')):
            raise ValueError('Decoded native context/core hashes required')
    if product=='TOA':
        for key in ('decoded_display_context_sha256','decoded_display_core_sha256'):
            if not re.fullmatch('[a-f0-9]{64}',source.get(key,'')):
                raise ValueError('Derived TOA display RGB hashes required')


def canonical_from_native(ds,source,stored=None):
    validate_source(source)
    if (ds.crs!=CRS.from_user_input(source['source_crs']) or ds.shape!=(1360,1360)
        or list(ds.transform)[:6]!=source['native_context_transform'] or ds.count!=3
        or ds.dtypes!=(source['source_rgb_dtype'],)*3 or ds.colorinterp!=(ColorInterp.red,ColorInterp.green,ColorInterp.blue)):
        raise ValueError('Captured original EarthView context grid/RGB mismatch')
    rgb=ds.read();valid=np.all(ds.read_masks([1,2,3])==255,axis=0)
    display=apply_toa_mapping(rgb,source['rgb_mapping']) if source['input_product']=='TOA' else rgb
    context=area_hr(display,valid,source)
    context_hash=digest_bytes(rgb.tobytes());core_hash=digest_bytes(rgb[:,40:1320,40:1320].tobytes())
    if context_hash!=source['decoded_native_context_sha256'] or core_hash!=source['decoded_useful_core_sha256']:
        raise ValueError('Captured original EarthView RGB hash mismatch')
    if source['input_product']=='TOA' and (digest_bytes(display.tobytes())!=source['decoded_display_context_sha256']
        or digest_bytes(display[:,40:1320,40:1320].tobytes())!=source['decoded_display_core_sha256']):
        raise ValueError('Derived fixed-mapping TOA display RGB hash mismatch')
    if stored is not None:
        with np.load(io.BytesIO(stored),allow_pickle=False) as arrays:
            if not np.array_equal(arrays['hr_context'],context) or not np.array_equal(arrays['hr'],context[:,16:-16,16:-16]):
                raise ValueError('Stored EarthView canonical HR disagrees with native assembly')
    return context,source['useful_native_rgb_bytes'],context_hash,core_hash


def verify_original_assembly(store,source):
    """Reopen all16 pinned original files and reproduce the published context."""
    validate_source(source)
    chips={};masks={};items=[]
    for record in source['source_items']:
        item_raw=store.get(record['item_json'])
        if digest_bytes(item_raw)!=record['item_http_pin']['sha256'] or json.loads(item_raw)!=record['item']:
            raise ValueError('Pinned original STAC item bytes/metadata mismatch')
        raw=store.get(record['object'])
        if len(raw)!=record['http_pin']['bytes'] or digest_bytes(raw)!=record['http_pin']['sha256']:
            raise ValueError('Pinned original EarthView raster full-byte hash mismatch')
        item=record['item'];items.append(item)
        with MemoryFile(raw) as memory,memory.open() as ds:
            indices=validate_header(ds,item,product=source['input_product'])
            if record.get('selected_rgb_band_indices')!=indices:
                raise ValueError('Original declared RGB band selection mismatch')
            rgb=ds.read(indices);valid=np.all(ds.read_masks(indices)==255,axis=0)
        if digest_bytes(rgb.tobytes())!=record['decoded_rgb_sha256'] or digest_bytes(valid.tobytes())!=record['declared_nodata_mask_sha256']:
            raise ValueError('Pinned original EarthView decoded RGB/mask hash mismatch')
        chips[item['id']]=rgb;masks[item['id']]=valid
    plan=validate_mosaic(items,source['source_provenance'],product=source['input_product'])
    rgb,valid=stitch_original(plan,chips,masks)
    if digest_bytes(rgb[:,88:1448,88:1448].tobytes())!=source['decoded_native_context_sha256'] or digest_bytes(rgb[:,128:1408,128:1408].tobytes())!=source['decoded_useful_core_sha256']:
        raise ValueError('Original16-chip mosaic disagrees with captured context/core pixels')
    if source['input_product']=='TOA':
        mode='fixed' if source['rgb_mapping']==TOA_DISPLAY_MAPPING else 'robust'
        display,reproduced=map_toa_mosaic(rgb,valid,mode=mode)
        if reproduced!=source['rgb_mapping']:
            raise ValueError('Original16-file whole-mosaic TOA mapping parameter reproduction mismatch')
    else:
        display=rgb
    return canonical_context(display,valid,plan)


def process_sample(store,row,seed):
    if row.get('source_kind')!='earthview' or row.get('sample_id')!=row['source']['sample_id']:
        raise ValueError('EarthView selection/source identity mismatch')
    if row.get('split')!='train':
        raise ValueError('EarthView expansion must preserve existing evaluation population')
    from scripts.maritime_pilot import deterministic_npz
    from scripts.maritime_ultimate_release import pair_payload
    source=row['source'];recomputed=verify_original_assembly(store,source)
    with ExitStack() as stack:
        memory=stack.enter_context(MemoryFile(store.get(source['native_context'])))
        ds=stack.enter_context(memory.open())
        context,useful,context_hash,core_hash=canonical_from_native(ds,source,store.get(source['hr']))
    if not np.array_equal(context,recomputed):
        raise ValueError('Original assembly canonical HR reproduction mismatch')
    bounds=source['physical_core_bounds'];common={key:value for key,value in row.items() if key!='source'}
    common.update(crs=source['output_crs'],hr_transform=source['hr_core_transform'],
        lr_transform=[10,0,bounds[0],0,-10,bounds[3]],hr_sha256=array_hash(context[:,16:-16,16:-16]),
        context_sha256=array_hash(context),useful_native_rgb_bytes=useful,
        decoded_native_context_sha256=context_hash,decoded_native_core_sha256=core_hash,
        native_gsd_m=1.,original_full_support_verified=False,original_declared_nodata_support_verified=True,
        mask_provenance=source['original_mask_provenance'],mask_caveat=source['mask_caveat'],
        upstream_visual_rendering=source['upstream_visual_rendering'],source_provenance=source,
        native_context=source['native_context'],
        input_product=source['input_product'],source_rgb_dtype=source['source_rgb_dtype'],
        rgb_mapping=source.get('rgb_mapping'),
        native_byte_accounting='decoded original RGB '+source['source_rgb_dtype']+' owned core;16 same-acquisition original files retained; display conversion does not increase information')
    shared=deterministic_npz(dict(context=context,hr=np.ascontiguousarray(context[:,16:-16,16:-16]),
        hr_mask=np.ones((512,512),bool),provenance=np.asarray(json.dumps(common,sort_keys=True))))
    children={name:pair_payload(context,common,name,seed)[0] for name in RECIPE_VERSIONS}
    return common,context,shared,children
