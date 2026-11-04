"""Torch native RGB patches from immutable mother source-window references.

Patch size is in source pixels, preserving the original product spacing.
Set the epoch explicitly for reproducible changing crops. Downloads require GCE;
generation/CRC/SHA checks and a bounded file cache are shared with release code.
"""
import hashlib
import copy
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.windows import Window,transform


def native_patch(ds,core,patch_size,identity,epoch,seed,mask_ds=None,rgb_mapping=None):
    col,row,width,height=core
    if (patch_size<1 or min(width,height)<patch_size or min(col,row)<0
        or col+width>ds.width or row+height>ds.height):
        raise ValueError('Native patch must fit the observed original core')
    t=ds.transform
    if ds.dtypes==('uint16',)*3 and rgb_mapping is None:
        raise ValueError('Original uint16 TOA requires a saved versioned RGB display mapping')
    if ds.dtypes==('uint8',)*3 and rgb_mapping is not None:
        raise ValueError('A TOA mapping cannot be applied to original uint8 RGB')
    if (ds.count!=3 or ds.dtypes not in (('uint8',)*3,('uint16',)*3)
        or ds.colorinterp!=(ColorInterp.red,ColorInterp.green,ColorInterp.blue)
        or ds.crs is None or not ds.crs.is_projected or ds.crs.linear_units_factor[1]!=1
        or t.b!=0 or t.d!=0 or not 0<t.a<=1 or not np.isclose(t.e,-t.a)):
        raise ValueError('Original metric RGB product grid <=1m required')
    entropy=hashlib.sha256(json.dumps([identity,epoch,seed],separators=(',',':')).encode()).digest()
    rng=np.random.default_rng(int.from_bytes(entropy,'little'))
    x=col+int(rng.integers(width-patch_size+1));y=row+int(rng.integers(height-patch_size+1))
    window=Window(x,y,patch_size,patch_size)
    if mask_ds is None:valid=np.all(ds.read_masks([1,2,3],window=window)==255,axis=0)
    else:
        if (mask_ds.shape!=ds.shape or mask_ds.crs!=ds.crs or mask_ds.transform!=ds.transform
            or mask_ds.count!=1 or mask_ds.dtypes!=('uint8',)):
            raise ValueError('Original producer mask grid mismatch')
        valid=mask_ds.read(1,window=window)==255
    if not valid.all():raise ValueError('Original mask does not fully support patch')
    rgb=ds.read([1,2,3],window=window)
    metadata=dict(original_pixel_offset=[x,y],
        patch_transform=list(transform(window,t))[:6],crs=ds.crs.to_string(),product_spacing_m=t.a)
    if rgb.dtype==np.uint16:
        from scripts.maritime_earthview_intake import apply_toa_mapping
        rgb=apply_toa_mapping(rgb,rgb_mapping)
        metadata.update(original_rgb_dtype='uint16',
            output_representation='normalized_uint8_display_rgb_derivative',
            rgb_mapping=copy.deepcopy(rgb_mapping))
    return rgb.astype(np.float32)/np.float32(255),metadata


def validate_earthview_native(ds,source,core):
    """Require the observed original 1360 m context and its saved TOA mapping."""
    from rasterio.crs import CRS
    if (source.get('schema')!='earthview-candidate-v1' or source.get('source_kind')!='earthview'
        or ds.shape!=(1360,1360) or core!=[40,40,1280,1280]
        or source.get('product_spacing_m')!=1.0
        or list(ds.transform)[:6]!=source.get('native_context_transform')
        or ds.crs!=CRS.from_user_input(source['source_crs'])
        or source.get('source_rgb_dtype') not in ('uint8','uint16')
        or ds.dtypes!=(source['source_rgb_dtype'],)*3
        or source.get('original_mask_provenance')!='original_declared_nodata_zero_rgb_band_masks'):
        raise ValueError('Unsupported/inconsistent EarthView original native context')
    t=ds.transform
    context=[t.c,t.f-1360,t.c+1360,t.f]
    bounds=[t.c+40,t.f-1320,t.c+1320,t.f-40]
    if (t.a!=1 or t.e!=-1 or t.b!=0 or t.d!=0
        or not np.allclose(source.get('physical_context_bounds',[]),context,rtol=0,atol=1e-7)
        or not np.allclose(source.get('physical_core_bounds',[]),bounds,rtol=0,atol=1e-7)):
        raise ValueError('EarthView physical context/core geometry disagrees with original grid')
    if source['source_rgb_dtype']=='uint16' and not isinstance(source.get('rgb_mapping'),dict):
        raise ValueError('EarthView uint16 needs its saved full-mosaic display mapping')


def validate_aerial_native(ds,source,core):
    from src.maritime_aerial_source import source_adapter,validate_native_grid
    source_adapter(source).validate_source(source);validate_native_grid(ds,source)
    c,r,_,_=source['source_pixel_window'];x,y,w,h=source['owned_core_pixel_window']
    if core!=[x-c,y-r,w,h]:raise ValueError('Aerial native owned core mismatch')


class NativeMaritimeDataset:
    def __init__(self,manifest,cache,split='train',patch_size=512,seed=20261006,client=None):
        from scripts.maritime_cloud_inventory import require_cloud_vm
        require_cloud_vm()
        from google.cloud import storage
        from scripts.maritime_ultimate_release import CloudStore
        self.manifest=json.loads(Path(manifest).read_text())
        if self.manifest.get('dataset')!='mother' or not self.manifest.get('training_format_verified'):
            raise ValueError('Verified mother manifest required')
        self.samples=[r for r in self.manifest['samples'] if r['split']==split]
        if not self.samples:raise ValueError('Empty native mother split')
        if any(sample.get('source_kind') not in ('skysat','skysat_raw_window','maxar','earthview','aerial_rgb') for sample in self.samples):
            raise ValueError('Unsupported native Mother source kind; an explicit source adapter is required')
        self.store=CloudStore(client or storage.Client(),'gs://unused/native-training-cache',cache)
        self.patch_size=patch_size;self.seed=seed;self.epoch=0

    def __len__(self):return len(self.samples)

    def set_epoch(self,epoch):
        if not isinstance(epoch,int) or epoch<0:raise ValueError('Nonnegative integer epoch required')
        self.epoch=epoch

    def __getitem__(self,index):
        import torch
        sample=self.samples[index];source=sample['source_provenance'];mask=None
        from contextlib import ExitStack
        with ExitStack() as stack:
            if sample.get('native_window_ref'):
                reference=sample['native_window_ref'];pin=reference['object'];core=reference['core_window']
                if reference['mask'].get('object'):
                    mask=stack.enter_context(rasterio.open(self.store.cached_source(reference['mask']['object'])))
            else:
                pin=sample['native_context']
                if source.get('owned_core_pixel_window'):
                    col,row,width,height=source['owned_core_pixel_window'];x,y,_,_=source['source_pixel_window']
                    core=[col-x,row-y,width,height]
                else:
                    core=[50,50,1600,1600]
                    if source['mask'].get('object'):
                        mask=stack.enter_context(rasterio.open(self.store.cached_source(source['mask']['object'])))
            ds=stack.enter_context(rasterio.open(self.store.cached_source(pin)))
            if sample['source_kind']=='earthview':validate_earthview_native(ds,source,core)
            if sample['source_kind']=='aerial_rgb':validate_aerial_native(ds,source,core)
            image,geometry=native_patch(ds,core,self.patch_size,sample['sample_id'],self.epoch,self.seed,mask,
                rgb_mapping=source.get('rgb_mapping') if sample['source_kind']=='earthview' else None)
            original_dtype=ds.dtypes[0]
        self.store.trim_cache()
        return dict(image=torch.from_numpy(image),sample_id=sample['sample_id'],
                    native_gsd_m=sample['native_gsd_m'],product_spacing_m=geometry['product_spacing_m'],
                    source_platform='aerial' if sample['source_kind']=='aerial_rgb' else 'satellite',
                    native_gsd_semantics=sample.get('native_gsd_semantics','documented_satellite_product_grid'),
                    original_native_rgb_dtype=original_dtype,
                    image_representation=geometry.get('output_representation','normalized_original_uint8_product_rgb'),
                    rgb_mapping_json=json.dumps(geometry.get('rgb_mapping'),sort_keys=True))
