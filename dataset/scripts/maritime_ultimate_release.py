"""Fast immutable mixed-native mother and three shared-HR training datasets.

Selection JSON schema ``ultimate-selection-v1`` contains review_policy,
splits.assignment and selection rows: sample_id, source_kind, site_id,
geographic_group_id, split, acquisition_id, country, and source. Source is an
unchanged normalized SkySat candidate, original-source SkySat window, or Maxar
intake record. Optional semantic labels remain unknown when absent. Mother
references generation-pinned native rasters/windows; its TARs carry shared
544-square context and 512-square HR. Children use identical sample IDs/splits.

Bulk imagery processing is GCE-only. No source raster is resized to inflate
native-byte accounting. Full target success is distinct from format validity.
"""
import argparse
from collections import Counter, defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import io
import json
import math
from pathlib import Path
import re
import tarfile
import threading

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import ColorInterp, MaskFlags
from rasterio.io import MemoryFile
from rasterio.transform import Affine, array_bounds
from rasterio.windows import Window, transform as window_transform

from scripts.maritime_cloud_inventory import (create_only, digest_bytes,
    file_sha256, json_bytes, parse_gs, require_cloud_vm, verify_output)
from scripts.maritime_maxar_intake import area_hr
from scripts.maritime_pilot import deterministic_npz
from scripts.maritime_release_build import MASK_PROVENANCE, pinned
from scripts.maritime_release_select import local_pair, near, polygon_wgs84
from src.maritime_dataset import MaritimeDataset
from src.maritime_recipes import RECIPE_VERSIONS, array_hash, canonical_hr, degrade
from src.cloud_config import GCP_PROJECT

# Some original GeoTIFF tags encode 0.8 as float32 (0.800000011920929).
# At 1600 native pixels the nominal 1280m edge differs by 19 micrometres.
# This tolerance concerns coordinate encoding only; native pixels/affines are
# preserved and material overlaps still fail closed.
CORE_BOUND_TOLERANCE_M = 1e-4


def source_geometry(row):
    source=row['source']
    if row['source_kind']=='aerial_rgb':
        from src.maritime_aerial_source import source_geometry as aerial_geometry
        return aerial_geometry(row)
    if row['source_kind']=='earthview':
        from src.maritime_earthview_source import source_geometry as earthview_geometry
        return earthview_geometry(row)
    if row['source_kind']=='academic_raster':
        from src.maritime_academic_source import source_geometry as academic_geometry
        return academic_geometry(row)
    if row['source_kind']=='maxar':
        return source.get('crs') or 'EPSG:'+str(source['source']['epsg']),source['physical_core_bounds']
    return source['crs'],source['core_bounds']


def native_bytes(row):
    if row['source_kind']=='aerial_rgb':
        from src.maritime_aerial_source import source_adapter
        return source_adapter(row['source']).native_bytes(row)
    if row['source_kind']=='earthview':
        from src.maritime_earthview_source import native_bytes as earthview_bytes
        return earthview_bytes(row)
    if row['source_kind']=='academic_raster':
        from src.maritime_academic_source import native_bytes as academic_bytes
        return academic_bytes(row)
    if row['source_kind']!='maxar':return 7680000
    owned=row['source']['owned_core_pixel_window']
    if len(owned)!=4 or any(not isinstance(v,int) or isinstance(v,bool) for v in owned) or min(owned)<0 or min(owned[2:])<1:
        raise ValueError('Invalid owned native pixel window')
    value=owned[2]*owned[3]*3
    if row['source']['useful_native_rgb_bytes']!=value:raise ValueError('Native byte count disagrees with owned pixels')
    return value


def target_gaps(rows,document):
    total=sum(native_bytes(row) for row in rows)
    gaps=[]
    if not 60000000000<=total<=60100000000:gaps.append('native_60gb_target')
    sites={row['site_id'] for row in rows}
    countries={row.get('country') for row in rows if row.get('country') and row.get('country')!='unknown'}
    if len(sites)<30 or len(countries)<3:gaps.append('site_country_diversity')
    for split,minimum in [('validation',5),('test',10)]:
        if len({row['geographic_group_id'] for row in rows if row['split']==split})<minimum:gaps.append(split+'_geographic_group_diversity')
    per_site=Counter()
    for row in rows:per_site[row['site_id']]+=native_bytes(row)
    if total and max(per_site.values())>total*.1:gaps.append('site_byte_share_10percent')
    categories=Counter(row.get('primary_category','unknown') for row in rows)
    targets={'marina_small_craft':.35,'commercial_fishing_port':.25,'coastline':.25,'open_water':.15}
    if any(abs(categories[name]/len(rows)-fraction)>.05 for name,fraction in targets.items()):gaps.append('semantic_category_coverage')
    if sum(row.get('vessel_presence') in ('present','confirmed') for row in rows)<.3*len(rows):gaps.append('vessel_coverage')
    if sum(row.get('small_craft_presence') in ('present','confirmed') for row in rows)<.15*len(rows):gaps.append('small_craft_coverage')
    if not .03<=sum(row.get('hard_negative') is True for row in rows)/len(rows)<=.07:gaps.append('hard_negative_coverage')
    # A declared exception must remain visible, rather than impersonating a
    # completed independent audit or an exhaustive original review policy.
    for gate in ('geography_verified','source_reuse_quality_verified','independent_source_audit',
                 'independent_numeric_audit','independent_size_audit'):
        if document.get('gates',{}).get(gate) is not True:gaps.append(gate)
    return sorted(set(gaps))


def validate_selection(document):
    if document.get('schema')!='ultimate-selection-v1' or not document.get('review_policy'):
        raise ValueError('Versioned selection and explicit review policy required')
    rows=document.get('selection',[])
    if not rows:raise ValueError('Nonempty frozen selection required')
    seen=set();groups={};sites={};captures=defaultdict(list);polygons=defaultdict(list)
    for row in rows:
        identity=row['sample_id']
        if identity in seen:raise ValueError('Duplicate sample ID')
        if not isinstance(identity,str) or not re.fullmatch('[A-Za-z0-9_-]+',identity):raise ValueError('Unsafe sample ID')
        seen.add(identity)
        kind=row['source_kind'];source=row['source']
        if kind not in ('skysat','skysat_raw_window','maxar','academic_raster','earthview','aerial_rgb'):raise ValueError('Unknown source kind')
        if kind=='aerial_rgb':
            from src.maritime_aerial_source import source_adapter
            source_adapter(source).validate_source(source)
        elif kind=='earthview':
            from src.maritime_earthview_source import validate_source as earthview_validate
            earthview_validate(source)
        elif kind=='academic_raster':
            from src.maritime_academic_source import validate_source as academic_validate
            academic_validate(source)
        else:
            gsd=source.get('native_gsd_m')
            if not isinstance(gsd,(int,float)) or not math.isfinite(gsd) or not 0<gsd<=1:raise ValueError('Native source GSD must be documented <=1m')
        if kind=='maxar':
            for key in ('native_context','hr'):pinned(source[key])
            if (source.get('original_mask_provenance')!='original_embedded_dataset_mask'
                or source.get('source',{}).get('event_license')!='CC-BY-NC-4.0'
                or source['source'].get('native_gsd_m')!=gsd):
                raise ValueError('Maxar original mask, source GSD or event license provenance missing')
            native_bytes(row)
        elif kind not in ('academic_raster','earthview','aerial_rgb'):
            pinned(source['source_object' if kind=='skysat_raw_window' else 'mother'])
            if source['mask']['provenance'] not in MASK_PROVENANCE:raise ValueError('Unresolved original mask provenance')
            if source['mask'].get('object'):pinned(source['mask']['object'])
            if kind=='skysat_raw_window':
                core,context=source['core_window'],source['context_window']
                if (len(core)!=4 or len(context)!=4 or core[2:]!=[1600,1600]
                    or context!=[core[0]-50,core[1]-50,1700,1700]
                    or any(not isinstance(v,int) or isinstance(v,bool) or v<0 for v in context)):
                    raise ValueError('1600-square native core and real 50-pixel halo required')
        if row['split'] not in ('train','validation','test'):raise ValueError('Training split unresolved')
        if document['splits']['assignment'].get(row['geographic_group_id'])!=row['split']:raise ValueError('Frozen split assignment mismatch')
        for key,mapping in [('site_id',sites),('geographic_group_id',groups)]:
            value=row[key]
            if not isinstance(value,str) or not value:raise ValueError('Site and group IDs required')
            if value in mapping and mapping[value]!=row['split']:raise ValueError('Cross-split site/group')
            mapping[value]=row['split']
        if not isinstance(row['acquisition_id'],str) or not row['acquisition_id']:raise ValueError('Acquisition identity required')
        crs,bounds=source_geometry(row)
        if len(bounds)!=4 or not np.isfinite(bounds).all() or not np.allclose([bounds[2]-bounds[0],bounds[3]-bounds[1]],[1280,1280],rtol=0,atol=CORE_BOUND_TOLERANCE_M):
            raise ValueError('Physical 1280m core required')
        metric=CRS.from_user_input(crs)
        if not metric.is_projected or metric.linear_units_factor[1]!=1:raise ValueError('Metric CRS required')
        shape=polygon_wgs84(dict(crs=crs,core_bounds=bounds))
        polygons[row['geographic_group_id']].append(shape)
        for old_crs,old_bounds,old_shape in captures[row['acquisition_id']]:
            if metric==CRS.from_user_input(old_crs):
                a,b,c,d=bounds;e,f,g,h=old_bounds
                overlap=(min(c,g)-max(a,e)>CORE_BOUND_TOLERANCE_M
                         and min(d,h)-max(b,f)>CORE_BOUND_TOLERANCE_M)
            else:
                a,b=local_pair(shape,old_shape)
                # An intersection only as wide as the encoded-coordinate
                # tolerance is an edge-rounding sliver, including when the
                # same acquisition is represented in two metric CRSs.
                overlap=a.intersection(b).buffer(-CORE_BOUND_TOLERANCE_M/2).area>1e-4
            if overlap:raise ValueError('Overlapping same-acquisition native cores')
        captures[row['acquisition_id']].append((crs,bounds,shape))
    from shapely.ops import unary_union
    shapes={key:unary_union(value) for key,value in polygons.items()}
    names=sorted(shapes)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            if groups[a]!=groups[b] and near(shapes[a],shapes[b],4000):raise ValueError('Cross-split 2km geographic buffers overlap')
    # Group original-source windows to download each raster once. IDs and shard
    # ordering remain deterministic independently of thread completion order.
    def order(row):
        s=row['source'];ref=s.get('source_object') or s.get('mother') or s.get('native_context')
        return row['split'],ref['uri'],str(ref['generation']),row['sample_id']
    return sorted(rows,key=order),target_gaps(rows,document)


def read_skysat(ds,source,mask_ds=None,raw=False):
    from scripts.maritime_cloud_inventory import validate_header
    validate_header(ds)
    window=Window(*source['context_window']) if raw else None
    transform=window_transform(window,ds.transform) if raw else ds.transform
    if (CRS.from_user_input(source['crs'])!=ds.crs or list(transform)[:6]!=source['context_transform']
        or (not raw and ds.shape!=(1700,1700))):raise ValueError('Original native context grid mismatch')
    if raw and (window.col_off<0 or window.row_off<0 or window.col_off+1700>ds.width or window.row_off+1700>ds.height):
        raise ValueError('Real halo outside original source')
    if not np.allclose(array_bounds(1600,1600,transform*Affine.translation(50,50)),source['core_bounds'],rtol=0,atol=1e-6):
        raise ValueError('Original native core bounds mismatch')
    if mask_ds is not None:
        if (mask_ds.shape!=ds.shape or mask_ds.count!=1 or mask_ds.dtypes!=('uint8',)
            or mask_ds.crs!=ds.crs or mask_ds.transform!=ds.transform):raise ValueError('Producer original mask grid mismatch')
        values=mask_ds.read(1,window=window)
        if not np.isin(values,[0,255]).all():raise ValueError('Producer mask must be binary')
        valid=values==255
    else:
        if source['mask']['provenance'] not in {'original_explicit_dataset_mask','original_embedded_alpha_mask',
            'original_embedded_dataset_mask','legacy_nodata_zero_conservative','documented_unmask_zero_nodata_zero_conservative'}:
            raise ValueError('Separate producer mask required by source provenance')
        if source['mask']['provenance'].startswith('original_') and not all(
            MaskFlags.per_dataset in flags or MaskFlags.alpha in flags for flags in ds.mask_flag_enums):
            raise ValueError('Declared original mask is absent from native raster')
        valid=np.all(ds.read_masks([1,2,3],window=window)==255,axis=0)
    rgb=ds.read([1,2,3],window=window)
    if rgb.shape!=(3,1700,1700) or valid.shape!=(1700,1700) or not valid.all():raise ValueError('Original core and halo full support required')
    return rgb,valid


def read_maxar(ds,source,stored):
    spacing=source['product_spacing_m'];crs,_=source_geometry(dict(source_kind='maxar',source=source))
    transform=ds.transform
    if (ds.count!=3 or ds.dtypes!=('uint8',)*3 or ds.colorinterp!=(ColorInterp.red,ColorInterp.green,ColorInterp.blue)
        or ds.crs!=CRS.from_user_input(crs) or not 0<spacing<=1
        or transform.a!=spacing or transform.e!=-spacing or transform.b!=0 or transform.d!=0):
        raise ValueError('Original Maxar native RGB/grid mismatch')
    if not all(MaskFlags.per_dataset in flags and MaskFlags.nodata not in flags for flags in ds.mask_flag_enums):
        raise ValueError('Original Maxar embedded mask missing')
    sx,sy,sw,sh=source['source_pixel_window']
    expected=Affine(*source['original_transform'])*Affine.translation(sx,sy)
    if list(transform)[:6]!=list(expected)[:6] or ds.shape!=(sh,sw):raise ValueError('Maxar source native window grid mismatch')
    left,bottom,right,top=source['physical_core_bounds']
    context=[left-40,bottom-40,right+40,top+40]
    phase=[context[0]-transform.c,transform.f-context[3]]
    if source['physical_context_bounds']!=context or not np.allclose(phase,source['area_phase_m'],rtol=0,atol=1e-7):raise ValueError('Maxar physical area phase mismatch')
    inv=~transform;col,row=inv*(left,top);endcol,endrow=inv*(right,bottom)
    x,y=math.ceil(col-.5-1e-9),math.ceil(row-.5-1e-9)
    w,h=math.ceil(endcol-.5-1e-9)-x,math.ceil(endrow-.5-1e-9)-y
    if source['owned_core_pixel_window']!=[x+sx,y+sy,w,h] or min(x,y)<0 or x+w>sw or y+h>sh:
        raise ValueError('Maxar owned native core accounting mismatch')
    rgb=ds.read();valid=np.all(ds.read_masks()==255,axis=0)
    result=area_hr(rgb,valid,source)
    with np.load(io.BytesIO(stored),allow_pickle=False) as arrays:
        if not np.array_equal(arrays['hr_context'],result) or not np.array_equal(arrays['hr'],result[:,16:-16,16:-16]):
            raise ValueError('Stored Maxar canonical HR disagrees with original source pixels')
    useful=rgb[:,y:y+h,x:x+w]
    context_hash=digest_bytes(rgb.tobytes());core_hash=digest_bytes(useful.tobytes())
    if context_hash!=source['decoded_native_context_sha256'] or core_hash!=source['decoded_useful_core_sha256']:
        raise ValueError('Decoded original Maxar pixel hash mismatch')
    return result,w*h*3,context_hash,core_hash


def common_sample(row,context,useful_bytes,native_context_hash,native_core_hash):
    crs,bounds=source_geometry(row)
    source=row['source']
    result={key:row.get(key) for key in ('sample_id','source_kind','site_id','geographic_group_id',
        'split','acquisition_id','country','primary_category','vessel_presence','small_craft_presence','hard_negative')}
    result.update(crs=crs,core_bounds=bounds,hr_transform=[2.5,0,bounds[0],0,-2.5,bounds[3]],
        lr_transform=[10,0,bounds[0],0,-10,bounds[3]],hr_sha256=array_hash(context[:,16:-16,16:-16]),
        context_sha256=array_hash(context),useful_native_rgb_bytes=useful_bytes,
        decoded_native_context_sha256=native_context_hash,decoded_native_core_sha256=native_core_hash,
        native_gsd_m=source['native_gsd_m'],original_full_support_verified=True,
        native_byte_accounting='decoded original uint8 RGB core pixels; no halos or training/storage bytes',
        source_provenance=source)
    if row['source_kind'] in ('skysat','skysat_raw_window'):
        result['coordinate_encoding']=dict(
            native_context_transform=source['context_transform'],
            native_core_span_m=[bounds[2]-bounds[0],bounds[3]-bounds[1]],
            nominal_native_spacing_m=.8,nominal_core_span_m=1280.,
            encoded_native_spacing_m=[source['context_transform'][0],-source['context_transform'][4]],
            core_edge_delta_from_nominal_m=[bounds[2]-bounds[0]-1280,bounds[3]-bounds[1]-1280],
            coordinate_encoding_tolerance_m=CORE_BOUND_TOLERANCE_M,
            canonical_hr_coordinate_basis='nominal 0.8m pixel grid to 2.5m HR; original encoded affine retained without native resampling')
    if row['source_kind']=='skysat_raw_window':
        result['native_window_ref']=dict(object=source['source_object'],context_window=source['context_window'],
            core_window=source['core_window'],mask=source['mask'])
    else:result['native_context']=source.get('native_context') or source['mother']
    return result


def pair_payload(context,common,recipe,seed):
    pair=degrade(context,common['sample_id'],recipe,release_seed=seed)
    metadata={key:common[key] for key in ('sample_id','split','site_id','geographic_group_id','crs','hr_transform','lr_transform','hr_sha256')}
    metadata.update(recipe=pair['metadata'],lr_origin='synthetic_hr_only',context_sha256=common['context_sha256'])
    payload=deterministic_npz(dict(hr=np.ascontiguousarray(context[:,16:-16,16:-16]),lr=pair['lr'],
        hr_mask=pair['hr_mask'],lr_mask=pair['lr_mask'],provenance=np.asarray(json.dumps(metadata,sort_keys=True))))
    return payload,metadata


def tar_member(archive,name,payload):
    entry=tarfile.TarInfo(name);entry.size=len(payload);entry.mode=0o644
    entry.mtime=0;entry.uid=0;entry.gid=0;entry.uname='';entry.gname=''
    archive.addfile(entry,io.BytesIO(payload))


def verify_shard(path,recipe,recompute_contexts,seed):
    dataset=MaritimeDataset([path],recipe)
    count=0
    for index in range(len(dataset)):
        arrays,meta=dataset.read_arrays(index)
        identity=meta['sample_id']
        if identity in recompute_contexts:
            expected=degrade(recompute_contexts[identity],identity,recipe,release_seed=seed)
            if (not np.array_equal(expected['lr'],arrays['lr'])
                or not np.array_equal(recompute_contexts[identity][:,16:-16,16:-16],arrays['hr'])
                or expected['metadata']!=meta['recipe']):raise ValueError('Serialized deterministic recipe mismatch')
            count+=1
    # Exercise actual PyTorch collation/tensor operations on a bounded batch;
    # all samples were separately decoded and hash/shape checked above.
    import torch
    from torch.utils.data import DataLoader, Subset
    batch=next(iter(DataLoader(Subset(dataset,range(min(2,len(dataset)))),batch_size=2,num_workers=0)))
    if not torch.isfinite(batch['hr'].mean()) or not torch.isfinite(batch['lr'].mean()):raise ValueError('Torch loader smoke failed')
    return dict(verified_samples=len(dataset),numeric_recomputed_samples=count,
        loader='torch.utils.data.DataLoader',torch_version=torch.__version__,
        hr_shape=list(batch['hr'].shape),lr_shape=list(batch['lr'].shape))


def release_state(format_verified,gaps):
    full=format_verified and not gaps
    return dict(status='FULL_TARGET_TRAINING_READY' if full else 'PROVISIONAL_TRAINING_READY',
        training_format_verified=format_verified,full_target_passed=full,strict_target_gaps=gaps)


class CloudStore:
    def __init__(self,client,prefix,scratch):
        self.client=client;self.bucket,self.root=parse_gs(prefix.rstrip('/'));self.prefix=prefix.rstrip('/')
        self.scratch=Path(scratch);self.scratch.mkdir(parents=True,exist_ok=True)
        self.cache=OrderedDict();self.lock=threading.Lock();self.cache_bytes=0

    def get(self,ref):return verify_output(self.client,pinned(ref))

    def publish(self,name,data):
        ref=create_only(self.client.bucket(self.bucket).blob(self.root+'/'+name),data)
        return dict(uri=self.prefix+'/'+name,**ref)

    def existing(self,name):
        blob=self.client.bucket(self.bucket).blob(self.root+'/'+name)
        if not blob.exists():return None
        blob.reload();payload=blob.download_as_bytes(if_generation_match=int(blob.generation),checksum='crc32c')
        return json.loads(payload)

    def cached_source(self,ref):
        pinned(ref);key=(ref['uri'],str(ref['generation']),ref['sha256'])
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key);return self.cache[key]
            path=self.scratch/('source-'+ref['sha256']+'.tif')
            if not path.exists() or path.stat().st_size!=ref['bytes'] or file_sha256(path)!=ref['sha256']:
                bucket,name=parse_gs(ref['uri']);blob=self.client.bucket(bucket).blob(name,generation=int(ref['generation']))
                blob.reload()
                if int(blob.size)!=ref['bytes'] or blob.crc32c!=ref['crc32c']:raise ValueError('Original source metadata pin mismatch')
                temporary=path.with_suffix('.partial')
                blob.download_to_filename(str(temporary),if_generation_match=int(ref['generation']),checksum='crc32c')
                if temporary.stat().st_size!=ref['bytes'] or file_sha256(temporary)!=ref['sha256']:raise ValueError('Original source file SHA/size mismatch')
                temporary.replace(path)
            self.cache[key]=path;self.cache_bytes+=ref['bytes']
            return path

    def trim_cache(self):
        # Called after worker completion: never evict a file still being read.
        while self.cache_bytes>2000000000 and len(self.cache)>1:
            key,path=self.cache.popitem(last=False);self.cache_bytes-=path.stat().st_size;path.unlink()


def process_sample(store,row,seed):
    if row['source_kind']=='aerial_rgb':
        from src.maritime_aerial_source import source_adapter
        return source_adapter(row['source']).process_sample(store,row,seed)
    if row['source_kind']=='earthview':
        from src.maritime_earthview_source import process_sample as earthview_process
        return earthview_process(store,row,seed)
    if row['source_kind']=='academic_raster':
        from src.maritime_academic_source import process_sample as academic_process
        return academic_process(store,row,seed)
    source=row['source'];kind=row['source_kind']
    with ExitStack() as stack:
        if kind=='maxar':
            memory=stack.enter_context(MemoryFile(store.get(source['native_context'])))
            ds=stack.enter_context(memory.open())
            context,useful,context_hash,core_hash=read_maxar(ds,source,store.get(source['hr']))
        else:
            raw=kind=='skysat_raw_window'
            if raw:ds=stack.enter_context(rasterio.open(store.cached_source(source['source_object'])))
            else:
                memory=stack.enter_context(MemoryFile(store.get(source['mother'])))
                ds=stack.enter_context(memory.open())
            mask=None
            if source['mask'].get('object'):
                if raw:mask=stack.enter_context(rasterio.open(store.cached_source(source['mask']['object'])))
                else:
                    memory=stack.enter_context(MemoryFile(store.get(source['mask']['object'])))
                    mask=stack.enter_context(memory.open())
            rgb,valid=read_skysat(ds,source,mask,raw)
            context,_=canonical_hr(rgb,valid);useful=7680000
            context_hash=digest_bytes(rgb.tobytes());core_hash=digest_bytes(rgb[:,50:-50,50:-50].tobytes())
    common=common_sample(row,context,useful,context_hash,core_hash)
    shared=deterministic_npz(dict(context=context,hr=np.ascontiguousarray(context[:,16:-16,16:-16]),
        hr_mask=np.ones((512,512),bool),provenance=np.asarray(json.dumps(common,sort_keys=True))))
    children={name:pair_payload(context,common,name,seed)[0] for name in RECIPE_VERSIONS}
    return common,context,shared,children


def run(args):
    instance=require_cloud_vm()
    import torch
    torch.set_num_threads(1)
    from google.cloud import storage
    payload=Path(args.selection).read_bytes();document=json.loads(payload)
    rows,gaps=validate_selection(document)
    client=storage.Client(project=GCP_PROJECT)
    store=CloudStore(client,args.output_prefix,args.scratch)
    config=dict(schema='ultimate-release-v1',selection_sha256=digest_bytes(payload),
        worker_sha256=file_sha256(__file__),code_sha256=args.code_sha256,image_id=args.image_id,
        release_seed=args.release_seed,shard_samples=args.shard_samples,workers=args.workers,
        recipes=RECIPE_VERSIONS,review_policy=document['review_policy'],splits=document['splits'],
        mother_format='generation-pinned native RGB context/window references plus shared HR TAR shards',
        target_native_rgb_bytes=[60000000000,60100000000])
    config['coordinate_encoding_tolerance_m']=CORE_BOUND_TOLERANCE_M
    config_sha=digest_bytes(json_bytes(config))
    store.publish('configuration.json',json_bytes(config));selection_ref=store.publish('frozen-selection.json',payload)
    stratified=defaultdict(list)
    for row in rows:stratified[(row['split'],row.get('primary_category','unknown'),row['source_kind'])].append(row['sample_id'])
    recompute={identity for identities in stratified.values() for identity in sorted(identities,key=lambda value:digest_bytes(value.encode()))[:math.ceil(len(identities)*.05)]}
    # Batch boundaries also split at original-source boundaries for bounded
    # cache ownership and efficient original raster reuse across many windows.
    batches=[];current=[];previous=None
    for row in rows:
        source=row['source'];ref=source.get('source_object') or source.get('mother') or source.get('native_context')
        key=(row['split'],ref['uri'],str(ref['generation'])) if row['source_kind']=='skysat_raw_window' else (row['split'],'materialized')
        if current and (len(current)==args.shard_samples or key!=previous):batches.append(current);current=[]
        current.append(row);previous=key
    if current:batches.append(current)
    completed=[];samples=[];decoded_seen={}
    with ThreadPoolExecutor(max_workers=args.workers) as workers:
        for index,batch in enumerate(batches):
            receipt_name=f'control/batches/{index:06d}.json'
            receipt=store.existing(receipt_name)
            if receipt is not None:
                if receipt['configuration_sha256']!=config_sha or receipt['sample_ids']!=[row['sample_id'] for row in batch]:raise ValueError('Immutable batch resume conflict')
                # Published shards already passed serialization/loader checks;
                # generation + CRC metadata checks suffice for bounded resume.
                for ref in receipt['objects'].values():
                    bucket,name=parse_gs(ref['uri']);blob=client.bucket(bucket).blob(name,generation=int(ref['generation']));blob.reload()
                    if str(blob.generation)!=str(ref['generation']) or int(blob.size)!=ref['bytes'] or blob.crc32c!=ref['crc32c']:raise ValueError('Resume shard metadata mismatch')
            else:
                # Two native readers may run concurrently, each with independent
                # raster handles. No GDAL auth globals or shared handles mutate.
                processed=list(workers.map(lambda row:process_sample(store,row,args.release_seed),batch))
                objects={};reports={};contexts={item[0]['sample_id']:item[1] for item in processed if item[0]['sample_id'] in recompute}
                paths={name:store.scratch/f'{name}-{index:06d}.tar' for name in ['mother',*RECIPE_VERSIONS]}
                try:
                    with ExitStack() as stack:
                        archives={name:stack.enter_context(tarfile.open(path,'w')) for name,path in paths.items()}
                        for common,context,shared,children in processed:
                            tar_member(archives['mother'],common['sample_id']+'.npz',shared)
                            for name,pair in children.items():tar_member(archives[name],common['sample_id']+'.npz',pair)
                    for recipe in RECIPE_VERSIONS:reports[recipe]=verify_shard(paths[recipe],recipe,contexts,args.release_seed)
                    # Shared arrays are independently reopened after TAR/NPZ
                    # serialization and compared to the frozen decoded hashes.
                    with tarfile.open(paths['mother'],'r') as archive:
                        for common,context,_,_ in processed:
                            with np.load(io.BytesIO(archive.extractfile(common['sample_id']+'.npz').read()),allow_pickle=False) as arrays:
                                if array_hash(arrays['context'])!=common['context_sha256'] or array_hash(arrays['hr'])!=common['hr_sha256'] or not arrays['hr_mask'].all():raise ValueError('Serialized shared mother hash/support mismatch')
                    for name,path in paths.items():objects[name]=store.publish(f'{name}/shards/{index:06d}.tar',path.read_bytes())
                    receipt=dict(state='COMPLETE',configuration_sha256=config_sha,
                        sample_ids=[row['sample_id'] for row in batch],samples=[item[0] for item in processed],
                        split=batch[0]['split'],objects=objects,verification=reports)
                    store.publish(receipt_name,json_bytes(receipt))
                finally:
                    for path in paths.values():path.unlink(missing_ok=True)
                del processed,contexts
            for sample in receipt['samples']:
                decoded=sample['decoded_native_core_sha256']
                if decoded in decoded_seen:raise ValueError('Duplicate decoded native core pixels: '+decoded_seen[decoded]+' and '+sample['sample_id'])
                decoded_seen[decoded]=sample['sample_id']
            completed.append(receipt);samples.extend(receipt['samples']);store.trim_cache()
            print(json.dumps(dict(completed_batches=len(completed),samples=len(samples),
                useful_native_rgb_bytes=sum(item['useful_native_rgb_bytes'] for item in samples))),flush=True)
    total=sum(sample['useful_native_rgb_bytes'] for sample in samples)
    if total!=sum(native_bytes(row) for row in rows):raise ValueError('Published verified native byte accounting mismatch')
    state=release_state(True,gaps)
    manifests={}
    for name in ['mother',*RECIPE_VERSIONS]:
        shards=[dict(object=receipt['objects'][name],split=receipt['split'],sample_ids=receipt['sample_ids']) for receipt in completed]
        manifest=dict(schema='ultimate-dataset-v1',dataset=name,**state,configuration=config,
            selection=selection_ref,samples=samples,shards=shards,useful_native_rgb_bytes=total,
            sample_count=len(samples),recipe=RECIPE_VERSIONS.get(name),
            sample_id_split_and_hr_shared_across_recipes=True)
        manifests[name]=store.publish(name+'/manifest.json',json_bytes(manifest))
        card=(f'# Maritime RGB {name}\n\n{state["status"]}. {len(samples)} samples. '
            f'{total} useful original uint8 RGB core bytes; halos, duplicates, float arrays, '
            'archives and synthetic LR are excluded. Native source GSD is <=1 m; '
            'product grid spacing is separately documented. Shared HR is 512x512 RGB '
            'float32 at 2.5 m, child LR is 128x128 at 10 m.\n\n'
            'The mother references immutable native source raster windows; shared TARs contain '
            'the common HR and real 544x544 context. Original masks are verified before recipes. '
            'Children are compatible with src.maritime_dataset.MaritimeDataset. Download '
            'the generation-pinned TARs to training storage and instantiate that loader with '
            'recipe d1, d2, or d3. Use manifest sample split assignments, rather than random '
            'splitting across shards.\n\nReview policy: '+json.dumps(document['review_policy'],sort_keys=True)+
            '\n\nStrict target gaps: '+', '.join(gaps)+
            '\n\nLicenses and radiometric/compression history remain in each sample source_provenance. '
            'Use only under the applicable source research license; Maxar RGB products can be '
            'JPEG-compressed visual imagery. Missing semantic labels are unknown. '
            'Numeric recomputation covers at least 5% in each split/category/source stratum; '
            'all serialized child samples are decoded and checked for hashes, shapes, masks '
            'and coordinate alignment. A bounded real PyTorch batch is collated per shard.\n')
        store.publish(name+'/DATASET.md',card.encode())
    receipt=dict(schema='ultimate-release-receipt-v1',state='COMPLETE',**state,
        configuration_sha256=config_sha,configuration=config,vm_instance_id=instance,
        manifests=manifests,sample_count=len(samples),useful_native_rgb_bytes=total,
        verification=dict(all_serialized_samples_verified=True,real_torch_loader=True,
            source_pixels_recomputed=True,numeric_recomputed_samples_per_recipe=len(recompute),
            independent_audit=False),split_counts=dict(Counter(sample['split'] for sample in samples)))
    ref=store.publish('release-receipt.json',json_bytes(receipt))
    print(json.dumps(dict(receipt=ref,status=state['status'],useful_native_rgb_bytes=total)),flush=True)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',required=True)
    parser.add_argument('--output-prefix',required=True)
    parser.add_argument('--scratch',required=True)
    parser.add_argument('--image-id',required=True)
    parser.add_argument('--code-sha256',required=True)
    parser.add_argument('--release-seed',type=int,default=20261006)
    parser.add_argument('--shard-samples',type=int,default=32)
    parser.add_argument('--workers',type=int,choices=range(1,9),default=2)
    args=parser.parse_args()
    if not 1<=args.shard_samples<=32:parser.error('shard-samples must be 1..32')
    if not re.fullmatch('[a-f0-9]{64}',args.code_sha256):parser.error('code-sha256 must be SHA256')
    run(args)


if __name__=='__main__':main()
