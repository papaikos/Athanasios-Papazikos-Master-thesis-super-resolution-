"""Build immutable mother/shared and isolated recipe releases on a GCE VM.

Run mother, each recipe, then finalize separately. Provisional output never
authorizes a full-size mother or training-ready release.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import io
import json
import math
from pathlib import Path
import re
import tarfile
import tempfile
import uuid

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine, array_bounds

from scripts.maritime_cloud_inventory import (create_only, digest_bytes, file_sha256,
    json_bytes, parse_gs, require_cloud_vm, validate_header, verify_output)
from scripts.maritime_pilot import deterministic_npz, validate_pair, write_geotiff
from scripts.maritime_release_select import accepted_review, local_pair, near, polygon_wgs84
from src.maritime_dataset import MaritimeDataset, smoke_batch
from src.maritime_recipes import RECIPE_VERSIONS, array_hash, canonical_hr, degrade

MOTHER = 'skysat-maritime-mother-v1'
CHILDREN = {name: 'skysat-maritime-'+name+'-v1' for name in RECIPE_VERSIONS}
COMMON = ('sample_id','hr_sha256','context_sha256','site_id','geographic_group_id',
          'split','core_bounds','crs','hr_transform')
MASK_PROVENANCE = {'original_explicit_dataset_mask','original_embedded_alpha_mask',
    'original_embedded_dataset_mask','legacy_nodata_zero_conservative',
    'documented_unmask_zero_nodata_zero_conservative','producer_original_min_rgb_mask',
    'original_udm_all_unusable_bits_zero','original_udm2_all_unusable_bits_zero'}


def pinned(ref):
    if not isinstance(ref,dict) or not all(key in ref for key in ('uri','generation','bytes','crc32c','sha256')):
        raise ValueError('Complete generation/checksum/SHA pinned object required')
    parse_gs(ref['uri'])
    if int(ref['generation'])<1 or ref['bytes']<1 or not re.fullmatch('[0-9a-f]{64}',ref['sha256']):
        raise ValueError('Invalid immutable object pin')
    return ref


def validate_selection(document, provisional):
    rows=document.get('selection',[])
    if not rows: raise ValueError('A positive fully reviewed selection is required')
    # The selector deliberately leaves these downstream gates unresolved;
    # independent evidence below replaces only those two placeholders.
    limitations=[name for name in document.get('gates',{}).get('missing',[])
                 if name not in ('independent_pixel_size_audit','published_mother_integrity_audit')]
    if document.get('gates',{}).get('metadata_selection_passed') is not True:
        limitations.append('metadata_selection_passed')
    reviewers={row.get('review_decision',{}).get('reviewer') for row in rows}
    for name in ('numeric','source','size'):
        audit=document.get('independent_audits',{}).get(name,{})
        good=(audit.get('passed') is True and audit.get('independent') is True
              and audit.get('reviewer') and audit['reviewer'] not in reviewers)
        try: pinned(audit.get('evidence'))
        except (ValueError,TypeError,KeyError): good=False
        if not good: limitations.append('independent_'+name+'_audit')
    release_rows=[row for row in rows if row.get('split')!='development']
    total=len(release_rows)*7680000
    if len(release_rows)!=7813 or not 60000000000<=total<=60100000000:
        limitations.append('unique_native_core_size_target')
    seen=set(); acquisitions=defaultdict(list); groups={}; sites={}; footprints=defaultdict(list)
    for row in rows:
        identity=row['canonical_core_id']
        if identity in seen: raise ValueError('Duplicate canonical sample identity')
        if not re.fullmatch('[A-Za-z0-9_-]+',identity): raise ValueError('Unsafe sample identity')
        seen.add(identity)
        reason=accepted_review(row,row.get('review_decision'))
        if reason or row.get('review_status')!='accepted_native_review':
            raise ValueError('Incomplete native review: '+str(reason))
        if row['mask']['provenance'] not in MASK_PROVENANCE:
            raise ValueError('Unresolved original mask support provenance')
        if row.get('split') not in ('train','validation','test','development'):
            raise ValueError('Unresolved split')
        for key, mapping in [('geographic_group_id',groups),('site_id',sites)]:
            value=row[key]
            if value in mapping and mapping[value]!=row['split']: raise ValueError('Cross-split site/group')
            mapping[value]=row['split']
        if document['splits']['assignment'].get(row['geographic_group_id'])!=row['split']:
            raise ValueError('Frozen split assignment mismatch')
        footprints[row['geographic_group_id']].append(polygon_wgs84(row))
        affine=Affine(*row['core_transform']); context=Affine(*row['context_transform'])
        if not np.allclose(array_bounds(1600,1600,affine),row['core_bounds'],rtol=0,atol=1e-6):
            raise ValueError('Core affine and source bounds mismatch')
        if not np.allclose(tuple(context*Affine.translation(50,50))[:6],tuple(affine)[:6],rtol=0,atol=1e-6):
            raise ValueError('Core/context source bounds mismatch')
        for previous in acquisitions[row['acquisition_id']]:
            if CRS.from_user_input(row['crs'])==CRS.from_user_input(previous['crs']):
                a,b,c,d=row['core_bounds']; e,f,g,h=previous['core_bounds']
                overlap=min(c,g)-max(a,e)>1e-6 and min(d,h)-max(b,f)>1e-6
            else:
                a,b=local_pair(polygon_wgs84(row),polygon_wgs84(previous))
                overlap=a.intersection(b).area>1e-4
            if overlap: raise ValueError('Overlapping same-acquisition source bounds')
        acquisitions[row['acquisition_id']].append(row)
    from shapely.ops import unary_union
    shapes={key:unary_union(value) for key,value in footprints.items()}
    names=sorted(shapes)
    for index,a in enumerate(names):
        for b in names[index+1:]:
            if groups[a]!=groups[b] and near(shapes[a],shapes[b],4000):
                raise ValueError('Cross-split geographic buffer/source overlap')
    limitations=sorted(set(limitations))
    if limitations and not provisional:
        raise ValueError('Full release gates failed; explicit --provisional required: '+','.join(limitations))
    return sorted(rows,key=lambda row:row['canonical_core_id']),limitations


def read_native(dataset,row,mask_dataset=None):
    validate_header(dataset)
    if dataset.shape!=(1700,1700) or CRS.from_user_input(row['crs'])!=dataset.crs:
        raise ValueError('Native mother shape/CRS mismatch')
    if tuple(dataset.transform)[:6]!=tuple(row['context_transform']):
        raise ValueError('Native mother exact affine mismatch')
    if mask_dataset is not None:
        if (mask_dataset.shape!=dataset.shape or mask_dataset.count!=1 or mask_dataset.dtypes!=('uint8',)
            or mask_dataset.crs!=dataset.crs or mask_dataset.transform!=dataset.transform):
            raise ValueError('Native separate mask grid mismatch')
        values=mask_dataset.read(1)
        if not np.isin(values,[0,255]).all(): raise ValueError('Nonbinary original support mask')
        support=values==255
    else:
        allowed=('original_explicit_dataset_mask','original_embedded_alpha_mask',
                 'original_embedded_dataset_mask','legacy_nodata_zero_conservative',
                 'documented_unmask_zero_nodata_zero_conservative')
        if row['mask']['provenance'] not in allowed:
            raise ValueError('Embedded mask lacks documented original support provenance')
        support=np.all(dataset.read_masks()==255,axis=0)
    if not support.all(): raise ValueError('Mother core and real halo require full support')
    return dataset.read(),support


def recompute_ids(rows,pilot=False):
    if pilot: return {row['sample_id'] for row in rows}
    strata=defaultdict(list)
    for row in rows:
        strata[(row['split'],row.get('primary_category'),row.get('vessel_presence'),
                row.get('small_craft_presence'))].append(row['sample_id'])
    return {identity for members in strata.values()
            for identity in sorted(members,key=lambda s:digest_bytes(s.encode()))[:math.ceil(.05*len(members))]}


def validate_children(mother,children):
    expected={row['sample_id']:row for row in mother['samples']}
    if set(children)!=set(RECIPE_VERSIONS): raise ValueError('All three recipes required')
    for recipe,child in children.items():
        if child.get('state')!='COMPLETE' or child['recipe']!=recipe: raise ValueError('Active/incomplete recipe task')
        actual={row['sample_id']:row for row in child['samples']}
        if len(actual)!=len(child['samples']) or set(actual)!=set(expected):
            raise ValueError('Child sample coverage mismatch')
        for identity,row in actual.items():
            if any(row.get(key)!=expected[identity].get(key) for key in COMMON):
                raise ValueError('Common frozen HR/site/split/bounds mismatch')
        if 'shards' in child:
            members=[identity for shard in child['shards'] for identity in shard['sample_ids']]
            if len(members)!=len(expected) or set(members)!=set(expected):
                raise ValueError('Shard sample coverage mismatch')


class Store:
    def __init__(self,client,prefix):
        self.client=client; self.prefix=prefix.rstrip('/')
        self.bucket,self.root=parse_gs(self.prefix)

    def put(self,name,payload):
        blob=self.client.bucket(self.bucket).blob(self.root+'/'+name)
        return dict(uri=self.prefix+'/'+name,**create_only(blob,payload))

    def get(self,ref): return verify_output(self.client,pinned(ref))

    def existing(self,name):
        blob=self.client.bucket(self.bucket).blob(self.root+'/'+name)
        if not blob.exists(): return None
        blob.reload()
        data=blob.download_as_bytes(if_generation_match=int(blob.generation),checksum='crc32c')
        return dict(uri=self.prefix+'/'+name,generation=str(blob.generation),bytes=len(data),
                    crc32c=blob.crc32c,sha256=digest_bytes(data))

    def document(self,name):
        ref=self.existing(name)
        if ref is None: raise ValueError('Missing completed artifact: '+name)
        return json.loads(self.get(ref)),ref


def read_shared(store,row):
    with np.load(io.BytesIO(store.get(row['shared'])),allow_pickle=False) as data:
        context=np.array(data['context']); hr=np.array(data['hr']); mask=np.array(data['hr_mask'])
    if (context.shape!=(3,544,544) or hr.shape!=(3,512,512) or context.dtype!=np.float32
        or hr.dtype!=np.float32 or mask.shape!=(512,512) or mask.dtype!=np.bool_ or not mask.all()
        or not np.isfinite(context).all() or (context<0).any() or (context>1).any()
        or not np.array_equal(hr,context[:,16:-16,16:-16])
        or array_hash(context)!=row['context_sha256'] or array_hash(hr)!=row['hr_sha256']):
        raise ValueError('Frozen shared input hash/array mismatch')
    return context,hr,mask


def restore_receipt(store,name,config_sha):
    ref=store.existing(name)
    if ref is None: return None
    receipt=json.loads(store.get(ref))
    if receipt['config_sha256']!=config_sha or receipt['state']!='COMPLETE':
        raise ValueError('Receipt configuration/state conflict')
    for artifact in receipt['objects'].values(): store.get(artifact)
    return receipt


def parquet_bytes(rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    simple=[dict(sample_id=row['sample_id'],site_id=row['site_id'],split=row['split'],
        hr_sha256=row['hr_sha256'],core_bytes=7680000,provenance=json.dumps(row,sort_keys=True)) for row in rows]
    output=io.BytesIO(); pq.write_table(pa.Table.from_pylist(simple),output,compression='zstd')
    return output.getvalue()


def source_raster_spacing(row):
    """Observed/grid spacing is separate from the vendor's acquisition GSD."""
    grid=row.get('source_grid') or {}
    terms=(row.get('source_raster_transform') or grid.get('crs_transform')
           or grid.get('original_transform') or grid.get('affine'))
    if terms is None and grid.get('affineTransform'):
        affine=grid['affineTransform']
        terms=[affine['scaleX'],affine.get('shearX',0),affine.get('translateX',0),
               affine.get('shearY',0),affine['scaleY'],affine.get('translateY',0)]
    if terms is None: return None
    definition=grid.get('crs') or grid.get('crsWkt') or grid.get('wkt') or row['crs']
    crs=CRS.from_user_input(definition)
    if not crs.is_projected: return None
    a,b,_,d,e,_=terms[:6]
    factor=crs.linear_units_factor[1]
    return [math.hypot(a,d)*factor,math.hypot(b,e)*factor]


def build_mother(store,document,rows,config,config_sha,scratch):
    samples=[]
    for selected in rows:
        identity=selected['canonical_core_id']; name='control/mother/'+identity+'.json'
        receipt=restore_receipt(store,name,config_sha)
        if receipt:
            sample=receipt['sample']; read_shared(store,sample); samples.append(sample); continue
        with tempfile.TemporaryDirectory(dir=scratch) as work:
            work=Path(work); rgbpath=work/'source.tif'; rgbpath.write_bytes(store.get(selected['mother']))
            maskref=selected['mask']['object']; separate=selected['mask']['kind']=='separate_validity'
            with ExitStack() as stack:
                ds=stack.enter_context(rasterio.open(rgbpath))
                ms=None
                if separate:
                    maskpath=work/'source-mask.tif'; maskpath.write_bytes(store.get(maskref))
                    ms=stack.enter_context(rasterio.open(maskpath))
                rgb,support=read_native(ds,selected,ms)
                affine=ds.transform; crs=ds.crs
            output=work/'rgb-context.tif'; write_geotiff(output,rgb,affine,crs)
            outputmask=work/'validity-context.tif'
            with rasterio.open(outputmask,'w',driver='GTiff',width=1700,height=1700,count=1,
                dtype='uint8',crs=crs,transform=affine,compress='DEFLATE',tiled=True,
                blockxsize=256,blockysize=256) as dst: dst.write(support.astype('uint8')*255,1)
            # Verify serialized native output and mask before deriving shared arrays.
            with rasterio.open(output) as ds,rasterio.open(outputmask) as ms:
                decoded,decoded_support=read_native(ds,selected,ms)
                if not np.array_equal(decoded,rgb): raise ValueError('Lossless mother serialization failed')
            context,hr=canonical_hr(decoded,decoded_support)
            sample=dict(selected,sample_id=identity,mother_id=identity,
                hr_sha256=array_hash(hr),context_sha256=array_hash(context),
                hr_transform=[2.5,0,selected['core_bounds'][0],0,-2.5,selected['core_bounds'][3]],
                lr_transform=[10,0,selected['core_bounds'][0],0,-10,selected['core_bounds'][3]],
                container_image_id=config['container_image_id'],code_sha256=config['code_sha256'],
                lr_origin='synthetic_hr_only',halo_pixels=50,core_pixel_window=[50,50,1600,1600],
                core_bytes=7680000,rgb_order='RGB',source_gsd_m=selected.get('native_gsd_m'),
                source_gsd_basis='vendor_acquisition_metadata' if selected.get('native_gsd_m') is not None else 'unresolved',
                source_raster_spacing_m=source_raster_spacing(selected),
                mother_grid_gsd_m=.8,hr_gsd_m=2.5,lr_gsd_m=10)
            sample['mother']=store.put(MOTHER+'/samples/'+identity+'/rgb-context.tif',output.read_bytes())
            sample['mask']=dict(selected['mask'],object=store.put(MOTHER+'/samples/'+identity+'/validity-context.tif',outputmask.read_bytes()),kind='separate_validity')
            shared=deterministic_npz(dict(context=context,hr=hr,hr_mask=np.ones((512,512),bool),
                provenance=np.array(json.dumps(sample,sort_keys=True))))
            sample['shared']=store.put('shared/'+identity+'.npz',shared)
            objects=dict(mother=sample['mother'],mask=sample['mask']['object'],shared=sample['shared'])
            store.put(name,json_bytes(dict(state='COMPLETE',config_sha256=config_sha,sample=sample,objects=objects)))
            samples.append(sample)
    metadata=dict(selection=store.put(MOTHER+'/selection.json',json_bytes(document)),
        sources=store.put(MOTHER+'/sources.json',json_bytes([dict(sample_id=r['sample_id'],
            source_objects=r.get('source_objects'),source_sha256=r['source_sha256'],
            original_mask_object=r.get('original_mask_object')) for r in samples])),
        splits=store.put(MOTHER+'/splits.json',json_bytes(document['splits'])),
        reviews=store.put(MOTHER+'/review-ledger.json',json_bytes([r['review_decision'] for r in rows])),
        audit=store.put(MOTHER+'/audit.json',json_bytes(dict(independent_audits=document.get('independent_audits',{}),
            size_audit=document.get('size_audit',{}),gates=document['gates']))),
        samples=store.put(MOTHER+'/samples.parquet',parquet_bytes(samples)))
    return dict(release=MOTHER,state='COMPLETE',samples=samples,metadata=metadata,
        useful_native_core_bytes=sum(r['split']!='development' for r in samples)*7680000,
        compressed_mother_bytes=sum(r['mother']['bytes']+r['mask']['object']['bytes'] for r in samples))


def tar_bytes(members):
    output=io.BytesIO()
    with tarfile.open(fileobj=output,mode='w',format=tarfile.USTAR_FORMAT) as archive:
        for name,payload in sorted(members):
            info=tarfile.TarInfo(name); info.size=len(payload); info.mtime=0; info.mode=0o644
            archive.addfile(info,io.BytesIO(payload))
    return output.getvalue()


def build_child(store,mother,recipe,config_sha,scratch,pilot):
    samples=[]; recompute=recompute_ids(mother['samples'],pilot)
    for frozen in mother['samples']:
        identity=frozen['sample_id']; receiptname='control/'+recipe+'/'+identity+'.json'
        receipt=restore_receipt(store,receiptname,config_sha)
        if receipt:
            samples.append(receipt['sample']); continue
        context,hr,hrmask=read_shared(store,frozen)
        result=degrade(context,identity,recipe)
        metadata=dict(frozen,recipe=result['metadata'],child_release=CHILDREN[recipe],
            pair_id=digest_bytes(json_bytes([identity,RECIPE_VERSIONS[recipe],'0'])))
        pair=dict(hr=hr,lr=result['lr'],hr_mask=hrmask,lr_mask=result['lr_mask'])
        validate_pair(pair)
        ref=store.put(CHILDREN[recipe]+'/pairs/'+frozen['split']+'/'+identity+'.npz',
            deterministic_npz(dict(**pair,provenance=np.array(json.dumps(metadata,sort_keys=True)))))
        sample=dict(frozen,pair=ref,recipe=result['metadata'])
        store.put(receiptname,json_bytes(dict(state='COMPLETE',config_sha256=config_sha,
            sample=sample,objects=dict(pair=ref,shared=frozen['shared']))))
        samples.append(sample)
    shards=[]; loader_reports=[]; checked=[]
    for split in sorted({r['split'] for r in samples}):
        splitrows=[row for row in samples if row['split']==split]
        for offset in range(0,len(splitrows),32):
            chunk=splitrows[offset:offset+32]
            payload=tar_bytes([(split+'/'+r['sample_id']+'.npz',store.get(r['pair'])) for r in chunk])
            ref=store.put(CHILDREN[recipe]+'/shards/'+split+'/'+str(offset//32).zfill(5)+'.tar',payload)
            with tempfile.TemporaryDirectory(dir=scratch) as work:
                path=Path(work)/'shard.tar'; path.write_bytes(store.get(ref))
                dataset=MaritimeDataset([path],recipe)
                for index in range(len(dataset)):
                    pair,metadata=dataset.read_arrays(index)
                    frozen=next(row for row in chunk if row['sample_id']==metadata['sample_id'])
                    if any(metadata.get(key)!=frozen.get(key) for key in COMMON):
                        raise ValueError('Serialized child common provenance mismatch')
                    if metadata['sample_id'] in recompute:
                        context,hr,mask=read_shared(store,frozen)
                        expected=degrade(context,metadata['sample_id'],recipe)
                        if (not np.array_equal(hr,pair['hr']) or not np.array_equal(mask,pair['hr_mask'])
                            or not np.allclose(expected['lr'],pair['lr'],atol=1e-6,rtol=1e-5)
                            or array_hash(expected['lr'])!=array_hash(pair['lr'])
                            or expected['metadata']!=metadata['recipe']):
                            raise ValueError('Deterministic recipe recomputation failed')
                        checked.append(metadata['sample_id'])
                loader_reports.append(smoke_batch(dataset))
            shards.append(dict(split=split,sample_ids=[r['sample_id'] for r in chunk],object=ref))
    if set(checked)!=recompute: raise ValueError('Missing stratified recipe recomputation')
    return dict(release=CHILDREN[recipe],recipe=recipe,recipe_version=RECIPE_VERSIONS[recipe],state='COMPLETE',
        samples=samples,shards=shards,loader_reports=loader_reports,recomputed_sample_ids=sorted(checked),
        recomputation=dict(atol=1e-6,rtol=1e-5,same_container_exact_hash_required=True),
        samples_parquet=store.put(CHILDREN[recipe]+'/samples.parquet',parquet_bytes(samples)))


def verify_manifest(store,manifest):
    for row in manifest['samples']:
        for ref in [row['mother'],row['mask']['object'],row['shared']]+([row['pair']] if 'pair' in row else []):
            store.get(ref)
    for shard in manifest.get('shards',[]): store.get(shard['object'])
    for ref in manifest.get('metadata',{}).values(): store.get(ref)
    if 'samples_parquet' in manifest: store.get(manifest['samples_parquet'])


def data_card(release,status,limitations,manifest):
    text=(f'# {release}\n\nStatus: {status}.\n\n'
        'RGB 0.8 m native cores; shared HR 2.5 m and synthetic LR 10 m. '
        'No model training or real Sentinel-2 pairing was performed.\n\n'
        f"Selected samples: {len(manifest['samples'])}. Unique native core bytes: {len(manifest['samples'])*7680000}.\n\n")
    if limitations: text+='Unmet full-release targets/audits: '+', '.join(limitations)+'. This release does not attest full training readiness.\n'
    text+='\nFull mother target: 7,813 unique cores, 60,000,000,000–60,100,000,000 native RGB core bytes. Development samples, halos and child arrays do not count toward this target.\n'
    text+='\nSource licenses and attribution are carried per sample; obey each source license.\n'
    return text.encode()


def manifest_summary(manifest):
    rows=manifest['samples']
    summary=dict(sample_count=len(rows),site_count=len({r['site_id'] for r in rows}),
        geographic_group_count=len({r['geographic_group_id'] for r in rows}),
        split_counts=dict(Counter(r['split'] for r in rows)),
        category_counts=dict(Counter(r.get('primary_category',r['review_decision']['primary_category']) for r in rows)),
        useful_native_core_bytes=sum(r['split']!='development' for r in rows)*7680000,
        compressed_mother_bytes=sum(r['mother']['bytes']+r['mask']['object']['bytes'] for r in rows),
        shared_npz_bytes=sum(r['shared']['bytes'] for r in rows))
    if 'shards' in manifest:
        summary.update(pair_count=len(rows),pair_npz_bytes=sum(r['pair']['bytes'] for r in rows),
            shard_tar_bytes=sum(r['object']['bytes'] for r in manifest['shards']),
            shard_count=len(manifest['shards']))
    if manifest.get('recipe')=='d3':
        parameters=['sigma_x_hr_pixels','sigma_y_hr_pixels','noise_sigma_normalized','clipping_fraction']
        summary['d3_parameters']={key:dict(minimum=min(r['recipe'][key] for r in rows),
            maximum=max(r['recipe'][key] for r in rows),mean=float(np.mean([r['recipe'][key] for r in rows])))
            for key in parameters}
        summary['d3_kernel_types']=dict(Counter(r['recipe']['kernel_type'] for r in rows))
    return summary


def verify_terminal_evidence(store,evidence,config_sha,image_id):
    """Verify supervisor-produced exit/log receipts for every started worker.

    Worker COMPLETE receipts attest artifacts only. The caller explicitly
    supplies independent SDK-supervisor evidence after containers have exited.
    """
    if not isinstance(evidence,dict) or not isinstance(evidence.get('receipts'),list):
        raise ValueError('Terminal evidence must list pinned supervisor receipts')
    prefix=store.root+'/control/attempts/'
    starts={}
    for blob in store.client.list_blobs(store.bucket,prefix=prefix):
        if not blob.name.endswith('-started.json'): continue
        name=blob.name[len(store.root)+1:]
        ref=store.existing(name); start=json.loads(store.get(ref))
        if start.get('config_sha256')!=config_sha:
            raise ValueError('Started attempt configuration mismatch')
        starts[ref['uri']]=(ref,start)
    seen=set(); successful=set(); receipts=[]; logs=[]
    for ref in evidence['receipts']:
        terminal=json.loads(store.get(ref))
        if terminal.get('config_sha256')!=config_sha or terminal.get('container_image_id')!=image_id:
            raise ValueError('Supervisor terminal configuration/image mismatch')
        if (terminal.get('producer')!='sdk_supervisor' or not terminal.get('supervisor_id')
            or terminal.get('state')!='EXITED' or terminal.get('worker_process_exited') is not True
            or terminal.get('logs_closed') is not True
            or type(terminal.get('exit_code')) is not int):
            raise ValueError('Independent terminal process exit and closed logs required')
        startref=pinned(terminal.get('start_record')); uri=startref['uri']
        if uri in seen or uri not in starts or startref!=starts[uri][0]:
            raise ValueError('Duplicate/unmatched supervisor job attempt')
        start=json.loads(store.get(startref))
        for key in ('phase','job_attempt_id','instance_id','container_image_id','config_sha256'):
            if not start.get(key) or terminal.get(key)!=start[key]:
                raise ValueError('Supervisor start/terminal job attempt identity mismatch')
        if start['phase'] not in ('mother','d1','d2','d3'):
            raise ValueError('Unknown supervised phase')
        log=pinned(terminal.get('closed_log')); store.get(log)
        seen.add(uri); receipts.append(ref); logs.append(log)
        if terminal['exit_code']==0: successful.add(start['phase'])
    if not starts or seen!=set(starts) or successful!={'mother','d1','d2','d3'}:
        raise ValueError('All started attempts must be terminal; four successful phases required')
    return dict(verified=True,scope='mother_d1_d2_d3_started_attempts',
        successful_phases=sorted(successful),terminal_receipts=receipts,closed_logs=logs,
        verification='Pinned independent SDK-supervisor exit and closed-log evidence')


def start_attempt(store,args,phase,config_sha,instance,scratch):
    checkpoint=scratch/(phase+'-attempt.json')
    explicit=getattr(args,'job_attempt_id',None)
    if explicit:
        identity=explicit
    elif checkpoint.exists():
        previous=json.loads(checkpoint.read_text())
        identity=previous['job_attempt_id'] if previous.get('instance_id')==instance else uuid.uuid4().hex
    else:
        identity=uuid.uuid4().hex
    if not re.fullmatch('[A-Za-z0-9_-]+',identity): raise ValueError('Unsafe job attempt identity')
    start=dict(phase=phase,job_attempt_id=identity,instance_id=instance,
        config_sha256=config_sha,container_image_id=args.container_image_id,
        producer='release_worker',process_terminal_verified=False)
    ref=store.put('control/attempts/'+phase+'/'+identity+'-started.json',json_bytes(start))
    checkpoint.write_bytes(json_bytes(dict(start,start_record=ref)))
    return ref


def run(args):
    instance=require_cloud_vm()
    from google.cloud import storage
    document=json.loads(Path(args.selection).read_text(encoding='utf-8-sig'))
    rows,limitations=validate_selection(document,args.provisional)
    if not args.container_image_id or not re.fullmatch(r'.+@sha256:[0-9a-f]{64}',args.container_image_id):
        raise ValueError('Pinned container image digest required')
    store=Store(storage.Client(),args.output_prefix)
    scratch=Path(args.scratch); scratch.mkdir(parents=True,exist_ok=True)
    code={name:file_sha256(Path(__file__).parents[1]/name) for name in (
        'scripts/maritime_release_build.py','scripts/maritime_pilot.py','src/maritime_recipes.py','src/maritime_dataset.py')}
    config=dict(selection_sha256=digest_bytes(json_bytes(document)),container_image_id=args.container_image_id,
        code_sha256=code,provisional=args.provisional,pilot=args.pilot,output_prefix=args.output_prefix.rstrip('/'),
        core=1600,halo=50,shard_samples=32,recipes=RECIPE_VERSIONS)
    config_sha=digest_bytes(json_bytes(config)); configref=store.put('control/config.json',json_bytes(config))
    # All externally supplied independent evidence is pinned and checked before building.
    for audit in document.get('independent_audits',{}).values():
        if audit.get('evidence'): store.get(audit['evidence'])
    phase=args.phase
    if phase=='finalize':
        mother,motherref=store.document(MOTHER+'/manifest.json'); children={}; refs={'mother':motherref}
        for name in ('mother','d1','d2','d3'):
            terminal,_=store.document('control/phases/'+name+'-complete.json')
            if terminal['config_sha256']!=config_sha or terminal['state']!='COMPLETE':
                raise ValueError('Incomplete artifact phase receipt in build configuration')
        verify_manifest(store,mother)
        for name in RECIPE_VERSIONS:
            children[name],refs[name]=store.document(CHILDREN[name]+'/manifest.json')
            verify_manifest(store,children[name])
        validate_children(mother,children)
        for name,manifest in [('mother',mother)]+list(children.items()):
            for sample in manifest['samples']:
                receipt=restore_receipt(store,'control/'+name+'/'+sample['sample_id']+'.json',config_sha)
                if receipt is None or receipt['sample']!=sample:
                    raise ValueError('Missing/mismatched per-sample durable receipt')
        for manifest in [mother]+list(children.values()):
            if manifest.get('config_sha256')!=config_sha: raise ValueError('Manifest configuration mismatch')
        terminal_path=getattr(args,'terminal_evidence',None)
        if terminal_path:
            evidence=json.loads(Path(terminal_path).read_text(encoding='utf-8-sig'))
            task_gate=verify_terminal_evidence(store,evidence,config_sha,args.container_image_id)
        else:
            task_gate=dict(verified=False,scope='mother_d1_d2_d3_started_attempts',
                reason='Worker artifact completion cannot attest process exit or closed logs')
            if not args.provisional:
                raise ValueError('Strict finalization requires independent --terminal-evidence')
            limitations=sorted(set(limitations+['terminal_supervisor_evidence_unverified']))
        status='PROVISIONAL_READY' if args.provisional else 'DATASET_READY'
        cards={}
        for name,manifest in [('mother',mother)]+list(children.items()):
            release=manifest['release']; cards[name]=store.put(release+'/DATA_CARD.md',data_card(release,status,limitations,manifest))
            marker='PROVISIONAL_READY' if args.provisional else 'MOTHER_READY' if name=='mother' else 'DATASET_READY'
            refs[name+'_ready']=store.put(release+'/'+marker+'.json',json_bytes(dict(status=marker,
                manifest=refs[name],config=configref,limitations=limitations)))
        result=dict(schema_version=1,status=status,config=configref,releases=refs,data_cards=cards,
            limitations=limitations,full_training_ready=not args.provisional,
            no_active_or_uncertain_build_tasks=task_gate['verified'],terminal_supervisor_audit=task_gate,
            common_sample_count=len(mother['samples']))
        store.put('release.json',json_bytes(result)); return result
    start_attempt(store,args,phase,config_sha,instance,scratch)
    if phase=='mother':
        manifest=build_mother(store,document,rows,config,config_sha,scratch); name=MOTHER
    else:
        mother,_=store.document(MOTHER+'/manifest.json')
        if mother['config_sha256']!=config_sha: raise ValueError('Mother configuration mismatch')
        manifest=build_child(store,mother,phase,config_sha,scratch,args.pilot); name=CHILDREN[phase]
    manifest.update(config_sha256=config_sha,config=configref,limitations=limitations,
        provisional=args.provisional,container_image_id=args.container_image_id)
    manifest['summary']=manifest_summary(manifest)
    verify_manifest(store,manifest)
    ref=store.put(name+'/manifest.json',json_bytes(manifest))
    terminal=dict(state='COMPLETE',phase=phase,config_sha256=config_sha,manifest=ref,
        evidence_scope='artifact_completion_only',process_terminal_verified=False)
    store.put('control/phases/'+phase+'-complete.json',json_bytes(terminal))
    (scratch/(phase+'-complete.json')).write_bytes(json_bytes(terminal))
    return terminal


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('selection','output-prefix','scratch','container-image-id'): parser.add_argument('--'+name,required=True)
    parser.add_argument('--phase',choices=['mother','d1','d2','d3','finalize'],required=True)
    parser.add_argument('--provisional',action='store_true')
    parser.add_argument('--pilot',action='store_true',help='Recompute every pair rather than 5%% per stratum')
    parser.add_argument('--job-attempt-id',help='SDK supervisor job identity for this worker invocation')
    parser.add_argument('--terminal-evidence',help='Finalizer-only JSON with pinned SDK-supervisor terminal receipts')
    args=parser.parse_args()
    print(json.dumps(run(args),sort_keys=True),flush=True)


if __name__=='__main__': main()
