"""Independent artifact auditor. All real raster reads run on GCE only.

This audit verifies published pixels and training inputs, not their equality
to original raw-source windows or an independently implemented degradation.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import ColorInterp
from rasterio.io import MemoryFile
from rasterio.transform import Affine, array_bounds

from scripts.maritime_cloud_inventory import create_only, json_bytes, parse_gs, require_cloud_vm
from src.maritime_dataset import MaritimeDataset, smoke_batch
from src.maritime_recipes import RECIPE_VERSIONS, array_hash, canonical_hr, degrade

NAMES = dict(mother='skysat-maritime-mother-v1',
             **{key:'skysat-maritime-'+key+'-v1' for key in RECIPE_VERSIONS})
COMMON = ('sample_id','hr_sha256','context_sha256','site_id','geographic_group_id',
          'split','core_bounds','crs','hr_transform','lr_transform','mother','mask','shared')


def validate_pin(ref):
    if not isinstance(ref,dict) or not all(k in ref for k in ('uri','generation','bytes','crc32c','sha256')):
        raise ValueError('Incomplete immutable object pin')
    parse_gs(ref['uri'])
    if int(ref['generation'])<1 or ref['bytes']<1 or not ref['crc32c'] or not re.fullmatch('[0-9a-f]{64}',ref['sha256']):
        raise ValueError('Invalid immutable object pin')
    return ref


class AuditStore:
    """Separate reader: generation, CRC and SHA checked on every object read."""
    def __init__(self,client,prefix):
        self.client=client; self.prefix=prefix.rstrip('/'); self.verified={}

    def get(self,ref):
        validate_pin(ref)
        bucket,name=parse_gs(ref['uri'])
        blob=self.client.bucket(bucket).blob(name,generation=int(ref['generation']))
        blob.reload(if_generation_match=int(ref['generation']))
        if (str(blob.generation)!=str(ref['generation']) or blob.size!=ref['bytes']
                or blob.crc32c!=ref['crc32c']):
            raise ValueError('Generation/size/CRC pin mismatch')
        data=blob.download_as_bytes(if_generation_match=int(ref['generation']),checksum='crc32c')
        if len(data)!=ref['bytes'] or hashlib.sha256(data).hexdigest()!=ref['sha256']:
            raise ValueError('Object SHA/size mismatch')
        self.verified[(ref['uri'],str(ref['generation']))]=dict(ref)
        return data

    def discover(self,name):
        uri=self.prefix+'/'+name; bucket,key=parse_gs(uri)
        blob=self.client.bucket(bucket).blob(key); blob.reload()
        data=blob.download_as_bytes(if_generation_match=int(blob.generation),checksum='crc32c')
        ref=dict(uri=uri,generation=str(blob.generation),bytes=len(data),crc32c=blob.crc32c,
                 sha256=hashlib.sha256(data).hexdigest())
        return json.loads(self.get(ref)),ref

    def starts(self):
        bucket,root=parse_gs(self.prefix)
        return [self.discover(blob.name[len(root)+1:]) for blob in self.client.list_blobs(
            bucket,prefix=root+'/control/attempts/') if blob.name.endswith('-started.json')]


def native_arrays(rgb_payload,mask_payload,row):
    with MemoryFile(rgb_payload) as rgbmem, MemoryFile(mask_payload) as maskmem:
        with rgbmem.open() as rgb,maskmem.open() as mask:
            expected=Affine(*row['context_transform']); crs=CRS.from_user_input(row['crs'])
            if (rgb.shape!=(1700,1700) or rgb.count!=3 or rgb.dtypes!=('uint8',)*3
                    or rgb.colorinterp!=(ColorInterp.red,ColorInterp.green,ColorInterp.blue)
                    or rgb.crs!=crs or rgb.transform!=expected or not crs.is_projected
                    or crs.linear_units_factor[1]!=1 or expected.a!=.8 or expected.e!=-.8
                    or expected.b!=0 or expected.d!=0):
                raise ValueError('Native RGB grid/type mismatch')
            if (mask.shape!=rgb.shape or mask.count!=1 or mask.dtypes!=('uint8',)
                    or mask.crs!=crs or mask.transform!=expected):
                raise ValueError('Native validity grid/type mismatch')
            support=mask.read(1)
            if not (support==255).all(): raise ValueError('Invalid core or real halo support')
            core=expected*Affine.translation(50,50)
            if (tuple(core)[:6]!=tuple(row['core_transform']) or not np.allclose(
                    array_bounds(1600,1600,core),row['core_bounds'],atol=1e-6,rtol=0)):
                raise ValueError('Core bounds/context halo mismatch')
            left,bottom,right,top=row['core_bounds']
            if (row['hr_transform']!=[2.5,0,left,0,-2.5,top]
                    or row['lr_transform']!=[10,0,left,0,-10,top]
                    or core.c!=left or core.f!=top):
                raise ValueError('Training grid origin/spacing differs from native core')
            footprint=array_bounds(1700,1700,expected)
            if (not np.allclose(footprint,[left-40,bottom-40,right+40,top+40],atol=1e-6,rtol=0)
                    or ('context_bounds' in row and not np.allclose(
                        footprint,row['context_bounds'],atol=1e-6,rtol=0))):
                raise ValueError('Context footprint differs from real native halo')
            return rgb.read(),support==255


def native_accounting(rows):
    identities=set(); acquisitions=defaultdict(list); splits={}; groups={}; count=0
    for row in rows:
        if row['sample_id'] in identities: raise ValueError('Duplicate sample ID')
        identities.add(row['sample_id'])
        if row['split'] not in ('train','validation','test','development'): raise ValueError('Unknown split')
        for field,mapping in [('site_id',splits),('geographic_group_id',groups)]:
            key=row[field]
            if key in mapping and mapping[key]!=row['split']: raise ValueError('Cross-split site/group')
            mapping[key]=row['split']
        crs=CRS.from_user_input(row['crs']); bounds=row['core_bounds']
        for previous in acquisitions[row['acquisition_id']]:
            if crs==CRS.from_user_input(previous['crs']):
                e,f,g,h=previous['core_bounds']; a,b,c,d=bounds
                if min(c,g)-max(a,e)>1e-6 and min(d,h)-max(b,f)>1e-6:
                    raise ValueError('Duplicate/overlapping physical acquisition core')
            else:
                # Cross-CRS same-capture cores require a geographic overlap audit.
                raise ValueError('Same acquisition uses mixed CRS; physical dedup unresolved')
        acquisitions[row['acquisition_id']].append(row)
        count+=row['split']!='development'
    return dict(unique_native_core_count=count,useful_native_core_bytes=count*1600*1600*3,
                split_counts=dict(Counter(row['split'] for row in rows)),
                excludes=['development','halo','masks','float_arrays','archive_copies'])


def terminal_audit(store,evidence,config_sha,image):
    starts={ref['uri']:(start,ref) for start,ref in store.starts()}
    seen=set(); success=set()
    for ref in evidence.get('terminal_receipts',evidence.get('receipts',[])):
        terminal=json.loads(store.get(ref)); startref=validate_pin(terminal.get('start_record'))
        uri=startref['uri']
        if uri in seen or uri not in starts or starts[uri][1]!=startref:
            raise ValueError('Unmatched/duplicate terminal attempt')
        start=json.loads(store.get(startref))
        for key in ('phase','job_attempt_id','instance_id','config_sha256','container_image_id'):
            if not start.get(key) or terminal.get(key)!=start[key]: raise ValueError('Terminal start identity mismatch')
        if (start['config_sha256']!=config_sha or start['container_image_id']!=image
                or start['phase'] not in NAMES or terminal.get('producer')!='sdk_supervisor'
                or not terminal.get('supervisor_id') or terminal.get('state')!='EXITED'
                or terminal.get('worker_process_exited') is not True or terminal.get('logs_closed') is not True
                or type(terminal.get('exit_code')) is not int):
            raise ValueError('Missing independent terminal exit evidence')
        store.get(validate_pin(terminal.get('closed_log')))
        seen.add(uri)
        if terminal['exit_code']==0: success.add(start['phase'])
    if not starts or seen!=set(starts) or success!=set(NAMES):
        raise ValueError('All started attempts must exit and all four phases succeed')
    return dict(verified=True,attempts=len(seen),successful_phases=sorted(success))


def audit_release(store,scratch):
    scratch=Path(scratch); scratch.mkdir(parents=True,exist_ok=True)
    release,release_ref=store.discover('release.json')
    config=json.loads(store.get(release['config'])); config_sha=hashlib.sha256(json_bytes(config)).hexdigest()
    manifests={key:json.loads(store.get(release['releases'][key])) for key in NAMES}
    provisional=config.get('provisional') is True
    if provisional and (release.get('full_training_ready') is not False or release['status']!='PROVISIONAL_READY'):
        raise ValueError('Provisional release falsely attests full readiness')
    for key,manifest in manifests.items():
        if (manifest.get('state')!='COMPLETE' or manifest.get('release')!=NAMES[key]
                or manifest.get('config_sha256')!=config_sha or manifest.get('config')!=release['config']
                or manifest.get('container_image_id')!=config['container_image_id']):
            raise ValueError('Manifest identity/configuration mismatch')
        marker=json.loads(store.get(release['releases'][key+'_ready']))
        expected='PROVISIONAL_READY' if provisional else 'MOTHER_READY' if key=='mother' else 'DATASET_READY'
        if marker['status']!=expected or marker['manifest']!=release['releases'][key] or marker['config']!=release['config']:
            raise ValueError('Readiness marker mismatch')
        store.get(release['data_cards'][key])
        phase,_=store.discover('control/phases/'+key+'-complete.json')
        if phase.get('state')!='COMPLETE' or phase.get('config_sha256')!=config_sha or phase.get('manifest')!=release['releases'][key]:
            raise ValueError('Phase artifact receipt mismatch')
        for pin in manifest.get('metadata',{}).values(): store.get(pin)
        if 'samples_parquet' in manifest: store.get(manifest['samples_parquet'])
        for row in manifest['samples']:
            receipt,_=store.discover('control/'+key+'/'+row['sample_id']+'.json')
            if receipt.get('state')!='COMPLETE' or receipt.get('config_sha256')!=config_sha or receipt.get('sample')!=row:
                raise ValueError('Per-sample durable receipt mismatch')
            for pin in receipt['objects'].values(): store.get(pin)
    mother=manifests['mother']; rows=mother['samples']; accounting=native_accounting(rows)
    if not rows or accounting['useful_native_core_bytes']!=mother['useful_native_core_bytes']:
        raise ValueError('Native core accounting mismatch')
    expected={row['sample_id']:row for row in rows}; shared={}
    for identity,row in expected.items():
        rgb,support=native_arrays(store.get(row['mother']),store.get(row['mask']['object']),row)
        context,hr=canonical_hr(rgb,support)
        with np.load(io.BytesIO(store.get(row['shared'])),allow_pickle=False) as values:
            mask=values['hr_mask']
            if (not np.array_equal(context,values['context']) or not np.array_equal(hr,values['hr'])
                    or mask.shape!=(512,512) or mask.dtype!=np.bool_ or not mask.all()
                    or array_hash(context)!=row['context_sha256'] or array_hash(hr)!=row['hr_sha256']):
                raise ValueError('Native-derived shared HR/context mismatch')
        shared[identity]=(context,hr)
    loader_reports={}
    for key in RECIPE_VERSIONS:
        child=manifests[key]; childrows=child['samples']; actual={r['sample_id']:r for r in childrows}
        if len(actual)!=len(childrows) or set(actual)!=set(expected) or child['recipe']!=key:
            raise ValueError('Child sample coverage mismatch')
        for identity,row in actual.items():
            if any(row.get(field)!=expected[identity].get(field) for field in COMMON):
                raise ValueError('Child common HR/splits/masks mismatch')
        seen=set(); paths=[]
        for number,shard in enumerate(child['shards']):
            payload=store.get(shard['object']); path=scratch/(key+'-'+str(number)+'.tar'); path.write_bytes(payload)
            paths.append(path)
            with tarfile.open(fileobj=io.BytesIO(payload),mode='r:') as archive:
                members=archive.getmembers(); ids=[]
                for member in members:
                    identity=Path(member.name).stem
                    if (not member.isfile() or member.name!=shard['split']+'/'+identity+'.npz'
                            or identity not in actual or identity in seen or actual[identity]['split']!=shard['split']):
                        raise ValueError('Unsafe/duplicate/wrong-split TAR member')
                    if archive.extractfile(member).read()!=store.get(actual[identity]['pair']):
                        raise ValueError('TAR/pair payload mismatch')
                    ids.append(identity); seen.add(identity)
                if ids!=shard['sample_ids']: raise ValueError('Shard member manifest mismatch')
        if seen!=set(expected): raise ValueError('Incomplete TAR sample coverage')
        dataset=MaritimeDataset(paths,key)
        for index in range(len(dataset)):
            pair,metadata=dataset.read_arrays(index); identity=metadata['sample_id']; context,hr=shared[identity]
            if metadata['recipe']['release_seed']!=20261006 or metadata['recipe']['variant_id']!='0':
                raise ValueError('Recipe realization differs from the frozen plan')
            result=degrade(context,identity,key)
            if (not np.array_equal(pair['hr'],hr) or not np.array_equal(pair['lr'],result['lr'])
                    or metadata['recipe']!=result['metadata'] or actual[identity]['recipe']!=result['metadata']
                    or any(metadata.get(field)!=expected[identity].get(field) for field in COMMON if field not in ('mother','mask','shared'))):
                raise ValueError('All-pair numeric/provenance/parameter recomputation mismatch')
        loader_reports[key]=smoke_batch(dataset)
    terminal=terminal_audit(store,release.get('terminal_supervisor_audit',{}),config_sha,config['container_image_id'])
    return dict(schema_version=1,passed=True,scope='published_artifact_integrity_and_training_input_audit',
        release=release_ref,provisional=provisional,full_training_ready_attested=False,
        audit_code_sha256={name:hashlib.sha256((Path(__file__).parents[1]/name).read_bytes()).hexdigest()
            for name in ('scripts/maritime_release_audit.py','src/maritime_dataset.py','src/maritime_recipes.py')},
        native_accounting=accounting,loader_reports=loader_reports,terminal_audit=terminal,
        numeric_recomputation='Every pair; uses the same versioned recipe implementation',
        original_raw_source_window_pixel_comparison_verified=False,
        independent_operator_implementation_verified=False,
        geographic_buffer_and_near_stereo_audit_verified=False,
        semantic_native_review_independently_repeated=False,
        limitations=['Original raw source-window equality unverified','Independent operator implementation unverified',
                     'Geographic buffer and near-stereo review not independently repeated',
                     'Semantic native review not independently repeated'],
        verified_objects=list(store.verified.values()) if hasattr(store,'verified') else [])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix',required=True); parser.add_argument('--scratch',required=True)
    parser.add_argument('--publish',help='Optional immutable gs:// audit object')
    args=parser.parse_args(); instance=require_cloud_vm()
    from google.cloud import storage
    client=storage.Client(); report=audit_release(AuditStore(client,args.prefix),args.scratch)
    report['auditor_instance_id']=instance
    payload=json_bytes(report); path=Path(args.scratch)/'release-audit.json'; path.write_bytes(payload)
    if args.publish:
        bucket,name=parse_gs(args.publish)
        ref=dict(uri=args.publish,**create_only(client.bucket(bucket).blob(name),payload))
        AuditStore(client,args.prefix).get(ref)
        print(json.dumps(dict(audit=ref,passed=True)))
    else: print(json.dumps(dict(path=str(path),passed=True)))


if __name__=='__main__': main()
