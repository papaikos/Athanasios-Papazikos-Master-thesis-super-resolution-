"""Candidate contract for original uint8 aerial RGB; scientific geometry is shared."""
import hashlib
import numpy as np
from scripts.maritime_cloud_inventory import json_bytes
from scripts.maritime_maxar_intake import area_hr


def planned_source_geometry(source):
    """Plan real physical windows from immutable metadata, without raster IO."""
    from types import SimpleNamespace
    from affine import Affine
    from rasterio.crs import CRS
    from rasterio.coords import BoundingBox
    from scripts.maritime_maxar_intake import planned_cores
    ds=SimpleNamespace(transform=Affine(*source['transform_affine']),crs=CRS.from_epsg(source['epsg']),
        height=source['shape'][0],width=source['shape'][1],bounds=BoundingBox(*source['projected_bounds']))
    return ds,planned_cores(ds,source)


def verified_core_plans(ds,source,doc):
    """Replay frozen planning geometry against the actual original header."""
    from scripts.maritime_maxar_intake import planned_cores
    actual=planned_cores(ds,source)
    if 'core_plans' not in doc:return actual
    frozen=doc['core_plans'][source['source_id']]
    index={tuple(p['world_grid_cell']):p for p in actual}
    seen=set()
    for plan in frozen:
        key=tuple(plan['world_grid_cell'])
        if key in seen or index.get(key)!=plan:
            raise ValueError('Frozen core planning and original geometry disagree')
        seen.add(key)
    return frozen


def sources_for_role(sources,allocation,role='train'):
    if role not in ('train','evaluation'):
        raise ValueError('Only explicit train or separately reserved evaluation sources allowed')
    return [s for s in sources if allocation[s['source_id']]['role']==role]


def aerial_area_hr(rgb, valid, geometry, count=544):
    """Exact aligned box sums, otherwise the frozen canonical sparse integrator."""
    spacing=geometry['product_spacing_m'];factor=2.5/spacing
    aligned=geometry['area_phase_m']==[0,0] and factor.is_integer()
    if not aligned:
        return area_hr(rgb,valid,geometry,count),'canonical_sparse_exact_area_v1'
    factor=int(factor)
    if not 0<spacing<=1 or factor*factor*255>np.iinfo(np.uint32).max:
        raise ValueError('Invalid original spacing or unsafe accumulator')
    side=count*factor
    if rgb.dtype!=np.uint8 or rgb.shape!=(3,side,side) or valid.dtype!=np.bool_ or valid.shape!=(side,side) or not valid.all():
        raise ValueError('Exact original RGB and complete area support required')
    sums=rgb.reshape(3,count,factor,count,factor).sum(axis=(2,4),dtype=np.uint32)
    hr=(sums.astype(np.float64)/(factor*factor)/255.).astype(np.float32)
    return hr,'integer_aligned_box_sum_uint32_v1'


def candidate_metadata(source, geometry, native_window_transform, original_object, evidence_pin):
    """Describe a supported source window after the caller has verified its pixels."""
    sid=hashlib.sha256(json_bytes(dict(source_id=source['source_id'],
        epsg=source['epsg'],world_grid_cell=geometry['world_grid_cell']))).hexdigest()
    return dict(schema='aerial-rgb-candidate-v1',source_kind='aerial_rgb',source_type=source['source_type'],
        sample_id=sid,physical_core_id=sid,source=source,source_href=source['href'],
        source_http_identity=source['original_http_identity'],source_original_object=original_object,
        source_evidence_pin=evidence_pin,output_crs='EPSG:'+str(source['epsg']),
        source_crs='EPSG:'+str(source['epsg']),original_transform=source['transform_affine'],
        native_window_transform=list(native_window_transform)[:6],
        native_gsd_m=source['native_gsd_m'],delivered_gsd_m=source['delivered_gsd_m'],
        explicit_gsd_semantics='original_delivered_orthomosaic_grid_not_camera_capture',
        original_camera_capture_gsd_m=source['original_camera_capture_gsd_m'],
        original_mask_provenance='original_alpha255_and_all_original_RGB_masks255',
        source_product='original_uint8_RGB_orthorectified_aerial_mosaic',
        source_radiometry='original_delivered_uint8_RGB_DN_no_new_stretch',
        hr_radiometry='original_RGB_DN_divided255_then_exact_area_integration',
        acquisition_day=source['acquisition_day'],publication_datetime=source['stac_datetime'],
        parent_tile_id=source['parent_tile_id'],capture_identity_type=source['capture_identity_type'],
        split='provisional_training_only',requires_semantic_review=True,
        requires_native_quality_review=True,final_eligible=False,**geometry)


def main():
    """Sequential bounded-cache GCE intake; every record still needs final QA."""
    import argparse,io,json
    from pathlib import Path
    import requests,rasterio
    from rasterio.enums import ColorInterp
    from rasterio.windows import Window,transform as window_transform
    from google.cloud import storage
    from scripts.maritime_cloud_inventory import require_cloud_vm,parse_gs,create_only,file_sha256,verify_output,digest_bytes
    from scripts.maritime_maxar_intake import planned_cores,write_native
    p=argparse.ArgumentParser();p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--scratch',type=Path,required=True);p.add_argument('--output-prefix',required=True)
    p.add_argument('--max-sources',type=int,default=250);p.add_argument('--max-cores',type=int,default=1000000)
    p.add_argument('--evaluation-policy',type=Path,required=True)
    p.add_argument('--source-role',choices=('train','evaluation'),default='train')
    args=p.parse_args();require_cloud_vm();args.scratch.mkdir(parents=True,exist_ok=True)
    client=storage.Client();bn,prefix=parse_gs(args.output_prefix);bucket=client.bucket(bn)
    policy_bytes=args.evaluation_policy.read_bytes()
    assert hashlib.sha256(policy_bytes).hexdigest()=='713b5022b92f330795962ed618a0ffff0d90ae22b869364700c3cd776eeddb4c'
    policy=json.loads(policy_bytes)
    from scripts.maritime_aerial_evaluation import allocate_source
    assert file_sha256(Path('scripts/maritime_aerial_evaluation.py'))==policy['module_sha256']
    doc=json.loads(args.sources.read_bytes());records=[];attempts=[];source_rejections=[]
    allocation_decisions={s['source_id']:allocate_source(s,policy['evaluation_ports']) for s in doc['sources']}
    training=sources_for_role(doc['sources'],allocation_decisions,args.source_role)
    size_order={'L':0,'M':1,'S':2,'V':3}
    training.sort(key=lambda s:(min(size_order.get(p['harbor_size'],4) for p in s['port_AOI_matches']),s['source_id']))
    def publish_file(path,name):
        blob=bucket.blob(prefix+'/'+name);blob.upload_from_filename(str(path),if_generation_match=0,checksum='crc32c');blob.reload()
        return dict(uri='gs://'+bn+'/'+blob.name,generation=str(blob.generation),bytes=path.stat().st_size,
            crc32c=blob.crc32c,sha256=file_sha256(path))
    def publish_json(data,name):
        blob=bucket.blob(prefix+'/'+name);return dict(uri='gs://'+bn+'/'+blob.name,**create_only(blob,json_bytes(data)))
    def checkpoint():
        state=dict(schema='aerial-rgb-intake-checkpoint-v1',records=records,attempts=attempts,
            source_rejections=source_rejections,source_role=args.source_role,allocation_decisions=allocation_decisions,
            source_inventory_sha256=digest_bytes(args.sources.read_bytes()),
            producer_sha256=file_sha256(Path(__file__)),evaluation_policy_sha256=digest_bytes(policy_bytes),
            approved_training_bytes=0,final_dataset_ready=False)
        temporary=args.scratch/'checkpoint.tmp.json';temporary.write_bytes(json_bytes(state))
        temporary.replace(args.scratch/'checkpoint.json')
    checkpoint()
    for source in training[:args.max_sources]:
        try:
            evidence=json.loads(verify_output(client,source['source_evidence_pin']))
            assert evidence['source']=={k:v for k,v in source.items() if k!='source_evidence_pin'}
            license_html=verify_output(client,evidence['official_dataset_metadata_object']).decode('utf8')
            assert 'CC0-1.0' in license_html and source['dataset']['license']=='CC0-1.0'
            original=args.scratch/'original.tif';identity=source['original_http_identity']
            with requests.get(source['href'],headers={'If-Match':identity['etag']},stream=True,timeout=120) as response:
                response.raise_for_status();assert response.headers['ETag']==identity['etag']
                with original.open('wb') as f:
                    for chunk in response.iter_content(1024*1024):f.write(chunk)
            assert original.stat().st_size==identity['bytes']
            original_pin=publish_file(original,'original/'+hashlib.sha256(source['source_id'].encode()).hexdigest()+'.tif')
            with rasterio.open(original) as ds:
                assert list(ds.transform)[:6]==source['transform_affine'] and ds.crs.to_epsg()==source['epsg']
                assert [ds.height,ds.width]==source['shape'] and ds.dtypes[:3]==('uint8',)*3
                assert ds.colorinterp[:3]==(ColorInterp.red,ColorInterp.green,ColorInterp.blue)
                alpha=source['alpha_index'];assert alpha is not None and ds.colorinterp[alpha-1]==ColorInterp.alpha
                for geometry in verified_core_plans(ds,source,doc):
                    if len(records)>=args.max_cores:break
                    outcome=dict(source_id=source['source_id'],grid=geometry['world_grid_cell'],status='STARTED')
                    attempts.append(outcome);checkpoint()
                    window=Window(*geometry['source_pixel_window']);rgb=ds.read([1,2,3],window=window)
                    valid=(ds.read(alpha,window=window)==255)&np.all(ds.read_masks([1,2,3],window=window)==255,axis=0)
                    if not valid.all():
                        outcome.update(status='REJECTED',reason='original_context_mask_not_fully_valid');checkpoint();continue
                    native_transform=window_transform(window,ds.transform)
                    row=candidate_metadata(source,geometry,native_transform,original_pin,source['source_evidence_pin'])
                    row['split']='provisional_'+('training' if args.source_role=='train' else 'evaluation')+'_only'
                    row['split_exclusion_evidence']=dict(aerial_reservation_sha256=hashlib.sha256(policy_bytes).hexdigest(),
                        allocation=allocation_decisions[source['source_id']],evaluation_ports_count=len(policy['evaluation_ports']))
                    sid=row['sample_id'];context,method=aerial_area_hr(rgb,valid,geometry);row['canonical_hr_method']=method
                    row['decoded_native_context_sha256']=digest_bytes(rgb.tobytes())
                    col,r,width,height=geometry['owned_core_pixel_window'];wc,wr,_,_=geometry['source_pixel_window']
                    useful=rgb[:,r-wr:r-wr+height,col-wc:col-wc+width]
                    row['decoded_useful_core_sha256']=digest_bytes(useful.tobytes())
                    native=args.scratch/'native.tif';write_native(native,rgb,valid,ds.crs,native_transform)
                    row['native_context']=publish_file(native,'native-context/'+sid+'.tif');native.unlink()
                    hr=args.scratch/'hr.npz';np.savez_compressed(hr,hr_context=context,hr=context[:,16:528,16:528],
                        metadata_json=np.asarray(json.dumps(row,sort_keys=True)))
                    row['hr']=publish_file(hr,'hr/'+sid+'.npz');hr.unlink()
                    ref=publish_json(row,'candidates/'+sid+'.json');row['candidate_record']=ref
                    records.append(row);outcome['status']='COMPLETED';checkpoint()
                    print(json.dumps(dict(candidates=len(records),sample_id=sid)),flush=True)
            original.unlink()
        except Exception as exc:
            for attempt in attempts:
                if attempt['source_id']==source['source_id'] and attempt['status']=='STARTED':
                    attempt.update(status='FAILED',reason=type(exc).__name__+':'+str(exc))
            source_rejections.append(dict(source_id=source['source_id'],reason=type(exc).__name__+':'+str(exc)))
            checkpoint()
            print(json.dumps(source_rejections[-1]),flush=True)
        if len(records)>=args.max_cores:break
    manifest=dict(schema='aerial-rgb-candidate-v1',status='PROVISIONAL_AERIAL_CANDIDATES',source_kind='aerial_rgb',
        candidates=records,candidate_count=len(records),attempts=attempts,source_rejections=source_rejections,
        allocation_decisions=allocation_decisions,requested_source_role=args.source_role,role_source_count=len(training),
        semantic_accepted_count=0,approved_training_bytes=0,final_dataset_ready=False)
    print(json.dumps(dict(manifest=publish_json(manifest,'candidate-manifest.json'),candidate_count=len(records))),flush=True)


if __name__=='__main__':main()
