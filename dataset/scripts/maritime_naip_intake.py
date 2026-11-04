"""Original NAIP RGB intake helpers; no assumed alpha or infrared decoding."""
from rasterio.enums import ColorInterp

def original_header(ds,source):
    if (ds.crs.to_epsg()!=source['epsg'] or list(ds.transform)[:6]!=source['transform_affine']
        or [ds.height,ds.width]!=source['shape'] or ds.count<3 or ds.dtypes[:3]!=('uint8',)*3
        or ds.colorinterp[:3]!=(ColorInterp.red,ColorInterp.green,ColorInterp.blue)):
        raise ValueError('Actual original NAIP RGB header disagrees with frozen grid or uint8 RGB')
    return dict(count=ds.count,dtypes=list(ds.dtypes),colorinterp=[v.name for v in ds.colorinterp],
        mask_flags=[[v.name for v in flags] for flags in ds.mask_flag_enums],nodata=ds.nodata)


import hashlib
import numpy as np
from scripts.maritime_cloud_inventory import json_bytes

def original_support(rgb,masks):
    """Conservative inferred zero guard in addition to genuine original masks."""
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[0]!=3 or masks.shape!=rgb.shape:
        raise ValueError('Original RGB and corresponding three masks required')
    return np.all(masks==255,axis=0)&~np.all(rgb==0,axis=0)

def contributor_geometry_supported(geometry):
    """Require every frozen positive-area contributor; never repair or pad pixels."""
    from scripts.maritime_maxar_intake import area_matrix
    spacing=geometry['product_spacing_m'];px,py=geometry['area_phase_m']
    if not np.isfinite([spacing,px,py]).all() or not 0<spacing<=1 or min(px,py)<0:return False
    width=area_matrix(px,544,spacing,2.5).shape[1]
    height=area_matrix(py,544,spacing,2.5).shape[1]
    return geometry['source_pixel_window'][2:]==[width,height]

def verify_reused_original(path,source,receipt):
    """Reuse only the byte-pinned original of this exact frozen HTTP parent."""
    if receipt['source_id']!=source['source_id'] or receipt['original_http_identity']!=source['original_http_identity']:
        raise ValueError('Reused original belongs to a different frozen parent identity')
    pin=receipt['object']
    if path.stat().st_size!=source['original_http_identity']['bytes'] or path.stat().st_size!=pin['bytes']:
        raise ValueError('Reused original byte length changed')
    with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    if sha!=pin['sha256']:raise ValueError('Reused original byte SHA changed')

def verified_naip_core_plans(ds,source,doc):
    """Legacy remains the default; V2 requires an explicit frozen per-core version."""
    from scripts.maritime_maxar_intake import planned_cores
    from scripts.maritime_noaa_intake import verified_core_plans
    from src.maritime_aerial_geometry import original_contributor_geometry
    if 'core_plans' not in doc:return verified_core_plans(ds,source,doc)
    frozen=doc['core_plans'][source['source_id']]
    if not any('geometry_version' in g for g in frozen):return verified_core_plans(ds,source,doc)
    index={tuple(p['world_grid_cell']):p for p in planned_cores(ds,source)};seen=set()
    for plan in frozen:
        key=tuple(plan['world_grid_cell'])
        if key in seen or key not in index:raise ValueError('Frozen NAIP core grid absent or duplicated')
        seen.add(key);version=plan.get('geometry_version','legacy-window-v1')
        if version=='original-contributor-window-v2':
            actual=dict(index[key],**original_contributor_geometry(ds,*key))
        elif version=='legacy-window-v1':actual=index[key]
        else:raise ValueError('Unknown frozen NAIP geometry version')
        if plan!=actual:raise ValueError('Frozen NAIP contributor planning and actual parent disagree')
    return frozen

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
    from scripts.maritime_noaa_intake import sources_for_role,verified_core_plans,candidate_metadata,aerial_area_hr
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
            evidence_objects=source['metadata_evidence_objects']
            documents={key:verify_output(client,pin) for key,pin in evidence_objects.items()}
            assert source['dataset']['license']=='Public-Domain'
            assert 'public domain' in documents['official_usda_policy_object'].decode('utf8').lower()
            assert 'public domain' in documents['official_usgs_metadata_object'].decode('utf8').lower()
            original=args.scratch/'original.tif';identity=source['original_http_identity']
            if 'original_reuse' in source:
                reuse=source['original_reuse'];original_pin=reuse['object']
                parent_bucket,parent_name=parse_gs(original_pin['uri'])
                blob=client.bucket(parent_bucket).blob(parent_name,generation=int(original_pin['generation']));blob.reload()
                if int(blob.size)!=original_pin['bytes'] or blob.crc32c!=original_pin['crc32c']:
                    raise ValueError('Reused original object metadata changed')
                blob.download_to_filename(str(original),if_generation_match=int(original_pin['generation']),checksum='crc32c',timeout=600)
                verify_reused_original(original,source,reuse)
            else:
                with requests.get(source['href'],headers={'If-Match':identity['etag']},stream=True,timeout=120) as response:
                    response.raise_for_status();assert response.headers['ETag']==identity['etag']
                    with original.open('wb') as f:
                        for chunk in response.iter_content(1024*1024):f.write(chunk)
                assert original.stat().st_size==identity['bytes']
                original_pin=publish_file(original,'original/'+hashlib.sha256(source['source_id'].encode()).hexdigest()+'.tif')
            with rasterio.open(original) as ds:
                header=original_header(ds,source)
                explicit_void=(header['nodata'] is not None or any(flags!=['all_valid'] for flags in header['mask_flags'][:3]))
                source=dict(source,original_raster_header=header,
                    original_support_policy='all_original_RGB_masks255_and_no_RGB000',
                    original_void_mask_available=explicit_void,
                    stac_datetime_role='capture_calendar_day_hour_unverified')
                evidence=dict(source=source,**evidence_objects)
                source['source_evidence_pin']=publish_json(evidence,'source-evidence/'+hashlib.sha256(source['source_id'].encode()).hexdigest()+'.json')
                print(json.dumps(dict(source_id=source['source_id'],original_raster_header=header,source_original_object=original_pin,source_evidence_pin=source['source_evidence_pin'])),flush=True)
                for geometry in verified_naip_core_plans(ds,source,doc):
                    if len(records)>=args.max_cores:break
                    outcome=dict(source_id=source['source_id'],grid=geometry['world_grid_cell'],status='STARTED')
                    attempts.append(outcome);checkpoint()
                    if not contributor_geometry_supported(geometry):
                        outcome.update(status='REJECTED',reason='frozen_geometry_area_contributor_count_mismatch')
                        checkpoint();continue
                    window=Window(*geometry['source_pixel_window']);rgb=ds.read([1,2,3],window=window)
                    valid=original_support(rgb,ds.read_masks([1,2,3],window=window))
                    if not valid.all():
                        outcome.update(status='REJECTED',reason='original_context_mask_not_fully_valid');checkpoint();continue
                    native_transform=window_transform(window,ds.transform)
                    row=candidate_metadata(source,geometry,native_transform,original_pin,source['source_evidence_pin'])
                    row['original_mask_provenance']='all_original_RGB_masks255'
                    row['original_support_policy']=source['original_support_policy']
                    row['original_void_mask_available']=explicit_void
                    row['publication_datetime']=None
                    row['stac_datetime']=source['stac_datetime']
                    row['stac_datetime_role']='capture_calendar_day_hour_unverified'
                    row['split']='provisional_'+('training' if args.source_role=='train' else 'evaluation')+'_only'
                    row['split_exclusion_evidence']=dict(aerial_reservation_sha256=hashlib.sha256(policy_bytes).hexdigest(),
                        allocation=allocation_decisions[source['source_id']],evaluation_ports_count=len(policy['evaluation_ports']))
                    if 'protected_geometry' in doc:
                        row['split_exclusion_evidence']['protected_geometry']=doc['protected_geometry']
                        row['split_exclusion_evidence']['protected_core_buffer_m']=4000
                    if 'marine_screening' in doc:
                        row['marine_screening']=dict(doc['marine_screening'][source['source_id']][str(geometry['world_grid_cell'])],
                            gshhg=doc['gshhg'],marine_prior_only=True)
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
