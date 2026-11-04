"""GCE-only provisional academic RGB intake with explicitly inferred validity.

This adapter preserves captured source RGB, uses real 1360 m context and exact
metric area integration, and never certifies maritime content or native GSD.
Geographic mosaics are unsupported until a separate audited reprojection exists.
"""
import argparse
import io
import json
import math
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from PIL import Image
import rasterio
from rasterio.crs import CRS
from rasterio.windows import Window, transform as window_transform

from scripts.maritime_cloud_inventory import (create_only, digest_bytes, file_sha256,
    json_bytes, parse_gs, require_cloud_vm, verify_output)
from scripts.maritime_maxar_intake import (area_hr, core_geometry, pinned_http, write_native)
from src.cloud_config import GCP_PROJECT


def validate_evidence(evidence):
    if evidence.get('license') != 'CC-BY-SA-4.0':
        raise ValueError('Explicit academic-compatible CC-BY-SA-4.0 license evidence required')
    for key in ('license_evidence_url', 'resolution_evidence_url'):
        url=urlsplit(evidence.get(key,''))
        if url.scheme!='https' or not url.netloc:
            raise ValueError('Public HTTPS license and resolution evidence required')
    for key in ('source_id', 'attribution'):
        if not isinstance(evidence.get(key),str) or not evidence[key].strip():
            raise ValueError('Source identity and attribution required')
    resolution=evidence.get('publisher_product_resolution_m')
    if isinstance(resolution,bool) or not isinstance(resolution,(float,int)) or not math.isfinite(resolution) or not 0<resolution<=1:
        raise ValueError('Documented publisher product resolution must be <=1m')
    if 'native_acquisition_gsd_m' not in evidence:
        raise ValueError('Native acquisition GSD status must be explicit, null when unknown')
    native=evidence['native_acquisition_gsd_m']
    if native is not None and (isinstance(native,bool) or not isinstance(native,(float,int)) or not math.isfinite(native) or not 0<native<=1 or not evidence.get('native_gsd_evidence_url')):
        raise ValueError('Claimed native acquisition GSD needs separate evidence')
    if not isinstance(evidence.get('upstream_processing_history'),list) or not evidence['upstream_processing_history']:
        raise ValueError('Upstream processing history must be recorded')


def validate_header(ds,evidence):
    validate_evidence(evidence)
    t=ds.transform
    if (ds.crs is None or not ds.crs.is_projected or ds.crs.linear_units!='metre'
        or not all(math.isfinite(v) for v in list(t)[:6]) or not 0<t.a<=1
        or abs(t.e+t.a)>1e-10 or t.b!=0 or t.d!=0):
        raise ValueError('Only north-up square metric product grids <=1m are supported; geographic adapter unsupported')
    if ds.count!=3 or ds.dtypes!=('uint8',)*3 or tuple(c.name for c in ds.colorinterp)!=('red','green','blue'):
        raise ValueError('Original declared three-band uint8 RGB required')
    if abs(t.a-evidence['publisher_product_resolution_m'])>1e-6:
        raise ValueError('Measured metric spacing disagrees with publisher product resolution')
    return t.a


def validate_audit_grid(ds,audit):
    if (ds.crs!=CRS.from_string(audit['crs']) or list(ds.transform)[:6]!=audit['transform'][:6]
        or ds.width!=audit['width'] or ds.height!=audit['height'] or ds.count!=audit['count']
        or list(ds.dtypes)!=audit['dtypes'] or [c.name for c in ds.colorinterp]!=audit['colorinterp']):
        raise ValueError('Raster grid differs from pinned header audit')


def inferred_support(rgb):
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[0]!=3:
        raise ValueError('Support inference requires original RGB uint8')
    return (rgb.max(axis=0)>0) & (rgb.min(axis=0)<255)


def canonical_context(rgb,geometry,count=544):
    return area_hr(rgb,inferred_support(rgb),geometry,count=count)


def planned_cores(ds,footprint=None):
    bounds=ds.bounds if hasattr(ds,'bounds') else rasterio.transform.array_bounds(ds.height,ds.width,ds.transform)
    left,bottom,right,top=bounds
    result=[]
    for gy in range(math.ceil((bottom+40)/1280),math.floor((top-40)/1280)):
        for gx in range(math.ceil((left+40)/1280),math.floor((right-40)/1280)):
            geometry=core_geometry(ds,gx,gy)
            if footprint is not None:
                from shapely.geometry import box
                if not footprint.covers(box(*geometry['physical_context_bounds'])):
                    continue
            result.append(geometry)
    return result


def check_http(audit):
    parsed=urlsplit(audit['url'])
    if parsed.scheme!='https' or parsed.hostname!='spacenet-dataset.s3.amazonaws.com' or not parsed.path.endswith('.tif'):
        raise ValueError('Initial adapter accepts audited public SpaceNet RGB TIFF objects only')
    identity=pinned_http({'visual_href':audit['url'],'visual_http_head':audit['http']})
    if identity['bytes']!=audit['http']['bytes']:
        raise ValueError('Source HTTP byte length changed after audit')
    return identity


def run(args):
    instance=require_cloud_vm()  # Before any original raster pixel access.
    from google.cloud import storage
    client=storage.Client(project=args.project)
    audits_payload=Path(args.source_audit).read_bytes()
    evidence_payload=Path(args.source_evidence).read_bytes()
    audits=[json.loads(line) for line in audits_payload.decode('utf8').splitlines() if line.strip()]
    evidence_document=json.loads(evidence_payload)
    evidence_by_url=evidence_document['sources']
    scratch=Path(args.scratch);scratch.mkdir(parents=True,exist_ok=True)
    bucket_name,prefix=parse_gs(args.output_prefix.rstrip('/'))
    bucket=client.bucket(bucket_name)
    def publish(name,data):
        return {'uri':f'gs://{bucket_name}/{prefix}/{name}',**create_only(bucket.blob(prefix+'/'+name),data)}
    configuration=dict(schema='academic-raster-intake-configuration-v1',
        source_audit_sha256=digest_bytes(audits_payload),source_evidence_sha256=digest_bytes(evidence_payload),
        worker_sha256=file_sha256(__file__),limit=args.limit,mode='metric-only',
        physical_core_m=1280,observed_context_m=1360,validity_tier='B',
        mask_method='inferred_maxRGB_gt0_and_minRGB_lt255_not_producer_verified',
        target_resolution_m=2.5,final_dataset_ready=False)
    publish('configuration.json',json_bytes(configuration))
    records=[];rejections=[]
    for audit in audits:
        if audit['url'] not in evidence_by_url:
            continue
        evidence=evidence_by_url[audit['url']]
        try:
            validate_evidence(evidence)
            identity=check_http(audit)
            footprint=None;footprint_ref=None
            if evidence.get('footprint_path'):
                from shapely.geometry import shape
                from shapely.ops import unary_union
                fp=Path(args.source_evidence).parent/evidence['footprint_path']
                raw=fp.read_bytes();document=json.loads(raw)
                # Evidence explicitly names footprint CRS; never silently guess it.
                if CRS.from_string(evidence['footprint_crs'])!=CRS.from_string(audit['crs']):
                    raise ValueError('Footprint CRS differs from source grid')
                footprint=unary_union([shape(f['geometry']) for f in document['features']])
                footprint_ref=publish('source-evidence/'+digest_bytes(raw)+'.geojson',raw)
            source_ref=publish('source-evidence/'+digest_bytes(json_bytes({'audit':audit,'evidence':evidence}))+'.json',json_bytes({'audit':audit,'evidence':evidence}))
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_HEADERS='If-Match: '+identity['etag'],
                CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif',GDAL_HTTP_MAX_RETRY=2,GDAL_HTTP_RETRY_DELAY=2):
                with rasterio.open('/vsicurl/'+audit['url']) as ds:
                    validate_header(ds,evidence);validate_audit_grid(ds,audit)
                    for geometry in planned_cores(ds,footprint):
                        if len(records)>=args.limit:
                            break
                        sample_id='academic-'+digest_bytes(json_bytes([audit['url'],identity['etag'],ds.crs.to_string(),geometry['world_grid_cell']]))[:32]
                        existing=bucket.blob(prefix+'/candidates/'+sample_id+'.json')
                        if existing.exists():
                            existing.reload()
                            record=json.loads(existing.download_as_bytes(if_generation_match=int(existing.generation),checksum='crc32c'))
                            for key in ('native_context','hr','overview'):
                                verify_output(client,record[key])
                            records.append(record);continue
                        win=Window(*geometry['source_pixel_window'])
                        rgb=ds.read([1,2,3],window=win)
                        mask=inferred_support(rgb)
                        if not mask.all():
                            rejections.append(dict(sample_id=sample_id,reason='inferred_context_support_incomplete',source_url=audit['url'],invalid_fraction=float(1-mask.mean())))
                            continue
                        context=area_hr(rgb,mask,geometry)
                        hr=np.ascontiguousarray(context[:,16:-16,16:-16])
                        c,r,w,h=geometry['owned_core_pixel_window'];wc,wr,_,_=geometry['source_pixel_window']
                        useful=rgb[:,r-wr:r-wr+h,c-wc:c-wc+w]
                        native_transform=window_transform(win,ds.transform)
                        metadata=dict(schema='academic-raster-candidate-v1',sample_id=sample_id,
                            source_kind='academic_raster',source_provenance=evidence,source_header_audit=audit,
                            source_evidence=source_ref,source_http_pin=identity,source_href=audit['url'],
                            license=evidence['license'],native_gsd_m=evidence['native_acquisition_gsd_m'],
                            publisher_product_resolution_m=evidence['publisher_product_resolution_m'],
                            original_transform=list(ds.transform)[:6],native_window_transform=list(native_transform)[:6],
                            output_crs=ds.crs.to_string(),source_crs=ds.crs.to_string(),validity_tier='B',
                            original_mask_provenance='inferred_rgb_extremes_not_producer_verified',
                            mask_inference_uncertainty='Black/white exclusions cannot detect all internal fill, mosaic seams, cloud or unobserved pixels; real black/white may be excluded.',
                            release_footprint=footprint_ref,canonical_hr_method='exact_metric_fractional_area_float64_then_normalized_float32',
                            decoded_native_context_sha256=digest_bytes(rgb.tobytes()),
                            decoded_useful_core_sha256=digest_bytes(useful.tobytes()),
                            requires_semantic_review=True,requires_native_quality_review=True,final_eligible=False,**geometry)
                        path=scratch/(sample_id+'.tif')
                        write_native(path,rgb,mask,ds.crs,native_transform)
                        native=publish('native-context/'+sample_id+'.tif',path.read_bytes());path.unlink()
                        buffer=io.BytesIO()
                        np.savez_compressed(buffer,hr_context=context,hr=hr,metadata_json=np.asarray(json.dumps(metadata,sort_keys=True)))
                        hrref=publish('hr/'+sample_id+'.npz',buffer.getvalue())
                        image=Image.fromarray(np.moveaxis(useful,0,-1));image.thumbnail((400,400))
                        buffer=io.BytesIO();image.save(buffer,format='PNG')
                        preview=publish('review/overview/'+sample_id+'.png',buffer.getvalue())
                        record=dict(**metadata,native_context=native,hr=hrref,overview=preview)
                        publish('candidates/'+sample_id+'.json',json_bytes(record));records.append(record)
                        print(json.dumps({'completed':len(records),'sample_id':sample_id,'status':'PROVISIONAL'}),flush=True)
            try:
                check_http(audit)  # Confirm original object identity after range reads.
            except Exception as exc:
                # Do not issue a usable manifest if an object changed during reads.
                raise RuntimeError('Source identity post-read verification failed; candidates remain unpublished in any manifest') from exc
        except (ValueError,rasterio.errors.RasterioError) as exc:
            rejections.append(dict(source_url=audit['url'],reason=str(exc)))
        if len(records)>=args.limit:
            break
    result=dict(schema='academic-raster-candidate-v1',configuration=configuration,
        vm_instance_id=instance,status='PROVISIONAL_NATIVE_CANDIDATES',final_dataset_ready=False,
        candidates=records,candidate_count=len(records),source_rejections=rejections,
        semantic_accepted_count=0)
    ref=publish('candidate-manifest.json',json_bytes(result))
    print(json.dumps({'manifest':ref,'candidate_count':len(records)}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-audit',required=True,help='Header probe JSONL; no pixel downloads occur locally')
    parser.add_argument('--source-evidence',required=True,help='JSON sources map keyed by audited URL')
    parser.add_argument('--output-prefix',required=True)
    parser.add_argument('--scratch',required=True)
    parser.add_argument('--limit',type=int,default=48)
    parser.add_argument('--project',default=GCP_PROJECT)
    args=parser.parse_args()
    if args.limit<=0:
        parser.error('--limit must be positive')
    run(args)
