"""GCE-only Maxar coastal COG intake preserving original sub-metre RGB pixels.

Physical cores are 1280m with a real 40m halo, independently area-integrated to
2.5m HR. Native useful bytes count source pixel centers in half-open cores once;
all positive-area contributors are retained in native context TIFFs. RGB JPEG
COG decoding is preserved losslessly; no extra uint8 resampling or upsampling.
This is intake evidence, never a semantic acceptance or final ready marker.
"""
import argparse
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import time
from urllib.request import Request, urlopen
from urllib.parse import urlsplit

import numpy as np
from PIL import Image
import rasterio
from rasterio.crs import CRS
from rasterio.enums import ColorInterp, MaskFlags
from rasterio.transform import Affine
from rasterio.windows import Window, transform as window_transform
from rasterio.warp import transform as project_xy

from scripts.maritime_cloud_inventory import (create_only, digest_bytes, file_sha256,
    json_bytes, parse_gs, require_cloud_vm, verify_output)
from scripts.maritime_planet_intake import area_matrix
from src.cloud_config import GCP_PROJECT


def validate_header(ds, native_gsd):
    try:
        gsd=float(native_gsd)
    except (ValueError, TypeError):
        raise ValueError('Native source GSD unknown')
    if not math.isfinite(gsd) or not 0<gsd<=1:
        raise ValueError('Native source GSD must be finite and <=1m')
    t=ds.transform
    if (ds.crs is None or not ds.crs.is_projected or ds.crs.linear_units!='metre'
        or not all(math.isfinite(v) for v in list(t)[:6]) or not 0<t.a<=1
        or abs(t.e+t.a)>1e-10 or t.b!=0 or t.d!=0):
        raise ValueError('Original metric north-up square pixel spacing <=1m required')
    if ds.count!=3 or ds.dtypes!=('uint8',)*3 or ds.colorinterp!=(ColorInterp.red,ColorInterp.green,ColorInterp.blue):
        raise ValueError('Original declared uint8 RGB required; no grayscale colour fabrication')
    if not all(MaskFlags.per_dataset in band and MaskFlags.nodata not in band
               for band in ds.mask_flag_enums):
        raise ValueError('Original embedded dataset mask required')
    return t.a


def core_geometry(ds, gx, gy):
    spacing=ds.transform.a
    left,bottom=gx*1280.,gy*1280.
    right,top=left+1280.,bottom+1280.
    context=[left-40,bottom-40,right+40,top+40]
    inverse=~ds.transform
    col,row=inverse*(context[0],context[3])
    endcol,endrow=inverse*(context[2],context[1])
    startcol,startrow=math.floor(col+1e-9),math.floor(row+1e-9)
    stopcol,stoprow=math.ceil(endcol-1e-9),math.ceil(endrow-1e-9)
    if min(startcol,startrow)<0 or stopcol>ds.width or stoprow>ds.height:
        raise ValueError('Real original source context outside raster')
    corecol,corerow=inverse*(left,top)
    coreendcol,coreendrow=inverse*(right,bottom)
    # Half-open center ownership removes native edge-pixel double counting.
    ownedcol,ownedrow=math.ceil(corecol-.5-1e-9),math.ceil(corerow-.5-1e-9)
    ownedstopcol,ownedstoprow=math.ceil(coreendcol-.5-1e-9),math.ceil(coreendrow-.5-1e-9)
    owned=[ownedcol,ownedrow,ownedstopcol-ownedcol,ownedstoprow-ownedrow]
    source_left,source_top=ds.transform*(startcol,startrow)
    return dict(world_grid_cell=[gx,gy],physical_core_bounds=[left,bottom,right,top],
        physical_context_bounds=context,source_pixel_window=[startcol,startrow,stopcol-startcol,stoprow-startrow],
        source_fractional_context_window=[col,row,endcol-col,endrow-row],
        owned_core_pixel_window=owned,useful_native_rgb_bytes=owned[2]*owned[3]*3,
        product_spacing_m=spacing,area_phase_m=[context[0]-source_left,source_top-context[3]],
        hr_context_transform=list(Affine(2.5,0,context[0],0,-2.5,context[3]))[:6],
        hr_core_transform=list(Affine(2.5,0,left,0,-2.5,top))[:6])


def validate_source_grid(ds,source):
    if (ds.crs!=CRS.from_epsg(source['epsg'])
        or list(ds.transform)[:6]!=source['visual_transform'][:6]
        or [ds.height,ds.width]!=source['visual_shape']):
        raise ValueError('COG and pinned STAC grid/dimensions disagree')


def area_hr(rgb, valid, geometry, count=544):
    spacing=geometry['product_spacing_m']
    px,py=geometry['area_phase_m']
    wx,wy=area_matrix(px,count,spacing,2.5),area_matrix(py,count,spacing,2.5)
    if (rgb.dtype!=np.uint8 or rgb.shape!=(3,wy.shape[1],wx.shape[1])
        or valid.dtype!=np.bool_ or valid.shape!=rgb.shape[1:]):
        raise ValueError('Original area contributor shape/type mismatch')
    if not valid.all():
        raise ValueError('Incomplete original contributor support')
    result=np.stack([np.asarray((wy@channel.astype(np.float64))@wx.T) for channel in rgb])
    return (result/255.).astype(np.float32)


def water_screen_result(values):
    if values.shape!=(17,17) or not {'water','coverage'}.issubset(values.dtype.names or ()):
        raise ValueError('Expected structured 17-square WorldCover coarse pixels')
    water,coverage=values['water'],values['coverage']
    if not np.isin(water,[0,1]).all() or not np.isin(coverage,[0,1]).all():
        raise ValueError('WorldCover water/coverage values must be binary')
    known_fraction=float(np.mean(coverage))
    water_fraction=float(np.mean(water))
    status=('UNKNOWN_COVERAGE' if known_fraction<.9 else
            'WATER_PROXY_PASSED' if water_fraction>=.15 else 'INLAND_REJECTED')
    return dict(status=status,discard=status=='INLAND_REJECTED',water_fraction=water_fraction,
        covered_fraction=known_fraction,maritime_support_known=status=='WATER_PROXY_PASSED',
        vessel_presence='unconfirmed',maritime_stratum='unconfirmed',
        dataset='ESA/WorldCover/v200',dataset_year=2021,landcover_class=80,
        method='nearest_class_sample_at_80m_centers_17x17_over_real_1360m_context',
        source_resolution_m=10,sampled_grid_spacing_m=80,coverage_threshold=.9,water_threshold=.15,
        pixel_payload_sha256=digest_bytes(values.tobytes()),
        license='CC-BY-4.0',temporal_caveat='2021 landcover proxy may differ from source acquisition')


def water_screen(ee,expression,geometry,epsg):
    left,bottom,right,top=geometry['physical_context_bounds']
    grid=dict(dimensions=dict(width=17,height=17),crsCode=f'EPSG:{epsg}',
        affineTransform=dict(scaleX=80,shearX=0,translateX=left,shearY=0,scaleY=-80,translateY=top))
    values=ee.data.computePixels(dict(expression=expression,fileFormat='NUMPY_NDARRAY',
        bandIds=['water','coverage'],grid=grid))
    result=water_screen_result(values)
    result.update(grid=grid,water_values=values['water'].tolist(),coverage_values=values['coverage'].tolist())
    return result


def planned_cores(ds, source):
    ports=source['port_AOI_matches']
    positions=[]
    for port in ports:
        x,y=project_xy('EPSG:4326',ds.crs,[port['longitude']],[port['latitude']])
        positions.append((x[0],y[0],port))
    bounds=ds.bounds
    candidates=[]
    for gx in range(math.ceil((bounds.left+40)/1280), math.floor((bounds.right-40)/1280)):
        for gy in range(math.ceil((bounds.bottom+40)/1280),math.floor((bounds.top-40)/1280)):
            cx,cy=(gx+.5)*1280,(gy+.5)*1280
            matches=[(math.hypot(cx-x,cy-y),port) for x,y,port in positions
                     if math.hypot(cx-x,cy-y)<=8000+math.sqrt(2)*640]
            if not matches:
                continue
            distance,port=min(matches,key=lambda pair:pair[0])
            geometry=core_geometry(ds,gx,gy)
            geometry.update(nearest_port=port,port_center_distance_m=distance)
            candidates.append(geometry)
    return sorted(candidates,key=lambda row:(row['port_center_distance_m'],row['world_grid_cell']))


def validate_source(source):
    href=source['visual_href']
    parsed=urlsplit(href)
    if parsed.scheme!='https' or parsed.hostname!='maxar-opendata.s3.amazonaws.com':
        raise ValueError('Expected official public Maxar Open Data HTTPS COG')
    if source['event_license']!='CC-BY-NC-4.0':
        raise ValueError('Event academic-free license grant unverified')
    if source['rgb_band_names']!=['red','green','blue'] or source['platform']=='WV01':
        raise ValueError('Source does not declare RGB')
    try:
        gsd=float(source['native_gsd_m'])
    except (ValueError,TypeError):
        raise ValueError('Native source GSD unknown')
    if not math.isfinite(gsd) or not 0<gsd<=1:
        raise ValueError('Native source GSD exceeds 1m')
    if not source['catalog_id'] or not source['port_AOI_matches']:
        raise ValueError('Acquisition identity and screened port AOI required')


def pinned_http(source):
    """Verify HTTP range access and freeze exact object identity before GDAL."""
    href=source['visual_href']
    with urlopen(Request(href,method='HEAD'),timeout=60) as response:
        headers=response.headers
        identity=dict(etag=headers.get('ETag'),bytes=int(headers.get('Content-Length','0')),
                      last_modified=headers.get('Last-Modified'))
    if not identity['etag'] or identity['bytes']<=0:
        raise ValueError('Original COG HTTP identity missing')
    screened=source['visual_http_head']
    if screened.get('etag') and screened['etag']!=identity['etag']:
        raise ValueError('Original COG changed after metadata screening')
    with urlopen(Request(href,headers={'Range':'bytes=0-0','If-Match':identity['etag']}),timeout=60) as response:
        if response.status!=206 or not response.headers.get('Content-Range','').startswith('bytes 0-0/') or len(response.read(2))!=1:
            raise ValueError('Original COG does not support bounded HTTP ranges')
    return identity


def write_native(path,rgb,valid,crs,transform):
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(path,'w',driver='GTiff',width=rgb.shape[2],height=rgb.shape[1],
            count=3,dtype='uint8',crs=crs,transform=transform,compress='deflate',tiled=True,
            blockxsize=256,blockysize=256) as out:
            out.write(rgb);out.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue)
            out.write_mask(valid.astype(np.uint8)*255)


def excluded_core_ids(manifests):
    identities=set()
    for manifest in manifests:
        if manifest.get('schema')!='maxar-mixed-native-v1':
            raise ValueError('Prior exclusion manifest schema must be mixed-native Maxar')
        rows=manifest.get('candidates')
        if not isinstance(rows,list) or manifest.get('candidate_count')!=len(rows):
            raise ValueError('Prior exclusion manifest candidate count mismatch')
        for row in rows:
            physical=dict(acquisition=row['source']['catalog_id'],crs=f'EPSG:{row["source"]["epsg"]}',
                          physical_core_bounds=row['physical_core_bounds'])
            identifier=digest_bytes(json_bytes(physical))
            if row['physical_core_id']!=identifier or row['sample_id']!=identifier:
                raise ValueError('Prior exclusion physical acquisition/core identity mismatch')
            identities.add(identifier)
    return identities


def run(args):
    instance=require_cloud_vm()
    from google.cloud import storage
    client=storage.Client(project=GCP_PROJECT)
    payload=Path(args.sources).read_bytes()
    inventory=json.loads(payload)
    sources=inventory['sources'][:args.max_sources]
    scratch=Path(args.scratch);scratch.mkdir(parents=True,exist_ok=True)
    bucket_name,prefix=parse_gs(args.output_prefix.rstrip('/'))
    bucket=client.bucket(bucket_name)
    def publish(name,data):
        return {'uri':f'gs://{bucket_name}/{prefix}/{name}',**create_only(bucket.blob(prefix+'/'+name),data)}
    exclusion_refs=[];exclusion_manifests=[]
    for uri in args.exclude_manifest:
        excluded_bucket,excluded_name=parse_gs(uri)
        blob=client.bucket(excluded_bucket).blob(excluded_name);blob.reload()
        data=blob.download_as_bytes(if_generation_match=int(blob.generation),checksum='crc32c')
        ref=dict(uri=uri,generation=str(blob.generation),bytes=len(data),crc32c=blob.crc32c,
                 sha256=digest_bytes(data))
        exclusion_refs.append(ref);exclusion_manifests.append(json.loads(data))
    excluded=excluded_core_ids(exclusion_manifests)
    configuration=dict(schema='maxar-mixed-native-v1',source_inventory_sha256=digest_bytes(payload),
        worker_sha256=file_sha256(__file__),max_cores=args.max_cores,max_sources=args.max_sources,physical_core_m=1280,real_halo_m=40,
        coarse_water_screen=args.coarse_water_screen,
        exclude_manifests=exclusion_refs,excluded_physical_core_count=len(excluded),
        excluded_core_ids_sha256=digest_bytes(json_bytes(sorted(excluded))),
        native_byte_accounting='source pixel centers in half-open physical core; original RGB bytes only',
        canonical_hr='exact_physical_area_float64_to_float32_2.5m_context544_core512')
    publish('configuration.json',json_bytes(configuration))
    publish('source-inventory.json',payload)
    ee,water_expression,water_error=None,None,None
    if args.coarse_water_screen:
        try:
            import ee
            import google.auth
            credentials,_=google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
            ee.Initialize(credentials=credentials,project=GCP_PROJECT,
                          opt_url='https://earthengine-highvolume.googleapis.com')
            landcover=ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
            water_expression=landcover.eq(80).unmask(0,False).rename('water').toUint8().addBands(
                landcover.mask().gt(0).unmask(0,False).rename('coverage').toUint8())
        except Exception as exc:
            water_error=str(exc)
    # Source metadata documents are archived exactly before any pixel intake.
    for source in sources:
        validate_source(source)
        for key in ('item_metadata','event_metadata','acquisition_metadata'):
            metadata=source[key]; path=Path(args.metadata_dir)/(metadata['sha256']+'.json')
            data=path.read_bytes()
            if len(data)!=metadata['bytes'] or digest_bytes(data)!=metadata['sha256']:
                raise ValueError('Pinned source STAC metadata changed')
            publish('metadata/'+metadata['sha256']+'.json',data)
    records=[];attempts=[];seen=set(excluded);rejections=[]
    # Geographic source/site balancing is frozen by inventory order, then one
    # core per source per round. Completed checkpoint pins are verified on resume.
    checkpoint=scratch/'checkpoint.json'
    if checkpoint.exists():
        old=json.loads(checkpoint.read_text())
        if old['configuration']!=configuration:
            raise ValueError('Immutable intake checkpoint configuration conflict')
        records=old['records'];attempts=old['attempts'];rejections=old['rejections']
        for row in records:
            for key in ('native_context','hr','overview'):
                verify_output(client,row[key])
            seen.add(row['physical_core_id'])
    def save():
        checkpoint.write_bytes(json_bytes(dict(configuration=configuration,records=records,attempts=attempts,rejections=rejections)))
    source_plans=[]
    for source in sources:
        if len(source_plans)>=args.max_sources:
            break
        try:
            identity=pinned_http(source)
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_HEADERS='If-Match: '+identity['etag'],
                CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif',GDAL_HTTP_MAX_RETRY=2,GDAL_HTTP_RETRY_DELAY=2):
                with rasterio.open('/vsicurl/'+source['visual_href']) as ds:
                    validate_header(ds,source['native_gsd_m'])
                    validate_source_grid(ds,source)
                    cores=planned_cores(ds,source)
            source_plans.append((source,identity,cores))
        except Exception as exc:
            rejections.append(dict(source_id=source['source_id'],reason=str(exc)));save()
    ordinal=0
    while len(records)<args.max_cores:
        has_planned_core=False
        for source,identity,cores in source_plans:
            if ordinal>=len(cores) or len(records)>=args.max_cores:
                continue
            has_planned_core=True
            geometry=cores[ordinal]
            physical=dict(acquisition=source['catalog_id'],crs=f'EPSG:{source["epsg"]}',
                          physical_core_bounds=geometry['physical_core_bounds'])
            core_id=digest_bytes(json_bytes(physical))
            if core_id in seen:
                continue
            previous=[row for row in attempts if row['core_id']==core_id]
            if previous and previous[-1]['status'] in ('REJECTED','FAILED'):
                continue
            coarse=None
            if args.coarse_water_screen:
                name='screening/coarse-water/'+core_id+'.json'
                previous_screen=bucket.blob(prefix+'/'+name)
                if previous_screen.exists():
                    previous_screen.reload()
                    coarse=json.loads(previous_screen.download_as_bytes(
                        if_generation_match=int(previous_screen.generation),checksum='crc32c'))
                else:
                    try:
                        if water_error:
                            raise RuntimeError(water_error)
                        coarse=water_screen(ee,water_expression,geometry,source['epsg'])
                    except Exception as exc:
                        coarse=dict(status='UNKNOWN_REQUEST_ERROR',discard=False,maritime_support_known=False,
                            vessel_presence='unconfirmed',maritime_stratum='unconfirmed',reason=str(exc))
                ref=publish(name,json_bytes(coarse))
                coarse['evidence']=ref
                if coarse['discard']:
                    attempts.append(dict(core_id=core_id,attempt=0,status='REJECTED',
                        source_id=source['source_id'],reason='coarse_worldcover_inland',coarse_water_screen=coarse))
                    save();continue
            for attempt in range(len(previous)+1,4):
                state=dict(core_id=core_id,attempt=attempt,status='STARTED',source_id=source['source_id'])
                attempts.append(state);save()
                try:
                    win=Window(*geometry['source_pixel_window'])
                    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_HEADERS='If-Match: '+identity['etag'],
                        CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif',GDAL_HTTP_MAX_RETRY=2,GDAL_HTTP_RETRY_DELAY=2):
                        with rasterio.open('/vsicurl/'+source['visual_href']) as ds:
                            validate_header(ds,source['native_gsd_m'])
                            validate_source_grid(ds,source)
                            rgb=ds.read([1,2,3],window=win)
                            mask=np.all(ds.read_masks([1,2,3],window=win)==255,axis=0)
                            native_transform=window_transform(win,ds.transform);crs=ds.crs
                    if not mask.all():
                        state.update(status='REJECTED',reason='original_context_mask_not_fully_valid');save();break
                    context=area_hr(rgb,mask,geometry)
                    hr=np.ascontiguousarray(context[:,16:-16,16:-16])
                    path=scratch/(core_id+'.tif')
                    write_native(path,rgb,mask,crs,native_transform)
                    native=publish('native-context/'+core_id+'.tif',path.read_bytes());path.unlink()
                    metadata=dict(schema='maxar-mixed-native-v1',sample_id=core_id,physical_core_id=core_id,
                        source=source,source_http_identity=identity,source_href=source['visual_href'],
                        native_gsd_m=source['native_gsd_m'],original_transform=source['visual_transform'][:6],
                        original_mask_provenance='original_embedded_dataset_mask',**geometry)
                    metadata['coarse_water_screen']=coarse
                    buffer=io.BytesIO()
                    np.savez_compressed(buffer,hr_context=context,hr=hr,metadata_json=np.asarray(json.dumps(metadata,sort_keys=True)))
                    hrref=publish('hr/'+core_id+'.npz',buffer.getvalue())
                    # Overview is triage only; it is excluded from native-byte totals.
                    col,row,width,height=geometry['owned_core_pixel_window'];wc,wr,_,_=geometry['source_pixel_window']
                    useful=rgb[:,row-wr:row-wr+height,col-wc:col-wc+width]
                    image=Image.fromarray(np.moveaxis(useful,0,-1));image.thumbnail((400,400))
                    buffer=io.BytesIO();image.save(buffer,format='PNG')
                    preview=publish('review/overview/'+core_id+'.png',buffer.getvalue())
                    record=dict(**metadata,native_context=native,hr=hrref,overview=preview,
                        decoded_native_context_sha256=digest_bytes(rgb.tobytes()),
                        decoded_useful_core_sha256=digest_bytes(useful.tobytes()),
                        requires_semantic_review=True,requires_native_quality_review=True,final_eligible=False)
                    ref=publish('candidates/'+core_id+'.json',json_bytes(record));record['candidate_record']=ref
                    records.append(record);seen.add(core_id);state['status']='COMPLETED';save()
                    print(json.dumps({'completed':len(records),'sample_id':core_id,
                        'useful_native_rgb_bytes':geometry['useful_native_rgb_bytes']}),flush=True)
                    del rgb,mask,context,hr,useful
                    break
                except Exception as exc:
                    state.update(status='FAILED' if attempt==3 else 'TRANSIENT_ERROR',reason=str(exc));save()
                    if attempt<3:
                        time.sleep(5*attempt)
        if not has_planned_core:
            break
        ordinal+=1
    result=dict(schema='maxar-mixed-native-v1',configuration=configuration,vm_instance_id=instance,
        status='PROVISIONAL_NATIVE_CANDIDATES',final_dataset_ready=False,candidates=records,
        candidate_count=len(records),useful_native_rgb_bytes=sum(row['useful_native_rgb_bytes'] for row in records),
        source_rejections=rejections,attempts=attempts,source_count=len(source_plans),semantic_accepted_count=0)
    ref=publish('candidate-manifest.json',json_bytes(result))
    print(json.dumps({'manifest':ref,'candidate_count':len(records),'useful_native_rgb_bytes':result['useful_native_rgb_bytes']}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',required=True)
    parser.add_argument('--metadata-dir',required=True)
    parser.add_argument('--scratch',required=True)
    parser.add_argument('--output-prefix',required=True)
    parser.add_argument('--max-cores',type=int,default=8)
    parser.add_argument('--max-sources',type=int,default=16)
    parser.add_argument('--coarse-water-screen',action='store_true')
    parser.add_argument('--exclude-manifest',action='append',default=[],
                        help='Completed prior Maxar manifest URI; generation/CRC/SHA frozen in configuration')
    run(parser.parse_args())
