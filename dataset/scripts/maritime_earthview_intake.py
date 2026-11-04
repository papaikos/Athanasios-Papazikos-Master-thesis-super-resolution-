"""GCE-only same-acquisition EarthView RGB mosaic candidate intake.

Sixteen adjacent original 384-square TOA chips are stitched without
resampling and converted using one recorded whole-mosaic display mapping.
The observed central 1360 m context is exact-area reduced to 2.5 m.
Every source contributor must pass original declared RGB nodata masks. Output
is provisional: radiometric continuity and maritime quality need independent
review; actual original RGB band interpretation and masks remain authoritative.
"""
import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import io
import json
import os
from pathlib import Path
import threading
import time
import uuid
from urllib.error import HTTPError,URLError
from urllib.parse import urlsplit
from urllib.request import Request,urlopen
from urllib3.exceptions import HTTPError as TransportHTTPError
from src.cloud_config import GCP_PROJECT

import numpy as np
from PIL import Image
import rasterio
import requests
from rasterio.crs import CRS
from rasterio.enums import ColorInterp,MaskFlags
from rasterio.io import MemoryFile
from rasterio.transform import Affine

from scripts.maritime_cloud_inventory import (create_only,digest_bytes,file_sha256,
    json_bytes,parse_gs,require_cloud_vm,verify_output)
from scripts.maritime_maxar_intake import area_hr

HOST='satellogic-earthview.s3.us-west-2.amazonaws.com'
ROOT='https://'+HOST+'/'


def official_url(url,suffix):
    parsed=urlsplit(url)
    if parsed.scheme!='https' or parsed.hostname!=HOST or parsed.query or parsed.fragment or not parsed.path.endswith(suffix):
        raise ValueError('Official public EarthView object URL required')
    return url


def validate_evidence(evidence):
    if evidence.get('license')!='CC-BY-4.0' or isinstance(evidence.get('native_resolution_m'),bool) or evidence.get('native_resolution_m')!=1.:
        raise ValueError('Documented native1m EarthView CC-BY-4.0 source required')
    for key in ('license_evidence_url','native_resolution_evidence_url'):
        parsed=urlsplit(evidence.get(key,''))
        if parsed.scheme!='https' or not parsed.netloc:
            raise ValueError('Official license and native-resolution evidence required')
    if not evidence.get('attribution') or not isinstance(evidence.get('upstream_processing_history'),list) or not evidence['upstream_processing_history']:
        raise ValueError('Attribution and upstream VISUAL processing history required')


def validate_mosaic(items,evidence,product='VISUAL'):
    validate_evidence(evidence)
    if product not in ('VISUAL','TOA'):
        raise ValueError('Explicit EarthView VISUAL or TOA product required')
    if len(items)!=16:
        raise ValueError('Exactly16 original adjacent chips required; no missing-tile filling')
    identities=set();ids=set();urls=set();placements=[]
    for item in items:
        p=item['properties'];t=p['proj:transform']
        time=datetime.fromisoformat(p['datetime'])
        if time.tzinfo is None:
            raise ValueError('Full timezone-aware acquisition timestamp required')
        identity=(p['datetime'],p['platform'],int(p['proj:epsg']),p['gsd'],p['satl:product_name'])
        identities.add(identity)
        if (p['license']!='CC-BY-4.0' or isinstance(p['gsd'],bool) or p['gsd']!=1. or p['satl:product_name']!='L1'
            or p['proj:shape']!=[384,384] or len(t)<6
            or t[:2]!=[1.,0.] or t[3:5]!=[0.,-1.] or not np.isfinite(t[:6]).all()):
            raise ValueError('Exact1m north-up384 RGB product metadata required')
        crs=CRS.from_epsg(int(p['proj:epsg']))
        if not crs.is_projected or crs.linear_units!='metre':
            raise ValueError('Metric projected native chip grid required')
        asset='visual' if product=='VISUAL' else 'analytic'
        url=official_url(item['assets'][asset]['href'],'_'+product+'.tif')
        if item['id'] in ids or url in urls:
            raise ValueError('Duplicate EarthView chip identity or original URL')
        ids.add(item['id']);urls.add(url);placements.append((float(t[2]),float(t[5]),item))
    if len(identities)!=1:
        raise ValueError('Exact full acquisition timestamp/platform/EPSG/GSD/product match required; no temporal mosaicking')
    xs=sorted({x for x,y,item in placements});ys=sorted({y for x,y,item in placements},reverse=True)
    if len(xs)!=4 or len(ys)!=4 or np.diff(xs).tolist()!=[384.]*3 or np.diff(ys).tolist()!=[-384.]*3:
        raise ValueError('Contiguous4x4 nonoverlapping original chip grid required')
    cells=[];occupied=set()
    for x,y,item in placements:
        col=xs.index(x)*384;row=ys.index(y)*384
        if (col,row) in occupied:
            raise ValueError('Duplicate original grid cell')
        occupied.add((col,row));cells.append(dict(item=item,col=col,row=row))
    left,top=xs[0],ys[0];identity=next(iter(identities))
    return dict(acquisition_identity=list(identity),input_product=product,
        source_rgb_dtype='uint8' if product=='VISUAL' else 'uint16',source_crs='EPSG:'+str(identity[2]),
        output_crs='EPSG:'+str(identity[2]),mosaic_bounds=[left,top-1536,left+1536,top],
        mosaic_transform=[1.,0.,left,0.,-1.,top],mosaic_shape=[1536,1536],
        source_pixel_window=[88,88,1360,1360],owned_core_pixel_window=[128,128,1280,1280],
        physical_context_bounds=[left+88,top-1448,left+1448,top-88],
        physical_core_bounds=[left+128,top-1408,left+1408,top-128],
        product_spacing_m=1.,area_phase_m=[0.,0.],useful_native_rgb_bytes=1280*1280*3*(1 if product=='VISUAL' else 2),
        native_context_transform=[1.,0.,left+88,0.,-1.,top-88],
        hr_context_transform=[2.5,0,left+88,0,-2.5,top-88],
        hr_core_transform=[2.5,0,left+128,0,-2.5,top-128],
        cells=sorted(cells,key=lambda cell:(cell['row'],cell['col'])))


def validate_header(ds,item,product='VISUAL'):
    p=item['properties'];t=Affine(*p['proj:transform'][:6])
    if (ds.crs!=CRS.from_epsg(int(p['proj:epsg'])) or ds.transform!=t
        or ds.width!=384 or ds.height!=384):
        raise ValueError('Actual original chip header must match exact STAC grid')
    if product=='VISUAL':
        if ds.count!=3 or ds.dtypes!=('uint8',)*3 or ds.colorinterp!=(ColorInterp.red,ColorInterp.green,ColorInterp.blue):
            raise ValueError('Original VISUAL3-band RGB uint8 required')
        indices=[1,2,3]
    elif product=='TOA':
        if ds.count!=4 or ds.dtypes!=('uint16',)*4:
            raise ValueError('Original TOA4-band uint16 required')
        indices=[]
        for color in (ColorInterp.red,ColorInterp.green,ColorInterp.blue):
            matches=[i+1 for i,value in enumerate(ds.colorinterp) if value==color]
            if len(matches)!=1:
                raise ValueError('Original TOA explicit unique RGB color interpretation required')
            indices.append(matches[0])
    else:
        raise ValueError('Unsupported original product')
    if ds.nodata!=0 or not all(MaskFlags.nodata in flags for flags in ds.mask_flag_enums):
        raise ValueError('Original declared nodata0 RGB band masks required')
    return indices


def stitch_original(plan,chips,masks):
    dtype=np.dtype(plan['source_rgb_dtype'])
    rgb=np.empty((3,1536,1536),dtype);valid=np.empty((1536,1536),bool)
    for cell in plan['cells']:
        identity=cell['item']['id'];pixels=chips[identity];mask=masks[identity]
        if pixels.dtype!=dtype or pixels.shape!=(3,384,384) or mask.dtype!=np.bool_ or mask.shape!=(384,384):
            raise ValueError('Original384 RGB pixel/mask types and shapes required')
        r,c=cell['row'],cell['col']
        rgb[:,r:r+384,c:c+384]=pixels;valid[r:r+384,c:c+384]=mask
    return rgb,valid


def canonical_context(rgb,valid,plan):
    if rgb.dtype!=np.uint8 or rgb.shape!=(3,1536,1536) or valid.dtype!=np.bool_ or valid.shape!=(1536,1536):
        raise ValueError('Unresampled original1536 mosaic required')
    return area_hr(rgb[:,88:1448,88:1448],valid[88:1448,88:1448],plan)


TOA_DISPLAY_MAPPING=dict(version='earthview-toa-fixed-display-rgb-v1',
    reflectance_scale=.0001,reflectance_offset=0.,gain=1.,gamma=2.2,
    formula='uint8=floor(255*clip(DN*0.0001,0,1)**(1/2.2)+0.5)',
    scope='one fixed mapping for every pixel, RGB band, chip, mosaic and acquisition; no histogram stretch',
    reflectance_encoding_evidence_url='https://satellogic.com/wp-content/uploads/2022/08/satellogic-imagery-product-specifications.pdf',
    encoding_evidence_pages=[10,17],
    offset_basis='Reflectance multiplied by10000 implies zero offset; no separate per-item offset is supplied.',
    scientific_caveat='Derived8bit displayRGB, not original native8bit observations or linear reflectance; originaluint16 TOA retained.')

TOA_ROBUST_RECIPE=dict(version='earthview-toa-whole-mosaic-robust-display-rgb-v1',
    percentiles=[.1,99.9],percentile_method='linear',parameter_dtype='float32',gamma=2.2,
    formula='uint8=floor(255*clip((DN-low_band)/(high_band-low_band),0,1)**(1/2.2)+0.5)',
    scope='One common per-RGB-band parameter set fitted over entire original1536m same-acquisition mosaic using only original allRGB-band-valid pixels; applied unchanged to all16chips and1360context.',
    scientific_caveat='Derived8bit displayRGB, with explicit scene-wide contrast mapping and tail clipping; not original native8bit or linear reflectance. Originaluint16 TOA retained.')


def toa_display_rgb(dn):
    if dn.dtype!=np.uint16 or dn.ndim!=3 or dn.shape[0]!=3:
        raise ValueError('Original RGB uint16 TOA DN required')
    linear=np.clip(dn.astype(np.float64)*TOA_DISPLAY_MAPPING['reflectance_scale'],0.,1.)
    return np.floor(255*np.power(linear,1/TOA_DISPLAY_MAPPING['gamma'])+.5).astype(np.uint8)


def apply_toa_mapping(dn,mapping):
    if dn.dtype!=np.uint16 or dn.ndim!=3 or dn.shape[0]!=3:
        raise ValueError('Original RGB uint16 TOA DN required')
    if mapping==TOA_DISPLAY_MAPPING:
        return toa_display_rgb(dn)
    if any(mapping.get(key)!=value for key,value in TOA_ROBUST_RECIPE.items()):
        raise ValueError('Versioned whole-mosaic TOA display mapping required')
    low=np.asarray(mapping['low_dn'],np.float64);high=np.asarray(mapping['high_dn'],np.float64)
    if low.shape!=(3,) or high.shape!=(3,) or not np.isfinite(low).all() or not np.isfinite(high).all() or not (high>low).all():
        raise ValueError('Finite nondegenerate common RGB percentile parameters required')
    normalized=np.clip((dn.astype(np.float64)-low[:,None,None])/(high-low)[:,None,None],0.,1.)
    return np.floor(255*np.power(normalized,1/TOA_ROBUST_RECIPE['gamma'])+.5).astype(np.uint8)


def map_toa_mosaic(dn,valid,mode='fixed'):
    if dn.dtype!=np.uint16 or dn.shape!=(3,1536,1536) or valid.dtype!=np.bool_ or valid.shape!=(1536,1536):
        raise ValueError('Whole original1536m TOA mosaic and nodata support required')
    if mode=='fixed':
        return toa_display_rgb(dn),dict(TOA_DISPLAY_MAPPING)
    if mode!='robust' or not valid.any():
        raise ValueError('Fixed or robust mapping with observed original support required')
    quantiles=np.asarray([np.percentile(band[valid],TOA_ROBUST_RECIPE['percentiles'],method='linear') for band in dn],np.float32)
    mapping=dict(TOA_ROBUST_RECIPE,low_dn=quantiles[:,0].astype(float).tolist(),
        high_dn=quantiles[:,1].astype(float).tolist(),
        original_mosaic_rgb_sha256=digest_bytes(dn.tobytes()),
        original_mosaic_support_sha256=digest_bytes(valid.tobytes()),
        statistics_valid_pixels=int(valid.sum()))
    return apply_toa_mapping(dn,mapping),mapping


def write_native_rgb(path,rgb,valid,crs,transform):
    if rgb.dtype not in (np.dtype('uint8'),np.dtype('uint16')) or rgb.shape[0]!=3 or valid.dtype!=np.bool_ or valid.shape!=rgb.shape[1:]:
        raise ValueError('Original uint8 or uint16 RGB pixels and binary support required')
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(path,'w',driver='GTiff',width=rgb.shape[2],height=rgb.shape[1],
            count=3,dtype=str(rgb.dtype),crs=crs,transform=transform,compress='deflate',tiled=True,
            blockxsize=256,blockysize=256) as ds:
            ds.write(rgb);ds.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue)
            ds.write_mask(valid.astype(np.uint8)*255)


def seam_diagnostics(rgb):
    """Descriptive boundary gradients; does not certify radiometric continuity."""
    values=rgb.astype(np.float32)
    dx=np.mean(np.abs(np.diff(values,axis=2)),axis=0)
    dy=np.mean(np.abs(np.diff(values,axis=1)),axis=0)
    rows=[]
    for direction,gradient in [('vertical',dx),('horizontal',dy)]:
        for edge in (384,768,1152):
            band=gradient[:,edge-1] if direction=='vertical' else gradient[edge-1,:]
            neighbors=gradient[:,edge-5:edge+4] if direction=='vertical' else gradient[edge-5:edge+4,:]
            baseline=float(np.mean(neighbors));jump=float(np.mean(band))
            rows.append(dict(direction=direction,source_pixel_edge=edge,
                mean_absolute_rgb_jump=jump,mean_local_gradient=baseline,
                ratio=jump/max(baseline,1e-6)))
    return dict(method='mean_absolute_original_uint8_rgb_boundary_gradient_vs9_pixel_neighborhood',
        interpretation='Descriptive only; buildings, water and actual scene edges can produce large ratios. Upstream VISUAL stretch consistency is unproven.',
        boundaries=rows,requires_radiometry_review=True)


def _fetch_pinned_once(url,expected=None):
    official_url(url,'.json' if url.endswith('.json') else '_TOA.tif' if url.endswith('_TOA.tif') else '_VISUAL.tif')
    if expected is None:
        with urlopen(Request(url,method='HEAD'),timeout=60) as response:
            etag=response.headers.get('ETag');length=int(response.headers.get('Content-Length','0'))
            modified=response.headers.get('Last-Modified')
    else:
        etag=expected['etag'];length=expected['bytes'];modified=expected.get('last_modified')
    if not etag or length<=0 or length>10000000:
        raise ValueError('Bounded original EarthView object identity required')
    with urlopen(Request(url,headers={'If-Match':etag}),timeout=90) as response:
        data=response.read(length+1)
        if len(data)!=length or response.headers.get('ETag')!=etag:
            raise ValueError('Original EarthView object identity/length changed during download')
    return data,dict(url=url,etag=etag,bytes=length,last_modified=modified,sha256=digest_bytes(data))


_public_connections=threading.local()


def public_session():
    """One immutable-config public-source session per fetch thread."""
    if not hasattr(_public_connections,'session'):
        session=requests.Session();session.headers['Accept-Encoding']='identity'
        session.mount('https://',requests.adapters.HTTPAdapter(pool_connections=4,pool_maxsize=4,max_retries=0,pool_block=True))
        _public_connections.session=session
    return _public_connections.session


def _fetch_keepalive_once(url,expected=None):
    official_url(url,'.json' if url.endswith('.json') else '_TOA.tif' if url.endswith('_TOA.tif') else '_VISUAL.tif')
    session=public_session()
    def status(response):
        if response.status_code!=200:
            raise HTTPError(url,response.status_code,'Original source HTTP status',response.headers,None)
    if expected is None:
        with session.head(url,timeout=(15,60),allow_redirects=False) as response:
            status(response);etag=response.headers.get('ETag');length=int(response.headers.get('Content-Length','0'))
            modified=response.headers.get('Last-Modified')
    else:
        etag=expected['etag'];length=expected['bytes'];modified=expected.get('last_modified')
    if not etag or not 0<length<=10000000:raise ValueError('Bounded original EarthView object identity required')
    with session.get(url,headers={'If-Match':etag},stream=True,timeout=(15,90),allow_redirects=False) as response:
        status(response);data=response.raw.read(length+1,decode_content=False)
        if len(data)!=length or response.headers.get('ETag')!=etag:
            raise ValueError('Original EarthView object identity/length changed during download')
    return data,dict(url=url,etag=etag,bytes=length,last_modified=modified,sha256=digest_bytes(data))


def fetch_pinned(url,expected=None,transport='urllib'):
    if transport not in ('urllib','requests-keepalive'):raise ValueError('Unsupported source HTTP transport')
    for attempt in range(3):
        try:return (_fetch_pinned_once if transport=='urllib' else _fetch_keepalive_once)(url,expected)
        except HTTPError as exc:
            if exc.code not in (408,429,500,502,503,504):
                raise ValueError(f'Original HTTP source identity unavailable or changed: status{exc.code}') from exc
            if attempt==2:raise
        except (URLError,TimeoutError,ConnectionError,requests.exceptions.RequestException,TransportHTTPError):
            if attempt==2:raise
        time.sleep(.5*(attempt+1))


def persist_outcome(publish,record):
    """Keep transient attempts immutable while leaving the source retryable."""
    data=json_bytes(record)
    name=('attempts/'+record['sample_id']+'/'+digest_bytes(data)+'.json'
        if record.get('retryable') else 'results/'+record['sample_id']+'.json')
    return publish(name,data)


def metadata_reference(publish,data,frozen=None):
    ref=(frozen or {}).get('object')
    if ref is not None:
        if ref['bytes']!=len(data) or ref['sha256']!=digest_bytes(data):
            raise ValueError('Staged metadata object identity differs from verified original bytes')
        return ref
    return publish('source-items/'+digest_bytes(data)+'.json',data)


class ImmutableSourceCache:
    """SHA-verified disk cache with per-URL singleflight within this worker.

    Previously captured HTTP identities are retained on resume. Frozen metadata
    may be read through its generation-pinned GCS object instead of public HTTP.
    Cache corruption and identity changes fail closed rather than refetching.
    """
    def __init__(self,root,fetch=None,staged=None,max_bytes=8_000_000_000,transport='urllib'):
        self.root=Path(root);self.fetch=fetch;self.staged=staged
        self.transport=transport
        self.guard=threading.Lock();self.locks={}
        for name in ('objects','receipts'):(self.root/name).mkdir(parents=True,exist_ok=True)
        if max_bytes<=0:raise ValueError('Positive bounded cache size required')
        self.max_bytes=max_bytes
        entries=[(p.name,p.stat()) for p in (self.root/'objects').iterdir() if p.is_file() and not p.name.endswith('.tmp')]
        self.objects=OrderedDict((name,stat.st_size) for name,stat in sorted(entries,key=lambda entry:entry[1].st_mtime_ns))
        self.total_bytes=sum(self.objects.values())
        self.trim()

    @staticmethod
    def check(data,pin,frozen=None):
        if len(data)!=pin['bytes'] or digest_bytes(data)!=pin['sha256']:
            raise ValueError('Immutable source cache corrupt bytes or SHA')
        if frozen is not None and any(pin[key]!=frozen[key] for key in ('url','etag','bytes','sha256')):
            raise ValueError('Original source disagrees with frozen metadata pin')

    def atomic_write(self,path,data):
        temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            temporary.write_bytes(data);os.replace(temporary,path)
        finally:
            temporary.unlink(missing_ok=True)

    def _trim_locked(self):
        while self.total_bytes>self.max_bytes and self.objects:
            sha,size=self.objects.popitem(last=False)
            (self.root/'objects'/sha).unlink(missing_ok=True)
            self.total_bytes-=size

    def trim(self):
        """Evict oldest cached bytes only; immutable URL receipts are retained."""
        with self.guard:self._trim_locked()

    def get(self,url,frozen=None):
        key=digest_bytes(url.encode())
        with self.guard:lock=self.locks.setdefault(key,threading.Lock())
        with lock:
            receipt=self.root/'receipts'/(key+'.json')
            previous=None;data=None
            if receipt.exists():
                previous=json.loads(receipt.read_bytes());pin=previous
                with self.guard:
                    obj=self.root/'objects'/pin['sha256']
                    if obj.exists():
                        data=obj.read_bytes();self.check(data,pin,frozen)
                        self.objects.move_to_end(pin['sha256'])
            if data is None and frozen is not None and frozen.get('object') is not None and self.staged is not None:
                data=self.staged(frozen['object'])
                pin={key:frozen[key] for key in ('url','etag','bytes','sha256')}
            elif data is None:
                data,pin=self.fetch(url) if self.fetch is not None else fetch_pinned(url,expected=previous or frozen,transport=self.transport)
            if pin['url']!=url:raise ValueError('Immutable source cache URL identity differs')
            self.check(data,pin,frozen)
            self.check(data,pin,previous)
            with self.guard:
                obj=self.root/'objects'/pin['sha256']
                if obj.exists():self.check(obj.read_bytes(),pin)
                else:
                    self.atomic_write(obj,data)
                    self.objects[pin['sha256']]=len(data);self.total_bytes+=len(data)
                self.objects.move_to_end(pin['sha256'])
                if not receipt.exists():self.atomic_write(receipt,json_bytes(pin))
                self._trim_locked()
            return data,pin


def ordered_parallel_results(function,values,workers):
    """Bound pending work and memory while preserving source-plan order."""
    iterator=iter(values)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending=[]
        for _ in range(workers):
            try:pending.append(pool.submit(function,next(iterator)))
            except StopIteration:break
        while pending:
            yield pending.pop(0).result()
            try:pending.append(pool.submit(function,next(iterator)))
            except StopIteration:pass


def storage_client(project,pool_size=0):
    from google.cloud import storage
    if pool_size==0:return storage.Client(project=project)
    import google.auth
    from google.auth.transport.requests import AuthorizedSession
    credentials,_=google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
    session=AuthorizedSession(credentials)
    session.mount('https://',requests.adapters.HTTPAdapter(pool_connections=pool_size,pool_maxsize=pool_size,max_retries=0,pool_block=True))
    # This documented SDK constructor parameter is private; the science image
    # pins storage2.19.0 and the GCE proof verifies this exact runtime.
    return storage.Client(project=project,credentials=credentials,_http=session)


class OutcomeSpool:
    """Retain full canonical source records on disk rather than in RAM."""
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.paths={key:self.root/(key+'.jsonl') for key in ('candidates','rejections')}
        for path in self.paths.values():path.write_bytes(b'')
        self.candidate_count=0;self.rejection_count=0;self.retryable=False

    def append(self,record):
        candidate=record.get('schema')=='earthview-candidate-v1'
        key='candidates' if candidate else 'rejections'
        with self.paths[key].open('ab') as stream:stream.write(json_bytes(record)+b'\n')
        self.candidate_count+=int(candidate);self.rejection_count+=int(not candidate)
        self.retryable|=bool(record.get('retryable'))

    def finalize(self,common):
        metadata=dict(common,candidate_count=self.candidate_count)
        if set(metadata).intersection(self.paths):raise ValueError('Outcome arrays must be streamed')
        path=self.root/'manifest.json'
        with path.open('wb') as stream:
            stream.write(b'{')
            for index,key in enumerate(sorted(set(metadata)|set(self.paths))):
                if index:stream.write(b',')
                stream.write(json_bytes(key)+b':')
                if key not in self.paths:stream.write(json_bytes(metadata[key]));continue
                stream.write(b'[')
                with self.paths[key].open('rb') as records:
                    for number,line in enumerate(records):
                        if number:stream.write(b',')
                        stream.write(line.rstrip(b'\n'))
                stream.write(b']')
            stream.write(b'}')
        return path


def create_only_file(blob,path):
    """Conditional file upload and streamed generation/CRC/SHA readback."""
    path=Path(path);sha=file_sha256(path);size=path.stat().st_size
    if not blob.exists():
        try:blob.upload_from_filename(str(path),if_generation_match=0,checksum='crc32c')
        except Exception as exc:
            if getattr(exc,'code',None)!=412:raise
    blob.reload()
    if int(blob.size)!=size:raise ValueError('Immutable output file conflict')
    verify=path.with_name(path.name+'.'+uuid.uuid4().hex+'.verify')
    try:
        blob.download_to_filename(str(verify),if_generation_match=int(blob.generation),checksum='crc32c')
        if verify.stat().st_size!=size or file_sha256(verify)!=sha:raise ValueError('Immutable output file conflict')
    finally:verify.unlink(missing_ok=True)
    return dict(generation=str(blob.generation),bytes=size,crc32c=blob.crc32c,sha256=sha)


def run(args):
    instance=require_cloud_vm()
    from google.api_core.exceptions import GoogleAPICallError,RetryError
    client=storage_client(args.project,args.gcs_pool_size)
    payload=Path(args.sources).read_bytes();document=json.loads(payload)
    if document.get('schema')!='earthview-source-mosaics-v1':
        raise ValueError('Versioned same-acquisition EarthView mosaic source configuration required')
    evidence=document['source_evidence'];validate_evidence(evidence)
    scratch=Path(args.scratch);scratch.mkdir(parents=True,exist_ok=True)
    bucket_name,prefix=parse_gs(args.output_prefix.rstrip('/'));bucket=client.bucket(bucket_name)
    def publish(name,data):
        return {'uri':f'gs://{bucket_name}/{prefix}/{name}',**create_only(bucket.blob(prefix+'/'+name),data)}
    product=args.product
    mosaics=document['mosaics'][:args.limit]
    identities=[digest_bytes(json_bytes([product,sorted(mosaic['json_paths'])])) for mosaic in mosaics]
    if len(identities)!=len(set(identities)):
        raise ValueError('Duplicate original mosaic source identity in intake plan')
    config=dict(schema='earthview-intake-configuration-v1',sources_sha256=digest_bytes(payload),input_product=product,
        worker_sha256=file_sha256(__file__),limit=args.limit,source_evidence=evidence,
        mosaic_workers=args.workers,fetch_workers=args.fetch_workers,
        cache_max_bytes=args.cache_max_bytes,
        source_http_transport=args.http_transport,gcs_pool_size=args.gcs_pool_size,
        staged_metadata_policy='Reference existing verified generation pins when supplied; otherwise publish original STAC bytes.',
        cache_policy='SHA-verified captured HTTP identities; generation-pinned frozen metadata; per-URL singleflight',
        rgb_mapping_recipe=(TOA_DISPLAY_MAPPING if args.toa_mapping=='fixed' else TOA_ROBUST_RECIPE) if product=='TOA' else 'publisher VISUAL; unknown historical per-chip rendering',
        support_policy='all_original_declared_nodata_rgb_band_masks_over_real1360_context',
        final_dataset_ready=False)
    publish('configuration.json',json_bytes(config));publish('source-configuration.json',payload)
    cache=ImmutableSourceCache(scratch/'source-cache',staged=lambda ref:verify_output(client,ref),max_bytes=args.cache_max_bytes,transport=args.http_transport)
    fetch_pool=ThreadPoolExecutor(max_workers=args.fetch_workers)
    def process_mosaic(mosaic):
        mosaic_key=digest_bytes(json_bytes([product,sorted(mosaic['json_paths'])]))
        sample_id='earthview-'+mosaic_key[:32]
        existing=bucket.blob(prefix+'/results/'+sample_id+'.json')
        if existing.exists():
            existing.reload();record=json.loads(existing.download_as_bytes(if_generation_match=int(existing.generation),checksum='crc32c'))
            if record.get('schema')=='earthview-candidate-v1':
                for key in ('native_context','hr','overview'):verify_output(client,record[key])
            return record
        try:
            items=[];item_refs={};item_http={}
            paths=mosaic['json_paths']
            futures=[fetch_pool.submit(cache.get,path if path.startswith('https://') else ROOT+path,document.get('metadata_pins',{}).get(path)) for path in paths]
            for path,future in zip(paths,futures):
                raw,pin=future.result();item=json.loads(raw)
                frozen=document.get('metadata_pins',{}).get(path)
                if frozen is not None and any(pin[key]!=frozen[key] for key in ('etag','bytes','sha256')):
                    raise ValueError('Original STAC metadata changed after source census')
                items.append(item);item_refs[item['id']]=metadata_reference(publish,raw,frozen)
                item_http[item['id']]=pin
            plan=validate_mosaic(items,evidence,product=product)
            if int(mosaic['epsg'])!=int(plan['acquisition_identity'][2]) or mosaic['mosaic_bounds']!=plan['mosaic_bounds']:
                raise ValueError('Frozen mosaic index geometry disagrees with actual original item grids')
            chips={};masks={};sources=[]
            asset='analytic' if product=='TOA' else 'visual'
            futures=[fetch_pool.submit(cache.get,cell['item']['assets'][asset]['href']) for cell in plan['cells']]
            for cell,future in zip(plan['cells'],futures):
                item=cell['item'];raw,http=future.result()
                obj=publish('source-objects/'+http['sha256']+'.tif',raw)
                with MemoryFile(raw) as memory,memory.open() as ds:
                    indices=validate_header(ds,item,product=product)
                    rgb=ds.read(indices);valid=np.all(ds.read_masks(indices)==255,axis=0)
                    header=dict(crs=ds.crs.to_string(),transform=list(ds.transform)[:6],shape=list(ds.shape),
                        count=ds.count,dtypes=list(ds.dtypes),colorinterp=[c.name for c in ds.colorinterp],
                        nodata=ds.nodata,mask_flags=[[f.name for f in flags] for flags in ds.mask_flag_enums])
                chips[item['id']]=rgb;masks[item['id']]=valid
                sources.append(dict(item=item,item_json=item_refs[item['id']],item_http_pin=item_http[item['id']],
                    object=obj,http_pin=http,header=header,selected_rgb_band_indices=indices,mosaic_pixel_offset=[cell['col'],cell['row']],
                    decoded_rgb_sha256=digest_bytes(rgb.tobytes()),declared_nodata_mask_sha256=digest_bytes(valid.tobytes())))
            original,valid=stitch_original(plan,chips,masks)
            if product=='TOA':
                rgb,mapping=map_toa_mosaic(original,valid,mode=args.toa_mapping)
            else:
                rgb=original;mapping=None
            context_valid=valid[88:1448,88:1448]
            if not context_valid.all():
                rejection=dict(sample_id=sample_id,reason='original_declared_nodata_context_support_incomplete',
                    valid_fraction=float(context_valid.mean()),source_items=sources,port=mosaic.get('port'),
                    seam_diagnostics=seam_diagnostics(rgb))
                publish('results/'+sample_id+'.json',json_bytes(rejection))
                return rejection
            context=canonical_context(rgb,valid,plan);hr=np.ascontiguousarray(context[:,16:-16,16:-16])
            native_rgb=np.ascontiguousarray(original[:,88:1448,88:1448]);useful=original[:,128:1408,128:1408]
            mapping_statistics=None
            if product=='TOA':
                observed=np.broadcast_to(valid,original.shape)
                values=original[observed]
                mapping_statistics=dict(observed_rgb_channel_values=int(values.size),
                    original_dn_min=int(values.min()),original_dn_max=int(values.max()),
                    dn_above_reflectance1_fraction=float(np.mean(values>10000)),
                    display_zero_fraction=float(np.mean(rgb[observed]==0)),
                    display255_fraction=float(np.mean(rgb[observed]==255)),
                    scope='1536square mosaic original validRGB bands; not a quality-acceptance claim')
            metadata=dict(schema='earthview-candidate-v1',sample_id=sample_id,source_kind='earthview',
                source_provenance=evidence,source_items=sources,license='CC-BY-4.0',native_gsd_m=1.,
                original_mask_provenance='original_declared_nodata_zero_rgb_band_masks',
                mask_caveat='Producer-declared per-band nodata support; not a separately supplied producer dataset mask.',
                upstream_visual_rendering=('Original uint16 TOA RGB reflectance DN stitched before one common documented whole-mosaic display mapping; no publisher VISUAL chips used.' if product=='TOA'
                    else 'Publisher VISUAL uint8 rendering; exact historical stretch version and interchip radiometric consistency unproven.'),
                rgb_mapping=mapping,
                rgb_mapping_statistics=mapping_statistics,
                decoded_native_context_sha256=digest_bytes(native_rgb.tobytes()),decoded_useful_core_sha256=digest_bytes(useful.tobytes()),
                decoded_display_context_sha256=digest_bytes(rgb[:,88:1448,88:1448].tobytes()),
                decoded_display_core_sha256=digest_bytes(rgb[:,128:1408,128:1408].tobytes()),
                seam_diagnostics=seam_diagnostics(rgb),port=mosaic.get('port'),
                port_center_distance_m=mosaic.get('port_center_distance_m'),
                marine_screening=mosaic.get('marine_screening'),
                split_exclusion_evidence=mosaic.get('split_exclusion_evidence'),
                requires_semantic_review=True,requires_native_quality_review=True,requires_radiometry_review=True,
                final_eligible=False,**{key:value for key,value in plan.items() if key!='cells'})
            path=scratch/(sample_id+'.tif')
            write_native_rgb(path,native_rgb,context_valid,CRS.from_string(plan['source_crs']),Affine(*plan['native_context_transform']))
            native=publish('native-context/'+sample_id+'.tif',path.read_bytes());path.unlink()
            buffer=io.BytesIO();np.savez_compressed(buffer,hr_context=context,hr=hr,metadata_json=np.asarray(json.dumps(metadata,sort_keys=True)))
            hrref=publish('hr/'+sample_id+'.npz',buffer.getvalue())
            image=Image.fromarray(np.moveaxis(rgb[:,128:1408,128:1408],0,-1));image.thumbnail((400,400));buffer=io.BytesIO();image.save(buffer,format='PNG')
            overview=publish('review/overview/'+sample_id+'.png',buffer.getvalue())
            record=dict(**metadata,native_context=native,hr=hrref,overview=overview)
            publish('candidates/'+sample_id+'.json',json_bytes(record))
        except (ValueError,rasterio.errors.RasterioError) as exc:
            record=dict(sample_id=sample_id,reason=str(exc),port=mosaic.get('port'))
        except (OSError,GoogleAPICallError,RetryError,requests.exceptions.RequestException,TransportHTTPError) as exc:
            record=dict(sample_id=sample_id,reason=str(exc),error_type=type(exc).__name__,retryable=True,port=mosaic.get('port'))
        persist_outcome(publish,record)
        return record
    spool=OutcomeSpool(scratch/'outcome-spool')
    try:
        for index,record in enumerate(ordered_parallel_results(process_mosaic,mosaics,args.workers),1):
            cache.trim()
            spool.append(record)
            print(json.dumps({'completed':index,'candidate_count':spool.candidate_count,'rejection_count':spool.rejection_count,'sample_id':record['sample_id']}),flush=True)
    finally:
        fetch_pool.shutdown(wait=True,cancel_futures=True)
    common=dict(schema='earthview-candidate-v1',configuration=config,vm_instance_id=instance,
        status='PROVISIONAL_NATIVE_CANDIDATES',final_dataset_ready=False,quality_accepted_count=0)
    path=spool.finalize(common);sha=file_sha256(path)
    name='incomplete-manifests/'+sha+'.json' if spool.retryable else 'candidate-manifest.json'
    ref=dict(uri=f'gs://{bucket_name}/{prefix}/{name}',**create_only_file(bucket.blob(prefix+'/'+name),path))
    summary=dict(manifest=ref,candidate_count=spool.candidate_count,rejection_count=spool.rejection_count)
    print(json.dumps(summary),flush=True)
    if spool.retryable:raise RuntimeError('Retryable source failures retained; resume the same frozen configuration to complete intake')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',required=True)
    parser.add_argument('--output-prefix',required=True)
    parser.add_argument('--scratch',required=True)
    parser.add_argument('--limit',type=int,default=6)
    parser.add_argument('--product',choices=['VISUAL','TOA'],default='TOA')
    parser.add_argument('--toa-mapping',choices=['fixed','robust'],default='fixed')
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--fetch-workers',type=int,default=16)
    parser.add_argument('--cache-max-bytes',type=int,default=8_000_000_000)
    parser.add_argument('--http-transport',choices=['urllib','requests-keepalive'],default='urllib')
    parser.add_argument('--gcs-pool-size',type=int,default=0)
    parser.add_argument('--project',default=GCP_PROJECT)
    args=parser.parse_args()
    if args.limit<=0:parser.error('--limit must be positive')
    if not 1<=args.workers<=8:parser.error('--workers must be between1 and8')
    if not 1<=args.fetch_workers<=32:parser.error('--fetch-workers must be between1 and32')
    if args.cache_max_bytes<=0:parser.error('--cache-max-bytes must be positive')
    if args.gcs_pool_size not in (0,32,64):parser.error('--gcs-pool-size must be0,32 or64')
    run(args)
