"""GCE-only pinned CC-BY-NC Planet disaster SkySat intake; review candidates only."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import re
import tempfile
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import Affine
from rasterio.windows import Window

from scripts.maritime_cloud_inventory import (create_only, digest_bytes, file_sha256,
    json_bytes, parse_gs, require_cloud_vm)
from scripts.maritime_planet_intake import (area_matrix, downsample, overlap,
    validate_header, write_candidate)

MASK_DOC = 'https://docs.planet.com/data/imagery/udm/'
USER_AGENT = 'MaritimeThesisDataset/1.0'
PUBLIC_S3_DOCUMENTATION = 'https://source.coop/planet/disasterdata/philippines-earthquake-2026-06-08/llms.txt'


def direct_transport(url, pin):
    """Provider-documented disasterdata bucket, verified against original pins.

    The public Source Cooperative object pages publish the us-west-2 bucket
    URI. Path-style S3 HTTPS avoids dotted-bucket TLS wildcard ambiguity.
    """
    parsed = urlsplit(url)
    if parsed.scheme!='https' or parsed.netloc!='data.source.coop' or not parsed.path.startswith('/planet/disasterdata/') or parsed.query or parsed.fragment:
        raise ValueError('No documented direct transport for this source')
    endpoint = 'https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop' + parsed.path
    if pin.get('version_id') is not None:
        endpoint += '?' + urlencode({'versionId':pin['version_id']})
    return endpoint


def validate_source(row):
    if row['license'] != 'CC-BY-NC-4.0' or row['constellation'] != 'skysat':
        raise ValueError('Free academic NC SkySat source required')
    gsd = float(row['native_gsd_m'])
    if not math.isfinite(gsd) or not 0 < gsd <= .8 or row['product_grid_m'] != .5:
        raise ValueError('Original native GSD or product spacing ineligible')
    if abs(float(row['source_stac']['properties']['gsd']) - gsd) > 1e-9 or not row['strip_id']:
        raise ValueError('Native GSD or acquisition identity inconsistent')


def metadata_status(row):
    return 'candidate' if row.get('intake_status') == 'candidate_requires_pixel_and_geometry_review' else 'skipped_metadata_geometry'


def verify_headers(headers, pin):
    headers = {k.lower(): v for k, v in headers.items()}
    if headers.get('etag') != pin['etag'] or int(headers.get('content-length', -1)) != int(pin['content_length']):
        raise ValueError('HTTP source ETag or Content-Length changed')
    if pin.get('version_id') is not None and headers.get('x-amz-version-id') != pin['version_id']:
        raise ValueError('HTTP source version ID changed or missing')


def download_pinned(url, pin, path, *, checksum=None, opener=urlopen):
    if not url.startswith('https://data.source.coop/') or not pin.get('etag') or int(pin['content_length']) <= 0:
        raise ValueError('Expected pinned public Source Cooperative HTTPS object')
    path = Path(path)
    partial = path.with_name(path.name + '.part')
    sha, md5, total = hashlib.sha256(), hashlib.md5(), 0
    headers = {'If-Match': pin['etag'], 'Accept-Encoding': 'identity', 'User-Agent': USER_AGENT}
    transport_url = url
    try:
        try:
            response = opener(Request(url, headers=headers), timeout=120)
        except HTTPError as exc:
            # Only transport denial triggers fallback. Integrity failures and
            # source changes are never retried against a weaker identity.
            if exc.code != 403:
                raise
            transport_url = direct_transport(url, pin)
            with opener(Request(transport_url, method='HEAD', headers=headers), timeout=120) as head:
                if head.status != 200:
                    raise ValueError('Direct public S3 transport HEAD failed')
                verify_headers(head.headers,pin)
            response = opener(Request(transport_url, headers=headers), timeout=120)
        with response:
            if response.status != 200:
                raise ValueError('Expected complete HTTP 200 source response')
            verify_headers(response.headers, pin)
            with partial.open('wb') as output:
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > int(pin['content_length']):
                        raise ValueError('HTTP source exceeds pinned length')
                    sha.update(chunk)
                    md5.update(chunk)
                    output.write(chunk)
        if total != int(pin['content_length']):
            raise ValueError('Interrupted HTTP source length mismatch')
        if checksum is not None:
            if not re.fullmatch('1220[0-9a-fA-F]{64}', checksum) or checksum[4:].lower() != sha.hexdigest():
                raise ValueError('Original STAC SHA256 multihash checksum mismatch')
        etag = pin['etag'].strip('"')
        if re.fullmatch('[0-9a-fA-F]{32}', etag) and md5.hexdigest() != etag.lower():
            raise ValueError('Single-part S3 ETag checksum mismatch')
        partial.replace(path)
        return dict(sha256=sha.hexdigest(), bytes=total, etag=pin['etag'],
                    version_id=pin.get('version_id'), stac_checksum=checksum,
                    source_url=url,transport_url=transport_url,
                    transport_documentation=PUBLIC_S3_DOCUMENTATION if transport_url!=url else None)
    finally:
        partial.unlink(missing_ok=True)


def udm_support(unusable):
    """Bit0 blackfill, remaining bits quality anomalies; reject every nonzero bit.

    This conservative policy also rejects cloud and suspect bands, using the
    producer's original UDM. UDM2 band8 is equivalent to the UDM1 bitfield.
    """
    if unusable.dtype != np.uint8:
        raise ValueError('Original UDM must be uint8')
    return unusable == 0


def mirror_original(bucket, name, path, source_ref, scratch):
    """Immutable upload with CRC32C plus streamed full SHA256 readback."""
    import google_crc32c
    crc = google_crc32c.Checksum()
    with open(path, 'rb') as stream:
        while chunk := stream.read(1024 * 1024):
            crc.update(chunk)
    import base64
    local_crc = base64.b64encode(crc.digest()).decode()
    blob = bucket.blob(name)
    if not blob.exists():
        blob.metadata = {'source_sha256': source_ref['sha256'], 'license': 'CC-BY-NC-4.0'}
        try:
            blob.upload_from_filename(str(path), if_generation_match=0, checksum='crc32c')
        except Exception as exc:
            if getattr(exc, 'code', None) != 412:
                raise
    blob.reload()
    if blob.size != source_ref['bytes'] or blob.crc32c != local_crc:
        raise ValueError('Immutable original GCS mirror size/CRC32C conflict')
    readback = Path(scratch) / 'mirror-readback.tif'
    try:
        blob.download_to_filename(str(readback), if_generation_match=int(blob.generation), checksum='crc32c')
        if file_sha256(readback) != source_ref['sha256']:
            raise ValueError('Immutable original GCS mirror SHA256 conflict')
    finally:
        readback.unlink(missing_ok=True)
    return dict(uri=f'gs://{bucket.name}/{name}', generation=str(blob.generation),
                bytes=blob.size, crc32c=blob.crc32c, sha256=source_ref['sha256'])


def same_grid(mask, ds):
    if mask.crs != ds.crs or mask.transform != ds.transform or (mask.width, mask.height) != (ds.width, ds.height):
        raise ValueError('Original UDM is not exactly aligned to RGB')


def run(args):
    instance = require_cloud_vm()
    from google.cloud import storage
    from PIL import Image
    manifest = json.loads(Path(args.sources).read_text(encoding='utf-8-sig'))
    rows = manifest['sources']
    if args.max_cores_per_source < 1:
        raise ValueError('Positive core cap required')
    client = storage.Client(project=args.project)
    bucket_name, prefix = parse_gs(args.output_prefix)
    bucket = client.bucket(bucket_name)
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    reports, strip_bounds, objects = [], {}, set()

    def publish(name, payload):
        return dict(uri=f'gs://{bucket_name}/{prefix}/{name}',
                    **create_only(bucket.blob(f'{prefix}/{name}'), payload))

    for source in sorted(rows, key=lambda r: r['id']):
        validate_source(source)
        report = dict(source=source, status=metadata_status(source), candidates=[], accepted_count=0)
        if report['status'] == 'candidate':
            with tempfile.TemporaryDirectory(dir=scratch) as temporary:
                visual_key = source['visual_asset_key']
                url, pin = source['asset_urls'][visual_key], source['public_head'][visual_key]
                identity = (url, pin.get('version_id'), pin['etag'])
                if identity in objects:
                    raise ValueError('Duplicate original HTTP asset identity')
                objects.add(identity)
                path = Path(temporary) / 'visual.tif'
                origin = download_pinned(url, pin, path,
                    checksum=source['source_stac']['assets'][visual_key].get('file:checksum'))
                report['original_http'] = origin
                report['original_mirror'] = mirror_original(bucket,
                    f'{prefix}/originals/{source["id"]}/visual.tif', path, origin, temporary)
                with rasterio.open(path) as ds:
                    # Embedded support is preferred. External UDM can establish
                    # original support when nodata/all-valid metadata cannot.
                    mask_ds, provenance = None, None
                    try:
                        provenance = validate_header(ds)
                    except ValueError:
                        t = ds.transform
                        if ds.crs is None or not ds.crs.is_projected or ds.crs.linear_units != 'metre' or not all(math.isfinite(v) for v in tuple(t)[:6]) or abs(t.a-.5)>1e-9 or abs(t.e+.5)>1e-9 or t.b or t.d or ds.count not in (3,4) or ds.dtypes[:3] != ('uint8',)*3 or ds.colorinterp[:3] != (ColorInterp.red,ColorInterp.green,ColorInterp.blue) or (ds.count == 4 and (ds.colorinterp[3] != ColorInterp.alpha or ds.dtypes[3] != 'uint8')):
                            report.update(status='rejected_rgb_grid_header')
                        else:
                            key = 'udm' if 'udm' in source['asset_urls'] else 'udm2' if 'udm2' in source['asset_urls'] else None
                            if key is None:
                                report.update(status='rejected_unresolved_original_mask')
                            else:
                                mask_path = Path(temporary) / 'mask.tif'
                                mask_ref = download_pinned(source['asset_urls'][key], source['public_head'][key], mask_path,
                                    checksum=source['source_stac']['assets'][key].get('file:checksum'))
                                report['mask_mirror'] = mirror_original(bucket, f'{prefix}/originals/{source["id"]}/{key}.tif', mask_path, mask_ref, temporary)
                                mask_ds = rasterio.open(mask_path)
                                try:
                                    same_grid(mask_ds, ds)
                                    if mask_ds.count != (1 if key == 'udm' else 8) or mask_ds.dtypes != ('uint8',)*mask_ds.count:
                                        raise ValueError('Unexpected original UDM band layout')
                                    provenance = f'original_{key}_all_unusable_bits_zero'
                                except ValueError as exc:
                                    report.update(status='rejected_original_mask_grid', reason=str(exc))
                                    mask_ds.close()
                                    mask_ds = None
                    try:
                        if provenance is not None:
                            report.update(status='review_pending', mask_provenance=provenance,
                                mask_documentation=MASK_DOC, resampling='exact_area_float64_half_up_uint8_0.5_to_0.8',
                                crs=ds.crs.to_string(), original_transform=list(ds.transform)[:6])
                            image = io.BytesIO()
                            overview = ds.read([1,2,3], out_shape=(3, max(1,round(ds.height/ds.width*640)),640))
                            Image.fromarray(np.moveaxis(overview,0,-1)).save(image,format='PNG')
                            report['source_overview'] = publish(f'review/overview/{source["id"]}.png', image.getvalue())
                            target = Affine(.8,0,ds.transform.c,0,-.8,ds.transform.f)
                            width,height = math.floor(ds.width*.5/.8),math.floor(ds.height*.5/.8)
                            existing = strip_bounds.setdefault((source['strip_id'],ds.crs.to_string()),[])
                            rejects = report.setdefault('rejections',{})
                            for row in range(50,height-1650+1,1600):
                                for col in range(50,width-1650+1,1600):
                                    if len(report['candidates'])>=args.max_cores_per_source: break
                                    left,top = target*(col,row)
                                    bounds=[left,top-1280,left+1280,top]
                                    if any(overlap(bounds,old) for old in existing):
                                        rejects['same_strip_overlap']=rejects.get('same_strip_overlap',0)+1
                                        continue
                                    x,y=(col-50)*.8,(row-50)*.8
                                    sx,sy=math.floor(x/.5+1e-10),math.floor(y/.5+1e-10)
                                    px,py=max(0,x-sx*.5),max(0,y-sy*.5)
                                    wx,wy=area_matrix(px,1700),area_matrix(py,1700)
                                    win=Window(sx,sy,wx.shape[1],wy.shape[1])
                                    rgb=ds.read([1,2,3],window=win)
                                    valid=(udm_support(mask_ds.read(1 if mask_ds.count==1 else 8,window=win)) if mask_ds else np.all(ds.read_masks([1,2,3],window=win)==255,axis=0))
                                    out,support=downsample(rgb,valid,px,py,1700)
                                    if not support.all():
                                        rejects['incomplete_original_support']=rejects.get('incomplete_original_support',0)+1
                                        continue
                                    sid=digest_bytes(json_bytes(dict(asset=source['id'],strip=source['strip_id'],bounds=bounds)))[:24]
                                    candidate=Path(temporary)/'candidate.tif'
                                    write_candidate(candidate,out,support,ds.crs,target*Affine.translation(col-50,row-50))
                                    mother=publish(f'candidates/{sid}.tif',candidate.read_bytes())
                                    preview=io.BytesIO()
                                    Image.fromarray(np.moveaxis(out[:,50:1650,50:1650],0,-1)).save(preview,format='PNG')
                                    report['candidates'].append(dict(sample_id=sid,mother=mother,
                                        preview=publish(f'review/native/{sid}.png',preview.getvalue()),core_bounds=bounds,
                                        source_pixel_window=[sx,sy,wx.shape[1],wy.shape[1]],
                                        mask_provenance=provenance,license=source['license'],
                                        source_sha256=origin['sha256'],requires_semantic_review=True,
                                        requires_quality_and_near_duplicate_review=True,final_eligible=False))
                                    existing.append(bounds)
                    finally:
                        if mask_ds is not None: mask_ds.close()
        reports.append(publish(f'sources/{source["id"]}.json',json_bytes(report)))
        (scratch/'checkpoint.json').write_bytes(json_bytes(dict(reports=reports)))
        print(json.dumps(dict(asset=source['id'],status=report['status'],candidates=len(report['candidates']))),flush=True)
    return publish('summary.json',json_bytes(dict(instance=instance,sources=reports,accepted_count=0,
        status='provisional_review_only',license='CC-BY-NC-4.0',worker_sha256=file_sha256(__file__),
        manifest_sha256=digest_bytes(json_bytes(manifest)),max_cores_per_source=args.max_cores_per_source)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('project','sources','output-prefix','scratch'): parser.add_argument('--'+name,required=True)
    parser.add_argument('--max-cores-per-source',type=int,default=20)
    print(json.dumps(run(parser.parse_args())))


if __name__=='__main__': main()
