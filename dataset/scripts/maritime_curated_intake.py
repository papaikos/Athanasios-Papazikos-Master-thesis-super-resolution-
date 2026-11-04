"""GCE-only academic ShipRSImageNet intake; no fabricated geo or ready marker.

Original images and annotations are preserved, with <=1 metre qualification
only when an explicit per-image numeric GSD is present. The upstream dataset
contains xView, HRSC2016 and FGSD, so overlapping datasets cannot be treated as
independent evaluation sources. No RGB upsampling contributes qualified bytes.
"""
import argparse
import hashlib
from html.parser import HTMLParser
import io
import json
import math
from pathlib import Path, PurePosixPath
import time
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, build_opener, HTTPCookieProcessor
from http.cookiejar import CookieJar
import zipfile
from xml.etree import ElementTree
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

from PIL import Image
from scripts.maritime_cloud_inventory import (create_only, file_sha256,
    json_bytes, parse_gs, require_cloud_vm)
from src.cloud_config import GCP_PROJECT

FILE_ID = '1_6FYGZQowFC_l5PHZkRtL70VB-vwT8Td'
REPO_COMMIT = '3a4ca4b15475c17b46a254e00f3b11c2273a6d39'
SOURCE_URL = f'https://drive.google.com/file/d/{FILE_ID}/view'
LICENSE_URL = f'https://github.com/zzndream/ShipRSImageNet/blob/{REPO_COMMIT}/README.md#image-source-and-usage-license'


def accepted_gsd(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and 0 < value <= 1 else None


def safe_member(name):
    name = name.replace('\\', '/')
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or ':' in name:
        raise ValueError('Unsafe ZIP member')
    return str(path)


def inspect_rgb(data):
    with Image.open(io.BytesIO(data)) as image:
        if image.mode != 'RGB':
            raise ValueError('Original three-channel RGB required')
        image.load()
        return image.width, image.height, image.width * image.height * 3


def pixel_identity(data):
    with Image.open(io.BytesIO(data)) as image:
        if image.mode != 'RGB':
            raise ValueError('Original three-channel RGB required')
        return hashlib.sha256(json_bytes([image.width, image.height, image.mode])
                              + image.tobytes()).hexdigest()


class ConfirmationParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action = None
        self.fields = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'form':
            self.action = attrs.get('action')
        if tag == 'input' and attrs.get('name'):
            self.fields[attrs['name']] = attrs.get('value', '')


def confirm_url(document):
    parser = ConfirmationParser()
    parser.feed(document)
    target = urlsplit(parser.action or '')
    if target.scheme != 'https' or target.hostname != 'drive.usercontent.google.com':
        raise ValueError('Unexpected Drive confirmation form; no external requests issued')
    return parser.action + '?' + urlencode(parser.fields)


def download_archive(path):
    """Public-cookie confirmation, three persisted resumable HTTP attempts."""
    if path.exists():
        if not zipfile.is_zipfile(path):
            raise ValueError('Existing archive is not a ZIP')
        return
    partial = path.with_suffix('.partial')
    attempts = path.with_suffix('.attempts.json')
    state = json.loads(attempts.read_text()) if attempts.exists() else {'attempts': 0}
    opener = build_opener(HTTPCookieProcessor(CookieJar()))
    while state['attempts'] < 3:
        state['attempts'] += 1
        attempts.write_text(json.dumps(state))
        try:
            url = 'https://drive.google.com/uc?' + urlencode(dict(export='download', id=FILE_ID))
            with opener.open(Request(url, headers={'User-Agent': 'MaritimeThesisAcademicIntake/1.0'}), timeout=120) as initial:
                if 'text/html' in initial.headers.get('Content-Type', ''):
                    url = confirm_url(initial.read(2 * 1024 * 1024).decode('utf-8'))
                else:
                    url = initial.url
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {'User-Agent': 'MaritimeThesisAcademicIntake/1.0'}
            if offset:
                headers['Range'] = f'bytes={offset}-'
            with opener.open(Request(url, headers=headers), timeout=120) as response:
                if 'text/html' in response.headers.get('Content-Type', ''):
                    raise ValueError('Drive returned HTML instead of dataset; no consent or registration bypass')
                append = response.status == 206 and offset > 0
                if append and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise ValueError('Resume offset mismatch')
                with partial.open('ab' if append else 'wb') as output:
                    total = offset if append else 0
                    previous = total
                    while True:
                        chunk = response.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        total += len(chunk)
                        if total - previous >= 128 * 1024 * 1024:
                            print(json.dumps({'download_bytes': total}), flush=True)
                            previous = total
            if not zipfile.is_zipfile(partial):
                raise ValueError('Downloaded original is not a ZIP')
            partial.replace(path)
            return
        except Exception as exc:
            print(json.dumps({'attempt': state['attempts'], 'error': str(exc)}), flush=True)
            if state['attempts'] == 3:
                raise
            time.sleep(10 * state['attempts'])


def archive_inventory(path):
    with zipfile.ZipFile(path) as archive:
        rows = []
        examples = []
        for item in archive.infolist():
            name = safe_member(item.filename)
            rows.append(dict(name=name, bytes=item.file_size, compressed_bytes=item.compress_size,
                             crc32=item.CRC))
            if not item.is_dir() and name.lower().endswith(('.xml', '.json', '.txt')) and len(examples) < 12:
                data = archive.read(item)
                if len(data) < 5 * 1024 * 1024:
                    examples.append(dict(name=name, text=data.decode('utf-8', errors='replace')[:16000]))
        return {'members': rows, 'annotation_examples': examples}


def qualified_annotation(xml_data, image_data=None):
    root = ElementTree.fromstring(xml_data)
    text = root.findtext('Img_Resolution')
    gsd = accepted_gsd(text)
    record = dict(filename=root.findtext('filename'),
                  source_family=root.findtext('source/dataset_source'),
                  source_database=root.findtext('source/database'),
                  port_id=root.findtext('port_id'),
                  annotation_gsd_raw=text, annotation_gsd_field='annotation/Img_Resolution',
                  gsd_units='metres per pixel',
                  gsd_evidence='upstream per-image annotation; not independent sensor calibration',
                  georeferencing=None, north_up_verified=False, real_halo_available=False)
    if gsd is None:
        try:
            raw = float(text)
        except (TypeError, ValueError):
            raw = None
        record['status'] = ('REJECTED_GSD_COARSER_THAN_1M' if raw is not None and math.isfinite(raw) and raw > 1
                            else 'REJECTED_GSD_UNKNOWN')
        return record
    if image_data is None:
        raise ValueError('Qualified annotation requires original RGB pixels')
    width, height, native_bytes = inspect_rgb(image_data)
    dims = [root.findtext('size/width'), root.findtext('size/height'), root.findtext('size/depth')]
    if tuple(map(int, dims)) != (width, height, 3):
        raise ValueError('Original RGB and XML dimensions disagree')
    names = [item.findtext('name') for item in root.findall('object')]
    record.update(status='QUALIFIED_GSD_RGB_LABELED', native_gsd_m=gsd,
                  width=width, height=height, native_rgb_bytes=native_bytes,
                  physical_extent_m=[width * gsd, height * gsd],
                  ship_count=sum(name != 'Dock' for name in names),
                  dock_count=names.count('Dock'), annotated_classes=dict(Counter(names)),
                  decoded_pixel_sha256=pixel_identity(image_data),
                  official_annotations_preserved=True,
                  recommended_usage='training supplement; constituent/geographic holdouts not independently verified')
    return record


def qualify_archive(client, path, original_ref, prefix, scratch):
    """Keep original chips and XML, rejecting GSD unknown/coarse before pixels."""
    with zipfile.ZipFile(path) as archive:
        names = [safe_member(name) for name in archive.namelist()]
        images = {PurePosixPath(name).name: name for name in names
                  if '/JPEGImages/' in name and not name.endswith('/')}
        splits = {}
        for split in ('train', 'val', 'test'):
            name = next(name for name in names if name.endswith(f'/VOC_Format/ImageSets/{split}.txt'))
            for image in archive.read(name).decode('utf-8-sig').splitlines():
                splits.setdefault(image.strip(), []).append(split)
        rows = []
        pixel_ids = {}
        uploads = []
        def publish(item):
            row, image, xml = item
            identifier = row['decoded_pixel_sha256']
            for key, data, suffix in [('image_original', image, f'images/{identifier}.bmp'),
                                     ('annotation_original', xml, f'annotations/{identifier}.xml')]:
                bucket, object_name = parse_gs(prefix.rstrip('/') + '/qualified/' + suffix)
                row[key] = {'uri': f'gs://{bucket}/{object_name}',
                            **create_only(client.bucket(bucket).blob(object_name), data)}
            return row
        for name in names:
            if not name.endswith('.xml') or '/VOC_Format/Annotations/' not in name:
                continue
            xml = archive.read(name)
            root = ElementTree.fromstring(xml)
            filename = root.findtext('filename')
            data = archive.read(images[filename]) if accepted_gsd(root.findtext('Img_Resolution')) is not None else None
            row = qualified_annotation(xml, data)
            row.update(archive_annotation_member=name, official_splits=splits.get(filename, []),
                       upstream_archive=original_ref, repository_commit=REPO_COMMIT,
                       annotation_sha256=hashlib.sha256(xml).hexdigest())
            if data is not None:
                row['archive_image_member'] = images[filename]
                row['original_image_sha256'] = hashlib.sha256(data).hexdigest()
                identity = row['decoded_pixel_sha256']
                if identity in pixel_ids:
                    row['status'] = 'DUPLICATE_NATIVE_PIXELS'
                    row['duplicate_of'] = pixel_ids[identity]
                else:
                    pixel_ids[identity] = filename
                    uploads.append((row, data, xml))
            rows.append(row)
        with ThreadPoolExecutor(max_workers=8) as pool:
            # The bounded archive contains 532 qualified ~2.6MB RGB images.
            # Submit at most eight uploads concurrently, retaining original pixels.
            for index, row in enumerate(pool.map(publish, uploads)):
                if index % 32 == 0:
                    print(json.dumps({'qualified_uploaded': index + 1}), flush=True)
    accepted = [row for row in rows if row['status'] == 'QUALIFIED_GSD_RGB_LABELED']
    report = dict(schema=1, status='ANNOTATED_SUPPLEMENT_QUALIFIED', source_original=original_ref,
                  repository_commit=REPO_COMMIT, license='academic-only; constituent terms apply',
                  license_evidence_url=LICENSE_URL, records=rows,
                  status_counts=dict(Counter(row['status'] for row in rows)),
                  qualified_images=len(accepted), unique_native_rgb_bytes=sum(row['native_rgb_bytes'] for row in accepted),
                  qualified_nonoverlap_native_bytes=None,
                  byte_accounting='Exact full-image pixel duplicates excluded; known 150px xView overlap not removed. Exclude from geographically disjoint mother bytes until original scene mapping exists',
                  ship_instances=sum(row['ship_count'] for row in accepted),
                  dock_instances=sum(row['dock_count'] for row in accepted),
                  minimum_gsd_m=min((row['native_gsd_m'] for row in accepted), default=None),
                  maximum_gsd_m=max((row['native_gsd_m'] for row in accepted), default=None),
                  gsd_field_evidence='Original annotation/Img_Resolution, in metres per pixel; xView source documentation independently states 0.3m',
                  native_physical_support='Per-image extent = original dimensions multiplied by reported GSD; no fabricated halo or enlarged resolution',
                  overlap_warning='xView chips overlap by 150 pixels and are also distributed through xView. Keep entire constituent/port group in one training side until scene/geographic dedup is proven',
                  final_dataset_ready=False)
    output = json_bytes(report)
    bucket, name = parse_gs(prefix.rstrip('/') + '/source-reports/qualified-manifest.json')
    ref = {'uri': f'gs://{bucket}/{name}', **create_only(client.bucket(bucket).blob(name), output)}
    (scratch / 'qualified-manifest.json').write_bytes(output)
    print(json.dumps({'manifest': ref, **{key: report[key] for key in (
        'qualified_images', 'unique_native_rgb_bytes', 'ship_instances', 'dock_instances', 'status_counts')}}), flush=True)
    return report


def upload_original(client, path, uri):
    bucket, name = parse_gs(uri)
    blob = client.bucket(bucket).blob(name)
    digest = file_sha256(path)
    if not blob.exists():
        blob.metadata = {'sha256': digest, 'upstream_file_id': FILE_ID, 'license': 'academic-only'}
        blob.upload_from_filename(str(path), if_generation_match=0, checksum='crc32c', timeout=1800)
    blob.reload()
    if int(blob.size) != path.stat().st_size or blob.metadata.get('sha256') != digest:
        raise ValueError('Existing original archive conflicts')
    # Original resumable upload validates CRC32C end-to-end; readback proves exact bytes.
    check = path.with_suffix('.readback')
    blob.download_to_filename(str(check), if_generation_match=int(blob.generation), checksum='crc32c', timeout=1800)
    if file_sha256(check) != digest:
        raise ValueError('Original archive readback SHA256 mismatch')
    check.unlink()
    return dict(uri=uri, generation=str(blob.generation), bytes=int(blob.size),
                crc32c=blob.crc32c, sha256=digest)


def run(args):
    instance = require_cloud_vm()
    from google.cloud import storage
    client = storage.Client(project=GCP_PROJECT)
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    archive = scratch / 'ShipRSImageNet-v1.1.zip'
    cached_report = scratch / 'archive-inventory.json'
    if args.qualify and cached_report.exists():
        original = json.loads(cached_report.read_text())['source_original']
        bucket_name, object_name = parse_gs(original['uri'])
        blob = client.bucket(bucket_name).blob(object_name, generation=int(original['generation']))
        blob.reload()
        if (int(blob.size) != original['bytes'] or blob.crc32c != original['crc32c']
                or file_sha256(archive) != original['sha256']):
            raise ValueError('Verified original archive changed before qualification')
        qualify_archive(client, archive, original, args.output_prefix, scratch)
        return
    download_archive(archive)
    ref = upload_original(client, archive, args.output_prefix.rstrip('/') + '/original/ShipRSImageNet-v1.1.zip')
    report = archive_inventory(archive)
    report.update(schema=1, source_original=ref, vm_instance_id=instance,
                  repository_commit=REPO_COMMIT, license='academic-only; original constituent terms apply',
                  license_evidence_url=LICENSE_URL, upstream_url=SOURCE_URL,
                  status='ORIGINAL_ARCHIVED_AWAITING_GSD_QUALIFICATION',
                  overlap_warning='Contains xView, HRSC2016, FGSD and Airbus source images; do not mix constituent holdouts')
    bucket, name = parse_gs(args.output_prefix.rstrip('/') + '/source-reports/archive-inventory.json')
    pin = create_only(client.bucket(bucket).blob(name), json_bytes(report))
    print(json.dumps({'report_uri': f'gs://{bucket}/{name}', **pin,
                      'members': len(report['members']), 'original_bytes': ref['bytes']}), flush=True)
    (scratch / 'archive-inventory.json').write_bytes(json_bytes(report))
    if args.qualify:
        qualify_archive(client, archive, ref, args.output_prefix, scratch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch', required=True)
    parser.add_argument('--output-prefix', required=True)
    parser.add_argument('--qualify', action='store_true')
    run(parser.parse_args())
