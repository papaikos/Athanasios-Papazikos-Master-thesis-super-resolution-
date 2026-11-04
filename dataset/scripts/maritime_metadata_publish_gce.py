"""Publish byte-identical large JSON via verified compressed metadata on GCE."""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path

import google_crc32c

from scripts.maritime_cloud_inventory import parse_gs,require_cloud_vm,verify_output
from src.cloud_config import GCS_BUCKET


def file_identity(path):
    size=0;sha=hashlib.sha256();crc=google_crc32c.Checksum()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):
            size+=len(chunk);sha.update(chunk);crc.update(chunk)
    return dict(bytes=size,sha256=sha.hexdigest(),crc32c=base64.b64encode(crc.digest()).decode())


def expand_verified_metadata(source,destination,expected):
    partial=Path(str(destination)+'.partial')
    size=0
    try:
        with gzip.open(source,'rb') as incoming,partial.open('wb') as outgoing:
            for chunk in iter(lambda:incoming.read(1024**2),b''):
                size+=len(chunk)
                if size>expected['bytes']:raise ValueError('Expanded metadata exceeds frozen byte count')
                outgoing.write(chunk)
        if file_identity(partial)!=expected:raise ValueError('Expanded metadata byte/SHA/CRC mismatch')
        partial.replace(destination)
    finally:partial.unlink(missing_ok=True)


def publish(request_pin,scratch,expected_code_sha):
    require_cloud_vm()
    if file_identity(__file__)['sha256']!=expected_code_sha:raise ValueError('Metadata publisher code hash mismatch')
    from google.cloud import storage
    from google.api_core.exceptions import PreconditionFailed
    client=storage.Client();request=json.loads(verify_output(client,request_pin))
    if request.get('schema')!='maritime-large-metadata-publication-request-v1':
        raise ValueError('Versioned metadata publication request required')
    identity=request['plain_identity'];wire_pin=request['compressed_wire_pin']
    bucket,name=parse_gs(request['output_uri'])
    if (bucket!=GCS_BUCKET
        or not name.startswith('maritime-rgb/expansion-20261007/control/expanded-quality-pool-v2/')
        or not name.endswith('-'+identity['sha256']+'.json')
        or not wire_pin['uri'].startswith('gs://'+bucket+'/maritime-rgb/expansion-20261007/control/large-metadata-transfers-v1/')):
        raise ValueError('Exact immutable metadata output namespace required')
    folder=Path(scratch);folder.mkdir(parents=True,exist_ok=True)
    compressed=folder/(wire_pin['sha256']+'.json.gz')
    compressed.write_bytes(verify_output(client,wire_pin))
    plain=folder/(identity['sha256']+'.json')
    expand_verified_metadata(compressed,plain,identity)
    blob=client.bucket(bucket).blob(name);blob.chunk_size=8*1024**2
    if not blob.exists():
        try:blob.upload_from_filename(str(plain),if_generation_match=0,checksum='crc32c',timeout=300)
        except PreconditionFailed:pass
    blob.reload();generation=int(blob.generation)
    if int(blob.size)!=identity['bytes'] or blob.crc32c!=identity['crc32c']:
        raise ValueError('Existing metadata object differs from frozen identity')
    readback=folder/(identity['sha256']+'.readback.json')
    blob.download_to_filename(str(readback),if_generation_match=generation,checksum='crc32c',timeout=300)
    if file_identity(readback)!=identity:raise ValueError('Full metadata GCS readback mismatch')
    pin=dict(uri=request['output_uri'],generation=str(generation),**identity)
    result=dict(schema='maritime-large-metadata-publication-result-v1',output_pin=pin,
        request_pin=request_pin,publisher_code_sha256=expected_code_sha,
        compressed_wire_pin=wire_pin,full_expanded_and_gcs_readback_sha_crc_verified=True,
        image_processing_performed=False,dataset_ready=False)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for option in ('request-pin','scratch','code-sha'):parser.add_argument('--'+option,required=True)
    args=parser.parse_args()
    publish(json.loads(Path(args.request_pin).read_bytes()),args.scratch,args.code_sha)
