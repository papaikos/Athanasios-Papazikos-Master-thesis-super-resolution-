"""Download a frozen maritime training split with the Storage SDK, on GCE only."""
import argparse
import json
import shutil
from pathlib import Path

from scripts.maritime_cloud_inventory import (
    digest_bytes, file_sha256, json_bytes, parse_gs, require_cloud_vm, verify_output,
)
from scripts.maritime_release_build import pinned


def read_index(client, pin=None, uri=None):
    if pin is None:
        bucket, name = parse_gs(uri)
        blob = client.bucket(bucket).blob(name)
        blob.reload()
        payload = blob.download_as_bytes(if_generation_match=int(blob.generation), checksum='crc32c')
        pin = dict(uri=uri, generation=str(blob.generation), bytes=len(payload),
                   crc32c=blob.crc32c, sha256=digest_bytes(payload))
    else:
        pinned(pin)
        payload = verify_output(client, pin)
    index = json.loads(payload)
    if (index.get('schema') != 'maritime-super-resolution-training-index-v1'
            or index.get('state') != 'FOUR_TRAINING_FORMATS_VERIFIED'
            or set(index.get('datasets', {})) != {'mother', 'd1', 'd2', 'd3'}):
        raise ValueError('Completed four-dataset training index required')
    receipt = json.loads(verify_output(client, pinned(index['release_receipt'])))
    if (receipt.get('state') != 'COMPLETE' or not receipt.get('training_format_verified')
            or receipt['manifests'] != index['datasets']
            or receipt['sample_count'] != index['sample_count']
            or receipt['useful_native_rgb_bytes'] != index['unique_useful_native_rgb_bytes']):
        raise ValueError('Training index and completed receipt disagree')
    return index, pin


def selected_shards(manifest, dataset, split):
    if (manifest.get('dataset') != dataset or not manifest.get('training_format_verified')
            or len(manifest['samples']) != manifest['sample_count']):
        raise ValueError('Verified dataset manifest required')
    identities = [row['sample_id'] for row in manifest['samples'] if row['split'] == split]
    shards = [row for row in manifest['shards'] if row['split'] == split]
    members = [identity for shard in shards for identity in shard['sample_ids']]
    if not identities or len(set(identities)) != len(identities):
        raise ValueError('Empty split or duplicate manifest identities')
    if set(members) != set(identities) or len(members) != len(identities):
        raise ValueError('Shard membership does not match frozen split')
    return shards


def download_shard(client, pin, target):
    pinned(pin)
    bucket, name = parse_gs(pin['uri'])
    blob = client.bucket(bucket).blob(name, generation=int(pin['generation']))
    blob.reload()
    if (str(blob.generation) != str(pin['generation']) or int(blob.size) != pin['bytes']
            or blob.crc32c != pin['crc32c']):
        raise ValueError('Shard generation, size or CRC differs from manifest')
    path = target / (pin['sha256'] + '.tar')
    if path.exists():
        if path.stat().st_size != pin['bytes'] or file_sha256(path) != pin['sha256']:
            raise ValueError('Existing local shard is corrupt; remove that file and retry')
        return path
    temporary = path.with_suffix('.partial')
    try:
        blob.download_to_filename(str(temporary), if_generation_match=int(pin['generation']), checksum='crc32c')
        if temporary.stat().st_size != pin['bytes'] or file_sha256(temporary) != pin['sha256']:
            raise ValueError('Downloaded shard size or SHA-256 mismatch')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def run(args):
    instance = require_cloud_vm()
    from google.cloud import storage
    client = storage.Client()
    pin = json.loads(Path(args.index_pin).read_text(encoding='utf-8-sig')) if args.index_pin else None
    index, pin = read_index(client, pin=pin, uri=args.index_uri)
    manifest_pin = pinned(index['datasets'][args.dataset])
    payload = verify_output(client, manifest_pin)
    manifest = json.loads(payload)
    if manifest['sample_count'] != index['sample_count']:
        raise ValueError('Manifest and index sample counts disagree')
    shards = selected_shards(manifest, args.dataset, args.split)
    # Mother reads original source windows through its existing bounded cache.
    if args.dataset == 'mother':
        shards = []
    total = sum(row['object']['bytes'] for row in shards)
    if args.max_bytes < 1 or total > args.max_bytes:
        raise ValueError(f'Selected TAR bytes {total} exceed --max-bytes {args.max_bytes}')
    target = Path(args.output).resolve()
    target.mkdir(parents=True, exist_ok=True)
    missing = sum(row['object']['bytes'] for row in shards
                  if not (target / (pinned(row['object'])['sha256'] + '.tar')).exists())
    if shutil.disk_usage(target).free < missing + len(payload):
        raise ValueError('Insufficient free training disk space for selected split')
    paths = [str(download_shard(client, row['object'], target)) for row in shards]
    manifest_path = target / (args.dataset + '-manifest.json')
    manifest_path.write_bytes(payload)
    result = dict(schema='maritime-local-training-download-v1', vm_instance_id=instance,
                  index=pin, manifest=manifest_pin, manifest_path=str(manifest_path),
                  dataset=args.dataset, split=args.split, shards=paths, tar_bytes=total,
                  samples=sum(row['split'] == args.split for row in manifest['samples']))
    (target / 'download.json').write_bytes(json_bytes(result))
    print(json.dumps(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--index-pin', help='JSON object with URI, generation, bytes, CRC32C and SHA-256')
    source.add_argument('--index-uri', help='Explicit gs:// index URI; snapshot pin is saved in download.json')
    parser.add_argument('--dataset', choices=['mother', 'd1', 'd2', 'd3'], required=True)
    parser.add_argument('--split', choices=['train', 'validation', 'test'], default='train')
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-bytes', type=int, default=20_000_000_000,
                        help='Local TAR byte cap for this split (default 20 GB)')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
