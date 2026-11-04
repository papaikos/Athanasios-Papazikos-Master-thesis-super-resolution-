"""Close and publish generation-pinned job evidence with byte readback."""
import argparse
import json
from pathlib import Path

from scripts.maritime_cloud_inventory import create_only, json_bytes, parse_gs, require_cloud_vm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--log', required=True)
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    require_cloud_vm()
    from google.cloud import storage
    bucket, prefix = parse_gs(args.prefix)
    client = storage.Client()
    destination = client.bucket(bucket)
    log = Path(args.log)
    payload = log.read_bytes()
    name = f'{prefix}/{log.name}'
    ref = dict(uri=f'gs://{bucket}/{name}', **create_only(destination.blob(name), payload))
    receipt = Path(args.receipt)
    record = json.loads(receipt.read_text())
    record['closed_log'] = ref
    name = f'{prefix}/{receipt.name}'
    print(json.dumps(dict(uri=f'gs://{bucket}/{name}',
                         **create_only(destination.blob(name), json_bytes(record)))))


if __name__ == '__main__':
    main()
