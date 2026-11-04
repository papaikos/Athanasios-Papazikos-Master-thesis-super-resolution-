"""Fetch only small review previews and metadata to the local review directory."""
import json
from pathlib import Path
from scripts.maritime_build_control import client_from_gcloud, BUCKET


def main():
    client = client_from_gcloud()
    prefix = 'skysat/thesis-datasets/v1/control/runs/maritime-20261006/'
    root = Path('reports/maritime-build-20261006/review')
    rows = []
    paths = [('catalog',f'catalog-screen-v2/part-{i}/groups/') for i in range(4)]
    paths += [('native','pilot-proof-v1/review/native/')]
    for kind, path in paths:
        blobs = list(client.list_blobs(BUCKET, prefix=prefix + path))
        if kind == 'native':
            blobs = blobs[:12]
        for blob in blobs:
            if blob.size > 15_000_000:
                raise ValueError('Oversized local preview')
            destination = root / kind / Path(blob.name).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            payload = blob.download_as_bytes(if_generation_match=int(blob.generation))
            destination.write_bytes(payload)
            if kind == 'catalog':
                row = json.loads(payload)
                rows.append(row)
                if 'preview' in row and (row['priority'] >= 1):
                    name = row['preview']['uri'].split('/'+BUCKET+'/',1)[1]
                    image = client.bucket(BUCKET).blob(name, generation=int(row['preview']['generation']))
                    (destination.with_suffix('.png')).write_bytes(image.download_as_bytes(if_generation_match=int(row['preview']['generation'])))
    (root/'catalog-rows.json').write_text(json.dumps(rows, indent=2))
    print(json.dumps(dict(catalog_groups=len(rows), native_previews=len(list((root/'native').glob('*.png'))))))


if __name__ == '__main__':
    main()
