"""Marine geography prior for provisional candidates, never quality approval."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

from scripts.maritime_earthview_plan import marine_allowed, screen_marine
from scripts.maritime_expansion_inventory import normalize_manifest
from scripts.maritime_expansion_select import _geometry
from scripts.maritime_marine_geography import CoastIndex, SHA, SIZE, read_land


def screen_candidates(document, coast):
    rows, excluded = [], []
    for row in normalize_manifest(document):
        geometry = _geometry(row)
        epsg = int(geometry['crs'].split(':')[1])
        prior = screen_marine(coast, dict(epsg=epsg, core_bounds=geometry['core_bounds']))
        row['marine_geography_prior'] = dict(prior, gshhg_sha256=SHA,
            marine_prior_only=True, establishes_ship_presence=False)
        if marine_allowed(prior):
            rows.append(row)
        else:
            excluded.append(dict(sample_id=row['sample_id'], prior=prior))
    return dict(schema='maritime-expansion-coast-prior-v1', candidates=rows,
        exclusions=excluded, input_count=len(rows)+len(excluded),
        statuses=dict(Counter(row['marine_geography_prior']['status'] for row in rows)),
        approval='Provisional geography only; source quality, semantics and final eligibility pending')


def main():
    import argparse
    from google.cloud import storage
    from scripts.maritime_cloud_inventory import create_only, json_bytes
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cloud-prefix', required=True)
    args = parser.parse_args()
    archive = Path(args.archive)
    if archive.stat().st_size != SIZE or hashlib.sha256(archive.read_bytes()).hexdigest() != SHA:
        raise ValueError('Unverified coastline archive')
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if n.endswith('gshhs_i.b')]
        if len(names) != 1:
            raise ValueError('Expected one intermediate coastline file')
        with z.open(names[0]) as stream:
            coast = CoastIndex(read_land(stream))
    raw = Path(args.input).read_bytes()
    result = screen_candidates(json.loads(raw), coast)
    result['input_sha256'] = hashlib.sha256(raw).hexdigest()
    payload = json_bytes(result)
    Path(args.output).write_bytes(payload)
    name, key = args.cloud_prefix.removeprefix('gs://').split('/', 1)
    blob = storage.Client().bucket(name).blob(key.rstrip('/')+'/manifest.json')
    pin = dict(uri=f'gs://{name}/{blob.name}', **create_only(blob, payload))
    Path(args.output+'.pin.json').write_bytes(json_bytes(pin))
    print(json.dumps(dict(input_count=result['input_count'], provisional=len(result['candidates']),
        excluded=len(result['exclusions']), statuses=result['statuses'], pin=pin)))


if __name__ == '__main__':
    main()
