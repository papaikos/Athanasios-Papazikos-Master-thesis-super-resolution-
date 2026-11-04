"""Per-image metadata table for Mother, read (only) from Mother/manifest.json.

Usage: python extract_metadata.py <Mother dir>
Writes thesis_data/mother_metadata.csv (one row per image) and thesis_data/mother_metadata.json
(field notes, manifest hash, counts). Needs pyproj for the image centre in longitude/latitude.
"""
import collections
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from pyproj import Transformer

HERE = Path(__file__).resolve().parent
# The manifest mixes ISO 3166-1 alpha-2 codes with country names (SkySat records); map names to codes.
COUNTRY_CODES = {'United States of America': 'US', 'Australia': 'AU', 'Micronesia (Federated States of)': 'FM',
                 'Jarvis Island': 'UM', 'Indonesia': 'ID', 'Solomon Islands': 'SB', 'Sri Lanka': 'LK',
                 'Peru': 'PE', 'Lebanon': 'LB'}
FIELDS = {
    'sample_id': 'image ID (also the NPZ name inside the TAR, and the same ID in D1/D2/D3)',
    'image': 'file name of the image inside its TAR',
    'split': 'train / validation / test as supplied',
    'tar': 'TAR file in Mother/<split>/ holding the image',
    'native_tif': 'native TIF in Mother/native/objects/ (native_window_ref or native_context, as the project loaders find it)',
    'source_kind': 'imagery source: aerial_rgb (NAIP), earthview, skysat_raw_window, maxar',
    'country_raw': 'country as written in the manifest',
    'country': 'ISO 3166-1 alpha-2 code (names mapped with COUNTRY_CODES)',
    'site_id': 'site identifier from the manifest',
    'geographic_group_id': 'geographic group used to keep splits apart',
    'acquisition_id': 'acquisition identifier from the manifest',
    'acquired': 'acquisition date/time: aerial source_provenance.acquisition_day; earthview source_items[0] datetime; '
                'maxar source.datetime; skysat source_metadata.acquisition_utc',
    'lon': 'longitude of the image core centre (WGS84), from core_bounds and crs',
    'lat': 'latitude of the image core centre (WGS84), from core_bounds and crs',
    'crs': 'projected CRS of the core bounds',
    'hr_gsd_m': 'HR pixel size in metres (hr_transform)',
    'native_gsd_m': 'native pixel size in metres',
    'nearest_port': 'nearest port name from the manifest (aerial, maxar) or port (earthview), if any',
    'port_center_distance_m': 'distance from the image to that port, if recorded',
    'requires_native_quality_review': 'dataset flag: native quality review still pending',
    'requires_semantic_review': 'dataset flag: semantic review still pending',
    'final_eligible': 'dataset flag',
    'skysat_review_status': 'SkySat source_provenance.status',
    'vessel_presence': 'manifest vessel label (mostly empty or unconfirmed)',
    'small_craft_presence': 'manifest small-craft label',
    'primary_category': 'manifest category label (mostly empty)',
    'hard_negative': 'manifest hard-negative label (mostly empty)',
}


def acquired(sample):
    provenance = sample['source_provenance']
    kind = sample['source_kind']
    if kind == 'aerial_rgb':
        return provenance.get('acquisition_day')
    if kind == 'earthview':
        items = provenance.get('source_items') or [{}]
        return items[0].get('item', {}).get('properties', {}).get('datetime')
    if kind == 'maxar':
        return provenance.get('source', {}).get('datetime')
    return provenance.get('source_metadata', {}).get('acquisition_utc')


def native_tif(sample):
    window = sample.get('native_window_ref')
    reference = window['object'] if window else sample.get('native_context')
    return reference['sha256'] + '.tif' if reference else None


def main(mother):
    manifest_path = mother / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    tar_of = {}
    for shard in manifest['shards']:
        name = shard['object']['sha256'] + '.tar'
        if not (mother / shard['split'] / name).exists():
            raise SystemExit(f'{shard["split"]}/{name} from the manifest is not on disk')
        tar_of.update({sample_id: name for sample_id in shard['sample_ids']})
    transformers = {}
    rows = []
    for sample in manifest['samples']:
        provenance = sample['source_provenance']
        crs = sample['crs']
        if crs not in transformers:
            transformers[crs] = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
        left, bottom, right, top = sample['core_bounds']
        lon, lat = transformers[crs].transform((left + right) / 2, (bottom + top) / 2)
        port = provenance.get('nearest_port') or provenance.get('port') or {}
        rows.append(dict(
            sample_id=sample['sample_id'], image=sample['sample_id'] + '.npz', split=sample['split'],
            tar=tar_of[sample['sample_id']], native_tif=native_tif(sample), source_kind=sample['source_kind'],
            country_raw=sample['country'], country=COUNTRY_CODES.get(sample['country'], sample['country']),
            site_id=sample.get('site_id'), geographic_group_id=sample.get('geographic_group_id'),
            acquisition_id=sample.get('acquisition_id'), acquired=acquired(sample),
            lon=round(lon, 6), lat=round(lat, 6), crs=crs, hr_gsd_m=sample['hr_transform'][0],
            native_gsd_m=sample.get('native_gsd_m'), nearest_port=port.get('name'),
            port_center_distance_m=provenance.get('port_center_distance_m'),
            requires_native_quality_review=provenance.get('requires_native_quality_review'),
            requires_semantic_review=provenance.get('requires_semantic_review'),
            final_eligible=provenance.get('final_eligible'),
            skysat_review_status=provenance.get('status') if sample['source_kind'] == 'skysat_raw_window' else None,
            vessel_presence=sample.get('vessel_presence'), small_craft_presence=sample.get('small_craft_presence'),
            primary_category=sample.get('primary_category'), hard_negative=sample.get('hard_negative')))
    out = HERE / 'thesis_data'
    out.mkdir(exist_ok=True)
    with (out / 'mother_metadata.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    groups = collections.defaultdict(set)
    for row in rows:
        groups[row['geographic_group_id']].add(row['split'])
    (out / 'mother_metadata.json').write_text(json.dumps(dict(
        created_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        manifest=str(manifest_path), manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        manifest_status=manifest.get('status'), images=len(rows), fields=FIELDS, country_codes=COUNTRY_CODES,
        counts=dict(split=collections.Counter(r['split'] for r in rows),
                    source_kind=collections.Counter(r['source_kind'] for r in rows),
                    split_by_source={f'{s}/{k}': n for (s, k), n in collections.Counter(
                        (r['split'], r['source_kind']) for r in rows).items()},
                    countries=len({r['country'] for r in rows}), sites=len({r['site_id'] for r in rows}),
                    geographic_groups=len(groups),
                    geographic_groups_in_more_than_one_split=sum(len(v) > 1 for v in groups.values()),
                    pending_quality_review=sum(r['requires_native_quality_review'] is True for r in rows))),
        indent=2), encoding='utf-8')
    print(f'{len(rows)} images -> {out / "mother_metadata.csv"}')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
