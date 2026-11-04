"""Conservative marine-coast screening using pinned GSHHG L1 land interiors.

Lakes and inland rivers are intentionally not holes in land. The 2017 coast
is a topology prior, not contemporary marine ground truth or vessel evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import zipfile

import numpy as np
from rasterio.warp import transform
from shapely.geometry import GeometryCollection, Polygon, box
from shapely.ops import transform as geometry_transform, unary_union

URL = 'https://github.com/GenericMappingTools/gshhg-gmt/releases/download/2.3.7/gshhg-bin-2.3.7.zip'
SHA = '28600e8f7a08645aab43079326df6504212ec5ccb2b4bcf3b5f4f12ed60e82bc'
SIZE = 118617033


def unwrap_ring(points):
    points = np.asarray(points, dtype=np.float64).copy()
    points[:, 0] = np.rad2deg(np.unwrap(np.deg2rad(points[:, 0])))
    return points


def read_land(stream):
    rings = []
    while True:
        header = stream.read(44)
        if not header:
            break
        if len(header) != 44:
            raise ValueError('Truncated GSHHG header')
        fields = struct.unpack('>3I4i2I2i', header)
        count, level = fields[1], fields[2] & 255
        data = stream.read(count * 8)
        if len(data) != count * 8:
            raise ValueError('Truncated GSHHG coordinate data')
        if level == 1:
            rings.append(unwrap_ring(np.frombuffer(data, dtype='>i4').reshape(-1, 2) * 1e-6))
    return rings


def classify_metric(core, land, margin_m=500):
    if not core.is_valid or not land.is_valid:
        return dict(status='UNKNOWN_INVALID_GEOMETRY', marine_area_m2=None, shore_distance_m=None)
    ocean_area = core.difference(land).area
    distance = None if land.is_empty else core.distance(land.boundary)
    if ocean_area > 1:
        status = 'MARINE_INTERSECTION'
    elif distance is not None and distance <= margin_m:
        status = 'COAST_MARGIN_UNCERTAIN'
    else:
        status = 'INLAND_FAR_FROM_MARINE_COAST'
    return dict(status=status, marine_area_m2=ocean_area, shore_distance_m=distance)


def missing_calibration_ids(ledger, evaluated):
    present = {r['sample_id'] for r in evaluated}
    return sorted({r['sample_id'] for r in ledger['reviews']} - present)


class CoastIndex:
    def __init__(self, rings):
        self.rings = rings
        self.bounds = np.array([(p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()) for p in rings]).reshape(-1, 4)
        self.polygons = {}

    def classify(self, record, margin_m=500):
        crs = f"EPSG:{record['source']['epsg']}"
        left, bottom, right, top = record['physical_core_bounds']
        xs, ys = transform(crs, 'EPSG:4326', [left, right, right, left], [bottom, bottom, top, top])
        ll = unwrap_ring(np.column_stack([xs, ys]))
        lon, lat = ll.mean(axis=0)
        if abs(lat) > 75 or np.ptp(ll[:, 0]) > 1:
            return dict(status='UNKNOWN_UNSUPPORTED_GEOGRAPHY', marine_area_m2=None, shore_distance_m=None)
        clip = box(ll[:, 0].min() - .1, ll[:, 1].min() - .1,
                   ll[:, 0].max() + .1, ll[:, 1].max() + .1)
        local = []
        for shift in (-720, -360, 0, 360, 720):
            b = self.bounds
            indices = np.where((b[:, 0] + shift <= clip.bounds[2]) & (b[:, 2] + shift >= clip.bounds[0]) &
                               (b[:, 1] <= clip.bounds[3]) & (b[:, 3] >= clip.bounds[1]))[0]
            for i in indices:
                key = (int(i), shift)
                if key not in self.polygons:
                    ring = self.rings[i].copy()
                    ring[:, 0] += shift
                    polygon = Polygon(ring)
                    self.polygons[key] = (polygon, polygon.is_valid)
                polygon, valid = self.polygons[key]
                if not valid:
                    return dict(status='UNKNOWN_INVALID_GEOMETRY', marine_area_m2=None, shore_distance_m=None)
                clipped = polygon.intersection(clip)
                if not clipped.is_empty:
                    def project(x, y, z=None):
                        # Longitude wrapping does not change the physical position.
                        x = (np.asarray(x) + 180) % 360 - 180
                        return transform('EPSG:4326', crs, x.tolist(), list(y))
                    local.append(geometry_transform(project, clipped))
        land = unary_union(local) if local else GeometryCollection()
        result = classify_metric(box(left, bottom, right, top), land, margin_m)
        result.update(center_wgs84=[float(lon), float(lat)], margin_m=margin_m,
                      shore_distance_is_clipped_lower_bound=True,
                      shore_distance_basis='L1 polygon boundary clipped to core WGS84 envelope plus 0.1 degree; artificial edges are farther than the 1 km rejection threshold at supported latitudes')
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--ledger', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    path = Path(args.archive)
    if path.stat().st_size != SIZE or hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() != SHA:
        raise ValueError('GSHHG publisher archive identity mismatch')
    with zipfile.ZipFile(path) as archive:
        members = [n for n in archive.namelist() if n.endswith('gshhs_f.b')]
        if len(members) != 1:
            raise ValueError('Expected exactly one full-resolution GSHHG binary')
        with archive.open(members[0]) as stream:
            coast = CoastIndex(read_land(stream))
    inputs = json.loads(Path(args.input).read_text())
    ledger = json.loads(Path(args.ledger).read_text())
    rows, seen = [], set()
    for record in inputs['candidates']:
        if record['sample_id'] in seen:
            continue
        seen.add(record['sample_id'])
        result = coast.classify(record, 500)
        result.update(sample_id=record['sample_id'], useful_native_rgb_bytes=record['useful_native_rgb_bytes'],
                      site_id=f"maxar-port-{record['nearest_port']['port_id']}")
        distance = result['shore_distance_m']
        result['margin_sensitivity'] = {str(m): result['status'] == 'MARINE_INTERSECTION' or
            (distance is not None and distance <= m) for m in (200, 500, 1000)}
        rows.append(result)
    index = {r['sample_id']: r for r in rows}
    missing = missing_calibration_ids(ledger, rows)
    calibration = [{**r, 'geography': index.get(r['sample_id'])} for r in ledger['reviews']]
    false_rejects = [r['sample_id'] for r in calibration if r['decision'] == 'accept' and
                    r['geography'] and r['geography']['status'] == 'INLAND_FAR_FROM_MARINE_COAST']
    # Use the most conservative 1 km exclusion threshold, not the 500 m display prior.
    rejects = [dict(sample_id=r['sample_id'], decision='reject', reason='GSHHG L1 land interior >1 km from marine coast',
                    evidence=r) for r in rows if r['status'] == 'INLAND_FAR_FROM_MARINE_COAST' and
               r['shore_distance_m'] > 1000]
    calibrated_false = [r['sample_id'] for r in calibration if r['decision'] == 'accept' and
                        r['geography'] and r['geography']['shore_distance_m'] is not None and
                        r['geography']['marine_area_m2'] == 0 and r['geography']['shore_distance_m'] > 1000]
    if calibrated_false or missing:
        rejects = []  # Fail open until legacy geometry conflicts are reviewed.
    rejected_ids = {r['sample_id'] for r in rejects}
    result = dict(schema='gshhg-marine-geography-v1', vector=dict(url=URL, sha256=SHA, bytes=SIZE, version='2.3.7'),
                  policy='L1 interiors include lakes/rivers; no vessel labels; SkySat exempt; unsupported/invalid retained UNKNOWN',
                  legacy_accuracy_caveat='2017 topology prior; 1 km conservative exclusion margin is heuristic, not verified coast accuracy',
                  input_sha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
                  ledger_sha256=hashlib.sha256(Path(args.ledger).read_bytes()).hexdigest(),
                  candidate_count=len(rows), evaluated=rows, reviews=rejects, calibration=calibration,
                  calibration_false_rejects_500m=false_rejects, calibration_false_rejects_1000m=calibrated_false,
                  calibration_missing_ids=missing, calibration_complete=not missing,
                  exclusions_enabled=not calibrated_false and not missing,
                  retained_native_rgb_bytes=sum(r['useful_native_rgb_bytes'] for r in rows if r['sample_id'] not in rejected_ids),
                  rejected_native_rgb_bytes=sum(r['useful_native_rgb_bytes'] for r in rows if r['sample_id'] in rejected_ids),
                  retained_site_count=len({r['site_id'] for r in rows if r['sample_id'] not in rejected_ids}))
    Path(args.output).write_text(json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False))
    print(json.dumps({k: v for k, v in result.items() if k not in ('evaluated', 'reviews', 'calibration')}))


if __name__ == '__main__':
    main()
