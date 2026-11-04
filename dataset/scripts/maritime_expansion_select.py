"""Additive train-count selection; source qualification is an explicit input.

This module never creates a READY marker. It preserves the complete baseline,
blocks frozen evaluation/development exposure, and counts decoded uint8 RGB.
"""
import argparse
import copy
from collections import Counter, defaultdict, deque
import json
import math
from pathlib import Path

from scripts.maritime_release_select import near, polygon_wgs84
from scripts.maritime_ultimate_select import disjoint_native_rows

RGB_PAIR_BYTES = 835584


def _geometry(row):
    if 'core_bounds' in row and 'crs' in row:
        return row
    from scripts.maritime_ultimate_release import source_geometry
    crs, bounds = source_geometry(row)
    return dict(row, crs=crs, core_bounds=list(bounds))


def build_selection(baseline, candidates, *, development_sites=(), protected_rows=(),
                    minimum_bytes=10000000000, target_bytes=15000000000,
                    max_site_fraction=.15, max_open_water_fraction=.10):
    if (not 0 < max_site_fraction <= 1 or not 0 < minimum_bytes <= target_bytes
            or not 0 <= max_open_water_fraction <= 1):
        raise ValueError('Invalid decoded payload target or site fraction')
    result = copy.deepcopy(baseline)
    originals = result['selection']
    if len({row['sample_id'] for row in originals}) != len(originals):
        raise ValueError('Duplicate baseline identities')
    prior_ids = {row['sample_id'] for row in originals}
    old_sites = {row['site_id']: row for row in originals}
    protected = [_geometry(row) for row in originals if row['split'] != 'train']
    protected.extend(_geometry(row) for row in protected_rows)
    protected_captures = {row['acquisition_id'] for row in protected}
    protected_sites = {row['site_id'] for row in protected}
    development_sites = set(development_sites)
    for group in result['splits'].get('groups', []):
        if group.get('split') in ('validation', 'test', 'development'):
            protected_sites.update(group['site_ids'])
        if group.get('split') == 'development':
            development_sites.update(group['site_ids'])
    protected_shapes = [polygon_wgs84(row) for row in protected]
    from shapely.strtree import STRtree
    from scripts.maritime_aerial_evaluation_release import _near_exposure
    protected_index = STRtree(protected_shapes)
    rejected, eligible, seen = [], [], set(prior_ids)
    def reject(row, reason):
        rejected.append(dict(sample_id=row['sample_id'], reason=reason))
    for original in sorted(candidates, key=lambda row: row['sample_id']):
        identity = original['sample_id']
        if identity in seen:
            reject(original, 'duplicate_identity')
            continue
        seen.add(identity)
        row = copy.deepcopy(original)
        screening = row.get('screening', {})
        if screening.get('eligible') is not True:
            reject(row, 'screening_not_qualified')
            continue
        priority = screening.get('priority')
        if isinstance(priority, bool) or not isinstance(priority, int) or not 0 <= priority <= 5:
            raise ValueError('Qualified source requires documented rank 0..5')
        if screening.get('composition') == 'open_water_negative' and priority != 5:
            raise ValueError('Open-water negative requires lowest priority 5')
        if row['site_id'] in development_sites:
            reject(row, 'development_site')
            continue
        if row['site_id'] in protected_sites:
            reject(row, 'protected_site')
            continue
        if row['acquisition_id'] in protected_captures:
            reject(row, 'protected_acquisition')
            continue
        geometry = _geometry(row)
        footprint = polygon_wgs84(geometry)
        if _near_exposure(footprint, protected_shapes, protected_index):
            reject(row, 'protected_geography')
            continue
        row['split'] = 'train'
        row['crs'], row['core_bounds'] = geometry['crs'], geometry['core_bounds']
        if row['site_id'] in old_sites:
            row['geographic_group_id'] = old_sites[row['site_id']]['geographic_group_id']
        else:
            # New groups train only; original group IDs and assignments stay frozen.
            row['geographic_group_id'] = 'expansion-' + row['site_id']
        eligible.append(row)
    # Baseline wins every identity and same-acquisition overlap decision.
    normalized_old = [_geometry(row) for row in originals]
    disjoint = disjoint_native_rows(normalized_old + eligible)
    kept = {row['sample_id'] for row in disjoint}
    if not prior_ids.issubset(kept):
        raise ValueError('Existing baseline has unresolved capture overlap')
    for row in eligible:
        if row['sample_id'] not in kept:
            reject(row, 'capture_overlap_or_crs_duplicate')
    eligible = [row for row in eligible if row['sample_id'] in kept]
    queues = defaultdict(deque)
    for row in sorted(eligible, key=lambda row: (row['screening']['priority'],
            row['screening'].get('composition') == 'open_water_negative', row['sample_id'])):
        queues[row['site_id']].append(row)
    train = sum(row['split'] == 'train' for row in originals)
    target_pairs = math.ceil(target_bytes / RGB_PAIR_BYTES)
    cap = max(1, math.floor(target_pairs * max_site_fraction))
    counts = Counter(row['site_id'] for row in originals if row['split'] == 'train')
    additions = []
    added_negatives = 0
    def negative(row):
        return row['screening'].get('composition') == 'open_water_negative'

    def composition_allowed(row):
        return (not negative(row) or added_negatives + 1 <=
                max_open_water_fraction * (len(additions) + 1) + 1e-12)

    while train < target_pairs:
        sites = [site for site in queues if queues[site] and counts[site] < cap
                 and composition_allowed(queues[site][0])]
        if not sites:
            break
        site = min(sites, key=lambda site: (queues[site][0]['screening']['priority'], counts[site], site))
        additions.append(queues[site].popleft())
        added_negatives += negative(additions[-1])
        counts[site] += 1
        train += 1
    composition_limited = sum(negative(row) and not composition_allowed(row)
                              for queue in queues.values() for row in queue) if train < target_pairs else 0
    for row in additions:
        group = row['geographic_group_id']
        assignment = result['splits']['assignment']
        if group in assignment and assignment[group] != 'train':
            raise ValueError('Expansion would change frozen assignment')
        assignment[group] = 'train'
    existing_groups = {group['geographic_group_id'] for group in result['splits'].get('groups', [])}
    for site in sorted({row['site_id'] for row in additions}):
        row = next(row for row in additions if row['site_id'] == site)
        if row['geographic_group_id'] not in existing_groups:
            result['splits'].setdefault('groups', []).append(dict(
                geographic_group_id=row['geographic_group_id'], site_ids=[site], split='train',
                development=False, prior_training=False, buffer_m_each=2000))
            existing_groups.add(row['geographic_group_id'])
    result['selection'] = originals + additions
    result['rejections'] = rejected
    payload = train * RGB_PAIR_BYTES
    result['summary'] = dict(sample_count=len(result['selection']), train_pairs=train,
        added_train_pairs=len(additions), qualified_disjoint_candidates=len(eligible),
        added_open_water_negative_count=added_negatives,
        added_open_water_negative_fraction=added_negatives/len(additions) if additions else 0,
        maximum_added_open_water_negative_fraction=max_open_water_fraction,
        composition_limited_candidates=composition_limited,
        train_rgb_bytes_per_recipe=payload, rgb_bytes_per_pair=RGB_PAIR_BYTES,
        minimum_train_rgb_bytes=minimum_bytes, target_train_rgb_bytes=target_bytes,
        minimum_shortfall_bytes=max(0, minimum_bytes-payload), target_shortfall_bytes=max(0, target_bytes-payload),
        site_train_counts=dict(counts), new_site_count_cap=cap,
        ranking_evidence_counts=dict(Counter(row['screening'].get('semantic_evidence', 'unspecified') for row in additions)),
        rejected_count=len(rejected), unselected_qualified_candidates=len(eligible)-len(additions))
    result['review_policy']['expansion'] = dict(
        version='marine-expansion-train-count-v1', baseline='preserved unchanged',
        ranking='medium/large ship predictions first, then other ships, harbor and shoreline; site count balances ties',
        semantics='prediction and observed review remain distinct',
        composition='open-water negatives bounded against actual additions; baseline preserved independently',
        protected_population='frozen sites/acquisitions plus 4km edge separation; explicit development exclusions',
        accounting='decoded uint8 RGB HR+LR per TRAIN split; not TAR size or independent information entropy')
    result['expansion_gates'] = dict(source_rights_and_resolution=False, source_support_and_geometry=False,
        marine_content_and_quality_audit=False, duplicate_and_leakage_audit=False)
    result['expansion_gate_evidence'] = {}
    result['status'] = 'REVIEW_PENDING'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--qualified-candidates', required=True, nargs='+')
    parser.add_argument('--protected-rows')
    parser.add_argument('--output', required=True)
    parser.add_argument('--minimum-bytes', type=int, default=10000000000)
    parser.add_argument('--target-bytes', type=int, default=15000000000)
    parser.add_argument('--site-fraction', type=float, default=.15)
    parser.add_argument('--open-water-fraction', type=float, default=.10)
    args = parser.parse_args()
    candidates = []
    for name in args.qualified_candidates:
        candidates.extend(json.loads(Path(name).read_text())['candidates'])
    from scripts.maritime_fast_reuse_inventory import DEVELOPMENT
    document = build_selection(json.loads(Path(args.baseline).read_text()), candidates,
        development_sites=DEVELOPMENT,
        protected_rows=json.loads(Path(args.protected_rows).read_text()) if args.protected_rows else (),
        minimum_bytes=args.minimum_bytes, target_bytes=args.target_bytes,
        max_site_fraction=args.site_fraction, max_open_water_fraction=args.open_water_fraction)
    Path(args.output).write_text(json.dumps(document, sort_keys=True, indent=2))
    print(json.dumps(document['summary'], sort_keys=True))
