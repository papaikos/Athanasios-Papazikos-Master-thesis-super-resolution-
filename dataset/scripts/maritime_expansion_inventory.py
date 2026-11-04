"""Normalize provisional source metadata without granting training eligibility.

EarthView capture identity excludes delivery CRS/product. Two representations
of the same satellite observation must face the same overlap protection.
"""
import copy
from datetime import datetime, timezone


def normalize_candidate(original):
    source = copy.deepcopy(original)
    if source.get('schema') == 'earthview-candidate-v1':
        identity = source['acquisition_identity']
        timestamp = datetime.fromisoformat(identity[0].replace('Z', '+00:00'))
        if timestamp.tzinfo is None:
            raise ValueError('EarthView acquisition timestamp requires a timezone')
        if not isinstance(identity[1], str) or not identity[1]:
            raise ValueError('EarthView satellite identity is unresolved')
        capture = 'earthview:' + identity[1] + ':' + timestamp.astimezone(timezone.utc).isoformat(timespec='microseconds')
        site = 'earthview-port-' + str(source['port']['id'])
        return dict(sample_id=source['sample_id'], source_kind='earthview', source=source,
            acquisition_id=capture, site_id=site, geographic_group_id=site,
            crs=source['source_crs'], core_bounds=source['physical_core_bounds'],
            country=source['port']['country'], vessel_presence='unconfirmed',
            small_craft_presence='unconfirmed', requires_stratified_native_audit=True,
            trust_basis='original_TOA_support_and_frozen_mosaic_mapping; quality_and_semantics_pending')
    if source.get('schema') == 'maxar-mixed-native-v1':
        port = source['nearest_port']
        capture = source['source']
        site = 'maxar-port-' + str(port['port_id'])
        return dict(sample_id=source['sample_id'], source_kind='maxar', source=source,
            acquisition_id=capture['catalog_id'], site_id=site, geographic_group_id=site,
            crs='EPSG:'+str(capture['epsg']), core_bounds=source['physical_core_bounds'],
            country=port['country_code_as_published'], vessel_presence='unconfirmed',
            small_craft_presence='unconfirmed', requires_stratified_native_audit=True,
            trust_basis='original_masks_and_coastal_port_prior; quality_and_semantics_pending')
    if (source.get('source_kind') in ('skysat', 'skysat_raw_window', 'maxar', 'earthview')
            and isinstance(source.get('source'), dict)
            and all(key in source for key in ('sample_id', 'site_id', 'acquisition_id', 'crs', 'core_bounds'))):
        return source
    raise ValueError('Unsupported original candidate schema; explicit source adapter required')


def normalize_manifest(document):
    candidates = document.get('candidates')
    if not isinstance(candidates, list):
        raise ValueError('Candidate list required')
    rows = [normalize_candidate(row) for row in candidates]
    if len({row['sample_id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate candidate sample identity')
    return rows
