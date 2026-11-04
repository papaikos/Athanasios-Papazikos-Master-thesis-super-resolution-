import io
import struct

import numpy as np
from shapely.geometry import Polygon, box

from scripts.maritime_marine_geography import CoastIndex, classify_metric, missing_calibration_ids, read_land, unwrap_ring


def test_lakes_are_not_subtracted_from_land():
    def record(level, points):
        header = struct.pack('>3I4i2I2i', level, len(points), level, 0, 0, 0, 0, 0, 0, -1, -1)
        return header + np.asarray(points, dtype='>i4').tobytes()
    stream = io.BytesIO(record(1, [(0, 0), (4000000, 0), (4000000, 4000000), (0, 4000000), (0, 0)]) +
                        record(2, [(1000000, 1000000), (2000000, 1000000), (1000000, 2000000), (1000000, 1000000)]))
    rings = read_land(stream)
    assert len(rings) == 1
    assert Polygon(rings[0]).contains(box(1.1, 1.1, 1.2, 1.2))


def test_dateline_ring_preserves_small_island():
    ring = unwrap_ring(np.array([[179, 10], [-179, 10], [-179, 12], [179, 12], [179, 10.]]))
    assert np.ptp(ring[:, 0]) == 2
    assert Polygon(ring).contains(box(179.5, 10.5, 180.5, 11.5))


def test_open_ocean_and_shoreline_and_freshwater():
    land = box(0, 0, 10000, 10000)
    assert classify_metric(box(-2000, 0, -1000, 1000), land)['status'] == 'MARINE_INTERSECTION'
    assert classify_metric(box(-500, 1000, 500, 2000), land)['status'] == 'MARINE_INTERSECTION'
    inland = classify_metric(box(4000, 4000, 5000, 5000), land)
    assert inland['status'] == 'INLAND_FAR_FROM_MARINE_COAST'
    assert inland['marine_area_m2'] == 0
    assert inland['shore_distance_m'] == 4000


def test_legacy_coast_margin_is_uncertainty_not_marine_fact():
    result = classify_metric(box(300, 1000, 400, 1100), box(0, 0, 10000, 10000), margin_m=500)
    assert result['status'] == 'COAST_MARGIN_UNCERTAIN'
    assert result['marine_area_m2'] == 0


def test_invalid_land_is_unknown_not_rejected():
    invalid = Polygon([(0, 0), (1000, 1000), (0, 1000), (1000, 0), (0, 0)])
    assert classify_metric(box(200, 200, 300, 300), invalid)['status'] == 'UNKNOWN_INVALID_GEOMETRY'


def test_no_land_geometry_means_ocean():
    from shapely.geometry import GeometryCollection
    assert classify_metric(box(0, 0, 1000, 1000), GeometryCollection())['marine_area_m2'] == 1000000


def test_projected_guard_classifies_land_and_ocean():
    ring = np.array([[2., 0.], [4., 0.], [4., 2.], [2., 2.], [2., 0.]])
    coast = CoastIndex([ring])
    assert coast.classify({'source': {'epsg': 32631}, 'physical_core_bounds':
        [499000, 110000, 500280, 111280]})['status'] == 'INLAND_FAR_FROM_MARINE_COAST'
    assert CoastIndex([]).classify({'source': {'epsg': 32631}, 'physical_core_bounds':
        [499000, 110000, 500280, 111280]})['status'] == 'MARINE_INTERSECTION'


def test_missing_calibration_records_are_detected():
    ledger = {'reviews': [{'sample_id': 'accepted'}, {'sample_id': 'rejected'}]}
    assert missing_calibration_ids(ledger, []) == ['accepted', 'rejected']
    assert missing_calibration_ids(ledger, [{'sample_id': 'accepted'}]) == ['rejected']
    assert missing_calibration_ids(ledger, ledger['reviews']) == []
