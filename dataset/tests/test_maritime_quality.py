import numpy as np
import pytest


def test_clean_water_metrics_trigger_review_without_rejecting():
    from src.maritime_quality import quality_metrics
    rgb = np.empty((3, 512, 512), np.float32)
    rgb[0], rgb[1], rgb[2] = 0.1, 0.3, 0.4
    result = quality_metrics(rgb)
    assert result['gradient_strength_mean'] == 0
    assert result['nearblack_fraction'] == result['nearwhite_fraction'] == 0
    assert result['review_triggers']
    assert result['decision'] == 'review_metrics_only'
    assert result['claim'] == 'not a validated cloud or blur classifier'


def test_bright_docks_and_rgb_variation_are_measured():
    from src.maritime_quality import quality_metrics
    rgb = np.zeros((3, 544, 544), np.float32)
    rgb[:, :, 272:] = 1
    result = quality_metrics(rgb)
    assert result['nearblack_fraction'] == result['nearwhite_fraction'] == 0.5
    assert result['low_color_variation_fraction'] == 1
    assert result['rgb_std'] == [0.5, 0.5, 0.5]
    assert result['gradient_strength_mean'] > 0
    assert result['dynamic_range_p99_p01'] == [1, 1, 1]


@pytest.mark.parametrize('values', [np.zeros((3, 16, 16), np.float32),
    np.full((3, 512, 512), np.nan, np.float32),
    np.full((3, 512, 512), 2, np.float32), np.zeros((3, 512, 512), np.uint8)])
def test_invalid_observed_rgb_rejected(values):
    from src.maritime_quality import quality_metrics
    with pytest.raises(ValueError): quality_metrics(values)


def detection(x, y, length, width, confidence=0.9, class_id=1):
    return dict(class_id=class_id, confidence=confidence,
                polygon=[[x, y], [x + length, y], [x + length, y + width], [x, y + width]])


def test_ship_size_bins_dedup_and_half_open_core_ownership():
    from src.maritime_quality import summarize_ship_detections
    detections = [detection(10, 10, 100, 10), detection(10, 10, 100, 10, 0.8),
                  detection(200, 100, 300, 20), detection(600, 200, 20, 10),
                  detection(1000, 0, 40, 20), detection(-20, 50, 40, 20),
                  detection(500, 400, 100, 50, class_id=7)]
    result = summarize_ship_detections(detections, pixel_spacing_m=0.5,
                                        core_bounds=[0, 0, 1020, 1020])
    assert result['evidence_status'] == 'model_predicted'
    assert result['ship_counts'] == dict(small=2, medium=1, large=1)
    assert result['harbor_count'] == 1
    assert result['duplicates_removed'] == 1
    assert result['outside_core_removed'] == 1
    assert next(item for item in result['detections'] if item['size_bin'] == 'medium')['long_edge_m'] == 50
    assert any(item['polygon_clipped_to_core'] for item in result['detections'])


def test_rotated_polygon_length_is_physical_edge_not_axis_aligned_box():
    from src.maritime_quality import summarize_ship_detections
    polygon = [[100, 100], [160, 180], [152, 186], [92, 106]]
    result = summarize_ship_detections([dict(class_id=1, confidence=0.95, polygon=polygon)],
                                        pixel_spacing_m=0.5, core_bounds=[0, 0, 512, 512])
    assert result['detections'][0]['long_edge_m'] == 50
    assert result['ship_counts']['medium'] == 1


def test_context_ranking_never_treats_harbor_prediction_as_marina_proof():
    from src.maritime_quality import rank_scene, summarize_ship_detections
    summary = summarize_ship_detections([detection(10, 10, 200, 100, class_id=7)],
        pixel_spacing_m=0.5, core_bounds=[0, 0, 512, 512])
    ranking = rank_scene(summary)
    assert ranking['tier'] == 'harbor_predictions'
    assert ranking['evidence_status'] == 'model_predicted'
    assert rank_scene(summary, context_tags=['marina'])['context_tags'] == ['marina']


@pytest.mark.parametrize('change', ['nan', 'negative_spacing', 'bad_class', 'bad_polygon', 'bad_confidence'])
def test_malformed_detector_geometry_rejected(change):
    from src.maritime_quality import summarize_ship_detections
    item, spacing = detection(10, 10, 100, 10), 0.5
    if change == 'nan': item['polygon'][0][0] = float('nan')
    elif change == 'negative_spacing': spacing = -1
    elif change == 'bad_class': item['class_id'] = 'ship'
    elif change == 'bad_polygon': item['polygon'] = [[0, 0]]
    elif change == 'bad_confidence': item['confidence'] = 2
    with pytest.raises(ValueError):
        summarize_ship_detections([item], pixel_spacing_m=spacing, core_bounds=[0, 0, 512, 512])


def test_rgb_gradient_does_not_lose_equal_intensity_colored_edges():
    from src.maritime_quality import quality_metrics
    rgb = np.zeros((3, 512, 512), np.float32)
    rgb[0, :, :256] = 1
    rgb[1, :, 256:] = 1
    assert quality_metrics(rgb)['gradient_strength_mean'] > 0


def test_polygon_iou_suppresses_partial_overlap_and_keeps_different_classes():
    from src.maritime_quality import summarize_ship_detections
    result = summarize_ship_detections([detection(10, 10, 100, 10, 0.8),
        detection(30, 10, 100, 10, 0.9), detection(30, 10, 100, 10, 0.7, 7)],
        pixel_spacing_m=0.5, core_bounds=[0, 0, 512, 512])
    assert result['duplicates_removed'] == 1
    assert result['ship_counts']['medium'] == result['harbor_count'] == 1
    assert result['detections'][0]['confidence'] == 0.9


def test_unknown_context_cannot_be_ranked_as_marina():
    from src.maritime_quality import rank_scene, summarize_ship_detections
    summary = summarize_ship_detections([], pixel_spacing_m=1, core_bounds=[0, 0, 512, 512])
    assert rank_scene(summary, context_tags=['shoreline'])['tier'] == 'declared_shoreline_context'
    assert rank_scene(summary, context_tags=['marina'])['tier'] == 'declared_marina_context'
    with pytest.raises(ValueError): rank_scene(summary, context_tags=['confirmed_ships'])
