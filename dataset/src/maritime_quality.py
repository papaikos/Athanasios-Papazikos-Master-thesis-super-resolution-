"""Transparent review metrics and model-predicted maritime geometry summaries.

These measurements are triage aids, not validated cloud/blur classifications
or evidence that a ship/harbor prediction is correct. Geometry uses convex DOTA
quadrilaterals; no detector or geometry package is required.
"""
import math

import numpy as np


def quality_metrics(rgb):
    """Measure normalized observed 512-core or 544-context RGB; never reject."""
    if (not isinstance(rgb, np.ndarray) or rgb.shape not in ((3, 512, 512), (3, 544, 544))
            or rgb.dtype.kind != 'f' or not np.isfinite(rgb).all()
            or (rgb < 0).any() or (rgb > 1).any()):
        raise ValueError('Expected finite normalized observed RGB core/context')
    values = rgb.astype(np.float64)
    percentiles = np.percentile(values, [1, 5, 50, 95, 99], axis=(1, 2)).T
    dynamic = percentiles[:, 4] - percentiles[:, 0]
    gradients = np.concatenate((np.abs(np.diff(values, axis=1)).ravel(),
                                np.abs(np.diff(values, axis=2)).ravel()))
    nearblack = float(np.mean(np.max(values, axis=0) <= 0.02))
    nearwhite = float(np.mean(np.min(values, axis=0) >= 0.98))
    gradient = float(gradients.mean())
    triggers = []
    if max(dynamic) < 0.05:
        triggers.append('low_dynamic_range_may_include_clean_water')
    if gradient < 0.005:
        triggers.append('low_gradient_may_include_clean_water')
    if nearblack > 0.1:
        triggers.append('nearblack_support_review')
    if nearwhite > 0.1:
        triggers.append('nearwhite_support_may_include_bright_docks')
    return dict(shape=list(rgb.shape), percentile_levels=[1, 5, 50, 95, 99],
        rgb_percentiles=percentiles.tolist(), dynamic_range_p99_p01=dynamic.tolist(),
        nearblack_fraction=nearblack, nearwhite_fraction=nearwhite,
        low_color_variation_fraction=float(np.mean(np.ptp(values, axis=0) <= 0.02)),
        rgb_std=values.std(axis=(1, 2)).tolist(), gradient_strength_mean=gradient,
        gradient_strength_p95=float(np.percentile(gradients, 95)),
        gradient_definition='absolute adjacent RGB channel differences, both spatial axes',
        thresholds=dict(nearblack_max_rgb=0.02, nearwhite_min_rgb=0.98,
            low_color_max_channel_range=0.02, review_max_dynamic_range=0.05,
            review_mean_gradient=0.005, review_extreme_fraction=0.1),
        review_triggers=triggers, decision='review_metrics_only',
        claim='not a validated cloud or blur classifier')


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _area(polygon):
    if len(polygon) < 3:
        return 0.0
    return abs(sum(_cross(a, b) for a, b in zip(polygon, polygon[1:] + polygon[:1]))) / 2


def _convex_quad(polygon):
    try:
        points = np.asarray(polygon, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError('Invalid detector polygon') from error
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError('Expected finite DOTA quadrilateral')
    edges = np.roll(points, -1, axis=0) - points
    turns = [_cross(edges[i], edges[(i + 1) % 4]) for i in range(4)]
    if not (all(v > 0 for v in turns) or all(v < 0 for v in turns)):
        raise ValueError('Detector polygon must be a nondegenerate convex quadrilateral')
    result = points.tolist()
    if sum(_cross(a, b) for a, b in zip(result, result[1:] + result[:1])) < 0:
        result.reverse()
    return result


def _intersection(subject, clip):
    """Sutherland-Hodgman convex polygon clipping (both polygons CCW)."""
    output = subject
    for a, b in zip(clip, clip[1:] + clip[:1]):
        if not output:
            break
        edge = np.subtract(b, a)
        source, output = output, []
        previous = source[-1]
        previous_distance = _cross(edge, np.subtract(previous, a))
        for current in source:
            distance = _cross(edge, np.subtract(current, a))
            if (distance >= 0) != (previous_distance >= 0):
                fraction = previous_distance / (previous_distance - distance)
                output.append((np.asarray(previous) + fraction * np.subtract(current, previous)).tolist())
            if distance >= 0:
                output.append(current)
            previous, previous_distance = current, distance
    return output


def _iou(a, b):
    intersection = _area(_intersection(a, b))
    union = _area(a) + _area(b) - intersection
    return intersection / union if union else 0.0


def summarize_ship_detections(detections, *, pixel_spacing_m, core_bounds,
                              confidence_threshold=0.25, iou_threshold=0.5):
    """Summarize DOTA class 1 ships/class 7 harbors in a half-open core.

    Polygons and core bounds share the detector's pixel coordinates. The pixel
    spacing must describe that grid (including any inference resampling).
    Length is the original quadrilateral's longest edge times spacing; clipped
    polygons indicate border support and do not pretend to be complete ships.
    Medium [50,150) m and large >=150 m are project bins, not vessel taxonomy.
    """
    if (not isinstance(pixel_spacing_m, (int, float)) or isinstance(pixel_spacing_m, bool)
            or not math.isfinite(pixel_spacing_m) or pixel_spacing_m <= 0):
        raise ValueError('Positive finite detector pixel spacing required')
    bounds = np.asarray(core_bounds, dtype=np.float64)
    if bounds.shape != (4,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
        raise ValueError('Invalid half-open core bounds')
    if not (0 <= confidence_threshold <= 1 and 0 < iou_threshold <= 1):
        raise ValueError('Invalid detector triage thresholds')
    left, top, right, bottom = bounds.tolist()
    clip = [[left, top], [right, top], [right, bottom], [left, bottom]]
    candidates, outside, below, other = [], 0, 0, 0
    for detection in detections:
        try:
            class_id, confidence = detection['class_id'], detection['confidence']
            if (not isinstance(class_id, int) or isinstance(class_id, bool)
                    or not isinstance(confidence, (int, float)) or isinstance(confidence, bool)
                    or not math.isfinite(confidence) or not 0 <= confidence <= 1):
                raise ValueError('Invalid detector class or confidence')
            polygon = _convex_quad(detection['polygon'])
        except (KeyError, TypeError) as error:
            raise ValueError('Malformed detector record') from error
        if class_id not in (1, 7):
            other += 1
            continue
        if confidence < confidence_threshold:
            below += 1
            continue
        centre = np.mean(polygon, axis=0)
        if not (left <= centre[0] < right and top <= centre[1] < bottom):
            outside += 1
            continue
        candidates.append(dict(class_id=class_id, confidence=float(confidence), polygon=polygon))
    candidates.sort(key=lambda item: (-item['confidence'], item['class_id'], tuple(np.asarray(item['polygon']).ravel())))
    selected, duplicates = [], 0
    counts = dict(small=0, medium=0, large=0)
    harbor_count = 0
    for candidate in candidates:
        if any(item['class_id'] == candidate['class_id'] and
               _iou(item['polygon'], candidate['polygon']) >= iou_threshold for item in selected):
            duplicates += 1
            continue
        polygon = candidate['polygon']
        long_edge = max(np.linalg.norm(np.subtract(b, a)) for a, b in zip(polygon, polygon[1:] + polygon[:1])) * pixel_spacing_m
        clipped = _intersection(polygon, clip)
        border = any(x < left or x > right or y < top or y > bottom for x, y in polygon)
        size = ('large' if long_edge >= 150 else 'medium' if long_edge >= 50 else 'small') if candidate['class_id'] == 1 else None
        if size:
            counts[size] += 1
        else:
            harbor_count += 1
        selected.append(dict(**candidate, core_polygon=clipped, polygon_clipped_to_core=border,
            long_edge_m=float(long_edge), size_bin=size, evidence_status='model_predicted'))
    return dict(evidence_status='model_predicted', ship_counts=counts, harbor_count=harbor_count,
        detections=selected, duplicates_removed=duplicates, outside_core_removed=outside,
        below_confidence_removed=below, other_classes_removed=other,
        pixel_spacing_m=float(pixel_spacing_m), core_bounds=bounds.tolist(),
        confidence_threshold=confidence_threshold, iou_threshold=iou_threshold,
        length_measure='original oriented polygon longest edge times detector grid spacing',
        size_bins_m=dict(small=[0, 50], medium=[50, 150], large=[150, None]),
        size_bin_convention='lower inclusive, upper exclusive; project bins',
        ownership='polygon centre in half-open core bounds',
        caution='model predictions require validation; harbor is not marina proof')


def rank_scene(summary, *, context_tags=()):
    """Return a priority tier, not a sample acceptance decision or quota."""
    tags = sorted(set(context_tags))
    if any(tag not in ('harbor', 'marina', 'shoreline') for tag in tags):
        raise ValueError('Unknown contextual scene tag')
    ships = summary['ship_counts']
    if ships['medium'] + ships['large']:
        tier, priority = 'medium_large_ship_predictions', 0
    elif sum(ships.values()):
        tier, priority = 'other_ship_predictions', 1
    elif summary['harbor_count']:
        tier, priority = 'harbor_predictions', 2
    elif 'marina' in tags:
        tier, priority = 'declared_marina_context', 3
    elif 'harbor' in tags:
        tier, priority = 'declared_harbor_context', 3
    elif 'shoreline' in tags:
        tier, priority = 'declared_shoreline_context', 4
    else:
        tier, priority = 'unclassified_context', 5
    return dict(tier=tier, priority=priority, context_tags=tags,
        evidence_status='model_predicted' if 'predictions' in tier else 'caller_declared_context',
        decision='ranking_only', caution='context tags require caller provenance; harbor is not marina proof')
