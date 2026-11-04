"""Bounded TRAIN-only blur comparison; diagnostic results never accept pairs.

Run from repository root: python -m scripts.compare_marine_calibration
"""

import argparse
from datetime import datetime, timezone
from itertools import islice
import json
from pathlib import Path
import tempfile

import numpy as np

from src.data import degrade, fit_acquisition_group, sha256
from src.marine_prepare import candidate_patches, copy_source, validate_manifest, write_json


def validate_comparison_manifest(manifest):
    records = manifest['records']
    if not 1 <= len(records) <= 6 or any(r['split'] != 'train' for r in records):
        raise ValueError('Comparison requires 1 to 6 TRAIN records only')
    validate_manifest(manifest)
    return records


def compare_candidates(candidates, source_id, *, max_patches=8):
    if not 1 <= max_patches <= 32:
        raise ValueError('Comparison requires 1 to 32 patches per source')
    baseline, blurred, references, supports, original = [], [], [], [], []
    for patch in islice(candidates, max_patches):
        lowpass, mask = degrade(patch['hr'], patch['valid'])
        baseline.append(patch['dn'])
        blurred.append(lowpass)
        references.append(patch['sr'])
        original.append(patch['support'])
        supports.append(patch['support'] & mask)
    result = {'candidate_count': len(baseline),
              'common_support_pixels': sum(int(s.sum()) for s in supports),
              'original_support_pixels': sum(int(s.sum()) for s in original),
              'variants': {}}
    for name, inputs, masks in (
        ('area4_original', baseline, original),
        ('area4_common', baseline, supports),
        ('gaussian_sigma2_common', blurred, supports),
    ):
        try:
            if not inputs:
                raise ValueError('No complete coastal patches with stable land')
            mapping = fit_acquisition_group(
                np.concatenate(inputs, axis=2), np.concatenate(references, axis=2),
                np.concatenate(masks, axis=1), site_ids=[source_id])
            result['variants'][name] = {'status': 'passed', 'calibration': mapping}
        except ValueError as error:
            result['variants'][name] = {
                'status': 'failed', 'error_type': type(error).__name__,
                'error': str(error), 'diagnostics': getattr(error, 'diagnostics', None)}
    return result


def compare(manifest_path, output, *, max_patches=8):
    manifest_path, output = Path(manifest_path), Path(output)
    records = validate_comparison_manifest(json.loads(manifest_path.read_text()))
    if not 1 <= max_patches <= 32:
        raise ValueError('Comparison requires 1 to 32 patches per source')
    # Never overwrite another diagnostic run or imply source acceptance.
    if output.exists():
        raise ValueError('Comparison output already exists; choose a new path')
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        'purpose': 'TRAIN-only hypothesis comparison, not acceptance or training',
        'source_manifest_sha256': sha256(manifest_path),
        'max_patches_per_source': max_patches, 'status': 'running', 'sources': [],
    }
    def save():
        report['updated_at'] = datetime.now(timezone.utc).isoformat()
        write_json(output, report)
    save()
    for record in records:
        outcome = {'id': record['id'], 'files': record['files'], 'status': 'running',
                   'candidate_diagnostics': {}}
        report['sources'].append(outcome)
        save()
        try:
            with tempfile.TemporaryDirectory(prefix='calibration-compare-', dir=output.parent) as temp:
                scratch = Path(temp)
                for kind in ('naip', 's2', 'nativevalid'):
                    copy_source(record['files'][kind], scratch / (kind + '.tif'))
                candidates = candidate_patches(
                    scratch / 'naip.tif', scratch / 's2.tif', scratch / 'nativevalid.tif',
                    diagnostics=outcome['candidate_diagnostics'])
                try:
                    outcome.update(compare_candidates(candidates, record['id'], max_patches=max_patches))
                finally:
                    candidates.close()
                outcome['status'] = 'complete'
        except Exception as error:
            outcome.update(status='error', error_type=type(error).__name__, error=str(error))
            save()
            report['status'] = 'failed'
            save()
            raise
        save()
    report['status'] = 'complete'
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-patches', type=int, default=8)
    args = parser.parse_args()
    compare(args.manifest, args.output, max_patches=args.max_patches)


if __name__ == '__main__':
    main()
