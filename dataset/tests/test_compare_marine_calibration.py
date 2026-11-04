import unittest

import numpy as np

from scripts import compare_marine_calibration as comparison
from src.data import area4
from test_marine_prepare import record


class CalibrationComparisonTests(unittest.TestCase):
    def test_rejects_holdouts_oversized_and_unpinned_manifests(self):
        for records in ([record('heldout', 'test')], [record(str(i)) for i in range(7)]):
            with self.assertRaises(ValueError):
                comparison.validate_comparison_manifest({'records': records})
        source = record('a')
        source['files']['naip']['generation'] = ''
        with self.assertRaises(ValueError):
            comparison.validate_comparison_manifest({'records': [source]})

    def test_both_variants_use_common_eroded_support_and_bounded_candidates(self):
        ramp = np.linspace(20, 200, 512, dtype=np.float32)
        hr = np.broadcast_to(ramp, (4, 512, 512)).copy()
        dn = area4(hr)
        patch = dict(hr=hr, valid=np.ones((512, 512), bool), dn=dn,
                     sr=dn * .002 + .01, support=np.ones((128, 128), bool))
        def candidates():
            yield patch
            raise AssertionError('Read beyond candidate limit')
        result = comparison.compare_candidates(candidates(), 'train-a', max_patches=1)
        self.assertEqual(result['candidate_count'], 1)
        self.assertEqual(result['common_support_pixels'], 15376)
        for name, variant in result['variants'].items():
            self.assertEqual(variant['status'], 'passed')
            self.assertEqual(variant['calibration']['support'], 16384 if name == 'area4_original' else 15376)

    def test_preserves_measured_rejection_diagnostics(self):
        hr = np.zeros((4, 512, 512), np.float32)
        patch = dict(hr=hr, valid=np.ones((512, 512), bool), dn=area4(hr),
                     sr=np.zeros((4, 128, 128)), support=np.ones((128, 128), bool))
        result = comparison.compare_candidates(iter([patch]), 'train-a')
        for variant in result['variants'].values():
            self.assertEqual(variant['status'], 'failed')
            self.assertEqual(variant['diagnostics']['bands'][0]['dn_p95_minus_p05'], 0)


    def test_download_failure_leaves_durable_diagnostic_and_removes_scratch(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / 'sources.json'
            output = root / 'comparison.json'
            manifest.write_text(json.dumps({'records': [record('a')]}))
            # Network transfer is the external boundary; all report IO is real.
            with patch.object(comparison, 'copy_source', side_effect=OSError('offline')):
                with self.assertRaisesRegex(OSError, 'offline'):
                    comparison.compare(manifest, output)
            report = json.loads(output.read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['sources'][0]['error'], 'offline')
            self.assertEqual(list(root.glob('calibration-compare-*')), [])
            self.assertEqual(list(root.glob('*.npz')), [])
