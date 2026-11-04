import copy
import unittest

from scripts.prepare_marine_addition import checksum, merge_dataset, validate_pair_arrays


class MarineDatasetTests(unittest.TestCase):
    def setUp(self):
        self.source = dict(source_sha256='a' * 64, source_object={'uri': 'gs://old'},
                           group_id='heldout', split='test', pairs=43, accepted_source_bytes=100)
        self.old = dict(version=1, sources=[self.source], bands=['R', 'G', 'B'], hr_m=2.5, lr_m=10)
        self.old['sha256'] = checksum(self.old)
        self.new = dict(source_sha256='b' * 64, source_object={'uri': 'gs://new'},
                        group_id='new', split='train', pairs=12, accepted_source_bytes=200)

    def test_preserves_frozen_heldouts_and_counts(self):
        original = copy.deepcopy(self.old)
        result = merge_dataset(self.old, [self.new])
        self.assertEqual(self.old, original)
        self.assertEqual(result['sources'][0], self.source)
        self.assertEqual(result['totals']['train']['pairs'], 12)
        self.assertEqual(result['marine_content_review']['verified_maritime_accepted_bytes'], 0)
        self.assertEqual(result['sha256'], checksum(result))

    def test_rejects_native_duplicate(self):
        self.new['source_sha256'] = self.source['source_sha256']
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            merge_dataset(self.old, [self.new])

    def test_rejects_heldout_overlap(self):
        self.new['group_id'] = 'heldout'
        with self.assertRaisesRegex(ValueError, 'heldouts'):
            merge_dataset(self.old, [self.new])

    def test_rejects_modified_frozen_manifest(self):
        self.old['hr_m'] = 1
        with self.assertRaisesRegex(ValueError, 'checksum'):
            merge_dataset(self.old, [self.new])

    def test_empty_duplicate_native_receipt_is_audit_only(self):
        empty = {**self.new, 'source_sha256': self.source['source_sha256'],
                 'pairs': 0, 'accepted_source_bytes': 0, 'shards': []}
        result = merge_dataset(self.old, [empty, self.new])
        self.assertEqual(len(result['sources']), 2)
        self.assertEqual(result['sources'][-1], self.new)

    def test_nonempty_zero_pair_receipt_rejected(self):
        self.new.update(pairs=0, shards=[])
        with self.assertRaisesRegex(ValueError, 'Zero-pair'):
            merge_dataset(self.old, [self.new])


class PairGeometryTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.hr = np.ones((3, 512, 512), np.float32)
        self.lr = np.ones((3, 128, 128), np.float32)
        self.mask = np.zeros((128, 128), bool)
        self.mask[2:-2, 2:-2] = True
        self.receipt = dict(source_id='new', group_id='group', split='train',
                            source_object={'uri': 'gs://new'}, manifest_sha256='a' * 64, recipe='rgb-v2')
        self.row = dict(asset_id='asset', group_id='group', grid={'crsCode': 'EPSG:32632'})
        self.info = dict(self.receipt, asset_id='asset', crs='EPSG:32632', cell=[1, 2],
                         native_covered_bytes=7680000,
                         transform=[2.5, 0, 1280, 0, -2.5, 3840],
                         lr_transform=[10, 0, 1280, 0, -10, 3840])

    def validate(self):
        return validate_pair_arrays(self.hr, self.lr, self.mask, self.info, self.row, self.receipt)

    def test_valid_aligned_pair(self):
        self.assertEqual(self.validate(), ('asset', 'EPSG:32632', 1, 2))

    def test_misalignment_rejected(self):
        self.info['lr_transform'][2] += 10
        with self.assertRaisesRegex(ValueError, 'grid'):
            self.validate()

    def test_noncanonical_mask_rejected(self):
        self.mask[0, 0] = True
        with self.assertRaisesRegex(ValueError, 'mask'):
            self.validate()

    def test_source_identity_rejected(self):
        self.info['source_id'] = 'other'
        with self.assertRaisesRegex(ValueError, 'provenance'):
            self.validate()

    def test_wrong_metric_crs_rejected(self):
        self.info['crs'] = 'EPSG:32633'
        with self.assertRaisesRegex(ValueError, 'CRS'):
            self.validate()
