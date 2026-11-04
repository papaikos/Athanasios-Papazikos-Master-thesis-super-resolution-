import unittest

from src.train import validate_gpu


class TrainingGPUContractTests(unittest.TestCase):
    def test_exactly_one_expected_gpu(self):
        for gpu, name in [("t4", "Tesla T4"), ("t4", "NVIDIA Tesla T4"), ("l4", "NVIDIA L4")]:
            validate_gpu(gpu, 1, name)
        for gpu, count, name in [
            ("l4", 2, "NVIDIA L4"), ("l4", 0, "NVIDIA L4"),
            ("l4", 1, "Tesla T4"), ("t4", 1, "NVIDIA L4"),
            ("a100", 1, "NVIDIA A100"), ("l4", 1, "NVIDIA L40"),
        ]:
            with self.subTest(gpu=gpu, count=count, name=name):
                with self.assertRaises(RuntimeError):
                    validate_gpu(gpu, count, name)
