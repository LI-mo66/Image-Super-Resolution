"""Test diagnostic compatibility only; no optimizer or training."""
import unittest
from unittest.mock import patch
import torch
from check_overlap_protocol import deterministic_cumsum_supported


class Compatibility(unittest.TestCase):
    def test_supported_cpu_restores_mode(self):
        before = torch.are_deterministic_algorithms_enabled()
        warn = torch.is_deterministic_algorithms_warn_only_enabled()
        self.assertTrue(deterministic_cumsum_supported('cpu'))
        self.assertEqual(torch.are_deterministic_algorithms_enabled(), before)
        self.assertEqual(torch.is_deterministic_algorithms_warn_only_enabled(), warn)

    def test_known_unsupported_kernel_is_explicit_fallback(self):
        before = torch.are_deterministic_algorithms_enabled()
        warn = torch.is_deterministic_algorithms_warn_only_enabled()
        with patch.object(torch, 'ones', side_effect=RuntimeError(
                'cumsum_cuda_kernel does not have a deterministic implementation')):
            self.assertFalse(deterministic_cumsum_supported('cuda'))
        self.assertEqual(torch.are_deterministic_algorithms_enabled(), before)
        self.assertEqual(torch.is_deterministic_algorithms_warn_only_enabled(), warn)

    def test_unrelated_runtime_failure_is_not_suppressed(self):
        with patch.object(torch, 'ones', side_effect=RuntimeError('CUDA out of memory')):
            with self.assertRaisesRegex(RuntimeError, 'out of memory'):
                deterministic_cumsum_supported('cuda')


if __name__ == '__main__':
    unittest.main()
