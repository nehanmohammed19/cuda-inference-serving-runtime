import unittest
import numpy as np
from runtime.reference import attention, rope, rmsnorm


class ReferenceTests(unittest.TestCase):
    def test_large_logits_are_finite(self):
        q = np.full((4, 8), 1e3, np.float32)
        k = np.full((3, 2, 8), 1e3, np.float32)
        v = np.arange(48, dtype=np.float32).reshape(3, 2, 8)
        out = attention(q, k, v)
        self.assertTrue(np.isfinite(out).all())
        np.testing.assert_allclose(out[0], v[:, 0].mean(axis=0))

    def test_rope_preserves_norm(self):
        x = np.random.default_rng(4).normal(size=(3, 2, 10)).astype(np.float32)
        y = rope(x, [0, 7, 100])
        np.testing.assert_allclose((x*x).sum(-1), (y*y).sum(-1), rtol=1e-6)

    def test_zero_norm(self):
        np.testing.assert_equal(rmsnorm(np.zeros((2, 8)), np.ones(8)), 0)
