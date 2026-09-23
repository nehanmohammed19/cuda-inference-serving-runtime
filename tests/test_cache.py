import unittest
import numpy as np
from runtime.cache import PagedCache


class CacheTests(unittest.TestCase):
    def test_fork_copy_on_write_and_reuse(self):
        c = PagedCache(np, 3, 2, 1, 2)
        c.create('a')
        c.append('a', np.ones((1, 2)), np.ones((1, 2)))
        c.fork('a', 'b')
        c.append('b', np.full((1, 2), 7), np.full((1, 2), 8))
        self.assertNotEqual(c.tables['a'], c.tables['b'])
        self.assertEqual(c.lengths['a'], 1)
        np.testing.assert_equal(c.dense('b')[0][1], [[7, 7]])
        c.check()
        c.release('a'); c.release('b'); c.check()
        self.assertEqual(len(c.free), 3)

    def test_exhaustion_is_atomic(self):
        c = PagedCache(np, 1, 1, 1, 1)
        c.create('a'); c.append('a', [[1]], [[2]])
        with self.assertRaises(MemoryError):
            c.append('a', [[3]], [[4]])
        self.assertEqual(c.lengths['a'], 1)
        c.check()

    def test_duplicate_and_scattered_pages(self):
        c = PagedCache(np, 5, 2, 1, 1)
        c.create('a'); c.create('b')
        with self.assertRaises(ValueError): c.create('a')
        for i in range(5):
            c.append('a', [[i]], [[i + 1]])
            if i < 2: c.append('b', [[9]], [[9]])
        np.testing.assert_equal(c.dense('a')[0].ravel(), range(5))
        c.release('b'); c.check()
