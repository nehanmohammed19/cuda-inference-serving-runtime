import unittest
import numpy as np
from runtime.model import Config, Model, toy_weights
from runtime.backend import Backend


class ModelTests(unittest.TestCase):
    def test_batched_matches_isolated(self):
        c=Config(); w=toy_weights(c)
        model=Model(Backend(),c,w,pages=16,page_size=2)
        model.create('a'); model.create('b')
        observed=[]
        for t in range(5): observed.append(model.step([t+1,t+10],['a','b']))
        for row,key in enumerate(['a','b']):
            isolated=Model(Backend(),c,w,pages=16,page_size=2); isolated.create(key)
            for t in range(5):
                expected=isolated.step([t+(1 if row==0 else 10)],[key])
                np.testing.assert_allclose(observed[t][row],expected[0],atol=3e-6,rtol=2e-5)
        model.release('a'); model.release('b')
        for cache in model.caches: cache.check()

    def test_preflight_does_not_partially_append(self):
        c=Config(); model=Model(Backend(),c,toy_weights(c),pages=1,page_size=1)
        model.create('a'); model.step([1],['a'])
        with self.assertRaises(MemoryError): model.step([2],['a'])
        self.assertTrue(all(cache.lengths['a']==1 for cache in model.caches))
