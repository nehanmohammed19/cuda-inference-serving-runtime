"""Storage, arithmetic and replay checks for the two reduced-precision formats."""
import os
import unittest
import numpy as np
from runtime.backend import Backend
from runtime.model import Config,Model,toy_weights


@unittest.skipUnless(os.environ.get('RUN_CUDA_TESTS')=='1','requires CUDA')
class PrecisionTests(unittest.TestCase):
    def test_model_storage_logits_and_replay(self):
        config=Config();weights=toy_weights(config)
        for dtype in ['float16','bfloat16']:
            backend=Backend('cuda',dtype=dtype)
            for mode in ['eager','buffered','graph']:
                model=Model(backend,config,weights,pages=16,page_size=2,execution=mode)
                oracle=Model(Backend(),config,weights,pages=16,page_size=2)
                for key in ['a','b']:model.create(key);oracle.create(key)
                self.assertTrue(all(w.dtype.itemsize==2 for w in model.weights.values()))
                self.assertTrue(all(cache.k.dtype.itemsize==2 for cache in model.caches))
                for token in range(6):
                    keys=['a','b'] if token%2==0 else ['a']
                    tokens=[token+1+i for i in range(len(keys))]
                    expected=oracle.step(tokens,keys)
                    actual=backend.numpy(model.step(tokens,keys)).astype(np.float32)
                    self.assertTrue(np.isfinite(actual).all())
                    relative_rmse=np.linalg.norm(actual-expected)/np.linalg.norm(expected)
                    self.assertLess(relative_rmse,0.006 if dtype=='float16' else 0.04)
                self.assertTrue(all(p.workspace.buffers['x'].dtype.itemsize==2
                                    for p in model.execution.cache.entries.values()))
                model.release('a');model.release('b');model.close()

    def test_norm_rope_and_split_attention(self):
        from runtime.cache import PagedCache
        from runtime import reference
        rng=np.random.default_rng(5)
        for dtype in ['float16','bfloat16']:
            backend=Backend('cuda',dtype=dtype);cp=backend.xp
            tolerance=0.002 if dtype=='float16' else 0.025
            x=rng.normal(size=(3,70)).astype(np.float32);w=rng.normal(size=70).astype(np.float32)
            dx,dw=backend.asarray(x),backend.asarray(w)
            with self.assertRaises(ValueError):backend.rmsnorm(cp.asarray(x),dw)
            expected=reference.rmsnorm(backend.numpy(dx),backend.numpy(dw))
            np.testing.assert_allclose(backend.numpy(backend.rmsnorm(dx,dw)),expected,
                                       atol=tolerance,rtol=tolerance)
            x=rng.normal(size=(3,4,10)).astype(np.float32);dx=backend.asarray(x)
            positions=np.array([0,7,500],np.int32)
            expected=reference.rope(backend.numpy(dx),positions)
            np.testing.assert_allclose(backend.numpy(backend.rope(dx,cp.asarray(positions))),
                                       expected,atol=tolerance,rtol=tolerance)
            cache=PagedCache(cp,8,4,2,10,dtype=backend.storage_dtype);cache.create('a')
            for _ in range(17):
                cache.append('a',backend.asarray(rng.normal(size=(2,10))),
                                  backend.asarray(rng.normal(size=(2,10))))
            q=backend.asarray(rng.normal(size=(1,4,10)))
            expected=reference.attention(backend.numpy(q[0]),*[backend.numpy(t) for t in cache.dense('a')])
            tables,lengths=cache.metadata(['a']);out=cp.empty_like(q)
            backend.launch_attention(q,cache,tables,lengths,out)
            np.testing.assert_allclose(backend.numpy(out)[0],expected,atol=tolerance,rtol=tolerance)
            partial,stats=backend.split_workspace(q,3)
            self.assertEqual(partial.dtype,np.float32)
            backend.launch_split(q,cache,tables,lengths,out,partial,stats,3)
            np.testing.assert_allclose(backend.numpy(out)[0],expected,atol=tolerance,rtol=tolerance)

    def test_bfloat16_rounding_preserves_special_values(self):
        backend=Backend('cuda',dtype='bfloat16')
        values=np.array([0.,-0.,np.inf,-np.inf,np.nan,1.,1.00390625],np.float32)
        out=backend.numpy(backend.asarray(values))
        self.assertTrue(np.isnan(out[4]));np.testing.assert_array_equal(out[:4],values[:4])
        self.assertTrue(np.signbit(out[1]));self.assertEqual(out[-1],1.)  # tie-to-even
