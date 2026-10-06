import unittest
import numpy as np
from runtime.backend import Backend
from runtime.cache import PagedCache
from runtime import reference


class CudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        if os.environ.get('RUN_CUDA_TESTS') != '1':
            raise unittest.SkipTest('set RUN_CUDA_TESTS=1 to require GPU validation')
        # Fail loudly on initialization errors when GPU validation was requested.
        cls.backend = Backend('cuda')

    def test_model_and_scheduler_match_cpu(self):
        from runtime.model import Config,Model,toy_weights
        from runtime.engine import Engine,Request
        c=Config(); w=toy_weights(c); outputs=[]
        for backend in [Backend(),self.backend]:
            model=Model(backend,c,w,pages=32,page_size=2)
            e=Engine(model,token_budget=3,prefill_chunk=2)
            for i,n in enumerate([1,7,3]): e.submit(Request(str(i),list(range(n)),5))
            outputs.append([r.generated for r in e.run()])
        self.assertEqual(outputs[0],outputs[1])

    def test_norm_rope(self):
        b = self.backend; cp = b.xp
        rng = np.random.default_rng(9)
        for d in [10, 64, 130, 512]:
            x = rng.normal(size=(3, d)).astype('float32')
            w = rng.normal(size=d).astype('float32')
            np.testing.assert_allclose(b.numpy(b.rmsnorm(cp.asarray(x), cp.asarray(w))),
                                       reference.rmsnorm(x, w), atol=2e-6, rtol=2e-5)
        x = rng.normal(size=(3, 4, 10)).astype('float32')
        pos = np.array([0, 17, 500], np.int32)
        np.testing.assert_allclose(b.numpy(b.rope(cp.asarray(x), cp.asarray(pos))),
                                   reference.rope(x, pos), atol=3e-5)

    def test_scattered_ragged_gqa_and_graph(self):
        b = self.backend; cp = b.xp; rng = np.random.default_rng(7)
        for d in [10, 64, 128, 256]:
            c = PagedCache(cp, 20, 4, 2, d)
            for key in ['a', 'b']: c.create(key)
            for t in range(17):
                for key in ['a', 'b'] if t < 7 else ['a']:
                    c.append(key, cp.asarray(rng.normal(size=(2,d)), dtype=cp.float32),
                             cp.asarray(rng.normal(size=(2,d)), dtype=cp.float32))
            q = cp.asarray(rng.normal(size=(2,4,d)), dtype=cp.float32)
            expected = np.stack([reference.attention(b.numpy(q[i]),
                                 *[b.numpy(x) for x in c.dense(key)])
                                 for i,key in enumerate(['a','b'])])
            out = b.attention(q,c,['a','b'])
            np.testing.assert_allclose(b.numpy(out), expected, atol=2e-6, rtol=2e-5)
            tables, lengths = c.metadata(['a','b']); dest = cp.empty_like(q)
            for splits in [1,3,16]:
                partial,stats=b.split_workspace(q,splits)
                b.launch_split(q,c,tables,lengths,dest,partial,stats,splits)
                np.testing.assert_allclose(b.numpy(dest),expected,atol=2e-6,rtol=2e-5)
            graph = b.capture_attention(q,c,tables,lengths,dest)
            graph.launch(); b.sync()
            np.testing.assert_allclose(b.numpy(dest),expected,atol=2e-6,rtol=2e-5)
