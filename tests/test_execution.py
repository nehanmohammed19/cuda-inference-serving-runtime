import os
import unittest
import numpy as np
from runtime.backend import Backend
from runtime.model import Model,Config,toy_weights
from runtime.engine import Engine,Request


@unittest.skipUnless(os.environ.get('RUN_CUDA_TESTS')=='1','requires CUDA')
class ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.backend=Backend('cuda')

    def test_graph_rebinds_data_and_varying_batch_without_corrupting_kv(self):
        # Tail dimensions and non-four-multiple head counts exercise warp masks.
        c=Config(hidden_size=30,num_attention_heads=3,num_key_value_heads=1,intermediate_size=70)
        weights=toy_weights(c)
        oracle=Model(Backend(),c,weights,pages=32,page_size=2)
        for mode in ['buffered','graph']:
            model=Model(self.backend,c,weights,pages=32,page_size=2,execution=mode)
            # New oracle per execution mode keeps histories independent.
            oracle=Model(Backend(),c,weights,pages=32,page_size=2)
            for key in ['a','b','c']:
                model.create(key);oracle.create(key)
            steps=[(['a','b'],[1,2]),(['a'],[3]),(['a','b','c'],[4,5,6]),
                   (['c','a'],[7,8]),(['b'],[9])]*3
            for keys,tokens in steps:
                expected=oracle.step(tokens,keys)
                actual=self.backend.numpy(model.step(tokens,keys))
                np.testing.assert_allclose(actual,expected,atol=8e-6,rtol=8e-5)
            report=model.execution.report()
            self.assertGreater(report['hits'],0)
            self.assertEqual(report['entries'],3)
            for key in ['a','b','c']:
                for left,right in zip(model.caches,oracle.caches):
                    np.testing.assert_allclose(self.backend.numpy(left.dense(key)[0]),right.dense(key)[0],
                                               atol=5e-6,rtol=5e-5)
                model.release(key)
            for cache in model.caches: cache.check()
            model.close()

    def test_scheduler_graph_matches_eager_and_reuses_pages(self):
        c=Config();weights=toy_weights(c);outputs=[]
        for mode in ['eager','buffered','graph']:
            model=Model(self.backend,c,weights,pages=16,page_size=2,execution=mode)
            engine=Engine(model,token_budget=2,prefill_chunk=3,policy='decode_first')
            for i,n in enumerate([1,7,3]):engine.submit(Request(str(i),list(range(n)),5))
            outputs.append([r.generated for r in engine.run()])
            for cache in model.caches:cache.check();self.assertEqual(len(cache.free),16)
            # Recycle physical pages with new sequence identities, same programs.
            model.create('new');self.assertTrue(np.isfinite(self.backend.numpy(model.step([4],['new']))).all())
            model.release('new');model.close()
        self.assertEqual(outputs[0],outputs[1]);self.assertEqual(outputs[0],outputs[2])

    def test_graph_program_eviction_and_copy_on_write(self):
        c=Config();w=toy_weights(c);model=Model(self.backend,c,w,pages=20,page_size=4,execution='graph')
        model.execution.cache.capacity=1
        model.create('a');model.step([1],['a']);model.backend.sync()
        for cache in model.caches:cache.fork('a','fork')
        out=self.backend.numpy(model.step([2,3],['a','fork']))
        self.assertFalse(np.allclose(out[0],out[1]))
        for cache in model.caches:cache.check();self.assertNotEqual(cache.tables['a'],cache.tables['fork'])
        self.assertEqual(model.execution.cache.evictions,1)
        model.release('a');model.release('fork');model.close()

    def test_device_event_dependency_and_deferred_retirement(self):
        from runtime.device import DeviceRuntime,StagingBuffer
        device=DeviceRuntime(self.backend);cp=self.backend.xp
        staging=StagingBuffer(cp,(17,),np.int32)
        ready=staging.upload(np.arange(17,dtype=np.int32),device.transfer)
        device.compute.wait(ready)
        with device.compute.stream:
            result=staging.device*2
            event=device.compute.record()
        released=[];device.retirement.retire(event,lambda:released.append(True),keepalive=result)
        device.retirement.collect(wait=True)
        np.testing.assert_array_equal(cp.asnumpy(result),np.arange(17)*2)
        self.assertEqual(released,[True]);device.close()
