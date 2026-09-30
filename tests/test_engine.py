import unittest
from runtime.model import Config,Model,toy_weights
from runtime.backend import Backend
from runtime.engine import Engine,Request


def make_engine(**kwargs):
    c=Config(); return Engine(Model(Backend(),c,toy_weights(c),pages=20,page_size=4),**kwargs)


class EngineTests(unittest.TestCase):
    def test_policies_produce_identical_tokens_and_release(self):
        outputs=[]
        for policy in ['serial','chunked','decode_first']:
            e=make_engine(policy=policy,token_budget=3,prefill_chunk=2)
            for i,length in enumerate([1,7,3]): e.submit(Request(str(i),list(range(length)),5))
            result=e.run(); outputs.append([r.generated for r in result])
            self.assertTrue(all(r.status=='finished' for r in result))
            self.assertFalse(e.reserved)
            for cache in e.model.caches: cache.check(); self.assertEqual(len(cache.free),20)
        self.assertEqual(outputs[0],outputs[1]); self.assertEqual(outputs[0],outputs[2])

    def test_cancellation_waiting_active_zero_output_and_budget(self):
        e=make_engine(token_budget=1)
        e.submit(Request('a',[1,2,3],3)); e.submit(Request('b',[1],3))
        e.cancel('b'); e.tick(); e.cancel('a')
        e.submit(Request('c',[1],0)); e.run()
        self.assertEqual(e.requests['a'].status,'cancelled')
        self.assertEqual(e.requests['c'].generated,[])
        self.assertTrue(all(t['processed_tokens']<=1 for t in e.trace))
        for c in e.model.caches: c.check()

    def test_reject_oversized_and_fairness(self):
        e=make_engine(token_budget=1,prefill_chunk=1)
        with self.assertRaises(ValueError): e.submit(Request('bad',[1]*100,1))
        for i in range(3): e.submit(Request(str(i),[1]*4,2))
        for _ in range(3): e.tick()
        self.assertTrue(all(r.cursor>0 for r in e.active))
        e.run()
