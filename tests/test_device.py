import unittest
import numpy as np
from runtime.device import Workspace,ProgramCache,RetirementQueue


class FakeEvent:
    def __init__(self, done=False): self.done=done
    def query(self): return self.done
    def synchronize(self): self.done=True


class DeviceTests(unittest.TestCase):
    def test_workspace_signatures_budget_and_sealing(self):
        w=Workspace(np,64)
        a=w.allocate('a',(4,),np.float32)
        self.assertIs(w.allocate('a',(4,),np.float32),a)
        with self.assertRaises(ValueError): w.allocate('a',(2,),np.float32)
        with self.assertRaises(MemoryError): w.allocate('big',(100,),np.float32)
        w.seal()
        with self.assertRaises(RuntimeError): w.allocate('new',(1,),np.float32)
        self.assertEqual(w.report()['bytes'],16)

    def test_program_lru_factory_failure_and_release(self):
        class Program:
            def __init__(self): self.closed=False; self.workspace=Workspace(np)
            def close(self): self.closed=True
        c=ProgramCache(2); a=c.get_or_create('a',Program); c.get_or_create('b',Program)
        self.assertIs(a,c.get_or_create('a',Program))
        def fail(): raise ValueError('compilation failed')
        with self.assertRaises(ValueError): c.get_or_create('bad',fail)
        c.get_or_create('c',Program)
        self.assertNotIn('b',c.entries); self.assertFalse(a.closed)
        self.assertEqual(c.report()['hits'],1); c.close(); self.assertTrue(a.closed)

    def test_retirement_waits_for_completion(self):
        queue=RetirementQueue(); event=FakeEvent(); result=[]
        queue.retire(event,lambda:result.append('released'),keepalive=object())
        queue.collect(); self.assertEqual(result,[])
        event.done=True; queue.collect(); self.assertEqual(result,['released'])
