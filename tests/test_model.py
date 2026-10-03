import unittest
import numpy as np
from runtime.model import Config, Model, toy_weights
from runtime.backend import Backend


class ModelTests(unittest.TestCase):
    def test_incremental_matches_full_sequence_reference(self):
        from runtime.reference import rmsnorm,rope,dense_causal
        c=Config(); w=toy_weights(c); tokens=[1,9,5,7,3]
        x=w['model.embed_tokens.weight'][tokens]
        for layer in range(c.num_hidden_layers):
            p=f'model.layers.{layer}.'
            def linear(z,name): return z @ w[p+name+'.weight'].T
            h=rmsnorm(x,w[p+'input_layernorm.weight'],c.rms_norm_eps)
            q=rope(linear(h,'self_attn.q_proj').reshape(-1,c.num_attention_heads,c.dim),range(len(tokens)))
            k=rope(linear(h,'self_attn.k_proj').reshape(-1,c.num_key_value_heads,c.dim),range(len(tokens)))
            v=linear(h,'self_attn.v_proj').reshape(-1,c.num_key_value_heads,c.dim)
            x=x+linear(dense_causal(q,k,v).reshape(-1,c.hidden_size),'self_attn.o_proj')
            h=rmsnorm(x,w[p+'post_attention_layernorm.weight'],c.rms_norm_eps)
            gate=linear(h,'mlp.gate_proj')
            x=x+linear(gate/(1+np.exp(-gate))*linear(h,'mlp.up_proj'),'mlp.down_proj')
        expected=rmsnorm(x,w['model.norm.weight'],c.rms_norm_eps)@w['lm_head.weight'].T
        model=Model(Backend(),c,w,page_size=2); model.create('a')
        actual=np.concatenate([model.step([t],['a']) for t in tokens])
        np.testing.assert_allclose(actual,expected,atol=4e-6,rtol=3e-5)

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
