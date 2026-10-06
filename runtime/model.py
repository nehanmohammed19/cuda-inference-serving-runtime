"""Llama forward inference: RMSNorm, split-half RoPE, GQA, SwiGLU.

Weights follow Hugging Face Llama names and [out,in] linear layouts. FP32 is
intentional for the first numerical baseline; GPU GEMMs are delegated to CuPy.
"""
from dataclasses import dataclass
import numpy as np
from .cache import PagedCache


@dataclass
class Config:
    vocab_size: int = 128
    hidden_size: int = 64
    intermediate_size: int = 128
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.
    max_position_embeddings: int = 2048

    def __post_init__(self):
        if min(self.vocab_size,self.hidden_size,self.intermediate_size,
               self.num_hidden_layers,self.num_attention_heads,self.num_key_value_heads) <= 0:
            raise ValueError('model dimensions must be positive')
        if self.hidden_size % self.num_attention_heads:
            raise ValueError('hidden size must divide query heads')
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError('query heads must divide KV heads')
        if self.dim % 2 or self.dim > 256:
            raise ValueError('supported head dimensions are even and <= 256')

    @property
    def dim(self): return self.hidden_size // self.num_attention_heads


def toy_weights(config, seed=42):
    rng = np.random.default_rng(seed); c = config; weights = {}
    def matrix(name, out, inp):
        weights[name] = (rng.normal(size=(out,inp)) / np.sqrt(inp)).astype(np.float32)
    matrix('model.embed_tokens.weight',c.vocab_size,c.hidden_size)
    matrix('lm_head.weight',c.vocab_size,c.hidden_size)
    weights['model.norm.weight'] = np.ones(c.hidden_size,np.float32)
    for layer in range(c.num_hidden_layers):
        prefix = f'model.layers.{layer}.'
        for norm in ['input_layernorm','post_attention_layernorm']:
            weights[prefix+norm+'.weight'] = np.ones(c.hidden_size,np.float32)
        for proj,out in [('q_proj',c.hidden_size),('k_proj',c.num_key_value_heads*c.dim),
                         ('v_proj',c.num_key_value_heads*c.dim),('o_proj',c.hidden_size)]:
            matrix(prefix+'self_attn.'+proj+'.weight',out,c.hidden_size)
        for proj in ['gate_proj','up_proj']:
            matrix(prefix+'mlp.'+proj+'.weight',c.intermediate_size,c.hidden_size)
        matrix(prefix+'mlp.down_proj.weight',c.hidden_size,c.intermediate_size)
    return weights


class Model:
    def __init__(self, backend, config, weights, pages=256, page_size=16):
        self.backend, self.config = backend, config
        self.xp = backend.xp
        # Validate before copying potentially large weights to the GPU.
        from .checkpoint import validate_weights
        validate_weights(config, weights)
        self.weights = {k:self.xp.asarray(v,dtype=self.xp.float32) for k,v in weights.items()}
        self.caches = [PagedCache(self.xp,pages,page_size,config.num_key_value_heads,config.dim)
                       for _ in range(config.num_hidden_layers)]

    def create(self,key):
        for cache in self.caches: cache.create(key)

    def release(self,key):
        self.backend.sync()
        for cache in self.caches: cache.release(key)

    def step(self, tokens, keys):
        """Consume one token per sequence; return next-token logits for every row."""
        c=self.config; xp=self.xp; b=self.backend
        if not keys or len(keys)!=len(tokens) or len(set(keys))!=len(keys):
            raise ValueError('one unique sequence key per token is required')
        positions = np.array([self.caches[0].lengths[k] for k in keys],np.int32)
        if positions.max() >= c.max_position_embeddings: raise ValueError('context limit exceeded')
        if any(t<0 or t>=c.vocab_size for t in tokens): raise ValueError('token out of range')
        # Preflight every layer before mutating any KV cache.
        for cache in self.caches:
            needed=sum(cache.lengths[k]%cache.page_size==0 or
                       cache.refs[cache.tables[k][-1]]>1 for k in keys)
            if needed>len(cache.free): raise MemoryError('insufficient KV pages for step')
        pos=xp.asarray(positions); x=self.weights['model.embed_tokens.weight'][xp.asarray(tokens)]
        for layer,cache in enumerate(self.caches):
            prefix=f'model.layers.{layer}.'
            def linear(z,name): return z @ self.weights[prefix+name+'.weight'].T
            h=b.rmsnorm(x,self.weights[prefix+'input_layernorm.weight'],c.rms_norm_eps)
            q=b.rope(linear(h,'self_attn.q_proj').reshape(-1,c.num_attention_heads,c.dim),pos,c.rope_theta)
            k=b.rope(linear(h,'self_attn.k_proj').reshape(-1,c.num_key_value_heads,c.dim),pos,c.rope_theta)
            v=linear(h,'self_attn.v_proj').reshape(-1,c.num_key_value_heads,c.dim)
            for row,key in enumerate(keys): cache.append(key,k[row],v[row])
            attended=b.attention(q,cache,keys).reshape(-1,c.hidden_size)
            x=x+linear(attended,'self_attn.o_proj')
            h=b.rmsnorm(x,self.weights[prefix+'post_attention_layernorm.weight'],c.rms_norm_eps)
            gate=linear(h,'mlp.gate_proj')
            sigmoid=1/(1+xp.exp(-xp.clip(gate,-80,80)))
            x=x+linear(gate*sigmoid*linear(h,'mlp.up_proj'),'mlp.down_proj')
        return b.rmsnorm(x,self.weights['model.norm.weight'],c.rms_norm_eps) @ self.weights['lm_head.weight'].T
