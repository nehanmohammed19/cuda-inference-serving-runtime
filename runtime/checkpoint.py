"""Load local, standard Hugging Face Llama safetensors without executing repo code."""
from dataclasses import fields
from pathlib import Path
import json
import numpy as np
from .model import Config


def load_checkpoint(directory):
    directory=Path(directory)
    raw=json.loads((directory/'config.json').read_text())
    if raw.get('model_type')!='llama': raise ValueError('only standard Llama is supported')
    if raw.get('rope_scaling') or raw.get('attention_bias') or raw.get('mlp_bias'):
        raise ValueError('scaled RoPE and biased projections are not implemented')
    if raw.get('hidden_act','silu')!='silu': raise ValueError('only SiLU is supported')
    c=Config(**{f.name:raw[f.name] for f in fields(Config) if f.name in raw})
    if raw.get('head_dim',c.dim)!=c.dim: raise ValueError('nonstandard head_dim')
    files=sorted(directory.glob('*.safetensors'))
    if not files: raise FileNotFoundError('no safetensors checkpoint found')
    weights={}
    # Read BF16 explicitly: NumPy does not natively support that safetensors dtype.
    for path in files:
        with path.open('rb') as stream:
            size=int.from_bytes(stream.read(8),'little')
            header=json.loads(stream.read(size))
        for name,meta in header.items():
            if name=='__metadata__': continue
            dtype={'F32':'<f4','F16':'<f2','BF16':'<u2'}.get(meta['dtype'])
            if dtype is None: raise ValueError('unsupported weight dtype '+meta['dtype'])
            offset,end=meta['data_offsets']
            if end-offset!=int(np.prod(meta['shape']))*np.dtype(dtype).itemsize:
                raise ValueError('invalid tensor size')
            mapped=np.memmap(path,dtype=dtype,mode='r',offset=8+size+offset,shape=tuple(meta['shape']))
            tensor=((np.asarray(mapped,dtype=np.uint32)<<16).view(np.float32)
                    if meta['dtype']=='BF16' else np.array(mapped,dtype=np.float32))
            if name in weights: raise ValueError('duplicate tensor '+name)
            weights[name]=tensor
    if raw.get('tie_word_embeddings',False):
        weights['lm_head.weight']=weights['model.embed_tokens.weight']
    validate_weights(c,weights)
    return c,weights,raw


def validate_weights(c,w):
    expected={'model.embed_tokens.weight':(c.vocab_size,c.hidden_size),
              'model.norm.weight':(c.hidden_size,), 'lm_head.weight':(c.vocab_size,c.hidden_size)}
    for layer in range(c.num_hidden_layers):
        p=f'model.layers.{layer}.'
        for norm in ['input_layernorm','post_attention_layernorm']:
            expected[p+norm+'.weight']=(c.hidden_size,)
        for name,out in [('q_proj',c.hidden_size),('k_proj',c.num_key_value_heads*c.dim),
                         ('v_proj',c.num_key_value_heads*c.dim),('o_proj',c.hidden_size)]:
            expected[p+'self_attn.'+name+'.weight']=(out,c.hidden_size)
        for name in ['gate_proj','up_proj']:
            expected[p+'mlp.'+name+'.weight']=(c.intermediate_size,c.hidden_size)
        expected[p+'mlp.down_proj.weight']=(c.hidden_size,c.intermediate_size)
    for name,shape in expected.items():
        if name not in w or w[name].shape!=shape: raise ValueError('missing or invalid '+name)
