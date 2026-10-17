"""Reusable whole-model programs with runtime argument rebinding.

A program signature fixes batch size and model identity, not sequence identity.
Token IDs, positions, physical write slots and block tables are mutable inputs.
The same captured graph therefore serves changing requests without rebuilding.
"""
from pathlib import Path
import time
import numpy as np
from .device import DeviceRuntime,ProgramCache,Workspace,StagingBuffer


class KernelLibrary:
    def __init__(self,backend):
        self.backend=backend; cp=backend.xp
        source=(Path(__file__).parents[1]/'kernels/execution.cu').read_text()
        self.module=cp.RawModule(code=source,options=('--std=c++11',))
        self.functions={name:self.module.get_function(name) for name in [
            'embedding','swiglu','residual_norm','rope_scatter','add_residual','warp_attention']}

    def __getitem__(self,name): return self.functions[name]


class ModelProgram:
    def __init__(self,runner,batch):
        started=time.perf_counter()
        self.runner=runner;self.model=runner.model;self.device=runner.device
        self.cp=self.model.xp;self.batch=batch;self.last_event=None
        self.graph=None;self.replays=0;self.closed=False
        c=self.model.config;self.width=(c.max_position_embeddings+runner.page_size-1)//runner.page_size
        self.workspace=Workspace(self.cp,runner.workspace_budget_bytes)
        self.input={name:StagingBuffer(self.cp,shape,np.int32) for name,shape in [
            ('tokens',(batch,)),('positions',(batch,)),('lengths',(batch,)),
            ('tables',(c.num_hidden_layers,batch,self.width)),('slots',(c.num_hidden_layers,batch))]}
        self.arrays={}
        for name,shape in [('x',(batch,c.hidden_size)),('norm',(batch,c.hidden_size)),
            ('q',(batch,c.num_attention_heads,c.dim)),('k',(batch,c.num_key_value_heads,c.dim)),
            ('v',(batch,c.num_key_value_heads,c.dim)),('attn',(batch,c.hidden_size)),
            ('proj',(batch,c.hidden_size)),('sum',(batch,c.hidden_size)),
            ('gate',(batch,c.intermediate_size)),('up',(batch,c.intermediate_size)),
            ('activated',(batch,c.intermediate_size)),('down',(batch,c.hidden_size)),
            ('logits',(batch,c.vocab_size)),('stats',(batch,c.num_attention_heads,1,2))]:
            self.arrays[name]=self.workspace.allocate(name,shape)
        self.workspace.seal()
        self.model.backend.sync()
        with self.device.compute.stream:
            for name,staging in self.input.items():
                staging.device.fill(1 if name=='lengths' else 0)
            # Warmup executes real writes, so preserve the location used by dummy
            # arguments. Otherwise compiling a new batch could corrupt live KV.
            saved=[(cache.k[0,0].copy(),cache.v[0,0].copy()) for cache in self.model.caches]
            self._ops();self._ops()
            self.device.compute.finish()
            if runner.mode=='graph':
                self.device.compute.stream.begin_capture()
                self._ops()
                self.graph=self.device.compute.stream.end_capture()
            for cache,(k,v) in zip(self.model.caches,saved):
                cache.k[0,0]=k;cache.v[0,0]=v
        self.device.compute.finish()
        self.compile_ms=(time.perf_counter()-started)*1000

    def _ops(self):
        m=self.model;c=m.config;cp=self.cp;b=m.backend;kernels=self.runner.kernels
        a=self.arrays; inputs={k:v.device for k,v in self.input.items()}
        B=self.batch;H=c.hidden_size;D=c.dim;I=c.intermediate_size
        i32=np.int32;f32=np.float32
        kernels['embedding'](((H+255)//256,B),(256,),
            (inputs['tokens'],m.weights['model.embed_tokens.weight'],a['x'],i32(H)))
        for layer,cache in enumerate(m.caches):
            prefix=f'model.layers.{layer}.'
            def linear(source,name,dest):
                self.runner.blas.linear(source,m.weights[prefix+name+'.weight'],dest)
            b.rms_kernel((B,),(256,),(a['x'],m.weights[prefix+'input_layernorm.weight'],
                                     a['norm'],i32(H),f32(c.rms_norm_eps)))
            linear(a['norm'],'self_attn.q_proj',a['q'].reshape(B,-1))
            linear(a['norm'],'self_attn.k_proj',a['k'].reshape(B,-1))
            linear(a['norm'],'self_attn.v_proj',a['v'].reshape(B,-1))
            b.rope_kernel(((c.num_attention_heads*D//2+255)//256,B),(256,),
                (a['q'],inputs['positions'],a['q'],i32(c.num_attention_heads),i32(D),f32(c.rope_theta)))
            kernels['rope_scatter'](((c.num_key_value_heads*D+255)//256,B),(256,),
                (a['k'],a['v'],inputs['positions'],inputs['slots'][layer],cache.k,cache.v,
                 i32(c.num_key_value_heads),i32(D),f32(c.rope_theta)))
            kernels['warp_attention'](((c.num_attention_heads+3)//4,B,1),(32,4),
                (a['q'],cache.k,cache.v,inputs['tables'][layer],inputs['lengths'],
                 a['attn'],a['stats'],i32(self.width),i32(cache.page_size),
                 i32(c.num_attention_heads),i32(c.num_key_value_heads),i32(D),i32(1)))
            linear(a['attn'],'self_attn.o_proj',a['proj'])
            kernels['residual_norm']((B,),(256,),
                (a['proj'],a['x'],m.weights[prefix+'post_attention_layernorm.weight'],a['sum'],
                 a['norm'],i32(H),f32(c.rms_norm_eps)))
            linear(a['norm'],'mlp.gate_proj',a['gate'])
            linear(a['norm'],'mlp.up_proj',a['up'])
            kernels['swiglu'](((B*I+255)//256,),(256,),
                (a['gate'],a['up'],a['activated'],i32(B*I)))
            linear(a['activated'],'mlp.down_proj',a['down'])
            kernels['add_residual'](((B*H+255)//256,),(256,),
                (a['down'],a['sum'],a['x'],i32(B*H)))
        b.rms_kernel((B,),(256,),(a['x'],m.weights['model.norm.weight'],a['norm'],i32(H),f32(c.rms_norm_eps)))
        self.runner.blas.linear(a['norm'],m.weights['lm_head.weight'],a['logits'])

    def run(self,tokens,keys):
        if self.closed: raise RuntimeError('program was closed')
        if self.last_event is not None: self.last_event.synchronize()
        # Make mutations on the same compute queue that last used the KV pool.
        positions=np.asarray([self.model.caches[0].lengths[k] for k in keys],np.int32)
        tables=np.full(self.input['tables'].host.shape,-1,np.int32)
        with self.device.compute.stream:
            slots=np.stack([cache.prepare_append(keys) for cache in self.model.caches])
        for layer,cache in enumerate(self.model.caches):
            for row,key in enumerate(keys):
                table=cache.tables[key];tables[layer,row,:len(table)]=table
        arguments={'tokens':np.asarray(tokens,np.int32),'positions':positions,'lengths':positions+1,
                   'tables':tables,'slots':slots}
        for name,value in arguments.items():
            ready=self.input[name].upload(value,self.device.transfer)
            self.device.compute.wait(ready)
        with self.device.compute.stream:
            if self.graph is None: self._ops()
            else: self.graph.launch(stream=self.device.compute.stream)
            self.last_event=self.device.compute.record()
        self.device.compute.submissions+=1;self.replays+=1
        for cache in self.model.caches:
            for key in keys: cache.lengths[key]+=1
        # Publish completion to the caller's stream before it consumes logits.
        self.cp.cuda.get_current_stream().wait_event(self.last_event)
        return self.arrays['logits']

    def close(self):
        if self.last_event is not None: self.last_event.synchronize()
        self.device.transfer.finish()
        self.closed=True;self.graph=None


class ExecutionRuntime:
    def __init__(self,model,mode='graph',capacity=4,workspace_budget_bytes=None):
        if mode not in ['graph','buffered']: raise ValueError('unknown execution mode')
        self.model=model;self.mode=mode;self.device=DeviceRuntime(model.backend)
        from .blas import Blas
        self.blas=Blas(self.device.compute.stream)
        self.kernels=KernelLibrary(model.backend);self.cache=ProgramCache(capacity)
        self.workspace_budget_bytes=workspace_budget_bytes
        self.page_size=model.caches[0].page_size

    def step(self,tokens,keys):
        signature=(len(keys),self.mode)
        program=self.cache.get_or_create(signature,lambda:ModelProgram(self,len(keys)))
        return program.run(tokens,keys)

    def report(self):
        return {**self.cache.report(),'mode':self.mode,
                'compute_submissions':self.device.compute.submissions,
                'transfer_submissions':self.device.transfer.submissions,
                'kernel_resources':{name:dict(fn.attributes) for name,fn in self.kernels.functions.items()},
                'kv_memory':[cache.report() for cache in self.model.caches],
                'programs':[{'batch':p.batch,'replays':p.replays,'compile_ms':p.compile_ms,
                             'staging_device_bytes':sum(s.device.nbytes for s in p.input.values()),
                             'pinned_host_bytes':sum(s.host.nbytes for s in p.input.values()),
                             'workspace':p.workspace.report()} for p in self.cache.entries.values()]}

    def close(self):
        self.cache.close();self.device.close();self.blas.close()
