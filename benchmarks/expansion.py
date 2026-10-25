"""Compare original eager, stable-buffer and full-graph programs on real hardware."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json
import subprocess
import time
import numpy as np
from runtime.backend import Backend
from runtime.cache import PagedCache
from runtime.checkpoint import load_checkpoint
from runtime.model import Model,Config,toy_weights
from runtime.engine import Engine,Request
from runtime.execution import KernelLibrary
from benchmarks.run import gpu_time,summary


def decode_run(backend,c,weights,mode,length,steps=16,trials=3):
    model=Model(backend,c,weights,pages=128,page_size=16,execution=mode)
    # Program compilation, library startup and weights upload are excluded.
    model.create('warm');model.step([1],['warm']);model.release('warm')
    records=[];reference=None
    for trial in range(trials):
        key=f'trial-{trial}';model.create(key)
        for token in range(length):model.step([token%c.vocab_size],[key])
        backend.sync();tokens=[];samples=[];previous=1
        for _ in range(steps):
            start=time.perf_counter()
            logits=model.step([previous],[key])
            previous=int(backend.numpy(model.xp.argmax(logits,axis=-1))[0]);backend.sync()
            samples.append((time.perf_counter()-start)*1000);tokens.append(previous)
        records.append({'trial':trial,'step_ms':summary(samples),'generated':tokens})
        if reference is None:reference=tokens
        assert tokens==reference
        model.release(key)
    report=model.execution.report() if model.execution else None
    model.close()
    return {'mode':mode,'prompt_length':length,'decode_tokens':steps,'trials':records,'runtime':report}


def scheduler_run(backend,c,w,mode,trials=3):
    model=Model(backend,c,w,pages=64,page_size=4,execution=mode)
    # Warm every possible batch in this bounded benchmark.
    for batch in range(1,5):
        keys=[f'warm-{i}' for i in range(batch)]
        for key in keys:model.create(key)
        model.step([1]*batch,keys)
        for key in keys:model.release(key)
    samples=[]
    for trial in range(trials):
        engine=Engine(model,token_budget=4,prefill_chunk=4,policy='decode_first')
        for i,n in enumerate([1,4,8,16]):engine.submit(Request(f'{trial}-{i}',list(range(n)),8))
        result=engine.run();elapsed=time.perf_counter()-engine.start
        samples.append({'trial':trial,'elapsed_s':elapsed,'output_tokens_per_s':32/elapsed,
            'generated':[r.generated for r in result],
            'ttft_ms':summary([(r.first_token_at-r.submitted_at)*1000 for r in result]),
            'itl_ms':summary([(t1-t0)*1000 for r in result for t0,t1 in zip(r.token_times,r.token_times[1:])])})
    report=model.execution.report() if model.execution else None
    model.close();return {'mode':mode,'trials':samples,'runtime':report}


def attention_grid(backend,repeats):
    cp=backend.xp;library=KernelLibrary(backend);rng=np.random.default_rng(31);rows=[]
    for batch in [1,4]:
        for length in [64,256,1024]:
            h,kv,d=8,2,64
            cache=PagedCache(cp,batch*((length+15)//16),16,kv,d)
            for row in range(batch):cache.create(str(row))
            # Build each sequence using deterministic arrays, outside timing.
            for row in range(batch):
                for t in range(length):
                    cache.append(str(row),cp.asarray(rng.normal(size=(kv,d)),dtype=cp.float32),
                                 cp.asarray(rng.normal(size=(kv,d)),dtype=cp.float32))
            tables,lengths=cache.metadata(list(cache.tables))
            q=cp.asarray(rng.normal(size=(batch,h,d)),dtype=cp.float32);out=cp.empty_like(q)
            baseline=lambda:backend.launch_attention(q,cache,tables,lengths,out)
            baseline();expected=backend.numpy(out).copy()
            candidates=[('cta_baseline',0)]
            for splits in [1,2,4,8]:candidates.append(('warp_shuffle',splits))
            for name,splits in candidates:
                partial,stats=backend.split_workspace(q,max(1,splits))
                def invoke():
                    if splits==0:baseline();return
                    target=out if splits==1 else partial
                    library['warp_attention'](((h+3)//4,batch,splits),(32,4),
                        (q,cache.k,cache.v,tables,lengths,target,stats,np.int32(tables.shape[1]),
                         np.int32(16),np.int32(h),np.int32(kv),np.int32(d),np.int32(splits)))
                    if splits>1:
                        backend.merge_kernel((batch,h),(256,),
                            (partial,stats,out,np.int32(h),np.int32(d),np.int32(splits)))
                invoke();actual=backend.numpy(out)
                np.testing.assert_allclose(actual,expected,atol=4e-6,rtol=4e-5)
                rows.append({'name':name,'splits':splits,'batch':batch,'context':length,
                    'query_heads':h,'kv_heads':kv,'head_dim':d,'dtype':'float32',
                    'latency_ms':gpu_time(cp,invoke,repeats),
                    'max_abs_error':float(np.max(np.abs(actual-expected)))})
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=Path('checkpoints/SmolLM2-135M'))
    parser.add_argument('--trials',type=int,default=3)
    parser.add_argument('--repeats',type=int,default=20)
    args=parser.parse_args();b=Backend('cuda');cp=b.xp;c=Config();w=toy_weights(c)
    output={'measured_at_utc':datetime.now(timezone.utc).isoformat(),
        'git_base_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
             for folder in ['runtime','kernels','benchmarks'] for p in sorted(Path(folder).glob('*'))
             if p.is_file() and p.suffix in ['.py','.cu']},
        'gpu':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),'cupy':cp.__version__,
        'driver_version':cp.cuda.runtime.driverGetVersion(),'dtype':'float32','warmups_excluded':True,
        'toy_decode':[],'scheduler':[],'attention':attention_grid(b,args.repeats)}
    for mode in ['eager','buffered','graph']:
        for length in [8,64]:output['toy_decode'].append(decode_run(b,c,w,mode,length,trials=args.trials))
        output['scheduler'].append(scheduler_run(b,c,w,mode,args.trials))
    for length in [8,64]:
        runs=[r for r in output['toy_decode'] if r['prompt_length']==length]
        assert all(r['trials'][0]['generated']==runs[0]['trials'][0]['generated'] for r in runs)
    assert all(r['trials'][0]['generated']==output['scheduler'][0]['trials'][0]['generated']
               for r in output['scheduler'])
    if args.checkpoint.exists():
        pc,pw,_=load_checkpoint(args.checkpoint)
        output['pretrained_model']='HuggingFaceTB/SmolLM2-135M'
        output['pretrained_decode']=[decode_run(b,pc,pw,mode,16,steps=16,trials=args.trials)
                                    for mode in ['eager','buffered','graph']]
        assert all(r['trials'][0]['generated']==output['pretrained_decode'][0]['trials'][0]['generated']
                   for r in output['pretrained_decode'])
    # A separate instrumented workload avoids contaminating the benchmark samples.
    traced=Model(b,c,w,pages=32,execution='graph')
    traced.create('trace');traced.step([1],['trace'])
    traced.execution.profiler.enabled=True
    for token in range(8):traced.step([token],['trace']);b.sync()
    traced.execution.profiler.export('results/expansion/trace.json')
    traced.release('trace');traced.close()
    path=Path('results/expansion/measurements.json');path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(output,indent=2));print('Wrote',path)


if __name__=='__main__':main()
