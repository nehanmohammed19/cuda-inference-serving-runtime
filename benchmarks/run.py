"""Actual device and wall-clock benchmarks; no scheduler simulation."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import platform
import subprocess
import time
import numpy as np
from runtime.backend import Backend
from runtime.cache import PagedCache
from runtime.model import Config,Model,toy_weights
from runtime.engine import Engine,Request


def summary(values):
    return {'median':float(np.median(values)),'p95':float(np.percentile(values,95)),
            'min':float(min(values)),'max':float(max(values)),'samples':values}


def gpu_time(cp,fn,repeats):
    for _ in range(3): fn()
    cp.cuda.get_current_stream().synchronize()
    samples=[]
    for _ in range(repeats):
        begin,end=cp.cuda.Event(),cp.cuda.Event()
        begin.record(); fn(); end.record(); end.synchronize()
        samples.append(float(cp.cuda.get_elapsed_time(begin,end)))
    return summary(samples)


def kernel_benchmark(b,repeats):
    cp=b.xp; rng=np.random.default_rng(19); results=[]
    for length in [16,64,256,1024]:
        batch,heads,kvheads,dim=4,8,2,64
        c=PagedCache(cp,batch*((length+15)//16),16,kvheads,dim)
        for row in range(batch): c.create(str(row))
        for t in range(length):
            for row in range(batch):
                c.append(str(row),cp.asarray(rng.normal(size=(kvheads,dim)),dtype=cp.float32),
                         cp.asarray(rng.normal(size=(kvheads,dim)),dtype=cp.float32))
        keys=list(c.tables); tables,lengths=c.metadata(keys)
        q=cp.asarray(rng.normal(size=(batch,heads,dim)),dtype=cp.float32); out=cp.empty_like(q)
        def custom(): b.launch_attention(q,c,tables,lengths,out)
        graph=b.capture_attention(q,c,tables,lengths,out)
        # Same FP32 operands, actual device operations. This high-level dense
        # oracle includes gathers and temporary arrays; it is NOT FlashAttention.
        def dense():
            answers=[]
            for row,key in enumerate(keys):
                k,v=c.dense(key); idx=cp.arange(heads)//(heads//kvheads)
                logits=cp.einsum('hd,thd->ht',q[row],k[:,idx])/np.sqrt(dim)
                prob=cp.exp(logits-logits.max(axis=-1,keepdims=True))
                prob/=prob.sum(axis=-1,keepdims=True)
                answers.append(cp.einsum('ht,thd->hd',prob,v[:,idx]))
            return cp.stack(answers)
        expected=b.numpy(dense()); custom(); b.sync()
        error=float(np.max(np.abs(b.numpy(out)-expected)))
        np.testing.assert_allclose(b.numpy(out),expected,atol=3e-6,rtol=3e-5)
        for name,fn in [('paged_cuda',custom),('paged_cuda_graph',graph.launch),('cupy_dense_oracle',dense)]:
            results.append({'implementation':name,'context':length,'batch':batch,'query_heads':heads,
                'kv_heads':kvheads,'head_dim':dim,'dtype':'float32','max_abs_error':error,
                'latency_ms':gpu_time(cp,fn,repeats)})
    return results


def serving_benchmark(device,trials):
    b=Backend(device); c=Config(); w=toy_weights(c); results=[]
    # Warm GEMM, allocator and custom kernel paths before starting request clocks.
    warm=Model(b,c,w,pages=64); warm.create('warm'); warm.step([1],['warm']); warm.release('warm')
    for policy in ['serial','chunked','decode_first']:
        for trial in range(trials):
            model=Model(b,c,w,pages=64,page_size=4)
            e=Engine(model,policy=policy,token_budget=8,prefill_chunk=4)
            for i,n in enumerate([1,4,8,16]):
                e.submit(Request(str(i),[j%c.vocab_size for j in range(n)],4))
            requests=e.run(); elapsed=time.perf_counter()-e.start
            ttft=[(r.first_token_at-r.submitted_at)*1000 for r in requests]
            itl=[(t1-t0)*1000 for r in requests for t0,t1 in zip(r.token_times,r.token_times[1:])]
            results.append({'device':device,'policy':policy,'trial':trial,'elapsed_s':elapsed,
                'output_tokens_per_s':sum(len(r.generated) for r in requests)/elapsed,
                'ttft_ms':summary(ttft),'itl_ms':summary(itl),
                'peak_pages_per_layer':max(cache.peak_pages for cache in model.caches),
                'generated':[r.generated for r in requests],'trace':e.trace})
    # Every scheduler is required to preserve greedy generation on the workload.
    assert all(r['generated']==results[0]['generated'] for r in results)
    return {'model_config':asdict(c),'weights':'random-toy-seed-42',
            'prompt_lengths':[1,4,8,16],'outputs_per_request':4,'arrival':'all at time zero',
            'token_budget':8,'prefill_chunk':4,'trials':results}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/measurements.json'))
    parser.add_argument('--repeats',type=int,default=15)
    parser.add_argument('--trials',type=int,default=3)
    parser.add_argument('--cpu-only',action='store_true')
    args=parser.parse_args()
    if min(args.repeats,args.trials)<=0: parser.error('repeat counts must be positive')
    data={'measured_at_utc':datetime.now(timezone.utc).isoformat(),'platform':platform.platform(),
          'python':platform.python_version(),'numpy':np.__version__,
          'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'history_note':'commit dates are reconstructed; measurement timestamp is actual',
          'cpu_serving':serving_benchmark('cpu',args.trials)}
    if not args.cpu_only:
        b=Backend('cuda'); cp=b.xp
        props=cp.cuda.runtime.getDeviceProperties(0)
        data['gpu']={'name':props['name'].decode(),'compute_capability':f"{props['major']}.{props['minor']}",
                     'memory_bytes':props['totalGlobalMem'],'cupy':cp.__version__,
                     'driver_version':cp.cuda.runtime.driverGetVersion(),
                     'runtime_version':cp.cuda.runtime.runtimeGetVersion()}
        data['kernels']=kernel_benchmark(b,args.repeats)
        data['cuda_serving']=serving_benchmark('cuda',args.trials)
    args.output.parent.mkdir(exist_ok=True,parents=True)
    args.output.write_text(json.dumps(data,indent=2))
    print(f'Wrote {args.output}')


if __name__=='__main__': main()
