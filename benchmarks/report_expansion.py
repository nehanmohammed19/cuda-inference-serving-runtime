"""Render the expansion report from the raw measurements, without invented data."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def latency(record):
    return float(np.median([t['step_ms']['median'] for t in record['trials']]))


def main():
    root=Path('results/expansion');data=json.loads((root/'measurements.json').read_text())
    colors=['#577590','#43aa8b','#f9c74f'];modes=['eager','buffered','graph']
    plt.rcParams.update({'figure.dpi':150,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,key,title in zip(axes,['toy_decode','pretrained_decode'],['Random tiny Llama','SmolLM2-135M pretrained']):
        rows=data.get(key,[])
        values=[latency(next(r for r in rows if r['mode']==m and r['prompt_length']==(8 if key=='toy_decode' else 16)))
                for m in modes]
        ax.bar(modes,values,color=colors)
        for x,value in enumerate(values):ax.text(x,value,f'{value:.2f}',ha='center',va='bottom')
        ax.set(title=title,ylabel='Wall latency per decode step (ms)')
    fig.suptitle('Whole-model execution · FP32 · median of 3 trial medians · initialization excluded')
    fig.tight_layout();fig.savefig(root/'decode.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,batch in zip(axes,[1,4]):
        for splits in [0,1,2,4,8]:
            rows=[r for r in data['attention'] if r['batch']==batch and r['splits']==splits]
            ax.plot([r['context'] for r in rows],[r['latency_ms']['median'] for r in rows],marker='o',
                    label='CTA baseline' if splits==0 else f'Warp, {splits} partition(s)')
        ax.set(xlabel='Context tokens',ylabel='CUDA-event latency (ms)',title=f'Batch {batch}')
        ax.set_xscale('log',base=2);ax.set_yscale('log');ax.legend(fontsize=8)
    fig.suptitle('Warp-shuffle and split-KV parameter sweep · 8 Q / 2 KV heads · D=64')
    fig.tight_layout();fig.savefig(root/'warp-sweep.png');plt.close(fig)
    load=json.loads((root/'http-load.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,rows,title in [(axes[0],data['scheduler'],'In-process scheduler · 3 trials'),
                           (axes[1],load['runs'],'Loopback HTTP · 12 requests · concurrency 4 · 1 run')]:
        values=[float(np.median([t['output_tokens_per_s'] for t in r['trials']]))
                if 'trials' in r else r['output_tokens_per_s'] for r in rows]
        ax.bar([r['mode'] for r in rows],values,color=colors)
        ax.set(title=title,ylabel='Output tokens / second')
    fig.suptitle('Measured toy-model serving · transport and scheduling overhead included')
    fig.tight_layout();fig.savefig(root/'serving.png');plt.close(fig)
    lines=['# Runtime expansion results','',f"Measured: {data['measured_at_utc']}.",
           f"Device: {data['gpu']}. CuPy {data['cupy']}. Precision FP32.",'',
           '## Whole-model decode','',
           'Batch 1; 16 decode steps per trial; 3 trials. Table values are the median',
           'of trial medians. Wall time includes host metadata preparation, pinned',
           'uploads, GPU execution, greedy selection and synchronization. Weights',
           'upload, compilation/capture, library startup and prompt processing are',
           'excluded. Pretrained model uses a deterministic token-ID prompt of length',
           '16. Timing is conditional on this small model and workload, not all LLMs.','',
           '| Model | Prompt | Eager ms | Buffered ms | Graph ms | Eager / graph |',
           '|---|---:|---:|---:|---:|---:|']
    for key,label in [('toy_decode','random tiny'),('pretrained_decode','SmolLM2-135M')]:
        for length in sorted({r['prompt_length'] for r in data.get(key,[])}):
            rows={r['mode']:r for r in data[key] if r['prompt_length']==length}
            a,b,c=[latency(rows[m]) for m in modes]
            lines.append(f'| {label} | {length} | {a:.3f} | {b:.3f} | {c:.3f} | {a/c:.2f}x |')
    lines+=['','![Decode](decode.png)','','## What changed','',
        '- Buffered execution uses stable intermediates, a dedicated cuBLAS handle, fused RoPE/KV scatter, residual RMSNorm and SwiGLU.',
        '- Graph execution captures the complete forward pass, including all library GEMMs and custom kernels. Runtime token/position/page inputs are rebound without recapture.',
        '- The scheduler uses a bounded batch-signature program cache, so changed request identities reuse programs.',
        '- Program workspaces, pinned staging, KV residency and compiler-reported kernel resources are present in the raw reports.',
        '- All compared execution modes generated identical tokens on the recorded workloads. The pretrained loader has not been compared against the Transformers framework.','',
        '## Attention parameter sweep','',
        'Twenty CUDA-event samples after three warmups per candidate; identical FP32',
        'inputs. Workspace allocation and metadata preparation excluded. Every',
        'candidate is checked against the original CTA implementation before timing.',
        'This is a parameter sweep, not an automatically trained tuning model. The',
        'whole-model graph path currently uses one warp partition per query head.','',
        '![Warp sweep](warp-sweep.png)','','## Serving experiments','',
        'In-process: prompts 1/4/8/16, eight output tokens per request, token budget',
        '4, decode-first scheduling; three trials. Every batch size is warmed first.',
        'HTTP: twelve requests, concurrency four, same prompt family/output count,',
        'one exploratory run per mode on loopback. No external load generator or',
        'network transit is involved. These are random toy-model workloads.','',
        '| Mode | In-process median tokens/s | HTTP tokens/s | HTTP p95 request ms |',
        '|---|---:|---:|---:|']
    for m in modes:
        serving=next(r for r in data['scheduler'] if r['mode']==m)
        http=next(r for r in load['runs'] if r['mode']==m)
        throughput=np.median([r['output_tokens_per_s'] for r in serving['trials']])
        lines.append(f"| {m} | {throughput:.1f} | {http['output_tokens_per_s']:.1f} | {http['http_latency_ms']['p95']:.2f} |")
    lines+=['','![Serving](serving.png)','','## Profiling and interpretation','',
        '`trace.json` contains actual host Chrome-trace intervals and separately',
        'labeled CUDA-event durations. GPU clocks are not aligned to host timestamps;',
        'no synthetic GPU timeline is drawn. Enable `BLOCKFORGE_NVTX=1` on a machine',
        'with NVTX support to add Nsight-visible ranges. Nsight and Compute Sanitizer',
        'were not installed/run here; register counts in the report are compiler',
        'attributes, not measured occupancy or stall counters.','',
        'The buffered-versus-eager comparison changes both fusion and execution',
        'infrastructure, so it cannot attribute gains to one kernel. Graph-versus-',
        'buffered is a closer comparison of dispatch paths but also changes host',
        'wrapper overhead. Laptop clocks, WDDM scheduling, trial order and background',
        'activity can influence results. No vendor-kernel or production-serving',
        'performance superiority is claimed.','',
        'Remaining gaps: parallel prefill, reduced-precision/Tensor Core attention,',
        'automatic prefix lookup, adaptive split dispatch, multi-GPU and a native',
        'C++ host runtime. Transfer/compute queues have explicit event dependencies;',
        'these experiments do not establish cross-request transfer/compute overlap.','',
        'Reproduce: `python -m benchmarks.expansion`, `python -m benchmarks.service_load`,',
        'then `python -m benchmarks.report_expansion`. Raw JSON preserves timestamps,',
        'software details, samples, output tokens and measured-source hashes.','']
    (root/'RESULTS.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
