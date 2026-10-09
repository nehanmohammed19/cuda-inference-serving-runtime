"""Derive a short results report directly from recorded JSON samples."""
from pathlib import Path
import json
import numpy as np

d=json.loads(Path('results/measurements.json').read_text())
lines=['# Measured results','',f"Measured at: {d['measured_at_utc']}.",
       f"GPU: {d.get('gpu',{}).get('name','not measured')}. Precision: FP32.",'',
       '## Attention microbenchmark','',
       'Batch 4, 8 query heads, 2 KV heads, head dimension 64. Times are CUDA-event',
       'medians over 15 samples after 3 warmups. Page metadata and split workspace',
       'are preallocated; preparation and allocation are excluded. The dense CuPy',
       'oracle includes gather/materialization and is a correctness baseline, not',
       'a competitive FlashAttention/cuDNN baseline.','',
       '| Context | Initial CUDA (ms) | Graph replay (ms) | Split-KV (ms) |',
       '|---|---:|---:|---:|']
for context in [16,64,256,1024]:
    rows={r['implementation']:r for r in d.get('kernels',[]) if r['context']==context}
    if rows:
        numbers=[rows[k]['latency_ms']['median'] for k in ['paged_cuda','paged_cuda_graph','split_kv_cuda']]
        lines.append(f'| {context} | '+' | '.join(f'{v:.4f}' for v in numbers)+' |')
lines+=['','![Attention](attention.png)','','## Full runtime workload','',
        'Random two-layer toy Llama, hidden size 64; four requests arriving at time',
        'zero with prompt lengths 1/4/8/16 and four output tokens each. Three trials.',
        'Token budget 2, prefill chunk 4, maximum 8 active requests. Wall time includes',
        'scheduling, metadata, model execution and synchronization.',
        'These numbers do not establish pretrained-model quality or realistic LLM throughput.','',
        '| Device | Policy | Median output tokens/s | Median trial p95 TTFT (ms) | Median trial p95 ITL (ms) |',
        '|---|---|---:|---:|---:|']
for device in ['cpu','cuda']:
    if device+'_serving' not in d: continue
    for policy in ['serial','chunked','decode_first']:
        rows=[r for r in d[device+'_serving']['trials'] if r['policy']==policy]
        values=[np.median([r['output_tokens_per_s'] for r in rows]),
                np.median([r['ttft_ms']['p95'] for r in rows]),np.median([r['itl_ms']['p95'] for r in rows])]
        lines.append(f'| {device} | {policy} | '+' | '.join(f'{v:.2f}' for v in values)+' |')
lines+=['','![Serving](serving.png)','','![Cache residency](cache.png)','',
        '## Interpretation and limits','',
        '- Split-KV trades an extra launch and workspace for more parallelism over long contexts.',
        '- CPU wins on this tiny model. GPU speedups over CPU are not claimed.',
        '- Policy comparisons require more workloads and trials; execution order and laptop clocks can influence these samples.',
        '- Three burst-workload trials and four requests are exploratory, not statistical evidence of production tail latency.',
        '- Graph replay covers attention only. Whole-model capture and host/device overlap are not implemented.',
        '- Page reservations use worst-case request lengths to prevent allocation deadlock; this can underutilize capacity.',
        '- Prefix fork/copy-on-write is tested at the allocator level, not wired to a global prefix lookup service.',
        '- No Tensor Core attention, FP16/BF16 execution, quantized KV, multi-GPU, or speculative decoding claim.',
        '- Nsight and Compute Sanitizer have not been run on this machine.',
        '- Raw measurements include source hashes, environment and actual timestamps; commit dates are reconstructed.','']
Path('results/RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
