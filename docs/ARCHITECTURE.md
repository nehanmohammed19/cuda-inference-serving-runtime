# Architecture and engineering notes

## Execution path

`Request -> Engine.tick -> Model.step -> Backend -> CUDA / library GEMM`

The engine chooses one token per ready sequence per microstep. Rows from different
requests share dense GEMMs. A chunk is a scheduling allowance: this implementation
still consumes prefill tokens sequentially. It does not execute an entire prompt
through a parallel FlashAttention prefill kernel. This is an important remaining
performance gap.

For each transformer layer, the model computes Q/K/V, applies RoPE to Q/K,
appends K/V, attends over each row's logical history, applies the output projection,
and evaluates the normalized SwiGLU feed-forward block. Attention at step t sees
positions 0 through t, enforcing causality without a materialized mask matrix.

## Memory ownership and admission

The physical layout is `[page, token_in_page, kv_head, head_dimension]` in FP32.
Every sequence owns a logical block table and current token count. Each layer has
its own pool. Append maps a logical token to a page and offset. GPU metadata is
rebuilt per microstep; eliminating these host allocations/transfers is future work.

Reference counts track logical owners, not GPU completion. The executor separately
synchronizes before releasing pages. Fork shares existing pages. Append to a
shared partial page copies the page before writing; appending at a page boundary
allocates a fresh page. Allocation failure leaves the sequence length unchanged.

Model.step preflights every layer before appending to any layer. Admission reserves
enough pages for the full declared prompt/output limit. This conservative approach
avoids multiple active sequences exhausting the pool while waiting for each other
to finish, at the cost of admitting fewer requests. A larger serving engine would
need eviction/recomputation or another explicit progress policy to overcommit safely.

The fork operation is a building block for shared prefixes, not a prefix lookup
service. It is not automatically used by requests with equal prompt tokens.

## Online softmax and split-KV

The initial CUDA kernel launches one block per request/query head, with 256
threads. Threads cooperate to reduce Q dot K over head coordinates. Each thread
owns one output coordinate and maintains a running weighted value accumulator.

For each new score s, update running maximum m and denominator l:

```
m_new = max(m, s)
alpha = exp(m - m_new)
beta = exp(s - m_new)
l_new = alpha*l + beta
acc_new = alpha*acc + beta*v
```

This avoids storing the full score vector. The initial kernel still synchronizes
the block repeatedly for every context token. That is a concrete bottleneck,
not a modern optimized FlashAttention implementation.

Split-KV assigns context partitions to separate blocks. Each partition emits its
maximum m_i, denominator l_i and normalized output o_i. A second kernel computes:

```
m = max_i(m_i)
w_i = exp(m_i - m) * l_i
o = sum_i(w_i * o_i) / sum_i(w_i)
```

Empty partitions have zero weight and are explicitly handled. Tests cover
partitions outnumbering tokens, scattered pages, odd tails and grouped heads.

The model's initial dispatcher selects a split per 128 context tokens, capped at
16. This is a transparent heuristic, not an autotuned optimum. Per-request split
counts, register pressure and merge overhead are good next experiments.

## Request scheduling

`serial` admits one request at a time. `chunked` rotates request order across
ticks. `decode_first` uses the same rotation but prioritizes already-decoding
requests. Each tick processes at most the token budget, advances prefills by at
most the chunk allowance and emits at most one decode token per request.

The final prompt token predicts the first output token. Subsequent microsteps
consume the previous output token. Consequently the final emitted token does not
need a KV entry. Cancellation removes queued requests or synchronously frees an
active request. Completed requests retain their output and timing records.

TTFT includes queueing from the declared arrival time. Inter-token latency is
measured between successive emitted tokens. Tiny burst workloads cannot establish
production p95 latency or fairness under sustained arrivals.

## CUDA Graph boundaries

Attention graph capture uses fixed pointers, fixed block-table width and fixed
launch geometry. Buffer contents may change only while respecting stream ordering
and capacity. The caller retains buffers and graph lifetime. Tests capture and
replay an actual CUDA graph.

The model is not graph-captured: it currently allocates intermediates, copies
metadata and performs host orchestration. Whole-model capture requires stable
workspace buffers, batch buckets, inactive-row masking and safe metadata updates.
Likewise the present executor does not overlap CPU scheduling with GPU execution.

## Profiling questions worth answering next

1. Nsight Systems: how much wall time is Python/metadata work versus GPU execution?
2. Nsight Compute: does split-KV improve occupancy or primarily expose more CTAs?
3. How much bandwidth is lost by repeated shared-memory reduction barriers?
4. Can warp-shuffle reductions and vectorized loads improve the baseline?
5. At which context/batch shapes does merge-launch overhead outweigh partitioning?
6. What is the TTFT/ITL tradeoff of parallel prefill and decode-first scheduling?
7. How does a fixed workspace change launch/allocation overhead and graph viability?

Only report answers after collecting traces. No Nsight or Compute Sanitizer result
is claimed by this repository. Sanitizer runs should precede production use.
