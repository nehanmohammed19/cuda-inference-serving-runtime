# Execution runtime invariants

## Program signatures and graph ownership

An execution runtime belongs to one model, device and compute stream. Its program
cache keys contain batch size and execution mode. Model architecture, dtype,
weights and cache-pool addresses stay fixed for that runtime. A new request ID
does not require a new program. Token IDs, logical positions, block tables and
physical write slots are input data, not graph topology.

Every program owns stable intermediate buffers and pinned/device input staging.
The sealed workspace refuses new allocations or changed buffer signatures.
The implementation intentionally uses one query-head partition in the graph
path; the separately benchmarked partition sweep does not silently change it.

The GEMM handle is owned by the runtime and bound to the compute queue before
capture. FP16/BF16 GEMM uses FP32 accumulation and fixed output buffers. This avoids the tested high-level
CuPy wrapper's capture restriction without changing cuBLAS handle state inside
the captured command sequence.

## Per-step order

1. Validate tokens, head geometry, sequence identities and context capacity.
2. Preflight physical-page capacity across all layers.
3. Find/create a compatible program. Warmup snapshots and restores the dummy KV
   slot it uses, so compiling another batch cannot corrupt a live sequence.
4. Reserve physical KV write slots, including copy-on-write when a partial page
   has multiple owners. Copy-on-write work uses the compute stream.
5. Populate pinned inputs and enqueue device transfers. Pinned buffers cannot be
   overwritten until their previous transfer event completes.
6. Order compute after those transfer events, then replay the model graph (or
   submit the same stable-buffer operator sequence for the buffered baseline).
7. Record completion, advance host sequence lengths and publish the completion
   dependency to the caller's CUDA stream.
8. Consume logits/greedy output before reusing borrowed program output storage.

Programs are not concurrently reentrant: their buffers belong to one in-flight
invocation. The next use waits for the last completion event. HTTP threads do
not call the model directly; one device worker owns the scheduler and programs.
There is no claim that the current executor overlaps different requests' IO and
compute. Events establish the ordering required to develop that optimization.

## Lifetime and errors

LRU eviction waits for the evicted program's last-use event before dropping its
graph/workspace. Model close synchronizes and closes program/stream/BLAS resources.
Page release still synchronizes before recycling physical pages. RetirementQueue
is a separate tested primitive for event-based deferred callbacks, not an
unimplemented asynchronous reclamation claim for the model allocator.

Validation failures happen before append. CUDA execution failures are fatal to
the serving worker; outstanding futures receive errors. The implementation does
not promise rollback of page mappings after a device/driver failure.

## Operator sequence and fusion

The full program contains embedding lookup, normalization, Q/K/V GEMMs, query
rotation, fused K rotation/KV scatter, warp-local online attention, output GEMM,
fused residual-add/RMSNorm, two MLP projections, fused SwiGLU, down projection,
residual-add, final normalization and vocabulary projection. The same workspace
buffers can be reused across layers because stream order preserves dependencies.

Warp attention uses one warp per query head with register accumulators and
shuffle QK reduction. It has no per-token CTA-wide reduction barrier. It is still
an attention kernel with selectable FP16/BF16 storage and FP32 softmax accumulation,
without Tensor Core MMA or parallel prefill. Split-KV
experiments launch separate context partitions and merge their normalized outputs
using maximum/denominator statistics. Empty partitions and tail heads are tested.

## Observability

Execution reports include program hits/misses/evictions, replay counts, compilation
wall time, named-buffer sizes, pinned staging bytes, KV residency and compiler
kernel resource attributes. Register counts do not establish achieved occupancy.
Host traces and device event durations are exported separately because their
clocks are not calibrated. Optional NVTX ranges support later Nsight analysis.

The HTTP service exposes `/health`, `/metrics`, and token-ID `/generate` with
JSON/SSE responses. It bounds inbox and outstanding scheduler work, propagates
disconnect cancellation for SSE writes, and retires completed request records.
The worker's trace retention is capped. There is no production authentication,
TLS, distributed scheduling or tokenizer/chat compatibility layer.
