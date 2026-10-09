# Implementation plan

Build a single-device Llama inference runtime with inspectable CUDA kernels and a
NumPy oracle. Target the available RTX 4060 (Ada, 8 GB); do not assume Hopper TMA
or WGMMA. This is an educational research system, not a production server.

## Twelve commit-sized stages

1. Record scope, architecture, reproducibility and reconstructed-history policy.
2. Implement and test a reference-counted paged KV allocator.
3. Implement stable reference attention and numerical primitives.
4. Add runtime-compiled CUDA RMSNorm, RoPE and paged online-softmax attention.
5. Add backend abstraction and fixed-shape attention CUDA Graph replay.
6. Implement Llama forward execution, GQA and deterministic toy checkpoints.
7. Load local Hugging Face safetensors checkpoints and tokenizers.
8. Implement continuous batching, bounded prefill chunks and token budgets.
9. Add generation CLI and lifecycle/cancellation checks.
10. Add reproducible kernel and mixed-request benchmarks with raw JSON.
11. Run correctness and performance experiments; plot actual measurements.
12. Document architecture, results, limitations and reproduction commands.

## Validation gates

- CPU unit tests: allocator exhaustion, sharing, copy-on-write, reuse, stable
  softmax, causal/paged/GQA attention and scheduler fairness/lifecycle.
- CUDA: compare kernels against NumPy, including tail dimensions, block boundaries,
  scattered pages and different sequence lengths; check graph replay results.
- Model: batched incremental execution must agree with isolated requests and a
  dense full-sequence reference; a toy model is not evidence of language quality.
- Benchmark: warmups and repeated samples; CUDA events for kernel timing, wall
  time for full requests. Report hardware, software, shapes and precision.
- No GPU result may be inferred from simulated scheduling or CPU measurements.

## Architecture

Python orchestrates scheduling and checkpoint loading. NumPy/CuPy supply dense
linear algebra; GPU matmuls use the installed library implementation. C++ CUDA
source owns normalization, positional rotation and paged attention. Cache metadata
stays on the host; cache values stay on the selected device. Synchronous iteration
boundaries guarantee a page is not recycled while kernels still reference it.

The first attention kernel prioritizes correctness and inspectability. It is not
FlashAttention-3 and does not claim Tensor Core attention or peak performance.
Graph replay initially covers fixed-shape attention only, not the whole model.
CPU/GPU overlap, multi-GPU, speculative decoding and quantization are follow-ons.

## Commit dates

The requested historical dates are a reconstructed presentation timeline, not
evidence of work performed on those dates. Commits and README disclose this. All
measurement timestamps record the actual experiment time.

## Completion record

All twelve stages were completed in the October 29 implementation session.
The host runtime is Python, CUDA kernels are runtime compiled, and dense GEMM is
delegated to NumPy/CuPy. Split-KV and fixed attention graph replay ran on the
RTX 4060. Sixteen tests passed. A SmolLM2-135M checkpoint completed generation.
The README and results explicitly delimit whole-model graphs, native C++ hosting,
Tensor Core attention and advanced serving features that remain future work.
