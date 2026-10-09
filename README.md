# BlockForge inference runtime

A working single-device Llama inference system with custom CUDA attention, a
paged KV allocator, continuous batching and reproducible experiments. Built to
study how memory ownership, launch overhead and request scheduling interact.

**Status:** research/learning implementation. Python owns orchestration; C++ CUDA
owns the custom kernels. This is not a native C++ server or a replacement for
TensorRT-LLM. Dense linear layers use NumPy/CuPy library GEMM.

## What works

- Incremental Llama forward inference: RMSNorm, RoPE, grouped-query attention,
  residual connections, SwiGLU and greedy generation.
- Local Hugging Face Llama safetensors loading, including FP16/BF16 storage
  converted to FP32 execution. Unsupported architecture variants fail explicitly.
- Physical KV pages with host block tables, reference counting, fork and
  copy-on-write. Release synchronizes the device before pages become reusable.
- Serial, rotating chunked and decode-first continuous-batching policies, bounded
  token budgets, arrival times, cancellation and memory admission.
- CUDA online-softmax paged attention and a split-KV variant with stable merging.
- Fixed-buffer CUDA Graph replay for the attention operator.
- CPU reference tests, GPU comparisons, a real pretrained checkpoint smoke run,
  raw benchmark samples and three generated plots.

## Results you can reproduce

Tests run on an RTX 4060 Laptop GPU (8 GB): **16 tests passed**, including CUDA
execution, graph replay, scattered pages, GQA, copy-on-write, cancellation and
full-sequence versus incremental model agreement. GPU validation is opt-in and
fails rather than silently skipping initialization errors when requested.

At context length 1,024, the measured split-KV kernel took **0.4710 ms**, versus
**1.1121 ms** for the initial single-partition kernel, about **2.36x** faster.
These are FP32 microbenchmark medians at batch 4, 8 query heads, 2 KV heads and
head dimension 64. This compares two kernels here; it is not a speedup over
FlashAttention, cuDNN or TensorRT-LLM.

![Attention measurements](results/attention.png)

The CPU was faster than CUDA on the tiny random model. That result is retained;
it is a useful warning about extrapolating GPU kernel timings to complete systems.

[Results and limitations](results/RESULTS.md) ·
[Raw samples and environment](results/measurements.json) ·
[Test output](results/tests.txt) ·
[Pretrained smoke output](results/pretrained-smoke.json)

The pretrained smoke test loaded **HuggingFaceTB/SmolLM2-135M** and generated 16
tokens on CUDA. It verifies checkpoint loading and execution, not language quality
or agreement with the Transformers implementation. Checkpoint hashes are saved in
`results/checkpoint-manifest.json`; weights are excluded from Git.

## Setup

Python 3.12 was used for the saved runs. From this directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For CPU-only use, only NumPy is needed for toy generation. Safetensors and
tokenizers support checkpoint use/tests; matplotlib generates plots.

CUDA needs a compatible NVIDIA driver and NVRTC/runtime/cuBLAS libraries.
No `nvcc` build step is required: CuPy runtime-compiles `kernels/ops.cu`.
This Windows machine used the versions in `requirements-tested.txt`, with CUDA
12.4 NVRTC/runtime/cuBLAS pip wheels installed alongside CuPy. The backend locates
their Windows DLL directories. On another machine, use toolkit packages compatible
with its driver rather than assuming those exact versions are appropriate.
The recorded CuPy runtime/driver reports are in the results JSON.

```powershell
python -m runtime --device cpu --tokens 1,2,3 --max-new-tokens 8
python -m runtime --device cuda --tokens 1,2,3 --max-new-tokens 8
```

Toy weights are random. Token IDs from this mode are not meaningful text.

For a real model, either supply a compatible local Llama checkpoint or explicitly
download the public example (~270 MB; see its model card and license):

```powershell
python scripts/download_example.py
python -m runtime --device cuda --checkpoint checkpoints/SmolLM2-135M --prompt "The capital of France is" --max-new-tokens 16 --pages 16
```

Standard Llama configurations with even head dimensions <=256 are supported.
Scaled RoPE, custom head dimensions, projection biases, quantized checkpoints and
non-SiLU activations are rejected. The CLI supports a single EOS token ID.
FP32 model weights and all-layer cache pools must fit device memory.

## Multiple requests

Create a JSON file whose prompts are token-ID lists:

```json
[
  {"key":"short", "prompt":[1,2,3], "max_new_tokens":8},
  {"key":"later", "prompt":[4,5,6,7], "max_new_tokens":8, "arrival_s":0.05}
]
```

```powershell
python -m runtime --device cuda --requests requests.json --policy decode_first --token-budget 2 --prefill-chunk 4
```

The Python `Engine` API also exposes `submit`, `tick` and `cancel`. It is a
single-threaded executor, with no HTTP service or concurrent producer API.

## Tests, benchmarks and graphs

```powershell
python -m unittest discover -s tests -v
$env:RUN_CUDA_TESTS="1"
python scripts/validate.py
python -m benchmarks.run
python -m benchmarks.plot
python scripts/summarize.py
```

Without a GPU: `python -m benchmarks.run --cpu-only`. Unset `RUN_CUDA_TESTS` for
the CPU suite. Linux uses `RUN_CUDA_TESTS=1 python scripts/validate.py`.

Microbenchmarks use CUDA events, warmups and 15 samples. Serving experiments use
wall time and three trials of an explicitly labeled tiny random model. Kernel
compilation is warmed before timing. Allocation and metadata preparation are
excluded from microbenchmarks and included in request measurements.

`git_revision` records the base revision at measurement start. Source SHA256 hashes
identify the exact measured files, including changes not yet committed then.
Measurement timestamps are actual UTC times (October 29 evening in Toronto).

## Architecture and next experiments

Read [the architecture notes](docs/ARCHITECTURE.md) for ownership invariants,
online-softmax merging, scheduling and profiler questions. The initial
[plan](PLAN.md) preceded implementation. The main remaining work is whole-model
graph capture, vectorized KV writes, true batched prefill, reduced-precision
Tensor Core paths and profiling against competitive vendor kernels. Multi-GPU,
quantization and speculative decoding are not implemented.

## History provenance

This repository was implemented with AI assistance in one session on **October
29, 2026**. At the user's request, its 12 commits use a reconstructed presentation
timeline spanning October 18–29. Those dates do **not** establish days worked or
manual authorship. Commit messages label this reconstruction; tests, checkpoint
smoke output and benchmarks retain actual execution timestamps. Nothing has been
pushed to a remote.

## References

- [PagedAttention](https://arxiv.org/abs/2309.06180): page-based KV memory management.
- [FlashInfer](https://arxiv.org/abs/2501.01005): attention execution and scheduling.
- [FlashAttention-3](https://arxiv.org/abs/2407.08608): advanced asynchronous pipelines.

This code is inspired by those system concepts; it does not reproduce their full
algorithms, kernels or published performance.
