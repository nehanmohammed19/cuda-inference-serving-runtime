# What the TT-Metal/TTNN inspection changed

Inspected revision: `c346c73b04b1a3074477389ac007fb0a62af437c`.
No Tenstorrent kernel/runtime source was copied into this runtime. These are design
analogies; a CUDA GPU is not a Tensix device and does not expose its NoC or firmware.

| Inspected source | Responsibility learned | Runtime implementation |
|---|---|---|
| [program_cache.hpp](https://github.com/tenstorrent/tt-metal/blob/c346c73b04b1a3074477389ac007fb0a62af437c/tt_metal/api/tt-metalium/program_cache.hpp) | Cached execution state has ownership distinct from per-invocation arguments | `ProgramCache`, `ModelProgram`, runtime token/position/page rebinding |
| [mesh_command_queue.hpp](https://github.com/tenstorrent/tt-metal/blob/c346c73b04b1a3074477389ac007fb0a62af437c/tt_metal/api/tt-metalium/mesh_command_queue.hpp) | Ordered dispatch and transfer lifetime are explicit | Compute/transfer CUDA streams, pinned staging, event dependencies |
| [mesh_event.hpp](https://github.com/tenstorrent/tt-metal/blob/c346c73b04b1a3074477389ac007fb0a62af437c/tt_metal/api/tt-metalium/mesh_event.hpp) | Completion is represented by owned runtime objects | CUDA completion events and retirement queue |
| [allocator report](https://github.com/tenstorrent/tt-metal/blob/c346c73b04b1a3074477389ac007fb0a62af437c/tech_reports/memory/allocator.md) | Allocation bookkeeping and memory movement are separate concerns | KV page ownership, sealed execution workspaces, resident/pool/payload reports |
| [trace and command queues report](https://github.com/tenstorrent/tt-metal/blob/c346c73b04b1a3074477389ac007fb0a62af437c/tech_reports/AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md) | Stable replay can address gaps caused by host dispatch; queue dependencies enable IO coordination | Full-forward CUDA Graph replay and separately ordered transfer inputs |
| [TTNN matmul API](https://github.com/tenstorrent/tt-metal/blob/c346c73b04b1a3074477389ac007fb0a62af437c/ttnn/cpp/ttnn/operations/matmul/matmul.hpp) | Operator execution includes configuration and output allocation choices | Explicit GEMM output buffers and fixed program signatures |

TT-Metal includes hardware support, firmware, distributed execution, tooling,
many operators, model integrations and extensive tests. Its repository size is
not a sensible target for a single-device research implementation. The relevant
comparison is whether execution responsibilities are implemented and observable.

## CUDA-specific decisions

The available GPU is Ada. The implementation uses warp shuffles, explicit stream
dependencies, runtime compilation, cuBLAS and CUDA Graphs. It does not introduce
Hopper-only TMA or WGMMA names as claimed capabilities.

The tested CuPy matmul wrapper rejects capture. `runtime/blas.py` therefore uses a
dedicated cuBLAS handle bound to one execution stream before capture. Operator
outputs are preallocated. This makes graph execution a real implementation rather
than an attention-only demo renamed as a model graph.

`runtime/device.py`, `runtime/execution.py`, `runtime/blas.py`,
`runtime/profiler.py`, `runtime/service.py` and `kernels/execution.cu` form the
new execution stack. Tests exercise real replay, different batch signatures,
memory reuse, event dependencies, API concurrency and output equality. The
results report records the measured benefit and the gaps that remain.
