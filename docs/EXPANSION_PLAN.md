# Runtime expansion after inspecting TT-Metal/TTNN

Reference revision: tenstorrent/tt-metal c346c73b04b1a3074477389ac007fb0a62af437c.
This is an architecture study, not a port of Tenstorrent hardware-specific code.

Inspected source: METALIUM_GUIDE.md; tt_metal/api/tt-metalium/program_cache.hpp,
mesh_command_queue.hpp and mesh_event.hpp; tech_reports/memory/allocator.md;
tech_reports/AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md;
ttnn/cpp/ttnn/operations/matmul/matmul.hpp.

The missing depth is primarily execution infrastructure, not a missing collection
of operation names. TT-Metal separates device ownership, buffer allocation,
program compilation, runtime-argument rebinding, command queues, events, trace
capture and operators. The runtime will adopt analogous responsibilities using
CUDA streams, events, pinned staging, reusable workspaces and CUDA Graphs.

## Implement and validate

1. Device command queues, event dependencies, pinned staging, deferred retirement,
   memory/workspace accounting and bounded program caching.
2. Reusable static execution programs that rebind runtime token IDs, positions,
   physical KV write slots and block tables without recapturing the program.
3. A whole-model CUDA Graph path: stable buffers, library GEMM output buffers,
   fused KV writes, fused residual RMSNorm, fused SwiGLU and custom attention.
4. Warp-shuffle attention to remove per-token CTA-wide reduction barriers.
5. Integrate graph execution with changing scheduler batches; retain eager CPU/GPU
   execution as an independent correctness and performance baseline.
6. Measure full runtime and pretrained model latency, launch/workspace statistics,
   cache pressure and graph-cache reuse. Export raw data, plots and a profiler
   trace; distinguish host intervals from device event timings.
7. Add CPU CI and optional GPU/sanitizer workflows, architecture documentation,
   experiment reproduction and explicit remaining limitations.

Tests must exercise actual GPU graph replay with changing data, request reuse,
boundary pages, GQA, tail dimensions, cache eviction, event ownership and output
agreement. No architecture-specific term is a capability claim without executable
code and a test/measurement. Native C++ hosting, multi-GPU and Hopper TMA/WGMMA
are outside this Ada implementation.

## Outcome

Implemented device queues/events, pinned inputs, sealed workspaces, bounded
program caching, a capture-safe cuBLAS binding, full-forward model graphs, fused
operators, warp attention, scheduler integration, a local JSON/SSE service,
profiling exports and CI workflows. Twenty-six tests passed locally including GPU
execution. Actual pretrained, kernel-sweep, scheduler and loopback HTTP results
are in results/expansion. Cross-request IO overlap, hardware-counter profiling and
native C++ hosting remain explicitly outside the demonstrated capabilities.
