# Experimental PySymmem collectives

These opt-in Python ProcessGroup backends package the tuned symmetric-memory
experiments. They do not change the default `symmem` backend. Validation and
performance measurements target eight H100 GPUs in one NVLink domain.

```python
import torch.distributed as dist
import torch.distributed._pysymmem.experimental

# Initialize the default NCCL group and set this rank's CUDA device first.
precision_group = dist.new_group(backend="symmem_fp32_accum")
uneven_group = dist.new_group(backend="symmem_uneven")
```

`symmem_fp32_accum` implements contiguous BF16/FP32 SUM all-reduce. BF16
inputs accumulate in FP32 and round to BF16 only after the full rank reduction.
Small buffers use one shot; larger buffers reduce owner slices then gather the
completed values. Ordinary CUDA tensors are staged into symmetric memory.
This is the selected v2 implementation, including the two-dimensional gather.
It does not detect or make claims about NCCL's selected accumulation algorithm.

`symmem_uneven` implements the selected pipelined uneven all-gather and v8
reduce-scatter paths for eight ranks. FP32 balanced reduce-scatter retains the
owner's local contribution for the final accumulation. BF16 balanced and
dominant-owner reductions retain their previously tested paths; they do not
promise the all-reduce backend's single-rounding BF16 semantics. Other supported
operations use the base SymmemBackend. Specialized AG/RS dispatch requires eight
ranks and CUDA compute capability at least 9; other inputs fall back to the base
implementation. Persistent kernels assume available SMs and serialized
collective execution. Concurrent or overlapped collectives are not validated.

The parent backend provides workspace allocation, rendezvous, barriers, streams
and Work objects. The modules here provide collective dispatch and Triton
kernels, with no imports from `agent_space`. Importing this package registers
the experimental backend names and requires Triton.

## Tests

Run from the repository root on an otherwise idle eight-GPU host:

```bash
python -m torch.distributed._pysymmem.experimental.tests.test_benchmark
python -m torch.distributed.run --standalone --nproc-per-node=8 --module torch.distributed._pysymmem.experimental.tests.test_all_reduce
python -m torch.distributed.run --standalone --nproc-per-node=8 --module torch.distributed._pysymmem.experimental.tests.test_uneven
```

The numerical suites cover eager and graph replay with changing inputs,
misaligned contiguous views, guards, empty and unequal shards, random inputs,
and large repeated workspace reuse. The all-reduce suite includes cancellation
cases checked against an FP64 reference. These are dedicated multi-process
tests and require the launch commands above.

## Benchmarks

```bash
python -m torch.distributed.run --standalone --nproc-per-node=8 --module torch.distributed._pysymmem.experimental.benchmarks.benchmark torch/distributed/_pysymmem/experimental/benchmarks/cases/all_reduce_fp32.json --output agent_space/ar_fp32.json
python -m torch.distributed.run --standalone --nproc-per-node=8 --module torch.distributed._pysymmem.experimental.benchmarks.benchmark torch/distributed/_pysymmem/experimental/benchmarks/cases/all_reduce_native.json --output agent_space/ar_native.json
python -m torch.distributed.run --standalone --nproc-per-node=8 --module torch.distributed._pysymmem.experimental.benchmarks.benchmark torch/distributed/_pysymmem/experimental/benchmarks/cases/uneven.json --output agent_space/uneven.json
python -m torch.distributed.run --standalone --nproc-per-node=8 --module torch.distributed._pysymmem.experimental.benchmarks.measure_bandwidth --output agent_space/ar_bandwidth.json
```

Outputs must not already exist. The latency harness alternates baseline and
candidate order across seven paired trials, reports the maximum rank latency,
and measures eager calls and single-collective graph replays separately. Input
refresh, allocations, warmup, JIT and capture are outside CUDA event timing;
staging, required barriers and output copies are inside it. AG/RS seed patterns
are preallocated for three changing generations. Reports include source hashes,
GPU identities, software versions, manifest and active NCCL environment.

The precision-matched all-reduce baseline includes BF16-to-FP32 conversion,
NCCL FP32 SUM and conversion back to BF16. The native baseline measures NCCL
BF16 SUM separately and does not assert equivalent numerical behavior.
Uneven manifests preserve whole-parameter ownership for model-derived
transformer-block, embedding and output-head cases; this is isolated collective
performance, not measured model throughput.

The bandwidth probe reads cumulative NVLink counters without resetting them.
It measures aggregate transmitted bytes across all eight GPUs in separate
windows, with idle-traffic checks. TB/s here is aggregate hardware counter
throughput, not per-GPU bandwidth or latency-derived algorithm bandwidth;
baselines can move different byte counts.

## Llama parameter-size all-reduce benchmarks

The `llama8b_fp32_accum.json` and `llama70b_fp32_accum.json` manifests
benchmark full parameter-sized BF16 buffers on each rank, with FP32
accumulation. Both preserve the fused QKV and gate/up layout convention.
They are modeled buffer sizes, not captured training buckets.

For each model, run the latency harness and the manifest-driven NVLink counter
probe separately. The following commands reproduce the 70B experiment; replace
`llama70b` with `llama8b` for the 8B experiment. Use fresh output paths on repeats.

```bash
python -m torch.distributed.run --standalone --nproc-per-node=8 --module torch.distributed._pysymmem.experimental.benchmarks.benchmark torch/distributed/_pysymmem/experimental/benchmarks/cases/llama70b_fp32_accum.json --output agent_space/llama70b_latency.json
python -m torch.distributed.run --standalone --nproc-per-node=8 --module torch.distributed._pysymmem.experimental.benchmarks.measure_bandwidth --manifest torch/distributed/_pysymmem/experimental/benchmarks/cases/llama70b_fp32_accum.json --output agent_space/llama70b_bandwidth.json
```

The counter probe reads element counts and workspace size from the manifest.
It compares the candidate against NCCL FP32 plus BF16 conversions using three
alternating counter windows per size. No scratch Python driver is required.

## Candidate 2: BF16 all-reduce with FP32 accumulation

`symmem_fp32_accum_candidate2` is a separate opt-in backend. It shares the
all-reduce Triton kernels and precision contract, using 16384-element reduction
blocks with 8 warps and 16384-element gather blocks with 4 warps. Candidate 1
remains available as `symmem_fp32_accum` with its existing settings.

```python
import torch.distributed._pysymmem.experimental

candidate2_group = dist.new_group(backend="symmem_fp32_accum_candidate2")
```

The following results compare candidate 2 only with NCCL FP32 SUM plus
BF16-to-FP32 and FP32-to-BF16 conversions. They were measured on eight H100
80GB GPUs with CUDA 12.9 and NCCL 2.30.7. Latency speedups use the median
paired ratio across seven alternating trials of 30 single-collective graph
replays. Parameter-sized BF16 buffers are replicated on each rank, using the
Llama 3.1 70B dimensions and fused QKV/gate-up convention described above.

| Parameter | BF16 buffer per rank | Speedup vs NCCL FP32 + casts | Candidate 2 TX bandwidth | NCCL TX bandwidth |
|---|---:|---:|---:|---:|
| Norm weights | 16 KiB | 1.34x | 0.025 TB/s | 0.010 TB/s |
| Attention output | 128 MiB | 1.65x | 2.177 TB/s | 1.718 TB/s |
| Fused QKV | 160 MiB | 1.65x | 2.216 TB/s | 1.750 TB/s |
| FFN down | 448 MiB | 1.65x | 2.313 TB/s | 1.819 TB/s |
| Fused FFN gate/up | 896 MiB | 1.57x | 2.340 TB/s | 1.910 TB/s |
| Embedding / output head | 2004 MiB | 1.56x | 2.348 TB/s | 1.925 TB/s |

Bandwidth was measured separately using NVLink endpoint counters, taking the
median of three windows per size/backend. It is aggregate transmitted traffic
across all eight GPUs, divided by collective-active CUDA-event time, not
per-GPU bandwidth. The two algorithms transfer different byte counts. The idle
TX/RX delta was 0 bytes. Tiny norm-buffer ratios are sensitive to timing
variation and should not be treated as stable throughput claims.

Validation passed: nine precision/reuse tests on every rank, all twelve
eager/graph latency rows and all thirty-six bandwidth windows. The precision
suite covers cancellation, dyadic random inputs, tails, guards, changed graph
inputs and repeated workspace reuse. Recorded benchmark source hashes were
rechecked after measurement.

Reproduce candidate 2 tests and NCCL-only measurements from the repository root:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=8 --module torch.distributed._pysymmem.experimental.tests.test_all_reduce_candidate2
python -m torch.distributed.run --standalone --nproc-per-node=8 --module torch.distributed._pysymmem.experimental.benchmarks.benchmark torch/distributed/_pysymmem/experimental/benchmarks/cases/llama70b_fp32_accum_candidate2.json --output agent_space/candidate2_latency.json
python -m torch.distributed.run --standalone --nproc-per-node=8 --module torch.distributed._pysymmem.experimental.benchmarks.measure_bandwidth --manifest torch/distributed/_pysymmem/experimental/benchmarks/cases/llama70b_fp32_accum_candidate2.json --output agent_space/candidate2_bandwidth.json
```

Use fresh output paths for repeated runs. No scratch module imports or launch
overrides are needed to obtain candidate 2 settings.
