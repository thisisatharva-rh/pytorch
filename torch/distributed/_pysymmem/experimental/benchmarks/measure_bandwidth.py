"""All-reduce NVLink endpoint traffic and public-call bandwidth probe."""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import TestCase

from .. import all_reduce as fp32_accum_v2
from . import nvlink_counters
from .benchmark import validate_manifest
from .nvlink_counters import counter_delta, read_counters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    sizes = parser.add_mutually_exclusive_group()
    sizes.add_argument("--sizes-mib", type=int, nargs="+")
    sizes.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new output path")
    manifest = None
    workspace_bytes = "268435456"
    candidate_backend = "symmem_fp32_accum"
    elements = [n * 1024**2 // 2 for n in (args.sizes_mib or [16, 128])]
    modes = ("pysymmem_bf16_fp32acc", "nccl_fp32_casts", "nccl_bf16_native")
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        validate_manifest(manifest)
        if (
            manifest["baseline"].get("backend") != "nccl"
            or manifest["baseline"].get("all_reduce_accumulation") != "float32"
            or manifest["candidate"].get("backend") not in ("symmem_fp32_accum", "symmem_fp32_accum_candidate2")
            or any(c["op"] != "all_reduce" or c["dtype"] != "bfloat16" for c in manifest["cases"])
        ):
            parser.error("manifest must compare BF16 FP32-accumulating all-reduce with NCCL FP32 plus casts")
        if int(os.environ.get("WORLD_SIZE", "0")) != manifest["world_size"]:
            parser.error("launch with the manifest world_size")
        candidate_backend = manifest["candidate"]["backend"]
        elements = [c["counts"][0] for c in manifest["cases"]]
        workspace_bytes = manifest["candidate"].get("env", {}).get("SYMMEM_WORKSPACE_BYTES", workspace_bytes)
        modes = ("pysymmem_bf16_fp32acc", "nccl_fp32_casts")
    if any(n <= 0 for n in elements):
        parser.error("sizes must be positive")
    paths = [
        Path(__file__),
        Path(nvlink_counters.__file__),
        Path(fp32_accum_v2.__file__),
    ]
    if args.manifest:
        paths.append(args.manifest)
    paths.extend(
        Path(module.__file__)
        for module in tuple(sys.modules.values())
        if getattr(module, "__file__", None)
        and "_pysymmem" in module.__file__
        and Path(module.__file__).is_file()
    )
    hashes = {
        str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in set(paths)
    }
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    timeout = timedelta(seconds=180)
    dist.init_process_group("nccl", device_id=device, timeout=timeout)
    rank, size = dist.get_rank(), dist.get_world_size()
    os.environ["SYMMEM_WORKSPACE_BYTES"] = workspace_bytes
    control = dist.new_group(backend="gloo", timeout=timeout)
    candidate = dist.new_group(backend=candidate_backend, timeout=timeout)
    identities = [None] * size
    dist.all_gather_object(
        identities,
        {
            "rank": rank,
            "uuid": str(torch.cuda.get_device_properties(device).uuid),
            "name": torch.cuda.get_device_name(device),
        },
        group=control,
    )
    stream = torch.cuda.Stream()
    checker = TestCase()
    trials = []
    idle = None
    graphs = {}
    try:
        torch.cuda.synchronize()
        dist.barrier(group=control)
        before = read_counters() if rank == 0 else None
        dist.barrier(group=control)
        time.sleep(0.2)
        dist.barrier(group=control)
        after = read_counters() if rank == 0 else None
        dist.barrier(group=control)
        if rank == 0:
            idle = {
                "before": before,
                "after": after,
                "delta": counter_delta(before, after),
            }
        for n in elements:
            mib = n * 2 / 1024**2
            inputs = {
                mode: torch.empty(n, dtype=torch.bfloat16, device=device)
                for mode in modes
            }
            seed = torch.full(
                (n,), (rank + 1) / 256, dtype=torch.bfloat16, device=device
            )
            temporary = torch.empty(n, dtype=torch.float32, device=device)

            def collective(mode):
                tensor = inputs[mode]
                if mode == "pysymmem_bf16_fp32acc":
                    dist.all_reduce(tensor, group=candidate)
                elif mode == "nccl_fp32_casts":
                    temporary.copy_(tensor)
                    dist.all_reduce(temporary)
                    tensor.copy_(temporary)
                else:
                    dist.all_reduce(tensor)

            stream.wait_stream(torch.cuda.current_stream())
            for mode in modes:
                with torch.cuda.stream(stream):
                    for _ in range(5):
                        inputs[mode].copy_(seed)
                        collective(mode)
                stream.synchronize()
                dist.barrier(group=control)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    collective(mode)
                graphs[mode] = graph
                del graph
            for trial, calls in enumerate((30, 60, 30)):
                order = modes[trial:] + modes[:trial]
                for mode in order:
                    torch.cuda.synchronize()
                    dist.barrier(group=control)
                    before = read_counters() if rank == 0 else None
                    dist.barrier(group=control)
                    pairs = []
                    wall_start = time.monotonic_ns()
                    with torch.cuda.stream(stream):
                        for _ in range(calls):
                            inputs[mode].copy_(seed)
                            start = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            start.record()
                            graphs[mode].replay()
                            end.record()
                            pairs.append((start, end))
                    stream.synchronize()
                    wall_end = time.monotonic_ns()
                    dist.barrier(group=control)
                    after = read_counters() if rank == 0 else None
                    dist.barrier(group=control)
                    timings = [None] * size
                    dist.all_gather_object(
                        timings,
                        {
                            "rank": rank,
                            "call_ms": [s.elapsed_time(e) for s, e in pairs],
                            "wall_start_ns": wall_start,
                            "wall_end_ns": wall_end,
                        },
                        group=control,
                    )
                    expected = size * (size + 1) / 2 / 256
                    checker.assertEqual(
                        inputs[mode],
                        torch.full_like(inputs[mode], expected),
                        atol=0,
                        rtol=0,
                    )
                    if rank == 0:
                        delta = counter_delta(before, after)
                        uuids = {
                            identity["uuid"].removeprefix("GPU-").lower()
                            for identity in identities
                        }
                        participating = {
                            gpu: data
                            for gpu, data in delta.items()
                            if data["uuid"].removeprefix("GPU-").lower() in uuids
                        }
                        if len(participating) != size:
                            raise RuntimeError(
                                f"counter devices do not match ranks: counters={[g['uuid'] for g in delta.values()]}, identities={identities}"
                            )
                        call_max = [
                            max(t["call_ms"][i] for t in timings) for i in range(calls)
                        ]
                        total_ms = sum(call_max)
                        tx = sum(g["tx_bytes"] for g in participating.values())
                        rx = sum(g["rx_bytes"] for g in participating.values())
                        per_gpu = {
                            gpu: {
                                "tx_bytes_per_call": g["tx_bytes"] / calls,
                                "rx_bytes_per_call": g["rx_bytes"] / calls,
                                "tx_GBps": g["tx_bytes"] / total_ms / 1e6,
                                "rx_GBps": g["rx_bytes"] / total_ms / 1e6,
                            }
                            for gpu, g in participating.items()
                        }
                        logical = n * 2
                        row = {
                            "bf16_mib_per_rank": mib,
                            "elements_per_rank": n,
                            "mode": mode,
                            "calls": calls,
                            "ms_per_call": total_ms / calls,
                            "bf16_payload_GBps_per_rank": logical
                            / (total_ms / calls)
                            / 1e6,
                            "aggregate_tx_bytes_per_call": tx / calls,
                            "aggregate_rx_bytes_per_call": rx / calls,
                            "aggregate_tx_GBps": tx / total_ms / 1e6,
                            "aggregate_rx_GBps": rx / total_ms / 1e6,
                            "per_gpu": per_gpu,
                            "before": before,
                            "after": after,
                            "delta": delta,
                            "timings": timings,
                            "rank_max_call_ms": call_max,
                            "validation": "exact result passed",
                        }
                        trials.append(row)
                        print(
                            json.dumps(
                                {
                                    k: v
                                    for k, v in row.items()
                                    if k
                                    not in (
                                        "before",
                                        "after",
                                        "delta",
                                        "timings",
                                        "per_gpu",
                                        "rank_max_call_ms",
                                    )
                                }
                            ),
                            flush=True,
                        )
            graphs.clear()
            torch.cuda.synchronize()
            dist.barrier(group=control)
        if any(
            hashlib.sha256(Path(p).read_bytes()).hexdigest() != digest
            for p, digest in hashes.items()
        ):
            raise RuntimeError("sources changed during bandwidth measurement")
        if rank == 0:
            report = {
                "metadata": {
                    "source_sha256": hashes,
                    "manifest": manifest,
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "nccl": torch.cuda.nccl.version(),
                    "identities": identities,
                    "command": sys.argv,
                    "counter": "nvidia-smi nvlink --getthroughput d; payload KiB converted to bytes",
                    "timing": "sum of rank-max CUDA-event call durations; single-collective graph replay; staging and FP32 baseline casts included; local input reset excluded",
                    "coordination": "Gloo only inside counter windows; no counter resets",
                    "caveats": [
                        "Device-wide endpoint counters include background traffic; inspect idle delta",
                        "TX and RX reported separately, never summed as unique traffic",
                        "NVLS switch reduction/multicast can make endpoint traffic asymmetric",
                        "Rates divide observed traffic by measured collective-active time, not the entire counter window",
                        "Input resets warm memory; per-call event/replay launch gaps can affect small sizes",
                    ],
                },
                "idle": idle,
                "trials": trials,
            }
            temporary_path = args.output.with_suffix(".tmp")
            temporary_path.write_text(json.dumps(report, indent=2) + "\n")
            temporary_path.replace(args.output)
    finally:
        graphs.clear()
        torch.cuda.synchronize()
        dist.destroy_process_group(candidate)
        dist.destroy_process_group(control)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
