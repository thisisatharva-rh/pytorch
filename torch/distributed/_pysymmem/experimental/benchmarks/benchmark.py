"""Manifest-driven public ProcessGroup correctness and paired latency harness."""

import argparse
import hashlib
import importlib
import json
import math
import os
import statistics
import sys
from datetime import timedelta
from pathlib import Path


OPS = {"all_reduce", "broadcast", "all_gather", "reduce_scatter", "all_to_all"}
DTYPES = {"float32", "float16", "bfloat16"}


def validate_manifest(manifest):
    size = manifest.get("world_size")
    if type(size) is not int or size < 1:
        raise ValueError("world_size must be a positive integer")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    names = set()
    for case in cases:
        name = case.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("case names must be nonempty and unique")
        names.add(name)
        if case.get("op") not in OPS or case.get("dtype") not in DTYPES:
            raise ValueError(f"{name}: unsupported operation or dtype")
        counts = case.get("counts")
        if (
            not isinstance(counts, list)
            or len(counts) != size
            or any(type(n) is not int or n < 0 for n in counts)
        ):
            raise ValueError(
                f"{name}: counts must contain one nonnegative integer per rank"
            )
        if (
            case["op"] in {"all_reduce", "broadcast", "all_to_all"}
            and len(set(counts)) != 1
        ):
            raise ValueError(
                f"{name}: this operation requires equal counts in this harness"
            )
        if case["op"] == "all_to_all" and counts[0] % size:
            raise ValueError(
                f"{name}: all_to_all count must be divisible by world_size"
            )
        root = case.get("root", 0)
        if type(root) is not int or not 0 <= root < size:
            raise ValueError(f"{name}: root must be a valid rank")
    timing = manifest.get("timing", {})
    for field, default in (
        ("warmup", 5),
        ("iterations", 20),
        ("trials", 7),
        ("validation_steps", 4),
        ("timeout_seconds", 120),
    ):
        value = timing.get(field, default)
        if type(value) is not int or value < 1:
            raise ValueError(f"timing.{field} must be a positive integer")
    modes = timing.get("modes", ["eager", "graph"])
    if (
        not isinstance(modes, list)
        or not modes
        or len(set(modes)) != len(modes)
        or any(m not in {"eager", "graph"} for m in modes)
    ):
        raise ValueError(
            "timing.modes must select eager and/or graph without duplicates"
        )
    for label in ("baseline", "candidate"):
        config = manifest.get(label)
        if not isinstance(config, dict) or not isinstance(config.get("backend"), str):
            raise ValueError(f"{label}.backend is required")
        if not isinstance(config.get("env", {}), dict) or any(
            not isinstance(k, str) or not isinstance(v, str)
            for k, v in config.get("env", {}).items()
        ):
            raise ValueError(f"{label}.env must map strings to strings")
        accumulation = config.get("all_reduce_accumulation", "native")
        if accumulation not in {"native", "float32"}:
            raise ValueError(
                f"{label}.all_reduce_accumulation must be native or float32"
            )
        if accumulation == "float32" and any(
            case["op"] != "all_reduce" for case in cases
        ):
            raise ValueError(
                "float32 accumulation wrapper supports only all_reduce cases"
            )
    return manifest


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(samples):
    base = samples["baseline"]
    candidate = samples["candidate"]
    return {
        label: {
            "samples_ms": values,
            "p50_ms": statistics.median(values),
            "p10_ms": percentile(values, 0.1),
            "p90_ms": percentile(values, 0.9),
        }
        for label, values in samples.items()
    } | {
        "paired_speedup_samples": [b / c for b, c in zip(base, candidate)],
        "median_paired_speedup": statistics.median(
            b / c for b, c in zip(base, candidate)
        ),
    }


def source_hashes(paths):
    return {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(paths))
    }


def run(manifest, output, manifest_path):
    import torch
    import torch.distributed as dist
    from torch.testing._internal.common_utils import TestCase

    if int(os.environ.get("WORLD_SIZE", "0")) != manifest["world_size"]:
        raise ValueError("launch with torchrun using the manifest world_size")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    timeout = timedelta(seconds=manifest.get("timing", {}).get("timeout_seconds", 120))
    paths = [Path(__file__), manifest_path]
    paths.extend(Path(p).resolve() for p in manifest.get("source_files", []))
    registration_modules = []
    for config in (manifest["baseline"], manifest["candidate"]):
        for module_name in config.get("imports", []):
            module = importlib.import_module(module_name)
            registration_modules.append(module_name)
            if module.__file__:
                paths.append(Path(module.__file__))
    # Include local Python backend dependencies imported during registration.
    repo = Path(__file__).resolve().parents[5]
    for module in tuple(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if file and "_pysymmem" in file and Path(file).is_file():
            paths.append(Path(file))
    hashes = source_hashes(paths)
    dist.init_process_group("nccl", device_id=device, timeout=timeout)
    rank, size = dist.get_rank(), dist.get_world_size()
    groups = {}
    rows = []
    checker = TestCase()
    timing = manifest.get("timing", {})
    warmup, iterations, trials, validation_steps = (
        timing.get(k, default)
        for k, default in (
            ("warmup", 5),
            ("iterations", 20),
            ("trials", 7),
            ("validation_steps", 4),
        )
    )
    stream = torch.cuda.Stream()
    devices = [None] * size
    dist.all_gather_object(
        devices,
        {
            "rank": rank,
            "local_rank": device.index,
            "host": os.uname().nodename,
            "name": torch.cuda.get_device_name(device),
            "total_memory": torch.cuda.get_device_properties(device).total_memory,
        },
    )
    try:
        for label in ("baseline", "candidate"):
            config = manifest[label]
            saved = {key: os.environ.get(key) for key in config.get("env", {})}
            try:
                os.environ.update(config.get("env", {}))
                groups[label] = dist.new_group(
                    backend=config["backend"], timeout=timeout
                )
            finally:
                for key, value in saved.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
        for case in manifest["cases"]:
            dtype = getattr(torch, case["dtype"])
            counts, op = case["counts"], case["op"]
            # Integer-valued patterns divided by a power of two remain exact in
            # BF16 for modest world sizes; bound every partial sum to <= 256.
            if size > 64:
                raise ValueError("exact validation currently supports at most 64 ranks")

            def pattern(n, peer, generation, offset=0):
                index = torch.arange(n, device=device, dtype=torch.int64) + offset
                return (((index + generation) % 3) - 1 + peer % 4).to(dtype) / 256

            states = {}
            for label in groups:
                if op == "reduce_scatter":
                    inputs = [
                        torch.empty(n, device=device, dtype=dtype) for n in counts
                    ]
                else:
                    inputs = torch.empty(counts[rank], device=device, dtype=dtype)
                if op == "all_gather":
                    result = [
                        torch.empty(n, device=device, dtype=dtype) for n in counts
                    ]
                elif op in {"all_reduce", "broadcast"}:
                    result = inputs
                else:
                    result = torch.empty(counts[rank], device=device, dtype=dtype)
                states[label] = (inputs, result)
            accumulators = {
                label: torch.empty_like(states[label][0], dtype=torch.float32)
                for label in groups
                if manifest[label].get("all_reduce_accumulation") == "float32"
            }

            generations = []
            for generation in range(3):
                if op == "reduce_scatter":
                    generations.append(
                        [
                            pattern(n, rank, generation, destination)
                            for destination, n in enumerate(counts)
                        ]
                    )
                else:
                    generations.append(pattern(counts[rank], rank, generation))

            def fill(label, generation):
                inputs, _ = states[label]
                if op == "reduce_scatter":
                    for destination, tensor in enumerate(inputs):
                        tensor.copy_(generations[generation % 3][destination])
                else:
                    inputs.copy_(generations[generation % 3])

            def collective(label):
                inputs, result = states[label]
                group = groups[label]
                if op == "all_reduce":
                    if label in accumulators:
                        temporary = accumulators[label]
                        temporary.copy_(inputs)
                        dist.all_reduce(temporary, group=group)
                        inputs.copy_(temporary)
                    else:
                        dist.all_reduce(inputs, group=group)
                elif op == "broadcast":
                    dist.broadcast(inputs, src=case.get("root", 0), group=group)
                elif op == "all_gather":
                    dist.all_gather(result, inputs, group=group)
                elif op == "reduce_scatter":
                    dist.reduce_scatter(result, inputs, group=group)
                else:
                    dist.all_to_all_single(result, inputs, group=group)

            def check(label, generation):
                inputs, result = states[label]
                if op == "all_gather":
                    for peer, tensor in enumerate(result):
                        checker.assertEqual(
                            tensor,
                            pattern(counts[peer], peer, generation),
                            atol=0,
                            rtol=0,
                        )
                elif op in {"all_reduce", "reduce_scatter"}:
                    expected = torch.zeros(
                        counts[rank], device=device, dtype=torch.float64
                    )
                    for peer in range(size):
                        expected.add_(
                            pattern(
                                counts[rank],
                                peer,
                                generation,
                                rank if op == "reduce_scatter" else 0,
                            ).double()
                        )
                    checker.assertEqual(result, expected.to(dtype), atol=0, rtol=0)
                elif op == "broadcast":
                    checker.assertEqual(
                        result,
                        pattern(counts[rank], case.get("root", 0), generation),
                        atol=0,
                        rtol=0,
                    )
                else:
                    chunk = counts[rank] // size
                    expected = torch.cat(
                        [
                            pattern(chunk, peer, generation, rank * chunk)
                            for peer in range(size)
                        ]
                    )
                    checker.assertEqual(result, expected, atol=0, rtol=0)
                if op in {"all_gather", "all_to_all", "reduce_scatter"}:
                    if isinstance(inputs, list):
                        for destination, tensor in enumerate(inputs):
                            checker.assertEqual(
                                tensor,
                                pattern(tensor.numel(), rank, generation, destination),
                                atol=0,
                                rtol=0,
                            )
                    else:
                        checker.assertEqual(
                            inputs,
                            pattern(inputs.numel(), rank, generation),
                            atol=0,
                            rtol=0,
                        )

            for label in groups:
                for generation in range(validation_steps):
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        fill(label, generation)
                        collective(label)
                    stream.synchronize()
                    check(label, generation)
            for mode in timing.get("modes", ["eager", "graph"]):
                graphs = {}
                samples = {label: [] for label in groups}
                try:
                    for label in groups:
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            for _ in range(warmup):
                                fill(label, 0)
                                collective(label)
                        stream.synchronize()
                        if mode == "graph":
                            dist.barrier()
                            torch.cuda.synchronize()
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph, stream=stream):
                                collective(label)
                            graphs[label] = graph
                            for generation in range(
                                validation_steps, 2 * validation_steps
                            ):
                                fill(label, generation)
                                torch.cuda.synchronize()
                                with torch.cuda.stream(stream):
                                    graph.replay()
                                stream.synchronize()
                                check(label, generation)
                            del graph
                    for trial in range(trials):
                        order = (
                            ["baseline", "candidate"]
                            if trial % 2 == 0
                            else ["candidate", "baseline"]
                        )
                        for label in order:
                            # In-place operations need a fresh input for each call.
                            # Fill outside each event interval on the same stream.
                            dist.barrier()
                            torch.cuda.synchronize()
                            event_pairs = []
                            with torch.cuda.stream(stream):
                                for iteration in range(iterations):
                                    fill(label, trial + iteration)
                                    start = torch.cuda.Event(enable_timing=True)
                                    end = torch.cuda.Event(enable_timing=True)
                                    start.record()
                                    if mode == "graph":
                                        graphs[label].replay()
                                    else:
                                        collective(label)
                                    end.record()
                                    event_pairs.append((start, end))
                            stream.synchronize()
                            check(label, trial + iterations - 1)
                            elapsed = torch.tensor(
                                [start.elapsed_time(end) for start, end in event_pairs],
                                device=device,
                                dtype=torch.float64,
                            )
                            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                            samples[label].append(elapsed.mean().item())
                    row = {
                        "case": case,
                        "timing": mode,
                        "validation": "exact changing-input and input-preservation checks passed",
                        **summarize(samples),
                    }
                    rows.append(row)
                    if rank == 0:
                        print(json.dumps(row), flush=True)
                finally:
                    graphs.clear()
        if hashes != source_hashes(paths):
            raise RuntimeError("source files changed during the experiment")
        if rank == 0:
            report = {
                "schema_version": 1,
                "manifest": manifest,
                "metadata": {
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "nccl": torch.cuda.nccl.version(),
                    "devices": devices,
                    "source_sha256": hashes,
                    "command": sys.argv,
                    "registration_modules": registration_modules,
                    "nccl_environment": {
                        key: os.environ[key]
                        for key in (
                            "NCCL_ALGO",
                            "NCCL_PROTO",
                            "NCCL_NVLS_ENABLE",
                            "NCCL_MIN_CTAS",
                            "NCCL_MAX_CTAS",
                            "NCCL_BUFFSIZE",
                            "NCCL_P2P_DISABLE",
                            "NCCL_SHM_DISABLE",
                            "NCCL_CUMEM_ENABLE",
                            "NCCL_MNNVL_ENABLE",
                        )
                        if key in os.environ
                    },
                    "repo": str(repo),
                    "timing_scope": "CUDA events around public collective calls; rank maximum per call; input refresh, allocation, warmup, JIT and capture excluded",
                    "graph_scope": "one collective per replay",
                    "limitations": [
                        "Isolated collective latency, not application throughput",
                        "Exact bounded patterns are not a full floating-point numerical suite",
                        "Backend names do not prove specialized dispatch",
                        "No hardware bandwidth counters or automatic winner promotion",
                        "Event instrumentation may affect tiny-collective latency",
                    ],
                },
                "results": rows,
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(output.suffix + ".tmp")
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            temporary.replace(output)
    finally:
        torch.cuda.synchronize()
        for group in reversed(list(groups.values())):
            dist.destroy_process_group(group)
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    manifest = validate_manifest(json.loads(args.manifest.read_text()))
    if args.validate_only:
        print("Manifest valid")
        return
    if args.output is None:
        parser.error("--output is required unless using --validate-only")
    if args.output.exists():
        parser.error("output already exists; use a new experiment path")
    run(manifest, args.output.resolve(), args.manifest.resolve())


if __name__ == "__main__":
    main()
