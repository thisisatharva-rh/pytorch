"""Distributed device-generic precision and repeated-use checks."""

import hashlib
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed._pysymmem.experimental import all_reduce as fp32_accum
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


GROUP = None
RECORDS = []


def values(n, peer, generation, kind, device):
    if kind == "cancellation":
        constants = [256.0, 1.0, -256.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        index = torch.arange(n, device=device)
        sign = ((index + generation) % 2 * 2 - 1).float()
        return (sign * constants[peer]).bfloat16()
    generator = torch.Generator().manual_seed(1000 + 97 * peer + generation)
    integers = torch.randint(-127, 128, (n,), generator=generator)
    scales = torch.where(torch.arange(n) % 2 == 0, 1.0, 1.0 / 128.0)
    return (integers.float() * scales / 1024).to(device=device, dtype=torch.bfloat16)


class TestPrecision(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        cls.primary_device = f"cuda:{local_rank}"

    @parametrize("n", [0, 1, 257, 32769])
    @parametrize("kind", ["cancellation", "random_dyadic"])
    def test_reuse_and_graph(self, device, n, kind):
        rank = dist.get_rank()
        storage = torch.full((n + 2,), 42.0, device=device, dtype=torch.bfloat16)
        tensor = storage[1:-1]
        stream = torch.cuda.Stream()
        graph = None
        try:
            for mode in ("eager", "graph"):
                if mode == "graph":
                    dist.barrier()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        dist.all_reduce(tensor, group=GROUP)
                for generation in range(6):
                    reference = torch.zeros(n, device=device, dtype=torch.float64)
                    for peer in range(dist.get_world_size()):
                        reference.add_(
                            values(n, peer, generation, kind, device).double()
                        )
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        tensor.copy_(values(n, rank, generation, kind, device))
                        if graph is None:
                            work = dist.all_reduce(tensor, group=GROUP, async_op=True)
                            work.wait()
                        else:
                            graph.replay()
                    stream.synchronize()
                    self.assertEqual(tensor, reference.bfloat16(), atol=0, rtol=0)
                    self.assertEqual(
                        storage[[0, -1]],
                        torch.full((2,), 42.0, device=device, dtype=torch.bfloat16),
                    )
                if graph is not None:
                    del graph
                    graph = None
            RECORDS.append(
                {
                    "n": n,
                    "kind": kind,
                    "eager_and_graph": "passed",
                    "generations_per_mode": 6,
                }
            )
        finally:
            if graph is not None:
                del graph

    def test_large_reuse(self, device):
        n = 1048577
        rank = dist.get_rank()
        stream = torch.cuda.Stream()
        tensor = torch.empty(n, device=device, dtype=torch.bfloat16)
        index = torch.arange(n, device=device)
        stream.wait_stream(torch.cuda.current_stream())
        for generation in range(32):
            with torch.cuda.stream(stream):
                tensor.copy_(((index + generation) % 7 - 3).float() * (rank + 1) / 256)
                dist.all_reduce(tensor, group=GROUP)
            stream.synchronize()
            factor = dist.get_world_size() * (dist.get_world_size() + 1) // 2
            expected = (
                ((index + generation) % 7 - 3).float() * factor / 256
            ).bfloat16()
            self.assertEqual(tensor, expected, atol=0, rtol=0)
        RECORDS.append(
            {"n": n, "kind": "large_reuse", "generations": 32, "status": "passed"}
        )


instantiate_device_type_tests(TestPrecision, globals(), only_for="cuda")

if __name__ == "__main__":
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=90))
    os.environ["SYMMEM_WORKSPACE_BYTES"] = "268435456"
    GROUP = dist.new_group(backend="symmem_fp32_accum", timeout=timedelta(seconds=90))
    try:
        try:
            run_tests()
        except SystemExit as error:
            if (
                error.code in (None, 0)
                and dist.get_rank() == 0
                and "PY_SYMMEM_VALIDATION_REPORT" in os.environ
            ):
                path = Path(fp32_accum.__file__)
                Path(os.environ["PY_SYMMEM_VALIDATION_REPORT"]).write_text(
                    json.dumps(
                        {
                            "world_size": dist.get_world_size(),
                            "candidate_sha256": hashlib.sha256(
                                path.read_bytes()
                            ).hexdigest(),
                            "test_source_sha256": hashlib.sha256(
                                Path(__file__).read_bytes()
                            ).hexdigest(),
                            "records": RECORDS,
                        },
                        indent=2,
                    )
                    + "\n"
                )
            raise
    finally:
        dist.destroy_process_group(GROUP)
        dist.destroy_process_group()
