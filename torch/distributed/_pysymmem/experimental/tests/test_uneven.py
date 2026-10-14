import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.distributed._pysymmem.experimental
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    subtest,
    TestCase,
)


class TestEight(TestCase):
    @parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_random_sum(self, device, dtype):
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        rank = dist.get_rank()
        counts = (131073, 32767, 0, 17, 8193, 0, 63, 1)
        for iteration in range(3):
            torch.manual_seed(9876 + rank * 100 + iteration)
            inputs = [torch.randn(n, device=device).to(dtype) for n in counts]
            originals = [t.clone() for t in inputs]
            result = torch.empty(counts[rank], device=device, dtype=dtype)
            reference = torch.empty(counts[rank], device=device, dtype=torch.float64)
            dist.reduce_scatter(result, inputs, group=GROUP)
            dist.reduce_scatter(reference, [t.double() for t in inputs])
            self.assertEqual(result, reference.to(dtype))
            self.assertEqual(inputs, originals, atol=0, rtol=0)

    @parametrize("dtype", [torch.bfloat16, torch.float32])
    @parametrize("graph", [False, True])
    @parametrize(
        "counts",
        [
            (0, 0, 0, 0, 0, 0, 0, 0),
            (1, 7, 63, 0, 8193, 2, 0, 33),
            (1048593, 0, 0, 0, 0, 0, 0, 0),
            (32767, 32768, 32769, 8192, 65537, 513, 1, 0),
            subtest((64 * 1024**2 + 17, 0, 0, 0, 0, 0, 0, 0), name="long_pipeline"),
        ],
    )
    def test_collectives(self, device, dtype, graph, counts):
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        rank = dist.get_rank()
        inp = torch.empty(counts[rank] + 2, device=device, dtype=dtype)
        inputs = [torch.empty(n + 2, device=device, dtype=dtype) for n in counts]
        ag_storage = [
            torch.full((n + 2,), -99, device=device, dtype=dtype) for n in counts
        ]
        ag = [t[1:-1] for t in ag_storage]
        ag_ref = [torch.empty(n, device=device, dtype=dtype) for n in counts]
        rs_storage = torch.full((counts[rank] + 2,), -99, device=device, dtype=dtype)
        rs = rs_storage[1:-1]
        rs_ref = torch.empty_like(rs)
        tensors = [t[1:-1] for t in inputs]
        local = inp[1:-1]

        def collective():
            dist.all_gather(ag, local, group=GROUP)
            dist.reduce_scatter(rs, tensors, group=GROUP)

        for t in [inp, *inputs]:
            t.zero_()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            collective()
        stream.synchronize()
        captured = None
        if graph:
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured, stream=stream):
                collective()
        for iteration in range(5):
            torch.manual_seed(1234 + rank * 100 + iteration)
            for t in [inp, *inputs]:
                t.copy_(torch.randint(-8, 9, t.shape, device=device).to(dtype) / 8)
            if captured is None:
                collective()
            else:
                captured.replay()
            torch.cuda.synchronize()
            dist.all_gather(ag_ref, local)
            dist.reduce_scatter(rs_ref, tensors)
            self.assertEqual(ag, ag_ref, atol=0, rtol=0)
            self.assertEqual(rs, rs_ref, atol=0, rtol=0)
            for t in [*ag_storage, rs_storage]:
                self.assertEqual(t[0].item(), -99)
                self.assertEqual(t[-1].item(), -99)
        if captured is not None:
            del captured


class TestLongBalancedRS(TestCase):
    @parametrize("dtype", [torch.bfloat16, torch.float32])
    @parametrize("graph", [False, True])
    def test_long_balanced(self, device, dtype, graph):
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        rank = dist.get_rank()
        counts = (33554449, 16777219, 8388613, 4194311, 4096, 4096, 0, 0)
        inputs = [torch.zeros(n, dtype=dtype, device=device) for n in counts]
        storage = torch.full((counts[rank] + 2,), -99, dtype=dtype, device=device)
        output = storage[1:-1]
        reference = torch.empty_like(output)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            dist.reduce_scatter(output, inputs, group=GROUP)
        stream.synchronize()
        captured = None
        if graph:
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured, stream=stream):
                dist.reduce_scatter(output, inputs, group=GROUP)
        for iteration in range(5):
            torch.manual_seed(2468 + rank * 100 + iteration)
            for tensor in inputs:
                tensor.copy_(
                    torch.randint(-8, 9, tensor.shape, device=device).to(dtype) / 8
                )
            originals = [tensor.clone() for tensor in inputs]
            if captured is None:
                dist.reduce_scatter(output, inputs, group=GROUP)
            else:
                captured.replay()
            torch.cuda.synchronize()
            dist.reduce_scatter(reference, inputs)
            self.assertEqual(output, reference, atol=0, rtol=0)
            self.assertEqual(inputs, originals, atol=0, rtol=0)
            self.assertEqual(storage[0].item(), -99)
            self.assertEqual(storage[-1].item(), -99)
        if captured is not None:
            del captured


instantiate_device_type_tests(TestEight, globals(), only_for="cuda")
instantiate_device_type_tests(TestLongBalancedRS, globals(), only_for="cuda")

if __name__ == "__main__":
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    os.environ["SYMMEM_WORKSPACE_BYTES"] = str(4 * 1024**3)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=120))
    GROUP = dist.new_group(backend="symmem_uneven", timeout=timedelta(seconds=120))
    try:
        run_tests()
    finally:
        dist.destroy_process_group(GROUP)
        dist.destroy_process_group()
