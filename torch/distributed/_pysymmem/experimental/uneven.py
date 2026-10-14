"""Eight-rank pipelined all-gather and uneven SUM reduce-scatter experiments."""

import triton

import torch
import torch.distributed as dist
from torch.distributed._pysymmem.backend import (
    cast_buffer,
    reduce_op_name,
    SymmemBackend,
)

from . import _uneven_kernels as kernels


class UnevenBackend(SymmemBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        props = torch.cuda.get_device_properties(self._device)
        self._eight = self._size == 8 and props.major >= 9
        self._ctas = min(128, props.multi_processor_count)
        self._order = tuple((p + self._rank) % self._size for p in range(self._size))

    def allgather(self, output_tensors, input_tensors, opts=None):
        inp = input_tensors[0]
        outs = output_tensors[0]
        if (
            not self._eight
            or inp.dtype not in (torch.bfloat16, torch.float32)
            or not inp.is_contiguous()
            or not all(t.is_contiguous() for t in outs)
        ):
            return super().allgather(output_tensors, input_tensors, opts)
        self._check_device(inp)
        if len(outs) != 8 or outs[self._rank].numel() != inp.numel():
            raise self._err(
                "allgather requires eight outputs and a matching local size"
            )
        for out in outs:
            self._check_device(out)
            if out.dtype != inp.dtype:
                raise self._err("allgather requires matching dtypes")
        block, warps = (16384 if inp.dtype == torch.float32 else 32768), 16
        counts = tuple(t.numel() for t in outs)
        chunks = tuple(triton.cdiv(n, 7 * block) * block for n in counts)
        slot = max(chunks)
        tiles = triton.cdiv(slot, block)
        ctas = min(self._ctas, tiles)
        data_bytes = slot * 8 * inp.element_size()
        flag_bytes = 8 * ctas * 128
        self._ensure_workspace(max(data_bytes + flag_bytes, 1))
        raw = [self._peer_scratch(p) for p in range(8)]
        bufs = tuple(cast_buffer(t, inp) for t in raw)
        flags = tuple(
            t[data_bytes : data_bytes + flag_bytes].view(torch.int32) for t in raw
        )
        if ctas:
            flags[self._rank].zero_()
        self._group_barrier()
        if ctas:
            kernels.ag_pipeline[(ctas,)](
                inp,
                bufs,
                flags,
                tuple(outs),
                counts,
                chunks,
                self._rank,
                slot,
                tiles,
                block,
                self._order,
                num_warps=warps,
            )
        self._group_barrier()
        return self._make_work(opts.asyncOp if opts else False)

    def reduce_scatter(self, output_tensors, input_tensors, opts=None):
        output = output_tensors[0]
        inputs = input_tensors[0]
        if (
            not self._eight
            or output.dtype not in (torch.bfloat16, torch.float32)
            or (opts and reduce_op_name(opts.reduceOp) != "sum")
            or not output.is_contiguous()
            or not all(t.is_contiguous() for t in inputs)
        ):
            return super().reduce_scatter(output_tensors, input_tensors, opts)
        self._check_device(output)
        if len(inputs) != 8 or inputs[self._rank].numel() != output.numel():
            raise self._err("reduce_scatter requires eight inputs and matching output")
        for tensor in inputs:
            self._check_device(tensor)
            if tensor.dtype != output.dtype:
                raise self._err("reduce_scatter requires matching dtypes")
        counts = tuple(t.numel() for t in inputs)
        if max(counts) * 4 > sum(counts) * 3:
            return self._dominant_reduce_scatter(output, inputs, counts, opts)
        if output.dtype == torch.float32:
            return self._fp32_reduce_scatter(output, inputs, counts, opts)
        return self._bf16_reduce_scatter(output, inputs, counts, opts)

    def _fp32_reduce_scatter(self, output, inputs, counts, opts):
        block = 4096
        starts = []
        result = 0
        for n in counts:
            starts.append(result)
            result += triton.cdiv(n, 256) * 256
        chunks = tuple(triton.cdiv(n, 7 * block) * block for n in counts)
        tiles = triton.cdiv(max(chunks), block)
        producers = min(32, tiles)
        consumers = min(96, tiles)
        if producers + consumers > min(
            128, torch.cuda.get_device_properties(output.device).multi_processor_count
        ):
            raise ValueError("split RS requires at most one block per SM")
        data_bytes = (result + triton.cdiv(max(counts), 64) * 64) * 4
        flag_bytes = (producers + consumers) * 128
        self._ensure_workspace(max(data_bytes + flag_bytes, 1))
        raw = [self._peer_scratch(p) for p in range(8)]
        bufs = tuple(cast_buffer(t, output) for t in raw)
        flags = tuple(
            t[data_bytes : data_bytes + flag_bytes].view(torch.int32) for t in raw
        )
        if producers:
            flags[self._rank].zero_()
        self._group_barrier()
        if consumers:
            order = tuple((p + self._rank) % 8 for p in range(8))
            kernels.drain_rs[(producers + consumers,)](
                tuple(inputs),
                bufs,
                flags,
                counts,
                tuple(starts),
                chunks,
                self._rank,
                result,
                tiles,
                block,
                producers,
                consumers,
                output,
                order,
                num_warps=16,
            )
        self._group_barrier()
        return self._make_work(opts.asyncOp if opts else False)

    def _bf16_reduce_scatter(self, output, inputs, counts, opts):
        block = 4096
        starts = []
        result = 0
        for n in counts:
            starts.append(result)
            result += triton.cdiv(n, 256) * 256
        chunks = tuple(triton.cdiv(n, 7 * block) * block for n in counts)
        tiles = triton.cdiv(max(chunks), block)
        producers = min(32, tiles)
        consumers = min(96, tiles)
        if producers + consumers > min(
            128, torch.cuda.get_device_properties(output.device).multi_processor_count
        ):
            raise ValueError("split RS requires at most one block per SM")
        data_bytes = (
            result + triton.cdiv(max(counts), 64) * 64
        ) * output.element_size()
        flag_bytes = producers * 128
        self._ensure_workspace(max(data_bytes + flag_bytes, 1))
        raw = [self._peer_scratch(p) for p in range(8)]
        bufs = tuple(cast_buffer(t, output) for t in raw)
        flags = tuple(
            t[data_bytes : data_bytes + flag_bytes].view(torch.int32) for t in raw
        )
        if producers:
            flags[self._rank].zero_()
        self._group_barrier()
        if consumers:
            kernels.split_rs[(producers + consumers,)](
                tuple(inputs),
                bufs,
                flags,
                counts,
                tuple(starts),
                chunks,
                self._rank,
                result,
                tiles,
                block,
                producers,
                consumers,
                num_warps=8,
            )
        self._group_barrier()
        if output.numel():
            output.reshape(-1).copy_(bufs[self._rank][result : result + output.numel()])
        self._group_barrier()
        return self._make_work(opts.asyncOp if opts else False)

    def _dominant_reduce_scatter(self, output, inputs, counts, opts):
        block, warps = 32768, 16
        starts = []
        result = 0
        for n in counts:
            starts.append(result)
            result += triton.cdiv(n, 256) * 256
        chunks = tuple(triton.cdiv(n, 7 * block) * block for n in counts)
        tiles = triton.cdiv(max(chunks), block)
        ctas = min(self._ctas, tiles)
        data_bytes = (
            result + triton.cdiv(max(counts), 64) * 64
        ) * output.element_size()
        flag_bytes = ctas * 128
        self._ensure_workspace(max(data_bytes + flag_bytes, 1))
        raw = [self._peer_scratch(p) for p in range(8)]
        bufs = tuple(cast_buffer(t, output) for t in raw)
        flags = tuple(
            t[data_bytes : data_bytes + flag_bytes].view(torch.int32) for t in raw
        )
        if ctas:
            flags[self._rank].zero_()
        self._group_barrier()
        if ctas:
            kernels.stream_rs_v2[(ctas,)](
                tuple(inputs),
                bufs,
                flags,
                counts,
                tuple(starts),
                chunks,
                self._rank,
                result,
                tiles,
                block,
                output,
                SKIP_WAIT=True,
                DRAIN=True,
                LOAD_POLL=False,
                num_warps=warps,
            )
        self._group_barrier()
        return self._make_work(opts.asyncOp if opts else False)


dist.Backend.register_backend(
    "symmem_uneven", UnevenBackend, extended_api=True, devices=["cuda"]
)
