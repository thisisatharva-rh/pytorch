"""BF16 communication with FP32 accumulation and one final BF16 conversion."""

import os

import triton

import torch
import torch.distributed as dist
from torch.distributed._pysymmem.backend import (
    cast_buffer,
    reduce_op_name,
    SymmemBackend,
)

from . import _all_reduce_kernels as kernels


class FP32AccumCandidate2Backend(SymmemBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not 1 <= self._size <= 8:
            raise RuntimeError("candidate 2 all-reduce supports 1 through 8 ranks")
        self._ar_block = int(os.environ.get("AR_BLOCK", "16384"))
        self._ar_warps = int(os.environ.get("AR_WARPS", "8"))
        self._one_shot_bytes = int(os.environ.get("AR_ONE_SHOT_BYTES", "65536"))
        self._gather_block = int(os.environ.get("AR_GATHER_BLOCK", "16384"))
        self._gather_warps = int(os.environ.get("AR_GATHER_WARPS", "4"))
        self._ar_views = {}
        self.ar_dispatch = {}
        if self._ar_block < 256 or self._ar_block & (self._ar_block - 1):
            raise ValueError("AR_BLOCK must be a power of two >= 256")
        if self._ar_warps not in (4, 8, 16):
            raise ValueError("AR_WARPS must be 4, 8, or 16")

    def allreduce(self, tensor_list, opts=None):
        if len(tensor_list) != 1:
            raise RuntimeError("candidate 2 all-reduce accepts one tensor")
        tensor = tensor_list[0]
        self._check_device(tensor)
        op = opts.reduceOp if opts else dist.ReduceOp.SUM
        if (
            reduce_op_name(op) != "sum"
            or tensor.dtype not in (torch.bfloat16, torch.float32)
            or not tensor.is_contiguous()
        ):
            raise RuntimeError(
                "candidate 2 precision contract requires contiguous BF16/FP32 SUM"
            )
        n = tensor.numel()
        async_op = opts.asyncOp if opts else False
        if n == 0:
            return self._make_work(async_op)
        chunk = triton.cdiv(triton.cdiv(n, self._size), 256) * 256
        offset = triton.cdiv(n, 256) * 256
        self._ensure_workspace((offset + chunk) * tensor.element_size())
        key = (tensor.dtype, n)
        if key not in self._ar_views:
            sources = tuple(
                cast_buffer(self._peer_scratch(peer), tensor)
                for peer in range(self._size)
            )
            partials = tuple(source[offset : offset + chunk] for source in sources)
            self._ar_views[key] = (sources, partials)
        sources, partials = self._ar_views[key]
        sources[self._rank][:n].copy_(tensor.reshape(-1))
        self._group_barrier()
        one_shot = n * tensor.element_size() <= self._one_shot_bytes
        tiles = triton.cdiv(n if one_shot else chunk, self._ar_block)
        kernels.reduce_slice[(tiles,)](
            sources,
            partials[self._rank],
            tensor,
            n,
            chunk,
            self._rank,
            one_shot,
            self._ar_block,
            num_warps=self._ar_warps,
        )
        if not one_shot:
            self._group_barrier()
            kernels.gather_slices[(self._size, triton.cdiv(chunk, self._gather_block))](
                partials,
                tensor,
                n,
                chunk,
                self._rank,
                self._gather_block,
                num_warps=self._gather_warps,
            )
        # Every peer must finish reading this generation before workspace reuse.
        self._group_barrier()
        self.ar_dispatch[key] = "one_shot_fp32" if one_shot else "two_shot_fp32"
        return self._make_work(async_op)


dist.Backend.register_backend(
    "symmem_fp32_accum_candidate2", FP32AccumCandidate2Backend, extended_api=True, devices=["cuda"]
)
