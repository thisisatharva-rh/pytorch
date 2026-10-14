import triton
import triton.language as tl


@triton.jit
def reduce_slice(
    sources,
    partial,
    output,
    N: tl.constexpr,
    CHUNK: tl.constexpr,
    RANK: tl.constexpr,
    ONE_SHOT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    local = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = local if ONE_SHOT else RANK * CHUNK + local
    mask = x < N
    if not ONE_SHOT:
        mask = mask & (local < CHUNK)
    acc = tl.zeros((BLOCK,), tl.float32)
    for peer in tl.static_range(len(sources)):
        acc += tl.load(sources[(peer + RANK) % len(sources)] + x, mask, other=0).to(
            tl.float32
        )
    if ONE_SHOT:
        tl.store(output + x, acc, mask)
    else:
        tl.store(partial + local, acc, mask)


@triton.jit
def gather_slices(
    partials,
    output,
    N: tl.constexpr,
    CHUNK: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    local = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    selected = (tl.program_id(0) + RANK) % len(partials)
    for peer in tl.static_range(len(partials)):
        if selected == peer:
            x = peer * CHUNK + local
            mask = (local < CHUNK) & (x < N)
            values = tl.load(partials[peer] + local, mask, other=0)
            tl.store(output + x, values, mask)
