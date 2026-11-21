import triton
import triton.language as tl


@triton.jit
def ag_pipeline(
    inp,
    bufs,
    flags,
    outs,
    counts: tl.constexpr,
    chunks: tl.constexpr,
    R: tl.constexpr,
    SLOT: tl.constexpr,
    TILES: tl.constexpr,
    B: tl.constexpr,
    order: tl.constexpr,
):
    pid = tl.program_id(0)
    lane = tl.arange(0, B)
    for tile in range(pid, TILES, tl.num_programs(0)):
        x = tile * B + lane
        if tile * B < chunks[R]:
            for p in tl.static_range(8):
                if p != R:
                    part = p if p < R else p - 1
                    src = part * chunks[R] + x
                    mask = (x < chunks[R]) & (src < counts[R])
                    v = tl.load(inp + src, mask, other=0)
                    tl.store(bufs[p] + R * SLOT + x, v, mask)
                    tl.store(outs[R] + src, v, mask)
                    tl.debug_barrier()
                    flag = flags[p] + (R * tl.num_programs(0) + pid) * 32
                    tl.atomic_xchg(flag, tile + 1, sem="release", scope="sys")
        for owner in tl.static_range(8):
            if owner != R:
                if tile * B < chunks[owner]:
                    for p in tl.static_range(8):
                        if order[p] != owner:
                            part = order[p] if order[p] < owner else order[p] - 1
                            dst = part * chunks[owner] + x
                            if part * chunks[owner] + tile * B < counts[owner]:
                                flag = (
                                    flags[order[p]]
                                    + (owner * tl.num_programs(0) + pid) * 32
                                )
                                while (
                                    tl.atomic_add(flag, 0, sem="acquire", scope="sys")
                                    < tile + 1
                                ):
                                    pass
                                tl.debug_barrier()
                                mask = (x < chunks[owner]) & (dst < counts[owner])
                                v = tl.load(
                                    bufs[order[p]] + owner * SLOT + x,
                                    mask,
                                    other=0,
                                    cache_modifier=".cg",
                                )
                                tl.store(outs[owner] + dst, v, mask)


@triton.jit
def stream_rs_v2(
    inputs,
    bufs,
    flags,
    counts: tl.constexpr,
    starts: tl.constexpr,
    chunks: tl.constexpr,
    R: tl.constexpr,
    RESULT: tl.constexpr,
    TILES: tl.constexpr,
    B: tl.constexpr,
    output,
    SKIP_WAIT: tl.constexpr,
    DRAIN: tl.constexpr,
    LOAD_POLL: tl.constexpr,
):
    pid = tl.program_id(0)
    lane = tl.arange(0, B)
    for tile in range(pid, TILES, tl.num_programs(0)):
        x = tile * B + lane
        for owner in tl.static_range(8):
            if tile * B < chunks[owner]:
                for part in tl.static_range(7):
                    src = part * chunks[owner] + x
                    mask = (x < chunks[owner]) & (src < counts[owner])
                    v = tl.load(inputs[owner] + src, mask, other=0)
                    tl.store(bufs[R] + starts[owner] + src, v, mask)
        tl.debug_barrier()
        tl.atomic_xchg(flags[R] + pid * 32, tile + 1, sem="release", scope="sys")
        work = tl.full((), False, tl.int1)
        for owner in tl.static_range(8):
            if owner != R:
                part = R if R < owner else R - 1
                work = work | (
                    (tile * B < chunks[owner])
                    & (part * chunks[owner] + tile * B < counts[owner])
                )
        if not SKIP_WAIT or work:
            for p in tl.static_range(8):
                if p != R:
                    ready = 0
                    while ready < tile + 1:
                        if LOAD_POLL:
                            ready = tl.inline_asm_elementwise(
                                "ld.acquire.sys.global.u32 $0, [$1];",
                                constraints="=r,l",
                                args=[flags[p] + pid * 32],
                                dtype=tl.int32,
                                is_pure=False,
                                pack=1,
                            )
                        else:
                            ready = tl.atomic_add(
                                flags[p] + pid * 32, 0, sem="acquire", scope="sys"
                            )
            tl.debug_barrier()
        for owner in tl.static_range(8):
            if owner != R:
                part = R if R < owner else R - 1
                dst = part * chunks[owner] + x
                if tile * B < chunks[owner]:
                    mask = (x < chunks[owner]) & (dst < counts[owner])
                    acc = tl.full((B,), 0, tl.float32)
                    for p in tl.static_range(8):
                        v = tl.load(
                            bufs[p] + starts[owner] + dst,
                            mask,
                            other=0,
                            cache_modifier=".cg",
                        )
                        acc += v.to(tl.float32)
                    tl.store(bufs[owner] + RESULT + dst, acc, mask)
        if DRAIN:
            tl.debug_barrier()
            tl.atomic_xchg(
                flags[R] + pid * 32 + 1, tile + 1, sem="release", scope="sys"
            )
            if tile * B < chunks[R]:
                for peer in tl.static_range(8):
                    if peer != R:
                        part = peer if peer < R else peer - 1
                        dst = part * chunks[R] + x
                        if part * chunks[R] + tile * B < counts[R]:
                            while (
                                tl.atomic_add(
                                    flags[peer] + pid * 32 + 1,
                                    0,
                                    sem="acquire",
                                    scope="sys",
                                )
                                < tile + 1
                            ):
                                pass
                            tl.debug_barrier()
                            mask = (x < chunks[R]) & (dst < counts[R])
                            value = tl.load(
                                bufs[R] + RESULT + dst,
                                mask,
                                other=0,
                                cache_modifier=".cg",
                            )
                            tl.store(output + dst, value, mask)


@triton.jit
def split_rs(
    inputs,
    bufs,
    flags,
    counts: tl.constexpr,
    starts: tl.constexpr,
    chunks: tl.constexpr,
    R: tl.constexpr,
    RESULT: tl.constexpr,
    TILES: tl.constexpr,
    B: tl.constexpr,
    PRODUCERS: tl.constexpr,
    CONSUMERS: tl.constexpr,
):
    pid = tl.program_id(0)
    lane = tl.arange(0, B)
    if pid < PRODUCERS:
        for tile in range(pid, TILES, PRODUCERS):
            x = tile * B + lane
            for owner in tl.static_range(8):
                if tile * B < chunks[owner]:
                    for part in tl.static_range(7):
                        src = part * chunks[owner] + x
                        mask = (x < chunks[owner]) & (src < counts[owner])
                        value = tl.load(inputs[owner] + src, mask, other=0)
                        tl.store(bufs[R] + starts[owner] + src, value, mask)
            tl.debug_barrier()
            tl.atomic_xchg(flags[R] + pid * 32, tile + 1, sem="release", scope="sys")
    else:
        for tile in range(pid - PRODUCERS, TILES, CONSUMERS):
            x = tile * B + lane
            producer = tile % PRODUCERS
            for peer in tl.static_range(8):
                while (
                    tl.atomic_add(
                        flags[peer] + producer * 32, 0, sem="acquire", scope="sys"
                    )
                    < tile + 1
                ):
                    pass
            tl.debug_barrier()
            for owner in tl.static_range(8):
                if owner != R:
                    part = R if R < owner else R - 1
                    dst = part * chunks[owner] + x
                    if tile * B < chunks[owner]:
                        mask = (x < chunks[owner]) & (dst < counts[owner])
                        acc = tl.full((B,), 0, tl.float32)
                        for peer in tl.static_range(8):
                            value = tl.load(
                                bufs[peer] + starts[owner] + dst,
                                mask,
                                other=0,
                                cache_modifier=".cg",
                            )
                            acc += value.to(tl.float32)
                        tl.store(bufs[owner] + RESULT + dst, acc, mask)


@triton.jit
def drain_rs(
    inputs,
    bufs,
    flags,
    counts: tl.constexpr,
    starts: tl.constexpr,
    chunks: tl.constexpr,
    R: tl.constexpr,
    RESULT: tl.constexpr,
    TILES: tl.constexpr,
    B: tl.constexpr,
    PRODUCERS: tl.constexpr,
    CONSUMERS: tl.constexpr,
    output,
    order: tl.constexpr,
):
    pid = tl.program_id(0)
    lane = tl.arange(0, B)
    if pid < PRODUCERS:
        for tile in range(pid, TILES, PRODUCERS):
            x = tile * B + lane
            for owner in tl.static_range(8):
                if owner != R and tile * B < chunks[owner]:
                    for part in tl.static_range(7):
                        src = part * chunks[owner] + x
                        mask = (x < chunks[owner]) & (src < counts[owner])
                        value = tl.load(inputs[owner] + src, mask, other=0)
                        tl.store(bufs[R] + starts[owner] + src, value, mask)
            tl.debug_barrier()
            tl.atomic_xchg(flags[R] + pid * 32, tile + 1, sem="release", scope="sys")
    else:
        for tile in range(pid - PRODUCERS, TILES, CONSUMERS):
            x = tile * B + lane
            producer = tile % PRODUCERS
            for step in tl.static_range(8):
                while (
                    tl.atomic_add(
                        flags[order[step]] + producer * 32,
                        0,
                        sem="acquire",
                        scope="sys",
                    )
                    < tile + 1
                ):
                    pass
            tl.debug_barrier()
            for owner in tl.static_range(8):
                if owner != R:
                    part = R if R < owner else R - 1
                    dst = part * chunks[owner] + x
                    if tile * B < chunks[owner]:
                        mask = (x < chunks[owner]) & (dst < counts[owner])
                        acc = tl.full((B,), 0, tl.float32)
                        for step in tl.static_range(8):
                            if order[step] != owner:
                                value = tl.load(
                                    bufs[order[step]] + starts[owner] + dst,
                                    mask,
                                    other=0,
                                    cache_modifier=".cg",
                                )
                                acc += value
                        tl.store(bufs[owner] + RESULT + dst, acc, mask)
            tl.debug_barrier()
            tl.atomic_xchg(flags[R] + pid * 32, tile + 1, sem="release", scope="sys")
            consumer = pid
            if tile * B < chunks[R]:
                for peer in tl.static_range(8):
                    if peer != R:
                        part = peer if peer < R else peer - 1
                        dst = part * chunks[R] + x
                        if part * chunks[R] + tile * B < counts[R]:
                            while (
                                tl.atomic_add(
                                    flags[peer] + consumer * 32,
                                    0,
                                    sem="acquire",
                                    scope="sys",
                                )
                                < tile + 1
                            ):
                                pass
                            tl.debug_barrier()
                            mask = (x < chunks[R]) & (dst < counts[R])
                            partial = tl.load(
                                bufs[R] + RESULT + dst,
                                mask,
                                other=0,
                                cache_modifier=".cg",
                            )
                            local = tl.load(inputs[R] + dst, mask, other=0)
                            tl.store(output + dst, partial + local, mask)
