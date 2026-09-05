# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 heterogeneous a16w4/a16w16 two-stage MoE kernels.

Each stage is one GPU launch whose grid contains two kinds of workgroups:

* routed experts: BF16 activations x MXFP4 weights;
* two shared experts represented as one wider BF16 MLP.

The routed and shared branches intentionally keep separate input/output shapes:

* routed: 3584 -> 2x384 -> 3584, top-16 over ``NE`` experts;
* shared: 7168 -> 2x768 -> 7168, one dense expert.

Both branches use SiTUv2.  Stage 2 atomically scatters the routed branch with
router weights, while the shared branch directly stores its dense result.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.act import situ_params
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import lds_typed_ptr, lds_vec_load
from aiter.ops.flydsl.kernels.tensor_shim import _to_raw as _raw

from .moe_2stage_a16wmix.gemm1 import _gemm1_body_a16w4
from .moe_2stage_a16wmix.gemm2 import _gemm2_body_a16w4
from .moe_2stage_a16wmix.utils import (
    _global_i32_at,
    _mma_bf16,
    _udiv,
    _umod,
    make_a_loader,
    make_b_loader,
)

KIMI_K3_ROUTED_HIDDEN = 3584
KIMI_K3_SHARED_HIDDEN = 7168
KIMI_K3_ROUTED_INTER = 384
KIMI_K3_SHARED_INTER = 768
KIMI_K3_TOPK = 16


@flyc.jit
def _direct_bf16_epilog(
    lds_acc_base_i32,
    accm,
    arg_out,
    n_block_idx,
    wave,
    lane,
    i32_M,
    *,
    BM,
    N_OUT,
    BN,
):
    """Store one dense GEMM tile without routing weights or atomics."""
    m_chunks = BM // 16
    m_reps = BM // 8
    n_per_wave = BN // 4
    num_acc_n = n_per_wave // 16
    store_steps = BN // 64
    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    lds_base_fptr = lds_typed_ptr(lds_acc_base_i32, T.f32)

    tx_i32 = fx.Int32(gpu.thread_id("x"))
    m_lane = tx_i32 // fx.Int32(32)
    n_lane = tx_i32 % fx.Int32(32)
    col_start = n_lane * fx.Int32(2)

    out_rsrc = buffer_ops.create_buffer_resource_from_addr(
        _raw(fx.Int64(arg_out)),
        num_records_bytes=_raw(fx.Int64(i32_M) * fx.Int64(N_OUT * 2)),
    )

    for i in range_constexpr(m_chunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        for j in range_constexpr(num_acc_n):
            col = wave * fx.Int32(n_per_wave) + fx.Int32(j * 16) + lane_mod_16
            vec = Vec(accm[i][j])
            for v in range_constexpr(4):
                idx = (row_base + fx.Int32(v)) * fx.Int32(BN) + col
                lds_base_fptr[idx] = fx.Float32(vec[v])

    gpu.barrier()

    for mr in range_constexpr(m_reps):
        row = fx.Int32(mr * 8) + m_lane
        valid = row < i32_M
        row_base_addr = row * fx.Int32(N_OUT) + n_block_idx * fx.Int32(BN) + col_start
        for s in range_constexpr(store_steps):
            idx0 = row * fx.Int32(BN) + col_start + fx.Int32(s * 64)
            v2 = Vec(
                lds_vec_load(
                    lds_acc_base_i32,
                    idx0 * fx.Int32(4),
                    Vec.make_type(2, fx.Float32),
                    fx.Float32,
                    align=8,
                )
            )
            packed = Vec.from_elements([v2[0], v2[1]], fx.Float32).to(fx.BFloat16)
            buffer_ops.buffer_store(
                packed,
                _raw(out_rsrc),
                _raw(row_base_addr + fx.Int32(s * 64)),
                mask=valid,
            )


def _gemm2_body_a16w16_direct(
    lds_raw_ptr,
    arg_a,
    arg_b,
    arg_out,
    bx_i32,
    lane,
    wave,
    i32_M,
    *,
    BM,
    TILE_N,
    TILE_K,
    N_OUT,
    INTER,
    b_cache_mod=0,
    use_k16=False,
    shared_weight_layout="preshuffled",
    rowmajor_b_to_lds=False,
):
    """Dense BF16 stage2 body used by the Kimi shared experts."""
    elem_bytes = 2
    kh_tile_bytes = TILE_K * elem_bytes
    lds_stride = TILE_K
    k_tiles_total = INTER // TILE_K
    m_repeat = BM // 16
    k_unroll = kh_tile_bytes // 64
    n_per_wave = TILE_N // 4
    num_acc_n = n_per_wave // 16
    k_blocks16 = kh_tile_bytes // 16

    if const_expr(rowmajor_b_to_lds):
        assert shared_weight_layout == "rowmajor"
        assert TILE_K == 128
        assert TILE_N % 64 == 0

    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    n_block_idx = bx_i32
    by_n = n_block_idx * fx.Int32(TILE_N)

    # BF16 weights are preshuffled N-major.  BF16 mode ignores the scale pointer,
    # so ``arg_b`` is also passed as the harmless dummy scale address.
    b_loader = make_b_loader(
        arg_b,
        arg_b,
        N_OUT=N_OUT,
        K=INTER,
        NE=1,
        e=fx.Int32(0),
        lane_div_16=lane_div_16,
        lane_mod_16=lane_mod_16,
        TILE_K=TILE_K,
        w_dtype="bf16",
        b_cache_mod=b_cache_mod,
        use_k16=use_k16,
        bf16_layout=shared_weight_layout,
    )

    c_k_div4 = (INTER * elem_bytes) // 4
    a_loader = make_a_loader(
        lds_raw_ptr,
        num_i32=BM * lds_stride // 2,
        BM=BM,
        TILE_K=TILE_K,
        KH_TILE_BYTES=kh_tile_bytes,
        k_blocks16=k_blocks16,
        lane_div_16=lane_div_16,
        lane_mod_16=lane_mod_16,
        swizzle=True,
        a_ptr=arg_a,
        a_num_bytes=fx.Int64(BM * INTER * elem_bytes),
        a_load_threads=256,
        row_base_dwords=lambda row_local: row_local * fx.Int32(c_k_div4),
        dma_cache_mod=b_cache_mod,
        dma_via_vgpr=use_k16,
    )

    n_tile_base = wave * fx.Int32(n_per_wave)
    cols = [
        b_loader.col(by_n, n_tile_base, fx.Int32(ni * 16))
        for ni in range_constexpr(num_acc_n)
    ]

    if const_expr(rowmajor_b_to_lds):
        # The direct row-major B path makes each MFMA lane fetch a different
        # output row, producing sixteen independent 16-byte memory transactions
        # per wave.  Cooperatively stage one [TILE_N, TILE_K] weight tile with
        # contiguous 16-byte loads, then use an XOR-swizzled LDS view for the
        # MFMA fragments.  The B arena follows the A tile and is reused for each
        # K iteration.
        tx_i32 = wave * fx.Int32(64) + lane
        b_lds_base_i32 = (
            fx.Int32(fx.ptrtoint(lds_raw_ptr)) + fx.Int32(BM * TILE_K * elem_bytes)
        )
        b_lds_ptr = lds_typed_ptr(b_lds_base_i32, T.bf16, align=16)
        b_rsrc = buffer_ops.create_buffer_resource_from_addr(
            _raw(fx.Int64(arg_b)),
            num_records_bytes=N_OUT * INTER * elem_bytes,
        )
        b_vec_elems = 8
        b_vecs_per_row = TILE_K // b_vec_elems
        b_tile_vecs = TILE_N * b_vecs_per_row
        assert b_tile_vecs % 256 == 0
        b_loads_per_thread = b_tile_vecs // 256

        def stage_rowmajor_b(base_k):
            for copy_i in range_constexpr(b_loads_per_thread):
                linear_vec = tx_i32 + fx.Int32(copy_i * 256)
                row = linear_vec // fx.Int32(b_vecs_per_row)
                k_vec = linear_vec % fx.Int32(b_vecs_per_row)
                global_elem = (
                    (by_n + row) * fx.Int32(INTER)
                    + base_k
                    + k_vec * fx.Int32(b_vec_elems)
                )
                raw = buffer_ops.buffer_load(
                    b_rsrc,
                    global_elem // fx.Int32(2),
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=b_cache_mod,
                )
                swizzled_vec = k_vec ^ (row & fx.Int32(b_vecs_per_row - 1))
                lds_elem = (
                    row * fx.Int32(TILE_K)
                    + swizzled_vec * fx.Int32(b_vec_elems)
                )
                fx.ptr_store(
                    Vec(raw).bitcast(fx.BFloat16),
                    b_lds_ptr + lds_elem,
                )

        def load_rowmajor_b(ni, ku):
            row = n_tile_base + fx.Int32(ni * 16) + lane_mod_16
            k_elem = fx.Int32((ku % 4) * 8) + lane_div_16 * fx.Int32(32)
            k_vec = k_elem // fx.Int32(b_vec_elems)
            swizzled_vec = k_vec ^ (row & fx.Int32(b_vecs_per_row - 1))
            byte_offset = (
                row * fx.Int32(TILE_K)
                + swizzled_vec * fx.Int32(b_vec_elems)
            ) * fx.Int32(elem_bytes)
            return Vec(
                lds_vec_load(
                    b_lds_base_i32,
                    byte_offset,
                    Vec.make_type(b_vec_elems, fx.BFloat16),
                    fx.BFloat16,
                    align=16,
                )
            )

    acc_layout = fx.make_layout(4, 1)
    accm = [
        [fx.make_rmem_tensor(acc_layout, fx.Float32) for _ in range(num_acc_n)]
        for _ in range(m_repeat)
    ]
    zero4 = Vec.filled(4, 0.0, fx.Float32)
    for mi in range_constexpr(m_repeat):
        for ni in range_constexpr(num_acc_n):
            accm[mi][ni].store(zero4)

    if const_expr(use_k16):
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
    else:
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
    mma = functools.partial(_mma_bf16, mma_atom, use_k16)

    for kt in range_constexpr(k_tiles_total):
        base_k = fx.Int32(kt * TILE_K)
        a_loader.store_tile(base_k)
        if const_expr(rowmajor_b_to_lds):
            stage_rowmajor_b(base_k)
        else:
            b_raw = [b_loader.load_raw(base_k, col) for col in cols]
            b_scale = [b_loader.load_scale(base_k, col) for col in cols]
        gpu.barrier()
        for ni in range_constexpr(num_acc_n):
            for ku in range_constexpr(k_unroll):
                if const_expr(rowmajor_b_to_lds):
                    bb = load_rowmajor_b(ni, ku)
                else:
                    bb = b_loader.upconvert(b_raw[ni], ku, b_scale[ni][ku])
                for mi in range_constexpr(m_repeat):
                    aa = a_loader.load(mi, ku)
                    mma(accm[mi][ni], aa, bb)
        gpu.barrier()

    gpu.barrier()
    lds_acc_base_i32 = fx.Int32(fx.ptrtoint(lds_raw_ptr))
    accm_values = [
        [accm[i][j].load().ir_value() for j in range(num_acc_n)]
        for i in range(m_repeat)
    ]
    _direct_bf16_epilog(
        lds_acc_base_i32,
        accm_values,
        arg_out,
        n_block_idx,
        wave,
        lane,
        i32_M,
        BM=BM,
        N_OUT=N_OUT,
        BN=TILE_N,
    )


@functools.cache
def compile_kimi_k3_fhmoe_stage1(
    *,
    NE,
    BM=32,
    SORT_BM=32,
    TILE_N=128,
    TILE_K=256,
    routed_b_cache_mod=2,
    shared_b_cache_mod=0,
    waves_per_eu=None,
    use_k16=False,
    shared_weight_layout="preshuffled",
    routed_xcd_swizzle=0,
    routed_k_wave=1,
    shared_tile_n=None,
    shared_tile_k=None,
    shared_k_wave=None,
    shared_first=False,
):
    """Compile the unified routed-MXFP4/shared-BF16 Kimi stage1 launch."""
    assert BM in (16, 32), f"Kimi stage1 compute BM must be 16 or 32, got {BM}"
    assert SORT_BM == 32, f"Kimi stage1 sorting BM must stay 32, got {SORT_BM}"
    assert BM <= SORT_BM
    assert KIMI_K3_ROUTED_INTER % TILE_N == 0
    assert KIMI_K3_SHARED_INTER % TILE_N == 0
    assert KIMI_K3_ROUTED_HIDDEN % TILE_K == 0
    assert KIMI_K3_SHARED_HIDDEN % TILE_K == 0
    assert shared_weight_layout in ("preshuffled", "rowmajor")
    assert routed_k_wave in (1, 2, 4)
    assert KIMI_K3_ROUTED_HIDDEN % (routed_k_wave * TILE_K) == 0
    assert KIMI_K3_SHARED_HIDDEN % (routed_k_wave * TILE_K) == 0
    shared_tile_n = TILE_N if shared_tile_n is None else shared_tile_n
    shared_tile_k = TILE_K if shared_tile_k is None else shared_tile_k
    shared_k_wave = routed_k_wave if shared_k_wave is None else shared_k_wave
    assert KIMI_K3_SHARED_INTER % shared_tile_n == 0
    assert KIMI_K3_SHARED_HIDDEN % shared_tile_k == 0
    assert shared_k_wave in (1, 2, 4)
    assert KIMI_K3_SHARED_HIDDEN % (shared_k_wave * shared_tile_k) == 0
    shared_num_n_waves = 4 // shared_k_wave
    assert shared_tile_n % shared_num_n_waves == 0, (
        "shared_tile_n must divide evenly across the shared N-waves, got "
        f"shared_tile_n={shared_tile_n}, shared_k_wave={shared_k_wave}"
    )
    shared_n_per_wave = shared_tile_n // shared_num_n_waves
    assert shared_n_per_wave >= 16 and shared_n_per_wave % 16 == 0, (
        "each shared stage1 N-wave must own a positive multiple of 16 columns, got "
        f"shared_tile_n={shared_tile_n}, shared_k_wave={shared_k_wave}, "
        f"shared_n_per_wave={shared_n_per_wave}"
    )

    routed_n_blocks = KIMI_K3_ROUTED_INTER // TILE_N
    shared_grid = KIMI_K3_SHARED_INTER // shared_tile_n
    # Match the generic A16W4 stage1 builder: each K-wave owns a double-buffered
    # A tile, then the same LDS is reused for the slice-K reduction.
    routed_a_lds_bytes = routed_k_wave * 2 * BM * TILE_K * 2
    if routed_k_wave > 1:
        num_n_waves = 4 // routed_k_wave
        num_acc_n = (TILE_N // num_n_waves) // 16
        m_repeat = BM // 16
        reduce_bytes = 4 * (num_acc_n * m_repeat) * 64 * 4 * 4
        routed_lds_bytes = max(routed_a_lds_bytes, reduce_bytes)
    else:
        routed_lds_bytes = routed_a_lds_bytes
    shared_a_lds_bytes = shared_k_wave * 2 * BM * shared_tile_k * 2
    if shared_k_wave > 1:
        shared_num_n_waves = 4 // shared_k_wave
        shared_num_acc_n = (shared_tile_n // shared_num_n_waves) // 16
        shared_m_repeat = BM // 16
        shared_reduce_bytes = (
            4 * (shared_num_acc_n * shared_m_repeat) * 64 * 4 * 4
        )
        shared_lds_bytes = max(shared_a_lds_bytes, shared_reduce_bytes)
    else:
        shared_lds_bytes = shared_a_lds_bytes
    lds_bytes = max(routed_lds_bytes, shared_lds_bytes)

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, lds_bytes, 16]

    @flyc.kernel(
        name=(
            f"kimi_k3_fhmoe_s1_ne{NE}_cbm{BM}_sortbm{SORT_BM}"
            f"_tn{TILE_N}_tk{TILE_K}"
            f"_sw{shared_weight_layout}"
            f"_rxcd{routed_xcd_swizzle}_rkw{routed_k_wave}"
            f"_stn{shared_tile_n}_stk{shared_tile_k}_skw{shared_k_wave}"
            f"_sf{int(shared_first)}"
        ),
        known_block_size=[256, 1, 1],
    )
    def stage1_kernel(
        arg_routed_x: fx.Int64,
        arg_shared_x: fx.Int64,
        arg_routed_w1: fx.Int64,
        arg_routed_w1_scale: fx.Int64,
        arg_shared_w1: fx.Int64,
        arg_routed_eids: fx.Int64,
        arg_routed_cumsum: fx.Int64,
        arg_routed_mind: fx.Int64,
        arg_shared_eids: fx.Int64,
        arg_shared_cumsum: fx.Int64,
        arg_shared_mind: fx.Int64,
        i32_M: fx.Int32,
        f32_situ_beta: fx.Float32,
        f32_situ_beta_rcp: fx.Float32,
        f32_situ_linbeta: fx.Float32,
        f32_situ_linbeta_rcp: fx.Float32,
        f32_swiglu_limit: fx.Float32,
        arg_routed_inter: fx.Int64,
        arg_shared_inter: fx.Int64,
    ):
        lds_raw_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        tx_i32 = fx.Int32(gpu.thread_id("x"))
        bx_i32 = fx.Int32(gpu.block_id("x"))
        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))

        routed_blocks = _global_i32_at(
            arg_routed_cumsum, fx.Int32(0)
        ) // fx.Int32(SORT_BM)
        routed_bound = routed_blocks * fx.Int32(routed_n_blocks)
        dispatch_bx = bx_i32
        if const_expr(shared_first):
            is_shared = bx_i32 < fx.Int32(shared_grid)
            shifted_bx = bx_i32 - fx.Int32(shared_grid)
            is_routed = shifted_bx < routed_bound
            dispatch_bx = is_shared.select(
                routed_bound + bx_i32,
                is_routed.select(
                    shifted_bx,
                    routed_bound + fx.Int32(shared_grid),
                ),
            )

        if dispatch_bx < routed_bound:
            routed_bx = dispatch_bx
            if const_expr(routed_xcd_swizzle > 0):
                nxcd = 8
                xq = _udiv(routed_bound, nxcd)
                xr = _umod(routed_bound, nxcd)
                xc = _umod(dispatch_bx, nxcd)
                wgid = (
                    xc * xq
                    + fx.Int32(arith.minsi(_raw(xc), _raw(xr)))
                    + _udiv(dispatch_bx, nxcd)
                )
                ng = fx.Int32(routed_xcd_swizzle * routed_n_blocks)
                group_id = wgid // ng
                first_pid_m = group_id * fx.Int32(routed_xcd_swizzle)
                remaining_m = routed_blocks - first_pid_m
                group_size_m = fx.Int32(
                    arith.minsi(
                        _raw(remaining_m), _raw(fx.Int32(routed_xcd_swizzle))
                    )
                )
                wig = wgid % ng
                m_block = first_pid_m + (wig % group_size_m)
                n_block = wig // group_size_m
                routed_bx = m_block * fx.Int32(routed_n_blocks) + n_block
            situ = situ_params(
                fx.Float32(f32_situ_beta),
                fx.Float32(f32_situ_beta_rcp),
                fx.Float32(f32_situ_linbeta),
                fx.Float32(f32_situ_linbeta_rcp),
                fx.Float32(f32_swiglu_limit),
            )
            _gemm1_body_a16w4(
                lds_raw_ptr,
                arg_routed_x,
                arg_routed_w1,
                arg_routed_w1_scale,
                arg_routed_eids,
                arg_routed_mind,
                arg_routed_cumsum,
                arg_routed_inter,
                routed_bx,
                lane,
                wave,
                i32_M,
                situ,
                BM=BM,
                SORT_BM=SORT_BM,
                TILE_N=TILE_N,
                TILE_K=TILE_K,
                K=KIMI_K3_ROUTED_HIDDEN,
                INTER=KIMI_K3_ROUTED_INTER,
                NE=NE,
                TOPK=KIMI_K3_TOPK,
                act="situv2",
                b_cache_mod=routed_b_cache_mod,
                w_dtype="fp4",
                w_layout="standard",
                k_wave=routed_k_wave,
                use_k16=use_k16,
            )
        else:
            shared_bx = dispatch_bx - routed_bound
            if shared_bx < fx.Int32(shared_grid):
                situ = situ_params(
                    fx.Float32(f32_situ_beta),
                    fx.Float32(f32_situ_beta_rcp),
                    fx.Float32(f32_situ_linbeta),
                    fx.Float32(f32_situ_linbeta_rcp),
                    fx.Float32(f32_swiglu_limit),
                )
                _gemm1_body_a16w4(
                    lds_raw_ptr,
                    arg_shared_x,
                    arg_shared_w1,
                    arg_shared_w1,
                    arg_shared_eids,
                    arg_shared_mind,
                    arg_shared_cumsum,
                    arg_shared_inter,
                    shared_bx,
                    lane,
                    wave,
                    i32_M,
                    situ,
                    BM=BM,
                    SORT_BM=SORT_BM,
                    TILE_N=shared_tile_n,
                    TILE_K=shared_tile_k,
                    K=KIMI_K3_SHARED_HIDDEN,
                    INTER=KIMI_K3_SHARED_INTER,
                    NE=1,
                    TOPK=1,
                    act="situv2",
                    b_cache_mod=shared_b_cache_mod,
                    w_dtype="bf16",
                    w_layout=(
                        "rowmajor"
                        if shared_weight_layout == "rowmajor"
                        else "standard"
                    ),
                    k_wave=shared_k_wave,
                    use_k16=use_k16,
                )

    @flyc.jit
    def launch(
        arg_routed_x: fx.Int64,
        arg_shared_x: fx.Int64,
        arg_routed_w1: fx.Int64,
        arg_routed_w1_scale: fx.Int64,
        arg_shared_w1: fx.Int64,
        arg_routed_eids: fx.Int64,
        arg_routed_cumsum: fx.Int64,
        arg_routed_mind: fx.Int64,
        arg_shared_eids: fx.Int64,
        arg_shared_cumsum: fx.Int64,
        arg_shared_mind: fx.Int64,
        i32_M: fx.Int32,
        i32_grid: fx.Int32,
        f32_situ_beta: fx.Float32,
        f32_situ_beta_rcp: fx.Float32,
        f32_situ_linbeta: fx.Float32,
        f32_situ_linbeta_rcp: fx.Float32,
        f32_swiglu_limit: fx.Float32,
        arg_routed_inter: fx.Int64,
        arg_shared_inter: fx.Int64,
        stream: fx.Stream,
    ):
        stage1_kernel(
            arg_routed_x,
            arg_shared_x,
            arg_routed_w1,
            arg_routed_w1_scale,
            arg_shared_w1,
            arg_routed_eids,
            arg_routed_cumsum,
            arg_routed_mind,
            arg_shared_eids,
            arg_shared_cumsum,
            arg_shared_mind,
            i32_M,
            f32_situ_beta,
            f32_situ_beta_rcp,
            f32_situ_linbeta,
            f32_situ_linbeta_rcp,
            f32_swiglu_limit,
            arg_routed_inter,
            arg_shared_inter,
            value_attrs={"rocdl.waves_per_eu": waves_per_eu} if waves_per_eu else None,
        ).launch(
            grid=(fx.Int64(i32_grid), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch


@functools.cache
def compile_kimi_k3_fhmoe_stage2(
    *,
    NE,
    BM=32,
    SORT_BM=32,
    TILE_N=128,
    TILE_K=128,
    routed_b_cache_mod=0,
    shared_b_cache_mod=0,
    waves_per_eu=None,
    use_k16=False,
    shared_weight_layout="preshuffled",
    routed_xcd_swizzle=0,
    routed_persist=False,
    shared_tile_n=None,
    shared_tile_k=None,
    shared_rowmajor_b_to_lds=False,
    shared_first=False,
):
    """Compile the unified routed-MXFP4/shared-BF16 Kimi stage2 launch."""
    assert BM in (16, 32), f"Kimi stage2 compute BM must be 16 or 32, got {BM}"
    assert SORT_BM == 32, f"Kimi stage2 sorting BM must stay 32, got {SORT_BM}"
    assert BM <= SORT_BM
    assert KIMI_K3_ROUTED_HIDDEN % TILE_N == 0
    assert KIMI_K3_SHARED_HIDDEN % TILE_N == 0
    assert KIMI_K3_ROUTED_INTER % TILE_K == 0
    assert KIMI_K3_SHARED_INTER % TILE_K == 0
    assert shared_weight_layout in ("preshuffled", "rowmajor")
    shared_tile_n = TILE_N if shared_tile_n is None else shared_tile_n
    shared_tile_k = TILE_K if shared_tile_k is None else shared_tile_k
    assert KIMI_K3_SHARED_HIDDEN % shared_tile_n == 0
    assert KIMI_K3_SHARED_INTER % shared_tile_k == 0
    assert shared_tile_n >= 64 and shared_tile_n % 64 == 0, (
        "shared stage2 splits N across four waves, so shared_tile_n must be a "
        f"positive multiple of 64, got {shared_tile_n}"
    )
    if shared_rowmajor_b_to_lds:
        assert shared_weight_layout == "rowmajor"
        assert shared_tile_k == 128
    if shared_first:
        assert not routed_persist

    routed_n_blocks = KIMI_K3_ROUTED_HIDDEN // TILE_N
    shared_grid = KIMI_K3_SHARED_HIDDEN // shared_tile_n
    # Match the conservative allocation in the generic stage2 builder.
    shared_b_lds_bytes = (
        shared_tile_n * shared_tile_k * 2 if shared_rowmajor_b_to_lds else 0
    )
    lds_bytes = max(
        BM * TILE_K * 2 + BM * TILE_N * 4,
        BM * shared_tile_k * 2
        + shared_b_lds_bytes
        + BM * shared_tile_n * 4,
    )

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, lds_bytes, 16]

    @flyc.kernel(
        name=(
            f"kimi_k3_fhmoe_s2_ne{NE}_cbm{BM}_sortbm{SORT_BM}"
            f"_tn{TILE_N}_tk{TILE_K}"
            f"_sw{shared_weight_layout}"
            f"_rxcd{routed_xcd_swizzle}_rp{int(routed_persist)}"
            f"_stn{shared_tile_n}_stk{shared_tile_k}"
            f"_sblds{int(shared_rowmajor_b_to_lds)}"
            f"_sf{int(shared_first)}"
        ),
        known_block_size=[256, 1, 1],
    )
    def stage2_kernel(
        arg_routed_inter: fx.Int64,
        arg_shared_inter: fx.Int64,
        arg_routed_w2: fx.Int64,
        arg_routed_w2_scale: fx.Int64,
        arg_shared_w2: fx.Int64,
        arg_routed_eids: fx.Int64,
        arg_routed_cumsum: fx.Int64,
        arg_routed_stids: fx.Int64,
        arg_routed_sweights: fx.Int64,
        i32_M: fx.Int32,
        arg_routed_out: fx.Int64,
        arg_shared_out: fx.Int64,
    ):
        lds_raw_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        tx_i32 = fx.Int32(gpu.thread_id("x"))
        bx_i32 = fx.Int32(gpu.block_id("x"))
        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))

        routed_blocks = _global_i32_at(
            arg_routed_cumsum, fx.Int32(0)
        ) // fx.Int32(SORT_BM)
        routed_bound = routed_blocks * fx.Int32(routed_n_blocks)
        dispatch_bx = bx_i32
        if const_expr(shared_first):
            is_shared = bx_i32 < fx.Int32(shared_grid)
            shifted_bx = bx_i32 - fx.Int32(shared_grid)
            is_routed = shifted_bx < routed_bound
            dispatch_bx = is_shared.select(
                routed_bound + bx_i32,
                is_routed.select(
                    shifted_bx,
                    routed_bound + fx.Int32(shared_grid),
                ),
            )

        def routed_tile(pid):
            if const_expr(routed_xcd_swizzle <= 0):
                return pid
            nxcd = 8
            xq = _udiv(routed_bound, nxcd)
            xr = _umod(routed_bound, nxcd)
            xc = _umod(pid, nxcd)
            wgid = (
                xc * xq
                + fx.Int32(arith.minsi(_raw(xc), _raw(xr)))
                + _udiv(pid, nxcd)
            )
            ng = fx.Int32(routed_xcd_swizzle * routed_n_blocks)
            group_id = wgid // ng
            first_pid_m = group_id * fx.Int32(routed_xcd_swizzle)
            remaining_m = routed_blocks - first_pid_m
            group_size_m = fx.Int32(
                arith.minsi(
                    _raw(remaining_m), _raw(fx.Int32(routed_xcd_swizzle))
                )
            )
            wig = wgid % ng
            m_block = first_pid_m + (wig % group_size_m)
            n_block = wig // group_size_m
            return m_block * fx.Int32(routed_n_blocks) + n_block

        def run_routed(pid):
            _gemm2_body_a16w4(
                lds_raw_ptr,
                arg_routed_inter,
                arg_routed_w2,
                arg_routed_w2_scale,
                arg_routed_eids,
                arg_routed_stids,
                arg_routed_sweights,
                arg_routed_out,
                routed_tile(pid),
                lane,
                wave,
                i32_M,
                BM=BM,
                SORT_BM=SORT_BM,
                TILE_N=TILE_N,
                TILE_K=TILE_K,
                N_OUT=KIMI_K3_ROUTED_HIDDEN,
                INTER=KIMI_K3_ROUTED_INTER,
                NE=NE,
                b_cache_mod=routed_b_cache_mod,
                w_dtype="fp4",
                use_k16=use_k16,
            )

        def run_shared(shared_bx):
            _gemm2_body_a16w16_direct(
                lds_raw_ptr,
                arg_shared_inter,
                arg_shared_w2,
                arg_shared_out,
                shared_bx,
                lane,
                wave,
                i32_M,
                BM=BM,
                TILE_N=shared_tile_n,
                TILE_K=shared_tile_k,
                N_OUT=KIMI_K3_SHARED_HIDDEN,
                INTER=KIMI_K3_SHARED_INTER,
                b_cache_mod=shared_b_cache_mod,
                use_k16=use_k16,
                shared_weight_layout=shared_weight_layout,
                rowmajor_b_to_lds=shared_rowmajor_b_to_lds,
            )

        if const_expr(routed_persist):
            grid_nb = fx.Int32(gpu.grid_dim.x)
            if bx_i32 < routed_bound:
                run_routed(bx_i32)
            for iv in range(bx_i32 + grid_nb, routed_bound, gpu.grid_dim.x):
                gpu.barrier()
                run_routed(fx.Int32(iv))
            if bx_i32 < fx.Int32(shared_grid):
                gpu.barrier()
                run_shared(bx_i32)
        elif dispatch_bx < routed_bound:
            run_routed(dispatch_bx)
        else:
            shared_bx = dispatch_bx - routed_bound
            if shared_bx < fx.Int32(shared_grid):
                run_shared(shared_bx)

    @flyc.jit
    def launch(
        arg_routed_inter: fx.Int64,
        arg_shared_inter: fx.Int64,
        arg_routed_w2: fx.Int64,
        arg_routed_w2_scale: fx.Int64,
        arg_shared_w2: fx.Int64,
        arg_routed_eids: fx.Int64,
        arg_routed_cumsum: fx.Int64,
        arg_routed_stids: fx.Int64,
        arg_routed_sweights: fx.Int64,
        i32_M: fx.Int32,
        i32_grid: fx.Int32,
        arg_routed_out: fx.Int64,
        arg_shared_out: fx.Int64,
        stream: fx.Stream,
    ):
        stage2_kernel(
            arg_routed_inter,
            arg_shared_inter,
            arg_routed_w2,
            arg_routed_w2_scale,
            arg_shared_w2,
            arg_routed_eids,
            arg_routed_cumsum,
            arg_routed_stids,
            arg_routed_sweights,
            i32_M,
            arg_routed_out,
            arg_shared_out,
            value_attrs={"rocdl.waves_per_eu": waves_per_eu} if waves_per_eu else None,
        ).launch(
            grid=(fx.Int64(i32_grid), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch
