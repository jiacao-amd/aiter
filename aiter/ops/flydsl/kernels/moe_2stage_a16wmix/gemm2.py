# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import dpp_utils
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import (
    global_typed_ptr,
    lds_typed_ptr,
    lds_vec_load,
)

from .utils import (
    _global_i32_at,
    _mma_bf16,
    _raw,
    _udiv,
    _umod,
    make_a_loader,
    make_b_loader,
)


def _wait_lds_barrier(vmcnt=63):
    """Wait for older direct-to-LDS VMEM while preserving younger VMEM loads."""
    waitcnt = (vmcnt & 0xF) | ((vmcnt & 0x30) << 10) | (7 << 4)
    rocdl.s_waitcnt(waitcnt)
    gpu.barrier()


# gfx950 CU count; caps the persistent gemm2 grid so high-expert launches (E896) do
# not over-launch ~max_m_blocks empty CTAs.
NUM_CU = 256


# @flyc.jit is LOAD-BEARING: it AST-rewrites ``if token_id < i32_M`` into an scf.if.
# Without it the guard runs as a plain Python if (dropped at trace), so the atomic-fadd
# scatter fires on padded/OOB rows -- ~13x s2 regression (39us -> ~490us at E896).
@flyc.jit
def _atomic_bf16_epilog(
    lds_acc_base_i32,
    accm,
    arg_out,
    arg_stids,
    arg_sweights,
    m_row,
    n_block_idx,
    wave,
    lane,
    i32_M,
    BM,
    N_OUT,
    BN,
    num_waves=4,
    routed_valid_m_cap=0,
    routed_m8_meta_broadcast=False,
    route_output=False,
    topk=1,
):
    _kMChunks = BM // 16
    _rows_per_pass = 2 * num_waves
    assert BM % _rows_per_pass == 0
    assert routed_valid_m_cap in (0, 8)
    assert isinstance(routed_m8_meta_broadcast, bool)
    assert isinstance(route_output, bool)
    assert topk > 0
    if routed_valid_m_cap == 8:
        assert BM == 16
        assert num_waves == 4
    if routed_m8_meta_broadcast:
        assert routed_valid_m_cap == 8
    # At M<=8, each token can route to an expert at most once, so one expert
    # has at most eight valid sorted rows.  The opt-in specialization removes
    # the second rows 8..15 metadata/atomic pass from the BM16 epilogue.
    M_REPS = 1 if routed_valid_m_cap == 8 else BM // _rows_per_pass
    # Each wave owns the same N slice when the wide-N candidate doubles both
    # BN and the wave count (128/4 == 256/8 == 32 columns per routed wave).
    _n_per_wave = BN // num_waves
    num_acc_n = _n_per_wave // 16
    _s_count = BN // 64  # each s-iter covers 64 cols (32 lanes x vec2)
    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    lds_base_fptr = lds_typed_ptr(lds_acc_base_i32, T.f32)

    tx_i32 = fx.Int32(gpu.thread_id("x"))
    m_lane = tx_i32 // fx.Int32(32)
    n_lane = tx_i32 % fx.Int32(32)
    col_start = n_lane * fx.Int32(2)

    def _flat_buffer(arg, elem_ty, align):
        ptr = global_typed_ptr(arg, elem_ty, align=align)
        view = fx.Tensor(fx.make_view(ptr, fx.make_layout((1, 1), (1, 1))))
        return fx.rocdl.make_buffer_tensor(view, max_size=True)

    stids = _flat_buffer(arg_stids, T.i32, 4)
    sweights = _flat_buffer(arg_sweights, T.f32, 4)
    out_bf16 = _flat_buffer(arg_out, T.bf16, 4)
    out_bf16_ptr = global_typed_ptr(arg_out, T.bf16, align=2)

    load_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
    load_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    atomic_bf16x2 = fx.make_copy_atom(
        fx.rocdl.BufferAtomicPkAdd(fx.BFloat16), fx.BFloat16
    )

    def load_scalar(atom, src, index, elem_ty):
        frag = fx.make_rmem_tensor(1, elem_ty)
        fx.copy(atom, src[None, index], frag)
        return Vec(frag.load())[0]

    packed = []
    weight = []
    packed_leaders = []
    weight_bits_leaders = []
    for mr in range_constexpr(M_REPS):
        sorted_pos = m_row + fx.Int32(mr * _rows_per_pass) + m_lane
        if const_expr(routed_m8_meta_broadcast):
            # The scatter transpose maps each 32-lane half-wave to one row:
            #   row = tx//32, col = tx%32 * 2.
            # Accumulator lanes wrote different 4-row stripes below, but after
            # the LDS barrier all 32 scatter lanes consume the same row. Have
            # lane 0/32 load that row's metadata once, then broadcast within
            # the corresponding half-wave through the LDS permute crossbar.
            packed_leader = fx.Int32(0)
            weight_bits_leader = fx.Int32(0)
            if n_lane == fx.Int32(0):
                packed_leader = load_scalar(
                    load_i32, stids, sorted_pos, fx.Int32
                )
                weight_bits_leader = load_scalar(
                    load_f32, sweights, sorted_pos, fx.Float32
                ).bitcast(fx.Int32)
            # Keep the leader values live while the independent accumulator
            # stores below execute.  Broadcasting immediately would force a
            # VMEM wait and lose that latency-hiding window.
            packed_leaders.append(packed_leader)
            weight_bits_leaders.append(weight_bits_leader)
        else:
            packed.append(load_scalar(load_i32, stids, sorted_pos, fx.Int32))
            weight.append(load_scalar(load_f32, sweights, sorted_pos, fx.Float32))

    for i in range_constexpr(_kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        for J in range_constexpr(num_acc_n):
            col = wave * fx.Int32(_n_per_wave) + fx.Int32(J * 16) + lane_mod_16
            vec = Vec(accm[i][J])
            for v in range_constexpr(4):
                idx = (row_base + fx.Int32(v)) * fx.Int32(BN) + col
                lds_base_fptr[idx] = fx.Float32(vec[v])

    if const_expr(routed_m8_meta_broadcast):
        leader_byte = (lane & fx.Int32(32)) * fx.Int32(4)
        for mr in range_constexpr(M_REPS):
            packed.append(
                fx.Int32(
                    rocdl.ds_bpermute(
                        T.i32, leader_byte, packed_leaders[mr]
                    )
                )
            )
            weight_bits = fx.Int32(
                rocdl.ds_bpermute(
                    T.i32, leader_byte, weight_bits_leaders[mr]
                )
            )
            weight.append(weight_bits.bitcast(fx.Float32))

    gpu.barrier()

    def store_row(mr, row_in_block, row_base_addr):
        for s in range_constexpr(_s_count):
            idx0 = row_in_block * fx.Int32(BN) + col_start + fx.Int32(s * 64)
            v2 = Vec(
                lds_vec_load(
                    lds_acc_base_i32,
                    idx0 * fx.Int32(4),
                    Vec.make_type(2, fx.Float32),
                    fx.Float32,
                    align=8,
                )
            )
            pk = Vec.from_elements(
                [v2[0] * weight[mr], v2[1] * weight[mr]], fx.Float32
            ).to(fx.BFloat16)
            if const_expr(route_output):
                out_off = row_base_addr + fx.Int64(s * 64)
                fx.ptr_store(pk, out_bf16_ptr + out_off)
            else:
                out_frag = fx.make_rmem_tensor(2, fx.BFloat16)
                out_frag.store(pk)
                out_off = row_base_addr + fx.Int32(s * 64)
                fx.copy(atomic_bf16x2, out_frag, out_bf16[None, out_off])

    for mr in range_constexpr(M_REPS):
        row_in_block = fx.Int32(mr * _rows_per_pass) + m_lane
        token_id = packed[mr] & fx.Int32(0x00FFFFFF)
        if token_id < i32_M:
            if const_expr(route_output):
                slot_id = packed[mr] >> fx.Int32(24)
                if slot_id < fx.Int32(topk):
                    route_row = fx.Int64(
                        token_id * fx.Int32(topk) + slot_id
                    )
                    row_base_addr = (
                        route_row * fx.Int64(N_OUT)
                        + fx.Int64(n_block_idx * fx.Int32(BN) + col_start)
                    )
                    store_row(mr, row_in_block, row_base_addr)
            else:
                row_base_addr = (
                    token_id * fx.Int32(N_OUT)
                    + n_block_idx * fx.Int32(BN)
                    + col_start
                )
                store_row(mr, row_in_block, row_base_addr)


# @flyc.jit is required for the lane/token guards, for the same reason as the
# LDS-backed epilogue above.  This opt-in M<=8 specialization keeps the MFMA
# accumulator layout in registers.  Adjacent lanes exchange each row value via
# DPP so lane 0..31 can emit packed adjacent-column BF16 atomics directly:
#
#   q = lane//16, c = lane%16, parity = c&1
#   row = q*4 + 2*row_pair + parity
#   col_pair = wave*32 + J*16 + (c&~1)
#
# Even lanes emit the even row in a row-pair; odd lanes emit the odd row.  All
# DPP exchanges execute before the row-dependent token guard, so a valid lane
# never depends on an inactive neighbour.
@flyc.jit
def _atomic_bf16_register_epilog(
    accm,
    arg_out,
    arg_stids,
    arg_sweights,
    m_row,
    n_block_idx,
    wave,
    lane,
    i32_M,
    BM,
    N_OUT,
    BN,
    num_waves=4,
    routed_m8_meta_broadcast=False,
):
    assert BM == 16
    assert BN == 128
    assert num_waves == 4
    assert isinstance(routed_m8_meta_broadcast, bool)

    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    parity = lane_mod_16 & fx.Int32(1)
    pair_col = lane_mod_16 & fx.Int32(14)
    is_odd_lane = parity == fx.Int32(1)

    def _flat_buffer(arg, elem_ty, align):
        ptr = global_typed_ptr(arg, elem_ty, align=align)
        view = fx.Tensor(fx.make_view(ptr, fx.make_layout((1, 1), (1, 1))))
        return fx.rocdl.make_buffer_tensor(view, max_size=True)

    stids = _flat_buffer(arg_stids, T.i32, 4)
    sweights = _flat_buffer(arg_sweights, T.f32, 4)
    out_bf16 = _flat_buffer(arg_out, T.bf16, 4)

    load_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
    load_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    atomic_bf16x2 = fx.make_copy_atom(
        fx.rocdl.BufferAtomicPkAdd(fx.BFloat16), fx.BFloat16
    )

    def load_scalar(atom, src, index, elem_ty):
        frag = fx.make_rmem_tensor(1, elem_ty)
        fx.copy(atom, src[None, index], frag)
        return Vec(frag.load())[0]

    def dpp_swap_adjacent(value):
        value_bits = fx.Float32(value).bitcast(fx.Int32)
        swapped_bits = fx.Int32(
            dpp_utils.update_dpp_i32(
                _raw(value_bits), _raw(value_bits), 0xB1, 0xF, 0xF, True
            )
        )
        return swapped_bits.bitcast(fx.Float32)

    if const_expr(routed_m8_meta_broadcast):
        # Every routed wave needs metadata for all eight potentially valid rows.
        # Lanes 0..3 and 16..19 load one row each.  Keeping each leader in the
        # same 16-lane quarter as its consumers also remains safe if scheduling
        # sinks a weight bpermute into the later token-valid region.
        packed_leader = fx.Int32(0)
        weight_bits_leader = fx.Int32(0)
        if lane < fx.Int32(32):
            if lane_mod_16 < fx.Int32(4):
                leader_row = lane_div_16 * fx.Int32(4) + lane_mod_16
                leader_pos = m_row + leader_row
                packed_leader = load_scalar(
                    load_i32, stids, leader_pos, fx.Int32
                )
                weight_bits_leader = load_scalar(
                    load_f32, sweights, leader_pos, fx.Float32
                ).bitcast(fx.Int32)

        for row_pair in range_constexpr(2):
            even_row = lane_div_16 * fx.Int32(4) + fx.Int32(2 * row_pair)
            row_in_block = even_row + parity

            # Execute all cross-lane operations before either the output-lane or
            # token guard diverges.  Lanes 32..63 read initialized-zero leader
            # slots for rows 8..15, and are discarded by the lane guard below.
            pairs = []
            for J in range_constexpr(2):
                vec = Vec(accm[0][J])
                even_value = fx.Float32(vec[2 * row_pair])
                odd_value = fx.Float32(vec[2 * row_pair + 1])
                even_peer = dpp_swap_adjacent(even_value)
                odd_peer = dpp_swap_adjacent(odd_value)
                pair_lo = is_odd_lane.select(odd_peer, even_value)
                pair_hi = is_odd_lane.select(odd_value, even_peer)
                pairs.append((pair_lo, pair_hi))

            leader_lane = (
                lane_div_16 * fx.Int32(16)
                + fx.Int32(2 * row_pair)
                + parity
            )
            leader_byte = leader_lane * fx.Int32(4)
            packed = fx.Int32(
                rocdl.ds_bpermute(T.i32, leader_byte, packed_leader)
            )
            weight_bits = fx.Int32(
                rocdl.ds_bpermute(T.i32, leader_byte, weight_bits_leader)
            )
            weight = weight_bits.bitcast(fx.Float32)
            token_id = packed & fx.Int32(0x00FFFFFF)

            if lane < fx.Int32(32):
                if token_id < i32_M:
                    row_base_addr = (
                        token_id * fx.Int32(N_OUT)
                        + n_block_idx * fx.Int32(BN)
                        + wave * fx.Int32(32)
                        + pair_col
                    )
                    for J in range_constexpr(2):
                        pair_lo, pair_hi = pairs[J]
                        pk = Vec.from_elements(
                            [pair_lo * weight, pair_hi * weight], fx.Float32
                        ).to(fx.BFloat16)
                        out_frag = fx.make_rmem_tensor(2, fx.BFloat16)
                        out_frag.store(pk)
                        out_off = row_base_addr + fx.Int32(J * 16)
                        fx.copy(
                            atomic_bf16x2,
                            out_frag,
                            out_bf16[None, out_off],
                        )
    else:
        # q=0,1 (lane 0..31) owns all eight rows that can be valid at M<=8.
        # Each lane loads its selected row metadata and reuses it for both J
        # atomics.  This branch intentionally preserves the register-only v1.
        if lane < fx.Int32(32):
            for row_pair in range_constexpr(2):
                even_row = lane_div_16 * fx.Int32(4) + fx.Int32(2 * row_pair)
                row_in_block = even_row + parity
                sorted_pos = m_row + row_in_block
                packed = load_scalar(load_i32, stids, sorted_pos, fx.Int32)
                weight = load_scalar(load_f32, sweights, sorted_pos, fx.Float32)
                token_id = packed & fx.Int32(0x00FFFFFF)

                pairs = []
                for J in range_constexpr(2):
                    vec = Vec(accm[0][J])
                    even_value = fx.Float32(vec[2 * row_pair])
                    odd_value = fx.Float32(vec[2 * row_pair + 1])
                    even_peer = dpp_swap_adjacent(even_value)
                    odd_peer = dpp_swap_adjacent(odd_value)
                    pair_lo = is_odd_lane.select(odd_peer, even_value)
                    pair_hi = is_odd_lane.select(odd_value, even_peer)
                    pairs.append((pair_lo, pair_hi))

                if token_id < i32_M:
                    row_base_addr = (
                        token_id * fx.Int32(N_OUT)
                        + n_block_idx * fx.Int32(BN)
                        + wave * fx.Int32(32)
                        + pair_col
                    )
                    for J in range_constexpr(2):
                        pair_lo, pair_hi = pairs[J]
                        pk = Vec.from_elements(
                            [pair_lo * weight, pair_hi * weight], fx.Float32
                        ).to(fx.BFloat16)
                        out_frag = fx.make_rmem_tensor(2, fx.BFloat16)
                        out_frag.store(pk)
                        out_off = row_base_addr + fx.Int32(J * 16)
                        fx.copy(
                            atomic_bf16x2,
                            out_frag,
                            out_bf16[None, out_off],
                        )


def _gemm2_body_a16w4(
    lds_raw_ptr,
    arg_a,
    arg_bq,
    arg_bscale,
    arg_eids,
    arg_stids,
    arg_sweights,
    arg_out,
    bx_i32,
    lane,
    wave,
    i32_M,
    *,
    BM,
    SORT_BM,
    TILE_N,
    TILE_K,
    N_OUT,
    INTER,
    NE,
    b_cache_mod=2,
    w_dtype="fp4",
    use_k16=False,
    double_buffer=False,
    num_waves=4,
    routed_valid_m_cap=0,
    routed_m8_meta_broadcast=False,
    routed_m8_register_epilogue=False,
    route_output=False,
    topk=1,
):
    """a16w4/a16wi4/a16w16 stage2 body. K=inter_dim (contraction), N=model_dim (N_OUT).

    A = bf16 stage1 intermediate by SORTED position. W2 = mxfp4/int4/bf16 (see gemm1).
    Output is either a routing-weighted bf16 atomic scatter to [tokens, model_dim]
    or a non-atomic token-slot store to [tokens, topk, model_dim].
    """
    elem_bytes = 2
    KH_TILE_BYTES = TILE_K * elem_bytes
    LDS_STRIDE = TILE_K
    K = INTER
    K_TILES_TOTAL = K // TILE_K
    # Kimi-K3 routed stage2 has exactly three K128 tiles at decode BM16. The
    # existing shared allocation is already large enough for two 4-KiB A slots
    # because the accumulator epilogue reuses the same LDS after the K loop.
    _PIPE = (
        double_buffer
        and not use_k16
        and w_dtype == "fp4"
        and BM == 16
        and TILE_N // num_waves == 32
        and TILE_K == 128
        and K_TILES_TOTAL == 3
    )
    A_LDS_STAGES = 2 if _PIPE else 1
    A_SLOT_BYTES = BM * KH_TILE_BYTES
    m_repeat = BM // 16
    k_unroll = KH_TILE_BYTES // 64
    assert num_waves in (4, 8)
    assert TILE_N % num_waves == 0
    assert isinstance(routed_m8_register_epilogue, bool)
    assert isinstance(route_output, bool)
    assert topk > 0
    if routed_m8_register_epilogue:
        assert not route_output
        assert routed_valid_m_cap == 8
        assert BM == 16 and TILE_N == 128 and num_waves == 4
    # Keep each routed wave on 32 columns in the TN256/8-wave candidate.
    _n_per_wave = TILE_N // num_waves
    assert _n_per_wave >= 16 and _n_per_wave % 16 == 0
    num_acc_n = _n_per_wave // 16
    k_blocks16 = KH_TILE_BYTES // 16
    _num_n_blocks = N_OUT // TILE_N

    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)

    m_block_idx = bx_i32 // fx.Int32(_num_n_blocks)
    n_block_idx = bx_i32 % fx.Int32(_num_n_blocks)
    e = rocdl.readfirstlane(T.i32, _raw(_global_i32_at(arg_eids, m_block_idx)))
    # Keep the sorted-workspace stride independent from the compute tile height.
    # Kimi decode can compute 16 rows while each expert still occupies 32 rows.
    m_row = m_block_idx * fx.Int32(SORT_BM)
    by_n = n_block_idx * fx.Int32(TILE_N)

    # ---- B (weight) operand path: layouts + buffer resources + load closures ----
    # Shared verbatim with gemm1 (see utils.make_b_loader); stage2's N is model_dim and
    # its K is inter_dim (the contraction).
    b_loader = make_b_loader(
        arg_bq,
        arg_bscale,
        N_OUT=N_OUT,
        K=K,
        NE=NE,
        e=e,
        lane_div_16=lane_div_16,
        lane_mod_16=lane_mod_16,
        TILE_K=TILE_K,
        w_dtype=w_dtype,
        b_cache_mod=b_cache_mod,
        use_k16=use_k16,
    )

    # ---- A path (shared with gemm1, see utils.make_a_loader) -------------------
    # A row = SORTED position m_row + row_local. The default block and the first four
    # waves of the 8-wave candidate stage one BM x TILE_K tile; all waves consume it.
    # Stage2 XOR-swizzles the A region to kill LDS bank conflicts.
    c_k_div4 = (K * elem_bytes) // 4
    a_loader = make_a_loader(
        lds_raw_ptr,
        num_i32=A_LDS_STAGES * BM * LDS_STRIDE // 2,
        BM=BM,
        TILE_K=TILE_K,
        KH_TILE_BYTES=KH_TILE_BYTES,
        k_blocks16=k_blocks16,
        lane_div_16=lane_div_16,
        lane_mod_16=lane_mod_16,
        swizzle=True,
        a_ptr=arg_a,
        a_num_bytes=fx.Int64(0xFFFFFFFF),
        a_load_threads=256,
        row_base_dwords=lambda row_local: (m_row + row_local) * fx.Int32(c_k_div4),
        dma_cache_mod=b_cache_mod,
        dma_via_vgpr=use_k16,
        k_grp_base_bytes=fx.Int32(0) if _PIPE else None,
        A_SLOT_BYTES=A_SLOT_BYTES,
        active_load_threads=256 if num_waves == 8 else None,
    )

    # ---- N-column addressing (W2 cols of model_dim; wave owns _n_per_wave) ------
    n_tile_base = wave * fx.Int32(_n_per_wave)
    cols = [
        b_loader.col(by_n, n_tile_base, fx.Int32(ni * 16))
        for ni in range_constexpr(num_acc_n)
    ]

    # ---- accumulators: accm[mi][ni] f32[4] (layout the atomic epilog expects) --
    acc_layout = fx.make_layout(4, 1)
    accm = [
        [fx.make_rmem_tensor(acc_layout, fx.Float32) for _ in range(num_acc_n)]
        for _ in range(m_repeat)
    ]
    zero4 = Vec.filled(4, 0.0, fx.Float32)
    for mi in range_constexpr(m_repeat):
        for ni in range_constexpr(num_acc_n):
            accm[mi][ni].store(zero4)

    # Arch-gate: gfx950 K=32 (one MFMA/K-step); gfx942 (use_k16) splits each v8bf16 into
    # two v4bf16 halves -> TWO 16x16x16 MFMAs into the same acc (no 16x16x32 on gfx942).
    if const_expr(use_k16):
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
    else:
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))

    _mma = functools.partial(_mma_bf16, mma_atom, use_k16)

    def load_b_tile(base_k):
        b_raw = [b_loader.load_raw(base_k, c) for c in cols]
        b_sc = [b_loader.load_scale(base_k, c) for c in cols]
        return b_raw, b_sc

    def preload_a(read_slot):
        # Materialize every current-slot LDS read before issuing the next direct-to-LDS
        # copy. This also reuses each A fragment across both N accumulators.
        return [
            [a_loader.load(mi, ku, slot=read_slot) for ku in range_constexpr(k_unroll)]
            for mi in range_constexpr(m_repeat)
        ]

    def compute_tile(b_tile, a_frags):
        b_raw, b_sc = b_tile
        for ni in range_constexpr(num_acc_n):
            for ku in range_constexpr(k_unroll):
                bb = b_loader.upconvert(b_raw[ni], ku, b_sc[ni][ku])
                for mi in range_constexpr(m_repeat):
                    _mma(accm[mi][ni], a_frags[mi][ku], bb)

    def issue_a_tile(base_k, slot):
        rocdl.sched_barrier(0)
        a_loader.store_tile(base_k, slot=slot)
        rocdl.sched_barrier(0)

    if const_expr(not _PIPE):
        for kt in range_constexpr(K_TILES_TOTAL):
            base_k = fx.Int32(kt * TILE_K)
            a_loader.store_tile(base_k)
            b_raw = [b_loader.load_raw(base_k, c) for c in cols]
            b_sc = [b_loader.load_scale(base_k, c) for c in cols]
            gpu.barrier()
            for ni in range_constexpr(num_acc_n):
                for ku in range_constexpr(k_unroll):
                    bb = b_loader.upconvert(b_raw[ni], ku, b_sc[ni][ku])
                    for mi in range_constexpr(m_repeat):
                        a8 = a_loader.load(mi, ku)
                        _mma(accm[mi][ni], a8, bb)
            gpu.barrier()
    else:
        # One raw-weight and one scale load per N accumulator for each K128 tile.
        B_VMEM_LOADS_PER_TILE = 2 * num_acc_n * (TILE_K // 128)
        issue_a_tile(fx.Int32(0), 0)
        b_cur = load_b_tile(fx.Int32(0))
        for kt in range_constexpr(K_TILES_TOTAL):
            cur_slot = kt % A_LDS_STAGES
            _wait_lds_barrier(B_VMEM_LOADS_PER_TILE)
            a_frags = preload_a(cur_slot)
            if const_expr(kt + 1 < K_TILES_TOTAL):
                next_k = fx.Int32((kt + 1) * TILE_K)
                issue_a_tile(next_k, (kt + 1) % A_LDS_STAGES)
                b_nxt = load_b_tile(next_k)
            compute_tile(b_cur, a_frags)
            if const_expr(kt + 1 < K_TILES_TOTAL):
                b_cur = b_nxt

    # ---- epilogue: atomic bf16 scatter (routing-weighted) ---------------------
    if const_expr(routed_m8_register_epilogue):
        accm_v = [
            [accm[i][J].load().ir_value() for J in range(num_acc_n)]
            for i in range(m_repeat)
        ]
        _atomic_bf16_register_epilog(
            accm_v,
            arg_out,
            arg_stids,
            arg_sweights,
            m_row,
            n_block_idx,
            wave,
            lane,
            i32_M,
            BM,
            N_OUT,
            TILE_N,
            num_waves,
            routed_m8_meta_broadcast,
        )
    else:
        # The legacy path reuses the A-LDS region for its f32 accumulator
        # transpose.  Wait for every wave's final A read before overwriting it.
        gpu.barrier()
        lds_acc_base_i32 = fx.Int32(fx.ptrtoint(lds_raw_ptr))
        accm_v = [
            [accm[i][J].load().ir_value() for J in range(num_acc_n)]
            for i in range(m_repeat)
        ]
        _atomic_bf16_epilog(
            lds_acc_base_i32,
            accm_v,
            arg_out,
            arg_stids,
            arg_sweights,
            m_row,
            n_block_idx,
            wave,
            lane,
            i32_M,
            BM,
            N_OUT,
            TILE_N,
            num_waves,
            routed_valid_m_cap,
            routed_m8_meta_broadcast,
            route_output,
            topk,
        )


def gemm2_a16w4_grid(BM, *, N_OUT, TILE_N, max_m_blocks, persist=False):
    """Flattened launch grid for a16w4 gemm2.

    Non-persistent (default): one CTA per (m-block x n-block) tile over padded
    ``max_m_blocks``. Persistent: cap to ``min(total_work, NUM_CU)`` CTAs (only when
    padded work > ``NUM_CU*4``); each CTA loops over its real work-tiles.
    """
    total_work = int(max_m_blocks) * (N_OUT // TILE_N)
    if persist and total_work > NUM_CU * 4:
        return min(total_work, NUM_CU)
    return total_work


@functools.cache
def compile_gemm2_a16w4_port(
    BM=32,
    *,
    NE,
    N_OUT,
    D_INTER,
    TILE_N=256,
    TILE_K=256,
    xcd_swizzle=1,
    b_cache_mod=2,
    waves_per_eu=None,
    w_dtype="fp4",
    persist=False,
    use_k16,
):
    """a16w4/a16wi4/a16w16 (bf16 intermediate A x mxfp4/int4/bf16 W2) stage2 builder.

    N_OUT = model_dim (down-proj output). D_INTER = inter_dim (contraction). Output
    bf16 [tokens, model_dim] via atomic (routing-weighted) scatter.

    ``xcd_swizzle`` (>0) bijectively round-robins the launch index across the 8 XCDs to
    balance per-XCD/HBM traffic (gemm2 is HBM-bound), + optional M-group swizzle for
    per-XCD L2 locality (group = xcd_swizzle m-blocks).
    """
    assert w_dtype in (
        "fp4",
        "int4",
        "bf16",
    ), f"w_dtype must be 'mxfp4', 'int4' or 'bf16', got {w_dtype!r}"
    # Arch-gate K=16 (gfx942) vs K=32 (gfx950); resolved by the caller and passed in.
    _use_k16 = use_k16
    _K = D_INTER
    assert _K % TILE_K == 0, f"D_INTER (K) must be a multiple of {TILE_K}, got {_K}"
    assert (
        N_OUT % TILE_N == 0
    ), f"model_dim (N_OUT) must be a multiple of {TILE_N}, got {N_OUT}"
    # 4 waves split TILE_N (TILE_N//4 cols each) -> num_acc_n = (TILE_N//4)//16.
    # num_acc_n==0 makes every accumulate/store loop empty -> silent all-zero
    # output that times fast (e.g. TILE_N=32). Require TILE_N >= 64.
    assert (
        TILE_N // 4
    ) >= 16, f"TILE_N//4 must be >= 16 (num_acc_n>=1), got TILE_N={TILE_N}"
    assert BM % 16 == 0, f"BM must be a multiple of 16, got {BM}"
    _num_n_blocks = N_OUT // TILE_N
    KH_TILE_BYTES = TILE_K * 2

    # LDS: A tile (BM x TILE_K bf16) then f32 accumulator region (BM x TILE_N f32).
    _a_bytes = BM * KH_TILE_BYTES
    _acc_bytes = BM * TILE_N * 4  # f32 accumulator region
    _lds_bytes = _a_bytes + _acc_bytes

    _wd_tag = "" if w_dtype == "fp4" else f"_{w_dtype}"
    _name = f"gemm2_a16w4{_wd_tag}_port_ne{NE}_h{N_OUT}_i{_K}_bm{BM}_tn{TILE_N}"
    if b_cache_mod != 2:
        _name += f"_bcm{b_cache_mod}"
    if xcd_swizzle > 0:
        _name += f"_xcd{xcd_swizzle}"
    if waves_per_eu:
        _name += f"_w{waves_per_eu}"
    if persist:
        _name += "_persist"

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, _lds_bytes, 16]

    @flyc.kernel(name=_name, known_block_size=[256, 1, 1])
    def gemm2_kernel(
        arg_a: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        arg_out: fx.Int64,
    ):
        lds_raw_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        tx_i32 = fx.Int32(gpu.thread_id("x"))
        bx_i32 = fx.Int32(gpu.block_id("x"))
        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))
        cumsum0 = _global_i32_at(arg_cumsum, fx.Int32(0))
        total_m_blocks = cumsum0 // fx.Int32(BM)
        bound = total_m_blocks * fx.Int32(_num_n_blocks)

        # Bijective XCD round-robin over valid tiles [0, bound) to balance per-XCD/HBM
        # traffic; xcd_swizzle>0 also M-group-swizzles for per-XCD L2 locality.
        _NXCD = 8
        _xq = _udiv(bound, _NXCD)
        _xr = _umod(bound, _NXCD)
        _SW = xcd_swizzle

        def _xcd_np(pid):
            xc = _umod(pid, _NXCD)
            wgid = (
                xc * _xq
                + fx.Int32(arith.minsi(_raw(xc), _raw(_xr)))
                + _udiv(pid, _NXCD)
            )
            if const_expr(_SW <= 0):
                return wgid
            _ng = fx.Int32(_SW * _num_n_blocks)
            group_id = wgid // _ng
            first_pid_m = group_id * fx.Int32(_SW)
            remaining_m = total_m_blocks - first_pid_m
            group_size_m = fx.Int32(arith.minsi(_raw(remaining_m), _raw(fx.Int32(_SW))))
            wig = wgid % _ng
            m_block = first_pid_m + (wig % group_size_m)
            n_block = wig // group_size_m
            return m_block * fx.Int32(_num_n_blocks) + n_block

        def _run_tile(tile):
            _gemm2_body_a16w4(
                lds_raw_ptr,
                arg_a,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_stids,
                arg_sweights,
                arg_out,
                tile,
                lane,
                wave,
                i32_M,
                BM=BM,
                SORT_BM=BM,
                TILE_N=TILE_N,
                TILE_K=TILE_K,
                N_OUT=N_OUT,
                INTER=_K,
                NE=NE,
                b_cache_mod=b_cache_mod,
                w_dtype=w_dtype,
                use_k16=_use_k16,
            )

        if const_expr(persist):
            # Persistent CU-limited grid (~NUM_CU CTAs): each CTA does tile bx_i32 then
            # strides by grid size over [0, bound); _xcd_np maps every visited index, so
            # each tile runs once (same mapping as non-persistent). Loop-top barrier
            # separates the prev tile's epilog LDS from the next tile's A-DMA.
            grid_nb = fx.Int32(gpu.grid_dim.x)
            if bx_i32 < bound:
                _run_tile(_xcd_np(bx_i32))
            for iv in range(bx_i32 + grid_nb, bound, gpu.grid_dim.x):
                gpu.barrier()
                _run_tile(_xcd_np(fx.Int32(iv)))
        else:
            if bx_i32 < bound:
                _run_tile(_xcd_np(bx_i32))

    @flyc.jit
    def launch_gemm2(
        arg_a: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        i32_grid: fx.Int32,
        arg_out: fx.Int64,
        stream: fx.Stream,
    ):
        grid_x = fx.Int64(i32_grid)
        gemm2_kernel(
            arg_a,
            arg_bq,
            arg_bscale,
            arg_eids,
            arg_cumsum,
            arg_stids,
            arg_sweights,
            i32_M,
            i32_max_m_blocks,
            arg_out,
            value_attrs={"rocdl.waves_per_eu": waves_per_eu} if waves_per_eu else None,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm2
