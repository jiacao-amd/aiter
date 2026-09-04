# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 TP8/B1 heterogeneous MoE helpers.

This intentionally narrow prototype validates the part of Kimi-K3 that cannot
be represented by the existing DeepSeek-V4 FHMoE ABI:

* routed experts consume a 3584-wide latent tensor;
* shared experts consume the original 7168-wide tensor; and
* stage 2 produces separate 3584-wide routed and 7168-wide shared outputs.

The production Kimi checkpoint uses packed MXFP4 weights.  This prototype uses
BF16 weights so the dual-input/dual-output scheduling contract can be validated
independently before adding packed-weight loading and tuned kernels.

The shared-expert-only entry point is also used by the production MXFP4
adapter: the routed branch stays on AITER's native MXFP4 MoE kernel while this
kernel computes Kimi's unquantized BF16 shared branch on an auxiliary stream.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from aiter.jit.utils.chip_info import get_gfx_runtime

_ROUTED_HIDDEN = 3584
_SHARED_HIDDEN = 7168
_INTERMEDIATE_PER_TP_RANK = 384
_TOPK = 16
_SHARED_EXPERTS = 2
_SHARED_INTERMEDIATE_PER_TP_RANK = _SHARED_EXPERTS * _INTERMEDIATE_PER_TP_RANK
_MAX_SHARED_BATCH = 4
_FUSED_INPUT_PROJECTION = (
    _ROUTED_HIDDEN + 2 * _SHARED_INTERMEDIATE_PER_TP_RANK
)
_SITU_BETA = 4.0
_SITU_LINEAR_BETA = 25.0
# 3584 / 14 = 256 and 7168 / 14 = 512.  This gives the B1 GEMV enough
# independent workgroups while keeping every split aligned to a 128-wide tile.
_STAGE1_SPLIT_K = 14
_STAGE1_PARTIAL_SHAPE = (
    _TOPK + _SHARED_EXPERTS,
    2,
    _STAGE1_SPLIT_K,
    _INTERMEDIATE_PER_TP_RANK,
)
_ROUTED_INTERMEDIATE_SHAPE = (_TOPK, _INTERMEDIATE_PER_TP_RANK)
_SHARED_INTERMEDIATE_SHAPE = (_SHARED_EXPERTS, _INTERMEDIATE_PER_TP_RANK)
_STAGE1_PARTIAL_ELEMENTS = (
    (_TOPK + _SHARED_EXPERTS) * 2 * _STAGE1_SPLIT_K * _INTERMEDIATE_PER_TP_RANK
)
_ROUTED_INTERMEDIATE_ELEMENTS = _TOPK * _INTERMEDIATE_PER_TP_RANK
_SHARED_INTERMEDIATE_ELEMENTS = _SHARED_EXPERTS * _INTERMEDIATE_PER_TP_RANK
_STAGE1_PARTIAL_BYTES = _STAGE1_PARTIAL_ELEMENTS * 4
_ROUTED_INTERMEDIATE_BYTES = _ROUTED_INTERMEDIATE_ELEMENTS * 2
_SHARED_INTERMEDIATE_BYTES = _SHARED_INTERMEDIATE_ELEMENTS * 2
_WORKSPACE_BYTES = (
    _STAGE1_PARTIAL_BYTES + _ROUTED_INTERMEDIATE_BYTES + _SHARED_INTERMEDIATE_BYTES
)
_SHARED_STAGE1_PARTIAL_SHAPE = (
    2,
    _STAGE1_SPLIT_K,
    _SHARED_INTERMEDIATE_PER_TP_RANK,
)
_SHARED_STAGE1_PARTIAL_ELEMENTS = (
    2 * _STAGE1_SPLIT_K * _SHARED_INTERMEDIATE_PER_TP_RANK
)
_SHARED_STAGE1_PARTIAL_BYTES_PER_TOKEN = _SHARED_STAGE1_PARTIAL_ELEMENTS * 4
_SHARED_INTERMEDIATE_BYTES_PER_TOKEN = _SHARED_INTERMEDIATE_PER_TP_RANK * 2
_SHARED_WORKSPACE_BYTES_PER_TOKEN = (
    _SHARED_STAGE1_PARTIAL_BYTES_PER_TOKEN
    + _SHARED_INTERMEDIATE_BYTES_PER_TOKEN
)


@triton.jit
def _tanh(x):
    return 2.0 * tl.sigmoid(2.0 * x) - 1.0


@triton.jit
def _situ(gate, up, BETA: tl.constexpr, LINEAR_BETA: tl.constexpr):
    gate = BETA * _tanh(gate / BETA) * tl.sigmoid(gate)
    up = LINEAR_BETA * _tanh(up / LINEAR_BETA)
    return gate * up


@triton.jit
def _kimi_k3_fhmoe_stage1_bf16(
    routed_x,
    shared_x,
    routed_w1,
    shared_w1,
    topk_ids,
    routed_intermediate,
    shared_intermediate,
    ROUTED_HIDDEN: tl.constexpr,
    SHARED_HIDDEN: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    TOPK: tl.constexpr,
    SHARED_EXPERTS: tl.constexpr,
    SITU_BETA: tl.constexpr,
    SITU_LINEAR_BETA: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Exact serial control retained for the checked-in before/after benchmark.
    route = tl.program_id(0)
    block_n = tl.program_id(1)
    offsets_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offsets_n < INTERMEDIATE

    gate = tl.zeros((BLOCK_N,), dtype=tl.float32)
    up = tl.zeros((BLOCK_N,), dtype=tl.float32)

    if route < TOPK:
        expert = tl.load(topk_ids + route).to(tl.int64)
        for k_start in range(0, ROUTED_HIDDEN, BLOCK_K):
            offsets_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offsets_k < ROUTED_HIDDEN
            x = tl.load(routed_x + offsets_k, mask=mask_k, other=0.0).to(tl.float32)
            gate_w = tl.load(
                routed_w1
                + (expert * (2 * INTERMEDIATE) + offsets_n[:, None]) * ROUTED_HIDDEN
                + offsets_k[None, :],
                mask=mask_n[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)
            up_w = tl.load(
                routed_w1
                + (expert * (2 * INTERMEDIATE) + INTERMEDIATE + offsets_n[:, None])
                * ROUTED_HIDDEN
                + offsets_k[None, :],
                mask=mask_n[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)
            gate += tl.sum(gate_w * x[None, :], axis=1)
            up += tl.sum(up_w * x[None, :], axis=1)

        activated = _situ(
            gate,
            up,
            BETA=SITU_BETA,
            LINEAR_BETA=SITU_LINEAR_BETA,
        )
        tl.store(
            routed_intermediate + route * INTERMEDIATE + offsets_n,
            activated,
            mask=mask_n,
        )
    else:
        shared_expert = route - TOPK
        if shared_expert < SHARED_EXPERTS:
            for k_start in range(0, SHARED_HIDDEN, BLOCK_K):
                offsets_k = k_start + tl.arange(0, BLOCK_K)
                mask_k = offsets_k < SHARED_HIDDEN
                x = tl.load(shared_x + offsets_k, mask=mask_k, other=0.0).to(tl.float32)
                gate_w = tl.load(
                    shared_w1
                    + (shared_expert * INTERMEDIATE + offsets_n[:, None])
                    * SHARED_HIDDEN
                    + offsets_k[None, :],
                    mask=mask_n[:, None] & mask_k[None, :],
                    other=0.0,
                ).to(tl.float32)
                up_w = tl.load(
                    shared_w1
                    + (
                        SHARED_EXPERTS * INTERMEDIATE
                        + shared_expert * INTERMEDIATE
                        + offsets_n[:, None]
                    )
                    * SHARED_HIDDEN
                    + offsets_k[None, :],
                    mask=mask_n[:, None] & mask_k[None, :],
                    other=0.0,
                ).to(tl.float32)
                gate += tl.sum(gate_w * x[None, :], axis=1)
                up += tl.sum(up_w * x[None, :], axis=1)

            activated = _situ(
                gate,
                up,
                BETA=SITU_BETA,
                LINEAR_BETA=SITU_LINEAR_BETA,
            )
            tl.store(
                shared_intermediate + shared_expert * INTERMEDIATE + offsets_n,
                activated,
                mask=mask_n,
            )


@triton.jit
def _kimi_k3_fhmoe_stage1_split_projection_bf16(
    routed_x,
    shared_x,
    routed_w1,
    shared_w1,
    topk_ids,
    partials,
    ROUTED_HIDDEN: tl.constexpr,
    SHARED_HIDDEN: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    TOPK: tl.constexpr,
    SHARED_EXPERTS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    route = tl.program_id(0)
    projection_split = tl.program_id(1)
    block_n = tl.program_id(2)
    projection = projection_split // SPLIT_K
    split = projection_split % SPLIT_K
    offsets_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

    if route < TOPK:
        expert = tl.load(topk_ids + route).to(tl.int64)
        split_size = ROUTED_HIDDEN // SPLIT_K
        for local_k in range(0, split_size, BLOCK_K):
            offsets_k = split * split_size + local_k + tl.arange(0, BLOCK_K)
            x = tl.load(routed_x + offsets_k).to(tl.float32)
            weight = tl.load(
                routed_w1
                + (
                    expert * (2 * INTERMEDIATE)
                    + projection * INTERMEDIATE
                    + offsets_n[:, None]
                )
                * ROUTED_HIDDEN
                + offsets_k[None, :],
                # Expert weights are streamed once; bypassing L1 avoids
                # evicting the repeatedly used activation vector.
                cache_modifier=".cg",
            ).to(tl.float32)
            accumulator += tl.sum(weight * x[None, :], axis=1)
    else:
        shared_expert = route - TOPK
        if shared_expert < SHARED_EXPERTS:
            split_size = SHARED_HIDDEN // SPLIT_K
            for local_k in range(0, split_size, BLOCK_K):
                offsets_k = split * split_size + local_k + tl.arange(0, BLOCK_K)
                x = tl.load(shared_x + offsets_k).to(tl.float32)
                weight = tl.load(
                    shared_w1
                    + (
                        projection * (SHARED_EXPERTS * INTERMEDIATE)
                        + shared_expert * INTERMEDIATE
                        + offsets_n[:, None]
                    )
                    * SHARED_HIDDEN
                    + offsets_k[None, :],
                    # Keep the shared activation resident while its much
                    # larger one-use expert weights stream through L2.
                    cache_modifier=".cg",
                ).to(tl.float32)
                accumulator += tl.sum(weight * x[None, :], axis=1)

    partial_offset = (
        (route * 2 + projection) * SPLIT_K + split
    ) * INTERMEDIATE + offsets_n
    tl.store(partials + partial_offset, accumulator)


@triton.jit
def _kimi_k3_fhmoe_stage1_split_reduce_bf16(
    partials,
    routed_intermediate,
    shared_intermediate,
    INTERMEDIATE: tl.constexpr,
    TOPK: tl.constexpr,
    SPLIT_K: tl.constexpr,
    SITU_BETA: tl.constexpr,
    SITU_LINEAR_BETA: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    route = tl.program_id(0)
    block_n = tl.program_id(1)
    offsets_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    gate = tl.zeros((BLOCK_N,), dtype=tl.float32)
    up = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for split in range(SPLIT_K):
        gate += tl.load(
            partials + ((route * 2) * SPLIT_K + split) * INTERMEDIATE + offsets_n
        )
        up += tl.load(
            partials + ((route * 2 + 1) * SPLIT_K + split) * INTERMEDIATE + offsets_n
        )

    activated = _situ(
        gate,
        up,
        BETA=SITU_BETA,
        LINEAR_BETA=SITU_LINEAR_BETA,
    )
    if route < TOPK:
        tl.store(
            routed_intermediate + route * INTERMEDIATE + offsets_n,
            activated,
        )
    else:
        tl.store(
            shared_intermediate + (route - TOPK) * INTERMEDIATE + offsets_n,
            activated,
        )


@triton.jit
def _kimi_k3_fhmoe_stage2_bf16(
    routed_intermediate,
    shared_intermediate,
    routed_w2,
    shared_w2,
    topk_ids,
    topk_weights,
    routed_output,
    shared_output,
    ROUTED_HIDDEN: tl.constexpr,
    SHARED_HIDDEN: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    TOPK: tl.constexpr,
    SHARED_EXPERTS: tl.constexpr,
    ROUTED_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    output_block = tl.program_id(0)
    offsets_i = tl.arange(0, BLOCK_K)

    if output_block < ROUTED_BLOCKS:
        offsets_n = output_block * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offsets_n < ROUTED_HIDDEN
        accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for route in range(TOPK):
            expert = tl.load(topk_ids + route).to(tl.int64)
            route_weight = tl.load(topk_weights + route).to(tl.float32)
            expert_accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for i_start in range(0, INTERMEDIATE, BLOCK_K):
                indices_i = i_start + offsets_i
                mask_i = indices_i < INTERMEDIATE
                intermediate = tl.load(
                    routed_intermediate + route * INTERMEDIATE + indices_i,
                    mask=mask_i,
                    other=0.0,
                ).to(tl.float32)
                weight = tl.load(
                    routed_w2
                    + (expert * ROUTED_HIDDEN + offsets_n[:, None]) * INTERMEDIATE
                    + indices_i[None, :],
                    mask=mask_n[:, None] & mask_i[None, :],
                    other=0.0,
                ).to(tl.float32)
                expert_accumulator += tl.sum(weight * intermediate[None, :], axis=1)
            accumulator += route_weight * expert_accumulator

        tl.store(routed_output + offsets_n, accumulator, mask=mask_n)
    else:
        shared_block = output_block - ROUTED_BLOCKS
        offsets_n = shared_block * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offsets_n < SHARED_HIDDEN
        accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for shared_expert in range(SHARED_EXPERTS):
            for i_start in range(0, INTERMEDIATE, BLOCK_K):
                indices_i = i_start + offsets_i
                mask_i = indices_i < INTERMEDIATE
                intermediate = tl.load(
                    shared_intermediate + shared_expert * INTERMEDIATE + indices_i,
                    mask=mask_i,
                    other=0.0,
                ).to(tl.float32)
                weight = tl.load(
                    shared_w2
                    + offsets_n[:, None] * (SHARED_EXPERTS * INTERMEDIATE)
                    + shared_expert * INTERMEDIATE
                    + indices_i[None, :],
                    mask=mask_n[:, None] & mask_i[None, :],
                    other=0.0,
                ).to(tl.float32)
                accumulator += tl.sum(weight * intermediate[None, :], axis=1)

        tl.store(shared_output + offsets_n, accumulator, mask=mask_n)


@triton.jit
def _kimi_k3_fhmoe_stage2_split_routes_bf16(
    routed_intermediate,
    shared_intermediate,
    routed_w2,
    shared_w2,
    topk_ids,
    topk_weights,
    routed_partials,
    shared_output,
    ROUTED_HIDDEN: tl.constexpr,
    SHARED_HIDDEN: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    TOPK: tl.constexpr,
    SHARED_EXPERTS: tl.constexpr,
    ROUTED_BLOCKS: tl.constexpr,
    ROUTE_SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Split the long top-k loop across workgroups and emit FP32 partials."""

    program = tl.program_id(0)
    routed_programs: tl.constexpr = ROUTED_BLOCKS * ROUTE_SPLITS
    offsets_i = tl.arange(0, BLOCK_K)

    if program < routed_programs:
        output_block = program // ROUTE_SPLITS
        route_split = program % ROUTE_SPLITS
        offsets_n = output_block * BLOCK_N + tl.arange(0, BLOCK_N)
        accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)
        routes_per_split: tl.constexpr = TOPK // ROUTE_SPLITS

        for local_route in range(routes_per_split):
            route = route_split * routes_per_split + local_route
            expert = tl.load(topk_ids + route).to(tl.int64)
            route_weight = tl.load(topk_weights + route).to(tl.float32)
            expert_accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for i_start in range(0, INTERMEDIATE, BLOCK_K):
                indices_i = i_start + offsets_i
                intermediate = tl.load(
                    routed_intermediate + route * INTERMEDIATE + indices_i
                ).to(tl.float32)
                weight = tl.load(
                    routed_w2
                    + (expert * ROUTED_HIDDEN + offsets_n[:, None]) * INTERMEDIATE
                    + indices_i[None, :]
                ).to(tl.float32)
                expert_accumulator += tl.sum(
                    weight * intermediate[None, :],
                    axis=1,
                )
            accumulator += route_weight * expert_accumulator

        tl.store(
            routed_partials + route_split * ROUTED_HIDDEN + offsets_n,
            accumulator,
        )
    else:
        shared_block = program - routed_programs
        offsets_n = shared_block * BLOCK_N + tl.arange(0, BLOCK_N)
        accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)
        shared_intermediate_size: tl.constexpr = SHARED_EXPERTS * INTERMEDIATE

        for i_start in range(0, shared_intermediate_size, BLOCK_K):
            indices_i = i_start + offsets_i
            intermediate = tl.load(shared_intermediate + indices_i).to(tl.float32)
            weight = tl.load(
                shared_w2
                + offsets_n[:, None] * shared_intermediate_size
                + indices_i[None, :]
            ).to(tl.float32)
            accumulator += tl.sum(weight * intermediate[None, :], axis=1)

        tl.store(shared_output + offsets_n, accumulator)


@triton.jit
def _kimi_k3_fhmoe_stage2_reduce_routes_bf16(
    routed_partials,
    routed_output,
    ROUTED_HIDDEN: tl.constexpr,
    ROUTE_SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offsets_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for route_split in range(ROUTE_SPLITS):
        accumulator += tl.load(
            routed_partials + route_split * ROUTED_HIDDEN + offsets_n
        )
    tl.store(routed_output + offsets_n, accumulator)


@triton.jit
def _kimi_k3_shared_stage1_split_projection_bf16(
    shared_x,
    shared_w1,
    partials,
    SHARED_HIDDEN: tl.constexpr,
    SHARED_INTERMEDIATE: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    token = tl.program_id(0)
    projection_split = tl.program_id(1)
    block_n = tl.program_id(2)
    projection = projection_split // SPLIT_K
    split = projection_split % SPLIT_K
    offsets_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    split_size = SHARED_HIDDEN // SPLIT_K
    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for local_k in range(0, split_size, BLOCK_K):
        offsets_k = split * split_size + local_k + tl.arange(0, BLOCK_K)
        weight = tl.load(
            shared_w1
            + (projection * SHARED_INTERMEDIATE + offsets_n[:, None])
            * SHARED_HIDDEN
            + offsets_k[None, :],
            cache_modifier=".cg",
        ).to(tl.float32)
        x = tl.load(shared_x + token * SHARED_HIDDEN + offsets_k).to(tl.float32)
        accumulator += tl.sum(weight * x[None, :], axis=1)

    partial_offset = (
        (token * 2 + projection) * SPLIT_K + split
    ) * SHARED_INTERMEDIATE + offsets_n
    tl.store(partials + partial_offset, accumulator)


@triton.jit
def _kimi_k3_shared_stage1_split_projection_m4_bf16(
    shared_x,
    shared_w1,
    partials,
    SHARED_HIDDEN: tl.constexpr,
    SHARED_INTERMEDIATE: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """M=4 specialization using MFMA while reusing each weight tile."""

    projection_split = tl.program_id(0)
    block_n = tl.program_id(1)
    projection = projection_split // SPLIT_K
    split = projection_split % SPLIT_K
    offsets_m = tl.arange(0, 16)
    offsets_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    split_size = SHARED_HIDDEN // SPLIT_K
    accumulator = tl.zeros((16, BLOCK_N), dtype=tl.float32)

    for local_k in range(0, split_size, BLOCK_K):
        offsets_k = split * split_size + local_k + tl.arange(0, BLOCK_K)
        x = tl.load(
            shared_x
            + offsets_m[:, None] * SHARED_HIDDEN
            + offsets_k[None, :],
            mask=offsets_m[:, None] < 4,
            other=0.0,
        )
        weight = tl.load(
            shared_w1
            + (projection * SHARED_INTERMEDIATE + offsets_n[None, :])
            * SHARED_HIDDEN
            + offsets_k[:, None],
            cache_modifier=".cg",
        )
        accumulator = tl.dot(x, weight, acc=accumulator)

    partial_offsets = (
        ((offsets_m[:, None] * 2 + projection) * SPLIT_K + split)
        * SHARED_INTERMEDIATE
        + offsets_n[None, :]
    )
    tl.store(
        partials + partial_offsets,
        accumulator,
        mask=offsets_m[:, None] < 4,
    )


@triton.jit
def _kimi_k3_shared_stage1_split_reduce_bf16(
    partials,
    shared_intermediate,
    SHARED_INTERMEDIATE: tl.constexpr,
    SPLIT_K: tl.constexpr,
    SITU_BETA: tl.constexpr,
    SITU_LINEAR_BETA: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    token = tl.program_id(0)
    offsets_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    gate = tl.zeros((BLOCK_N,), dtype=tl.float32)
    up = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for split in range(SPLIT_K):
        gate += tl.load(
            partials
            + ((token * 2) * SPLIT_K + split) * SHARED_INTERMEDIATE
            + offsets_n
        )
        up += tl.load(
            partials
            + ((token * 2 + 1) * SPLIT_K + split) * SHARED_INTERMEDIATE
            + offsets_n
        )

    tl.store(
        shared_intermediate + token * SHARED_INTERMEDIATE + offsets_n,
        _situ(
            gate,
            up,
            BETA=SITU_BETA,
            LINEAR_BETA=SITU_LINEAR_BETA,
        ),
    )


@triton.jit
def _kimi_k3_shared_stage2_bf16(
    shared_intermediate,
    shared_w2,
    shared_output,
    SHARED_HIDDEN: tl.constexpr,
    SHARED_INTERMEDIATE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    token = tl.program_id(0)
    offsets_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_i = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for i_start in range(0, SHARED_INTERMEDIATE, BLOCK_K):
        indices_i = i_start + offsets_i
        weight = tl.load(
            shared_w2
            + offsets_n[:, None] * SHARED_INTERMEDIATE
            + indices_i[None, :],
            cache_modifier=".cg",
        ).to(tl.float32)
        intermediate = tl.load(
            shared_intermediate + token * SHARED_INTERMEDIATE + indices_i
        ).to(tl.float32)
        accumulator += tl.sum(weight * intermediate[None, :], axis=1)

    tl.store(shared_output + token * SHARED_HIDDEN + offsets_n, accumulator)


@triton.jit
def _kimi_k3_shared_stage2_m4_bf16(
    shared_intermediate,
    shared_w2,
    shared_output,
    SHARED_HIDDEN: tl.constexpr,
    SHARED_INTERMEDIATE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """M=4 specialization using MFMA while reusing each weight tile."""

    offsets_m = tl.arange(0, 16)
    offsets_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((16, BLOCK_N), dtype=tl.float32)

    for i_start in range(0, SHARED_INTERMEDIATE, BLOCK_K):
        indices_i = i_start + tl.arange(0, BLOCK_K)
        intermediate = tl.load(
            shared_intermediate
            + offsets_m[:, None] * SHARED_INTERMEDIATE
            + indices_i[None, :],
            mask=offsets_m[:, None] < 4,
            other=0.0,
        )
        weight = tl.load(
            shared_w2
            + offsets_n[None, :] * SHARED_INTERMEDIATE
            + indices_i[:, None],
            cache_modifier=".cg",
        )
        accumulator = tl.dot(intermediate, weight, acc=accumulator)

    tl.store(
        shared_output
        + offsets_m[:, None] * SHARED_HIDDEN
        + offsets_n[None, :],
        accumulator,
        mask=offsets_m[:, None] < 4,
    )


@triton.jit
def _kimi_k3_split_routed_situ_bf16(
    fused_projection,
    routed_output,
    shared_intermediate,
    num_tokens,
    ROUTED_HIDDEN: tl.constexpr,
    SHARED_INTERMEDIATE: tl.constexpr,
    FUSED_OUTPUT: tl.constexpr,
    SITU_BETA: tl.constexpr,
    SITU_LINEAR_BETA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Split a fused input projection and apply SiTU to the shared half."""

    token = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    token_valid = token < num_tokens

    routed_mask = token_valid & (offsets < ROUTED_HIDDEN)
    routed = tl.load(
        fused_projection + token * FUSED_OUTPUT + offsets,
        mask=routed_mask,
        other=0.0,
    )
    tl.store(
        routed_output + token * ROUTED_HIDDEN + offsets,
        routed,
        mask=routed_mask,
    )

    shared_mask = token_valid & (offsets < SHARED_INTERMEDIATE)
    shared_base = token * FUSED_OUTPUT + ROUTED_HIDDEN
    gate = tl.load(
        fused_projection + shared_base + offsets,
        mask=shared_mask,
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        fused_projection + shared_base + SHARED_INTERMEDIATE + offsets,
        mask=shared_mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        shared_intermediate + token * SHARED_INTERMEDIATE + offsets,
        _situ(
            gate,
            up,
            BETA=SITU_BETA,
            LINEAR_BETA=SITU_LINEAR_BETA,
        ),
        mask=shared_mask,
    )


def _is_gfx950() -> bool:
    try:
        return get_gfx_runtime() == "gfx950"
    except (AssertionError, KeyError, RuntimeError):
        return False


def supports_kimi_k3_fhmoe_bf16(
    routed_x: torch.Tensor,
    shared_x: torch.Tensor,
    routed_w1: torch.Tensor,
    routed_w2: torch.Tensor,
    shared_w1: torch.Tensor,
    shared_w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> bool:
    """Return whether tensors satisfy the narrow TP8/B1 prototype contract."""

    expert_count = routed_w1.shape[0] if routed_w1.ndim == 3 else 0
    tensors = (
        routed_x,
        shared_x,
        routed_w1,
        routed_w2,
        shared_w1,
        shared_w2,
        topk_ids,
        topk_weights,
    )
    return (
        torch.cuda.is_available()
        and _is_gfx950()
        and all(tensor.is_cuda for tensor in tensors)
        and len({tensor.device for tensor in tensors}) == 1
        and all(tensor.is_contiguous() for tensor in tensors)
        and routed_x.dtype == torch.bfloat16
        and shared_x.dtype == torch.bfloat16
        and routed_w1.dtype == torch.bfloat16
        and routed_w2.dtype == torch.bfloat16
        and shared_w1.dtype == torch.bfloat16
        and shared_w2.dtype == torch.bfloat16
        and topk_ids.dtype in (torch.int32, torch.int64)
        and topk_weights.dtype in (torch.float32, torch.bfloat16)
        and tuple(routed_x.shape) == (1, _ROUTED_HIDDEN)
        and tuple(shared_x.shape) == (1, _SHARED_HIDDEN)
        and tuple(routed_w1.shape)
        == (expert_count, 2 * _INTERMEDIATE_PER_TP_RANK, _ROUTED_HIDDEN)
        and tuple(routed_w2.shape)
        == (expert_count, _ROUTED_HIDDEN, _INTERMEDIATE_PER_TP_RANK)
        and tuple(shared_w1.shape)
        == (2 * _SHARED_INTERMEDIATE_PER_TP_RANK, _SHARED_HIDDEN)
        and tuple(shared_w2.shape) == (_SHARED_HIDDEN, _SHARED_INTERMEDIATE_PER_TP_RANK)
        and tuple(topk_ids.shape) == (1, _TOPK)
        and tuple(topk_weights.shape) == (1, _TOPK)
        and expert_count >= _TOPK
    )


def supports_kimi_k3_shared_expert_bf16(
    shared_x: torch.Tensor,
    shared_w1: torch.Tensor,
    shared_w2: torch.Tensor,
) -> bool:
    """Return whether tensors satisfy Kimi-K3's TP8 small-batch shared MLP."""

    tensors = (shared_x, shared_w1, shared_w2)
    batch_size = shared_x.shape[0] if shared_x.ndim == 2 else 0
    return (
        torch.cuda.is_available()
        and _is_gfx950()
        and all(tensor.is_cuda for tensor in tensors)
        and len({tensor.device for tensor in tensors}) == 1
        and all(tensor.is_contiguous() for tensor in tensors)
        and all(tensor.dtype == torch.bfloat16 for tensor in tensors)
        and 1 <= batch_size <= _MAX_SHARED_BATCH
        and tuple(shared_x.shape) == (batch_size, _SHARED_HIDDEN)
        and tuple(shared_w1.shape)
        == (2 * _SHARED_INTERMEDIATE_PER_TP_RANK, _SHARED_HIDDEN)
        and tuple(shared_w2.shape)
        == (_SHARED_HIDDEN, _SHARED_INTERMEDIATE_PER_TP_RANK)
    )


def kimi_k3_shared_expert_bf16_workspace_size(num_tokens: int = 1) -> int:
    """Return scratch bytes required by ``kimi_k3_shared_expert_bf16``."""

    if not 1 <= num_tokens <= _MAX_SHARED_BATCH:
        raise ValueError(
            f"num_tokens must be in [1, {_MAX_SHARED_BATCH}], got {num_tokens}"
        )
    return num_tokens * _SHARED_WORKSPACE_BYTES_PER_TOKEN


def kimi_k3_split_routed_situ_bf16(
    fused_projection: torch.Tensor,
    *,
    routed_out: torch.Tensor | None = None,
    shared_intermediate_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ``[routed, shared_gate, shared_up]`` and apply Kimi SiTU.

    This is the epilogue for a single BF16 input projection formed by stacking
    Kimi-K3's routed down-projection and shared gate/up projection weights.
    """

    num_tokens = fused_projection.shape[0] if fused_projection.ndim == 2 else 0
    if (
        not torch.cuda.is_available()
        or not _is_gfx950()
        or not fused_projection.is_cuda
        or fused_projection.dtype != torch.bfloat16
        or not fused_projection.is_contiguous()
        or not 1 <= num_tokens <= _MAX_SHARED_BATCH
        or tuple(fused_projection.shape)
        != (num_tokens, _FUSED_INPUT_PROJECTION)
    ):
        raise NotImplementedError(
            "kimi_k3_split_routed_situ_bf16 requires a contiguous gfx950 "
            "BF16 tensor with shape [M, 5120] and M in [1, 4]"
        )

    if routed_out is None:
        routed_out = torch.empty(
            (num_tokens, _ROUTED_HIDDEN),
            dtype=torch.bfloat16,
            device=fused_projection.device,
        )
    elif (
        routed_out.device != fused_projection.device
        or routed_out.dtype != torch.bfloat16
        or not routed_out.is_contiguous()
        or tuple(routed_out.shape) != (num_tokens, _ROUTED_HIDDEN)
    ):
        raise ValueError(
            "routed_out must be contiguous BF16 shape [M, 3584] on the "
            "input device"
        )

    if shared_intermediate_out is None:
        shared_intermediate_out = torch.empty(
            (num_tokens, _SHARED_INTERMEDIATE_PER_TP_RANK),
            dtype=torch.bfloat16,
            device=fused_projection.device,
        )
    elif (
        shared_intermediate_out.device != fused_projection.device
        or shared_intermediate_out.dtype != torch.bfloat16
        or not shared_intermediate_out.is_contiguous()
        or tuple(shared_intermediate_out.shape)
        != (num_tokens, _SHARED_INTERMEDIATE_PER_TP_RANK)
    ):
        raise ValueError(
            "shared_intermediate_out must be contiguous BF16 shape [M, 768] "
            "on the input device"
        )

    return launch_kimi_k3_split_routed_situ_bf16(
        fused_projection,
        routed_out,
        shared_intermediate_out,
    )


def launch_kimi_k3_split_routed_situ_bf16(
    fused_projection: torch.Tensor,
    routed_out: torch.Tensor,
    shared_intermediate_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the prevalidated fused-input epilogue.

    Callers must satisfy the tensor contract checked by
    ``kimi_k3_split_routed_situ_bf16``. This low-overhead entry point is for
    serving integrations that validate and cache the fixed Kimi-K3 contract
    once per layer.
    """

    num_tokens = fused_projection.shape[0]
    block_size = 256
    _kimi_k3_split_routed_situ_bf16[
        (
            num_tokens,
            triton.cdiv(_ROUTED_HIDDEN, block_size),
        )
    ](
        fused_projection,
        routed_out,
        shared_intermediate_out,
        num_tokens,
        ROUTED_HIDDEN=_ROUTED_HIDDEN,
        SHARED_INTERMEDIATE=_SHARED_INTERMEDIATE_PER_TP_RANK,
        FUSED_OUTPUT=_FUSED_INPUT_PROJECTION,
        SITU_BETA=_SITU_BETA,
        SITU_LINEAR_BETA=_SITU_LINEAR_BETA,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return routed_out, shared_intermediate_out


def _shared_workspace_views(
    workspace: torch.Tensor | None,
    device: torch.device,
    num_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    workspace_bytes = kimi_k3_shared_expert_bf16_workspace_size(num_tokens)
    if workspace is None:
        workspace = torch.empty(
            workspace_bytes,
            dtype=torch.uint8,
            device=device,
        )
    elif (
        workspace.device != device
        or workspace.dtype != torch.uint8
        or not workspace.is_contiguous()
        or workspace.numel() < workspace_bytes
        or workspace.data_ptr() % 4
    ):
        raise ValueError(
            "workspace must be a contiguous, 4-byte-aligned uint8 tensor with "
            f"at least {workspace_bytes} elements on the input device"
        )

    workspace = workspace.view(-1)
    partial_end = num_tokens * _SHARED_STAGE1_PARTIAL_BYTES_PER_TOKEN
    shared_end = partial_end + num_tokens * _SHARED_INTERMEDIATE_BYTES_PER_TOKEN
    stage1_partials = (
        workspace[:partial_end]
        .view(torch.float32)
        .view(num_tokens, *_SHARED_STAGE1_PARTIAL_SHAPE)
    )
    shared_intermediate = (
        workspace[partial_end:shared_end]
        .view(torch.bfloat16)
        .view(num_tokens, _SHARED_INTERMEDIATE_PER_TP_RANK)
    )
    return stage1_partials, shared_intermediate


def kimi_k3_shared_expert_bf16(
    shared_x: torch.Tensor,
    shared_w1: torch.Tensor,
    shared_w2: torch.Tensor,
    *,
    shared_out: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run Kimi-K3's TP8 BF16 shared MLP for one to four tokens.

    The checkpoint stores two shared experts as one MLP with local intermediate
    width 768: ``7168 -> 2 * 768 -> 768 -> 7168``.
    """

    if not supports_kimi_k3_shared_expert_bf16(
        shared_x,
        shared_w1,
        shared_w2,
    ):
        raise NotImplementedError(
            "kimi_k3_shared_expert_bf16 requires contiguous gfx950 BF16 "
            "TP8 tensors with M in [1, 4], hidden=7168, and local "
            "intermediate=768"
        )

    device = shared_x.device
    num_tokens = shared_x.shape[0]
    if shared_out is None:
        shared_out = torch.empty_like(shared_x)
    elif (
        shared_out.device != device
        or shared_out.dtype != torch.bfloat16
        or not shared_out.is_contiguous()
        or tuple(shared_out.shape) != tuple(shared_x.shape)
    ):
        raise ValueError(
            "shared_out must be a contiguous BF16 tensor matching shared_x "
            "on the input device"
        )

    stage1_partials, shared_intermediate = _shared_workspace_views(
        workspace,
        device,
        num_tokens,
    )

    if num_tokens == _MAX_SHARED_BATCH:
        stage1_block_n = 16
        stage1_block_k = 64
        _kimi_k3_shared_stage1_split_projection_m4_bf16[
            (
                2 * _STAGE1_SPLIT_K,
                triton.cdiv(
                    _SHARED_INTERMEDIATE_PER_TP_RANK,
                    stage1_block_n,
                ),
            )
        ](
            shared_x,
            shared_w1,
            stage1_partials,
            SHARED_HIDDEN=_SHARED_HIDDEN,
            SHARED_INTERMEDIATE=_SHARED_INTERMEDIATE_PER_TP_RANK,
            SPLIT_K=_STAGE1_SPLIT_K,
            BLOCK_N=stage1_block_n,
            BLOCK_K=stage1_block_k,
            num_warps=4,
        )
    else:
        stage1_block_n = 16
        stage1_block_k = 128
        _kimi_k3_shared_stage1_split_projection_bf16[
            (
                num_tokens,
                2 * _STAGE1_SPLIT_K,
                triton.cdiv(
                    _SHARED_INTERMEDIATE_PER_TP_RANK,
                    stage1_block_n,
                ),
            )
        ](
            shared_x,
            shared_w1,
            stage1_partials,
            SHARED_HIDDEN=_SHARED_HIDDEN,
            SHARED_INTERMEDIATE=_SHARED_INTERMEDIATE_PER_TP_RANK,
            SPLIT_K=_STAGE1_SPLIT_K,
            BLOCK_N=stage1_block_n,
            BLOCK_K=stage1_block_k,
            num_warps=4,
        )

    stage1_reduce_block_n = 128
    _kimi_k3_shared_stage1_split_reduce_bf16[
        (
            num_tokens,
            triton.cdiv(
                _SHARED_INTERMEDIATE_PER_TP_RANK,
                stage1_reduce_block_n,
            ),
        )
    ](
        stage1_partials,
        shared_intermediate,
        SHARED_INTERMEDIATE=_SHARED_INTERMEDIATE_PER_TP_RANK,
        SPLIT_K=_STAGE1_SPLIT_K,
        SITU_BETA=_SITU_BETA,
        SITU_LINEAR_BETA=_SITU_LINEAR_BETA,
        BLOCK_N=stage1_reduce_block_n,
        num_warps=2,
    )

    if num_tokens == _MAX_SHARED_BATCH:
        stage2_block_n = 32
        stage2_block_k = 64
        _kimi_k3_shared_stage2_m4_bf16[
            (triton.cdiv(_SHARED_HIDDEN, stage2_block_n),)
        ](
            shared_intermediate,
            shared_w2,
            shared_out,
            SHARED_HIDDEN=_SHARED_HIDDEN,
            SHARED_INTERMEDIATE=_SHARED_INTERMEDIATE_PER_TP_RANK,
            BLOCK_N=stage2_block_n,
            BLOCK_K=stage2_block_k,
            num_warps=4,
        )
    else:
        stage2_block_n = 32
        stage2_block_k = 128
        _kimi_k3_shared_stage2_bf16[
            (
                num_tokens,
                triton.cdiv(_SHARED_HIDDEN, stage2_block_n),
            )
        ](
            shared_intermediate,
            shared_w2,
            shared_out,
            SHARED_HIDDEN=_SHARED_HIDDEN,
            SHARED_INTERMEDIATE=_SHARED_INTERMEDIATE_PER_TP_RANK,
            BLOCK_N=stage2_block_n,
            BLOCK_K=stage2_block_k,
            num_warps=8,
        )
    return shared_out


def kimi_k3_fhmoe_bf16_workspace_size() -> int:
    """Return the byte workspace required by ``kimi_k3_fhmoe_bf16``."""

    return _WORKSPACE_BYTES


def _workspace_views(
    workspace: torch.Tensor | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if workspace is None:
        workspace = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    elif (
        workspace.device != device
        or workspace.dtype != torch.uint8
        or not workspace.is_contiguous()
        or workspace.numel() < _WORKSPACE_BYTES
        or workspace.data_ptr() % 4
    ):
        raise ValueError(
            "workspace must be a contiguous, 4-byte-aligned uint8 tensor with "
            f"at least {_WORKSPACE_BYTES} elements on the input device"
        )

    workspace = workspace.view(-1)
    partial_end = _STAGE1_PARTIAL_BYTES
    routed_end = partial_end + _ROUTED_INTERMEDIATE_BYTES
    shared_end = routed_end + _SHARED_INTERMEDIATE_BYTES

    stage1_partials = (
        workspace[:partial_end].view(torch.float32).view(_STAGE1_PARTIAL_SHAPE)
    )
    routed_intermediate = (
        workspace[partial_end:routed_end]
        .view(torch.bfloat16)
        .view(_ROUTED_INTERMEDIATE_SHAPE)
    )
    shared_intermediate = (
        workspace[routed_end:shared_end]
        .view(torch.bfloat16)
        .view(_SHARED_INTERMEDIATE_SHAPE)
    )
    return stage1_partials, routed_intermediate, shared_intermediate


def kimi_k3_fhmoe_bf16(
    routed_x: torch.Tensor,
    shared_x: torch.Tensor,
    routed_w1: torch.Tensor,
    routed_w2: torch.Tensor,
    shared_w1: torch.Tensor,
    shared_w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    routed_out: torch.Tensor | None = None,
    shared_out: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the Kimi-K3 TP8/B1 BF16 FHMoE expert prototype.

    ``shared_w1`` and ``shared_w2`` use the native widened KimiMLP layout:
    ``[2 * 768, 7168]`` (gate followed by up) and ``[7168, 768]``.
    Pass a uint8 tensor of ``kimi_k3_fhmoe_bf16_workspace_size()`` bytes to
    reuse scratch storage across calls.

    The returned routed tensor is still in Kimi's 3584-wide latent space.  The
    caller must apply the model's all-reduce, RMSNorm, and 3584->7168
    up-projection before adding the returned shared tensor.
    """

    if not supports_kimi_k3_fhmoe_bf16(
        routed_x,
        shared_x,
        routed_w1,
        routed_w2,
        shared_w1,
        shared_w2,
        topk_ids,
        topk_weights,
    ):
        raise NotImplementedError(
            "kimi_k3_fhmoe_bf16 requires contiguous gfx950 BF16 TP8/B1 "
            "tensors with routed H=3584, shared H=7168, local I=384, "
            "top-k=16, and two shared experts"
        )

    device = routed_x.device
    if routed_out is None:
        routed_out = torch.empty(
            (1, _ROUTED_HIDDEN), dtype=torch.bfloat16, device=device
        )
    if shared_out is None:
        shared_out = torch.empty(
            (1, _SHARED_HIDDEN), dtype=torch.bfloat16, device=device
        )

    expected_outputs = (
        (routed_out, (1, _ROUTED_HIDDEN)),
        (shared_out, (1, _SHARED_HIDDEN)),
    )
    if any(
        output.device != device
        or output.dtype != torch.bfloat16
        or not output.is_contiguous()
        or tuple(output.shape) != shape
        for output, shape in expected_outputs
    ):
        raise ValueError(
            "output tensors must be contiguous BF16 tensors on the input device"
        )

    stage1_partials, routed_intermediate, shared_intermediate = _workspace_views(
        workspace,
        device,
    )

    stage1_block_n = 16
    stage1_block_k = 128
    _kimi_k3_fhmoe_stage1_split_projection_bf16[
        (
            _TOPK + _SHARED_EXPERTS,
            2 * _STAGE1_SPLIT_K,
            triton.cdiv(_INTERMEDIATE_PER_TP_RANK, stage1_block_n),
        )
    ](
        routed_x,
        shared_x,
        routed_w1,
        shared_w1,
        topk_ids,
        stage1_partials,
        ROUTED_HIDDEN=_ROUTED_HIDDEN,
        SHARED_HIDDEN=_SHARED_HIDDEN,
        INTERMEDIATE=_INTERMEDIATE_PER_TP_RANK,
        TOPK=_TOPK,
        SHARED_EXPERTS=_SHARED_EXPERTS,
        SPLIT_K=_STAGE1_SPLIT_K,
        BLOCK_N=stage1_block_n,
        BLOCK_K=stage1_block_k,
        num_warps=4,
    )
    stage1_reduce_block_n = 128
    _kimi_k3_fhmoe_stage1_split_reduce_bf16[
        (
            _TOPK + _SHARED_EXPERTS,
            triton.cdiv(_INTERMEDIATE_PER_TP_RANK, stage1_reduce_block_n),
        )
    ](
        stage1_partials,
        routed_intermediate,
        shared_intermediate,
        INTERMEDIATE=_INTERMEDIATE_PER_TP_RANK,
        TOPK=_TOPK,
        SPLIT_K=_STAGE1_SPLIT_K,
        SITU_BETA=_SITU_BETA,
        SITU_LINEAR_BETA=_SITU_LINEAR_BETA,
        BLOCK_N=stage1_reduce_block_n,
        num_warps=2,
    )

    # Split the top-16 routed experts over four workgroups.  The old Stage 1
    # partial buffer is dead after the SiTU reduction, so reuse its first
    # 4 * ROUTED_HIDDEN FP32 values for the Stage 2 route partials.
    stage2_route_splits = 4
    stage2_block_n = 32
    stage2_block_k = 128
    routed_blocks = triton.cdiv(_ROUTED_HIDDEN, stage2_block_n)
    shared_blocks = triton.cdiv(_SHARED_HIDDEN, stage2_block_n)
    routed_partials = stage1_partials.view(-1)
    _kimi_k3_fhmoe_stage2_split_routes_bf16[
        (routed_blocks * stage2_route_splits + shared_blocks,)
    ](
        routed_intermediate,
        shared_intermediate,
        routed_w2,
        shared_w2,
        topk_ids,
        topk_weights,
        routed_partials,
        shared_out,
        ROUTED_HIDDEN=_ROUTED_HIDDEN,
        SHARED_HIDDEN=_SHARED_HIDDEN,
        INTERMEDIATE=_INTERMEDIATE_PER_TP_RANK,
        TOPK=_TOPK,
        SHARED_EXPERTS=_SHARED_EXPERTS,
        ROUTED_BLOCKS=routed_blocks,
        ROUTE_SPLITS=stage2_route_splits,
        BLOCK_N=stage2_block_n,
        BLOCK_K=stage2_block_k,
        num_warps=8,
    )
    stage2_reduce_block_n = 256
    _kimi_k3_fhmoe_stage2_reduce_routes_bf16[
        (triton.cdiv(_ROUTED_HIDDEN, stage2_reduce_block_n),)
    ](
        routed_partials,
        routed_out,
        ROUTED_HIDDEN=_ROUTED_HIDDEN,
        ROUTE_SPLITS=stage2_route_splits,
        BLOCK_N=stage2_reduce_block_n,
        num_warps=4,
    )
    return routed_out, shared_out
