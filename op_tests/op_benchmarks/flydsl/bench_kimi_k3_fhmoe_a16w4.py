# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark Kimi-K3 routed-MXFP4/shared-BF16 unified FHMoE compute."""

from __future__ import annotations

import argparse
import statistics

import torch

import aiter
from aiter import QuantType, dtypes
from aiter.fused_moe import moe_sorting
from aiter.ops.flydsl.kimi_k3_fhmoe import (
    create_kimi_k3_fhmoe_workspace,
    kimi_k3_fhmoe_a16w4_from_sorted,
    prepare_kimi_k3_fhmoe_shared_weights,
)
from aiter.ops.flydsl.kernels.kimi_k3_fhmoe_a16w4 import (
    KIMI_K3_ROUTED_HIDDEN,
    KIMI_K3_ROUTED_INTER,
    KIMI_K3_SHARED_HIDDEN,
    KIMI_K3_SHARED_INTER,
    KIMI_K3_TOPK,
)
from aiter.ops.flydsl.kernels.moe_2stage_a16wmix import (
    flydsl_a16w4_gemm1,
    flydsl_a16w4_gemm2,
)
from aiter.ops.flydsl.moe_common import apply_gate_up
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4


def _measure(fn, warmup: int, iterations: int, samples: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    values = []
    for _ in range(samples):
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000.0 / iterations)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=1, choices=range(1, 33))
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--samples", type=int, default=7)
    args = parser.parse_args()

    M = args.tokens
    E = args.experts
    if E < KIMI_K3_TOPK:
        raise ValueError(f"--experts must be at least {KIMI_K3_TOPK}")

    torch.manual_seed(20260904)
    torch.cuda.manual_seed(20260904)
    routed_x = torch.randn(
        (M, KIMI_K3_ROUTED_HIDDEN), dtype=torch.bfloat16, device="cuda"
    )
    shared_x = torch.randn(
        (M, KIMI_K3_SHARED_HIDDEN), dtype=torch.bfloat16, device="cuda"
    )
    routed_w1_bf16 = (
        torch.randn(
            (E, 2 * KIMI_K3_ROUTED_INTER, KIMI_K3_ROUTED_HIDDEN),
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 16
    )
    routed_w2_bf16 = (
        torch.randn(
            (E, KIMI_K3_ROUTED_HIDDEN, KIMI_K3_ROUTED_INTER),
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 16
    )
    shared_w1 = (
        torch.randn(
            (2 * KIMI_K3_SHARED_INTER, KIMI_K3_SHARED_HIDDEN),
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 16
    )
    shared_w2 = (
        torch.randn(
            (KIMI_K3_SHARED_HIDDEN, KIMI_K3_SHARED_INTER),
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 16
    )

    quant = aiter.get_torch_quant(QuantType.per_1x32)
    routed_w1, routed_w1_scale = quant(routed_w1_bf16, quant_dtype=dtypes.fp4x2)
    routed_w2, routed_w2_scale = quant(routed_w2_bf16, quant_dtype=dtypes.fp4x2)
    del routed_w1_bf16, routed_w2_bf16
    routed_w1 = shuffle_weight_a16w4(
        routed_w1.view(E, 2 * KIMI_K3_ROUTED_INTER, KIMI_K3_ROUTED_HIDDEN // 2),
        16,
        False,
    )
    routed_w2 = shuffle_weight_a16w4(
        routed_w2.view(E, KIMI_K3_ROUTED_HIDDEN, KIMI_K3_ROUTED_INTER // 2),
        16,
        False,
    )
    routed_w1_scale = shuffle_scale_a16w4(routed_w1_scale, E, False)
    routed_w2_scale = shuffle_scale_a16w4(routed_w2_scale, E, False)
    shared_w1_shuffled, shared_w2_shuffled = prepare_kimi_k3_fhmoe_shared_weights(
        shared_w1, shared_w2
    )

    # Use an independent top-k set per token.  Repeating one set makes E=896
    # behave like E=16 and hides capacity-sized-grid regressions.
    topk_ids = torch.stack(
        [
            torch.randperm(E, device="cuda", dtype=torch.int64)[:KIMI_K3_TOPK]
            for _ in range(M)
        ]
    ).to(torch.int32)
    topk_weights = torch.softmax(
        torch.randn((M, KIMI_K3_TOPK), dtype=torch.float32, device="cuda"),
        dim=-1,
    )
    (
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        cumsum_tensor,
        _,
    ) = moe_sorting(
        topk_ids,
        topk_weights,
        E,
        KIMI_K3_ROUTED_HIDDEN,
        torch.bfloat16,
        32,
        accumulate=True,
    )
    capacity_blocks = int(sorted_expert_ids.numel())
    active_block_bound = min(M * KIMI_K3_TOPK, E, capacity_blocks)

    workspace = create_kimi_k3_fhmoe_workspace(
        max_sorted_tokens=sorted_token_ids.numel(),
        max_tokens=M,
        device=routed_x.device,
    )
    separate_inter = torch.empty_like(workspace.routed_inter)
    routed_baseline = torch.empty(
        (M, KIMI_K3_ROUTED_HIDDEN), dtype=torch.bfloat16, device="cuda"
    )
    routed_fused = torch.empty_like(routed_baseline)
    shared_fused = torch.empty(
        (M, KIMI_K3_SHARED_HIDDEN), dtype=torch.bfloat16, device="cuda"
    )

    def routed_only():
        routed_baseline.zero_()
        flydsl_a16w4_gemm1(
            a_bf16=routed_x,
            w1_u8=routed_w1,
            w1_scale_u8=routed_w1_scale,
            sorted_expert_ids=sorted_expert_ids,
            cumsum_tensor=cumsum_tensor,
            m_indices=sorted_token_ids,
            inter_sorted_bf16=separate_inter,
            n_tokens=M,
            NE=E,
            D_HIDDEN=KIMI_K3_ROUTED_HIDDEN,
            D_INTER=KIMI_K3_ROUTED_INTER,
            topk=KIMI_K3_TOPK,
            tile_m=32,
            tile_n=128,
            tile_k=256,
            act="situv2",
            situ_beta=4.0,
            situ_linear_beta=25.0,
        )
        flydsl_a16w4_gemm2(
            inter_sorted_bf16=separate_inter,
            w2_u8=routed_w2,
            w2_scale_u8=routed_w2_scale,
            sorted_expert_ids=sorted_expert_ids,
            cumsum_tensor=cumsum_tensor,
            sorted_token_ids=sorted_token_ids,
            sorted_weights=sorted_weights,
            flat_out=routed_baseline,
            M_logical=M,
            max_sorted=sorted_token_ids.numel(),
            NE=E,
            D_HIDDEN=KIMI_K3_ROUTED_HIDDEN,
            D_INTER=KIMI_K3_ROUTED_INTER,
            topk=KIMI_K3_TOPK,
            tile_m=32,
            tile_n=128,
            tile_k=128,
        )

    def shared_only():
        gate, up = (shared_x @ shared_w1.t()).chunk(2, dim=-1)
        middle = apply_gate_up(
            gate,
            up,
            "situv2",
            situ_beta=4.0,
            situ_linear_beta=25.0,
        ).to(torch.bfloat16)
        torch.mm(middle, shared_w2.t(), out=shared_fused)

    def separate():
        routed_only()
        shared_only()

    def fused():
        routed_fused.zero_()
        kimi_k3_fhmoe_a16w4_from_sorted(
            routed_x=routed_x,
            shared_x=shared_x,
            routed_w1=routed_w1,
            routed_w2=routed_w2,
            routed_w1_scale=routed_w1_scale,
            routed_w2_scale=routed_w2_scale,
            shared_w1=shared_w1_shuffled,
            shared_w2=shared_w2_shuffled,
            sorted_token_ids=sorted_token_ids,
            sorted_weights=sorted_weights,
            sorted_expert_ids=sorted_expert_ids,
            cumsum_tensor=cumsum_tensor,
            routed_out=routed_fused,
            workspace=workspace,
            shared_out=shared_fused,
        )

    routed_us = _measure(routed_only, args.warmup, args.iterations, args.samples)
    shared_us = _measure(shared_only, args.warmup, args.iterations, args.samples)
    separate_us = _measure(separate, args.warmup, args.iterations, args.samples)
    fused_us = _measure(fused, args.warmup, args.iterations, args.samples)

    routed_med = statistics.median(routed_us)
    shared_med = statistics.median(shared_us)
    separate_med = statistics.median(separate_us)
    fused_med = statistics.median(fused_us)
    print(f"Kimi-K3 FHMoE core M={M}, E={E}, topk={KIMI_K3_TOPK}")
    print(
        "routed block grid: "
        f"capacity={capacity_blocks}, active_bound={active_block_bound}"
    )
    print(f"routed two-stage: {routed_med:.3f} us")
    print(f"shared Torch:     {shared_med:.3f} us")
    print(f"separate total:   {separate_med:.3f} us")
    print(f"unified FHMoE:    {fused_med:.3f} us")
    print(f"speedup:          {separate_med / fused_med:.3f}x")
    print(f"routed samples:   {routed_us}")
    print(f"shared samples:   {shared_us}")
    print(f"separate samples: {separate_us}")
    print(f"unified samples:  {fused_us}")


if __name__ == "__main__":
    main()
