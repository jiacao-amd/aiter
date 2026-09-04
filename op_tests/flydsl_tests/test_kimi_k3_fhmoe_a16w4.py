# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for the Kimi-K3 heterogeneous a16w4/a16w16 FHMoE."""

import pytest
import torch

import aiter
from aiter import QuantType, dtypes
from aiter.fused_moe import moe_sorting
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kimi_k3_fhmoe import (
    KIMI_K3_BLOCK_M,
    create_kimi_k3_fhmoe_workspace,
    kimi_k3_fhmoe_a16w4,
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
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

_SKIP = pytest.mark.skipif(
    get_gfx() != "gfx950" or not is_flydsl_available(),
    reason="Kimi-K3 FHMoE requires gfx950 and FlyDSL",
)


def _cos_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.double()
    y = y.double()
    return float(1.0 - 2.0 * (x * y).sum() / (x.square() + y.square()).sum())


def _make_graph_case(M: int, E: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

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
    shared_w1, shared_w2 = prepare_kimi_k3_fhmoe_shared_weights(shared_w1, shared_w2)

    return {
        "routed_x": routed_x,
        "shared_x": shared_x,
        "routed_w1": routed_w1,
        "routed_w2": routed_w2,
        "routed_w1_scale": routed_w1_scale,
        "routed_w2_scale": routed_w2_scale,
        "shared_w1": shared_w1,
        "shared_w2": shared_w2,
        "topk_ids": torch.arange(E, dtype=torch.int32, device="cuda").repeat(M, 1),
        "topk_weights": torch.softmax(
            torch.randn((M, E), dtype=torch.float32, device="cuda"), dim=-1
        ),
    }


@_SKIP
@pytest.mark.parametrize("M", [1, 2, 4, 8, 16, 32])
def test_kimi_k3_fhmoe_matches_separate_kernels(M: int):
    """The unified launches match routed FlyDSL plus a dense shared BF16 oracle."""
    E = KIMI_K3_TOPK
    torch.manual_seed(7 + M)
    torch.cuda.manual_seed(7 + M)

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

    topk_ids = torch.arange(E, dtype=torch.int32, device="cuda").repeat(M, 1)
    topk_weights = torch.softmax(
        torch.randn((M, E), dtype=torch.float32, device="cuda"), dim=-1
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

    separate_inter = torch.empty(
        (sorted_token_ids.numel(), KIMI_K3_ROUTED_INTER),
        dtype=torch.bfloat16,
        device="cuda",
    )
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
    routed_ref = torch.zeros(
        (M, KIMI_K3_ROUTED_HIDDEN), dtype=torch.bfloat16, device="cuda"
    )
    flydsl_a16w4_gemm2(
        inter_sorted_bf16=separate_inter,
        w2_u8=routed_w2,
        w2_scale_u8=routed_w2_scale,
        sorted_expert_ids=sorted_expert_ids,
        cumsum_tensor=cumsum_tensor,
        sorted_token_ids=sorted_token_ids,
        sorted_weights=sorted_weights,
        flat_out=routed_ref,
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

    shared_gate, shared_up = (shared_x @ shared_w1.t()).chunk(2, dim=-1)
    shared_mid = apply_gate_up(
        shared_gate,
        shared_up,
        "situv2",
        situ_beta=4.0,
        situ_linear_beta=25.0,
    ).to(torch.bfloat16)
    shared_ref = shared_mid @ shared_w2.t()

    workspace = create_kimi_k3_fhmoe_workspace(
        max_sorted_tokens=sorted_token_ids.numel(),
        max_tokens=M,
        device=routed_x.device,
    )
    routed_actual = torch.zeros_like(routed_ref)
    routed_actual, shared_actual = kimi_k3_fhmoe_a16w4_from_sorted(
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
        routed_out=routed_actual,
        workspace=workspace,
    )
    torch.cuda.synchronize()

    assert not routed_actual.isnan().any().item()
    assert not shared_actual.isnan().any().item()
    # Routed stage2 uses BF16 atomics.  Combining the shared workgroups into the
    # launch changes CTA scheduling and therefore the non-associative add order.
    assert _cos_diff(routed_ref.float(), routed_actual.float()) < 1e-3
    assert _cos_diff(shared_ref.float(), shared_actual.float()) < 2e-3


@_SKIP
def test_kimi_k3_fhmoe_reusable_workspace_graph_replay():
    """The complete sorting + compute operator is safe to capture and replay."""
    M = 4
    E = KIMI_K3_TOPK
    case = _make_graph_case(M, E)
    max_sorted_tokens = M * KIMI_K3_TOPK + E * KIMI_K3_BLOCK_M - KIMI_K3_TOPK
    workspace = create_kimi_k3_fhmoe_workspace(
        max_sorted_tokens=max_sorted_tokens,
        max_tokens=M,
        num_experts=E,
        device=case["routed_x"].device,
    )
    shared_out = torch.empty(
        (M, KIMI_K3_SHARED_HIDDEN),
        dtype=torch.bfloat16,
        device=case["routed_x"].device,
    )
    kwargs = {
        **case,
        "workspace": workspace,
        "shared_out": shared_out,
    }

    for _ in range(3):
        eager_routed, eager_shared = kimi_k3_fhmoe_a16w4(**kwargs)
    torch.cuda.synchronize()
    eager_routed = eager_routed.clone()
    eager_shared = eager_shared.clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_routed, graph_shared = kimi_k3_fhmoe_a16w4(**kwargs)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    assert not graph_routed.isnan().any().item()
    assert not graph_shared.isnan().any().item()
    assert _cos_diff(eager_routed.float(), graph_routed.float()) < 1e-3
    assert _cos_diff(eager_shared.float(), graph_shared.float()) < 1e-6
