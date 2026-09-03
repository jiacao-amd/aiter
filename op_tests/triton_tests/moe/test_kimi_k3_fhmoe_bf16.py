# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.latent_moe_tail import latent_moe_tail
from aiter.ops.triton.moe.kimi_k3_fhmoe_bf16 import (
    kimi_k3_fhmoe_bf16,
    kimi_k3_fhmoe_bf16_workspace_size,
    supports_kimi_k3_fhmoe_bf16,
)

ROUTED_HIDDEN = 3584
SHARED_HIDDEN = 7168
INTERMEDIATE = 384
TOPK = 16
SHARED_EXPERTS = 2
SHARED_INTERMEDIATE = SHARED_EXPERTS * INTERMEDIATE
EXPERTS = 32
BETA = 4.0
LINEAR_BETA = 25.0


def _gfx950_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return get_gfx_runtime() == "gfx950"
    except (AssertionError, KeyError, RuntimeError):
        return False


pytestmark = pytest.mark.skipif(
    not _gfx950_available(),
    reason="Kimi-K3 FHMoE prototype requires gfx950",
)


def _inputs(seed: int = 20260903):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(shape):
        return (
            torch.randn(shape, generator=generator, device="cuda").mul_(0.02).bfloat16()
        )

    routed_x = randn((1, ROUTED_HIDDEN))
    shared_x = randn((1, SHARED_HIDDEN))
    routed_w1 = randn((EXPERTS, 2 * INTERMEDIATE, ROUTED_HIDDEN))
    routed_w2 = randn((EXPERTS, ROUTED_HIDDEN, INTERMEDIATE))
    shared_w1 = randn((2 * SHARED_INTERMEDIATE, SHARED_HIDDEN))
    shared_w2 = randn((SHARED_HIDDEN, SHARED_INTERMEDIATE))
    topk_ids = torch.randperm(EXPERTS, generator=generator, device="cuda")[:TOPK]
    topk_ids = topk_ids.to(torch.int32).unsqueeze(0).contiguous()
    topk_weights = torch.rand(
        (1, TOPK), generator=generator, device="cuda", dtype=torch.float32
    )
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    return (
        routed_x,
        shared_x,
        routed_w1,
        routed_w2,
        shared_w1,
        shared_w2,
        topk_ids,
        topk_weights,
    )


def _situ(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    gate = BETA * torch.tanh(gate / BETA) * torch.sigmoid(gate)
    up = LINEAR_BETA * torch.tanh(up / LINEAR_BETA)
    return gate * up


def _oracle(inputs):
    (
        routed_x,
        shared_x,
        routed_w1,
        routed_w2,
        shared_w1,
        shared_w2,
        topk_ids,
        topk_weights,
    ) = inputs
    selected_w1 = routed_w1[topk_ids[0].long()].float()
    selected_w2 = routed_w2[topk_ids[0].long()].float()

    routed_gate_up = torch.einsum("h,eoh->eo", routed_x[0].float(), selected_w1)
    routed_gate, routed_up = routed_gate_up.chunk(2, dim=-1)
    routed_intermediate = _situ(routed_gate, routed_up).bfloat16()
    routed_per_expert = torch.einsum(
        "ei,ehi->eh", routed_intermediate.float(), selected_w2
    )
    routed_output = (routed_per_expert * topk_weights[0].float().unsqueeze(-1)).sum(
        dim=0, keepdim=True
    )

    shared_gate_up = torch.mv(shared_w1.float(), shared_x[0].float())
    shared_gate, shared_up = shared_gate_up.chunk(2, dim=-1)
    shared_intermediate = _situ(shared_gate, shared_up).bfloat16()
    shared_output = torch.mv(shared_w2.float(), shared_intermediate.float()).unsqueeze(
        0
    )

    return routed_output.bfloat16(), shared_output.bfloat16()


def _relative_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    error = (actual.float() - expected.float()).square().mean().sqrt()
    scale = expected.float().square().mean().sqrt().clamp_min(1.0e-12)
    return float(error / scale)


@pytest.mark.parametrize("seed", [1, 17])
def test_kimi_k3_fhmoe_bf16_matches_oracle(seed: int):
    inputs = _inputs(seed)
    routed_before = inputs[0].clone()
    shared_before = inputs[1].clone()

    actual_routed, actual_shared = kimi_k3_fhmoe_bf16(*inputs)
    expected_routed, expected_shared = _oracle(inputs)
    torch.cuda.synchronize()

    assert _relative_rmse(actual_routed, expected_routed) < 0.01
    assert _relative_rmse(actual_shared, expected_shared) < 0.01
    torch.testing.assert_close(inputs[0], routed_before, rtol=0, atol=0)
    torch.testing.assert_close(inputs[1], shared_before, rtol=0, atol=0)


def test_kimi_k3_fhmoe_bf16_support_contract_is_narrow():
    inputs = _inputs()
    assert supports_kimi_k3_fhmoe_bf16(*inputs)

    wrong_shared = inputs[1][:, :ROUTED_HIDDEN].contiguous()
    assert not supports_kimi_k3_fhmoe_bf16(
        inputs[0],
        wrong_shared,
        *inputs[2:],
    )

    legacy_shared_w1 = inputs[4].view(
        SHARED_EXPERTS,
        2 * INTERMEDIATE,
        SHARED_HIDDEN,
    )
    legacy_shared_w2 = inputs[5].view(
        SHARED_EXPERTS,
        SHARED_HIDDEN,
        INTERMEDIATE,
    )
    assert not supports_kimi_k3_fhmoe_bf16(
        *inputs[:4],
        legacy_shared_w1,
        legacy_shared_w2,
        *inputs[6:],
    )


def test_kimi_k3_fhmoe_bf16_output_reuse():
    inputs = _inputs()
    routed_out = torch.empty_like(inputs[0])
    shared_out = torch.empty_like(inputs[1])

    actual_routed, actual_shared = kimi_k3_fhmoe_bf16(
        *inputs,
        routed_out=routed_out,
        shared_out=shared_out,
    )
    expected_routed, expected_shared = _oracle(inputs)
    torch.cuda.synchronize()

    assert actual_routed is routed_out
    assert actual_shared is shared_out
    assert _relative_rmse(actual_routed, expected_routed) < 0.01
    assert _relative_rmse(actual_shared, expected_shared) < 0.01


def test_kimi_k3_fhmoe_bf16_workspace_reuse():
    inputs = _inputs()
    workspace = torch.empty(
        kimi_k3_fhmoe_bf16_workspace_size(),
        dtype=torch.uint8,
        device="cuda",
    )
    routed_out = torch.empty_like(inputs[0])
    shared_out = torch.empty_like(inputs[1])

    actual_routed, actual_shared = kimi_k3_fhmoe_bf16(
        *inputs,
        routed_out=routed_out,
        shared_out=shared_out,
        workspace=workspace,
    )
    expected_routed, expected_shared = _oracle(inputs)
    torch.cuda.synchronize()

    assert actual_routed is routed_out
    assert actual_shared is shared_out
    assert _relative_rmse(actual_routed, expected_routed) < 0.01
    assert _relative_rmse(actual_shared, expected_shared) < 0.01

    with pytest.raises(ValueError, match="workspace must be"):
        kimi_k3_fhmoe_bf16(*inputs, workspace=workspace[:-1])


def test_kimi_k3_fhmoe_bf16_graph_replay_uses_updated_inputs():
    inputs = _inputs()
    routed_out = torch.empty_like(inputs[0])
    shared_out = torch.empty_like(inputs[1])
    workspace = torch.empty(
        kimi_k3_fhmoe_bf16_workspace_size(),
        dtype=torch.uint8,
        device="cuda",
    )

    kimi_k3_fhmoe_bf16(
        *inputs,
        routed_out=routed_out,
        shared_out=shared_out,
        workspace=workspace,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kimi_k3_fhmoe_bf16(
            *inputs,
            routed_out=routed_out,
            shared_out=shared_out,
            workspace=workspace,
        )

    inputs[0].add_(0.01)
    inputs[1].sub_(0.02)
    expected_routed, expected_shared = _oracle(inputs)
    graph.replay()
    torch.cuda.synchronize()

    assert _relative_rmse(routed_out, expected_routed) < 0.01
    assert _relative_rmse(shared_out, expected_shared) < 0.01


def test_kimi_k3_fhmoe_bf16_composes_with_latent_tail():
    inputs = _inputs()
    generator = torch.Generator(device="cuda").manual_seed(29)
    rms_weight = torch.randn(
        ROUTED_HIDDEN,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    up_weight = (
        torch.randn(
            (SHARED_HIDDEN, ROUTED_HIDDEN),
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )
        .mul_(ROUTED_HIDDEN**-0.5)
        .bfloat16()
    )
    epsilon = 1.0e-6

    routed, shared = kimi_k3_fhmoe_bf16(*inputs)
    actual = latent_moe_tail(routed, shared, rms_weight, up_weight, epsilon)

    expected_routed, expected_shared = _oracle(inputs)
    inverse_rms = torch.rsqrt(
        expected_routed.float().square().mean(dim=-1, keepdim=True) + epsilon
    )
    normalized = (expected_routed.float() * inverse_rms * rms_weight.float()).bfloat16()
    expected = (
        torch.mm(normalized.float(), up_weight.float().T).bfloat16().float()
        + expected_shared.float()
    ).bfloat16()
    torch.cuda.synchronize()

    assert _relative_rmse(actual, expected) < 0.01
