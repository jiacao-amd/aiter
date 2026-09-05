# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Runtime wrapper for the Kimi-K3 heterogeneous two-stage FHMoE kernels."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from flydsl.runtime.device import get_rocm_arch

from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from aiter.ops.shuffle import shuffle_weight

from .kernels.kimi_k3_fhmoe_a16w4 import (
    KIMI_K3_ROUTED_HIDDEN,
    KIMI_K3_ROUTED_INTER,
    KIMI_K3_SHARED_HIDDEN,
    KIMI_K3_SHARED_INTER,
    KIMI_K3_TOPK,
    compile_kimi_k3_fhmoe_stage1,
    compile_kimi_k3_fhmoe_stage2,
)

KIMI_K3_SORT_BLOCK_M = 32
# Backward-compatible public name: sorting/workspace rows remain fixed at 32.
KIMI_K3_BLOCK_M = KIMI_K3_SORT_BLOCK_M
KIMI_K3_MAX_DECODE_TOKENS = KIMI_K3_SORT_BLOCK_M
_STAGE1_TILE_N = 128
_STAGE2_TILE_N = 128
_GFX950_CU_COUNT = 256


def _routed_grid_block_upper_bound(
    tokens: int,
    num_experts: int,
    capacity_blocks: int,
) -> int:
    """Bound active routed blocks for decode shapes with tokens <= BLOCK_M."""
    return min(tokens * KIMI_K3_TOPK, num_experts, capacity_blocks)


def _decode_compute_block_m(tokens: int) -> int:
    """Use a half-height compute tile when no expert can receive over 16 routes."""
    if not 1 <= tokens <= KIMI_K3_MAX_DECODE_TOKENS:
        raise ValueError(
            f"tokens must be in [1, {KIMI_K3_MAX_DECODE_TOKENS}], got {tokens}"
        )
    return 16 if tokens <= 16 else KIMI_K3_SORT_BLOCK_M


def _decode_kernel_profile(tokens: int) -> dict[str, int | bool | None]:
    """Kimi-K3 A16W4 settings mirrored from the gfx950 tuned-FMoE rows."""
    if tokens <= 1:
        return dict(
            s1_tn=32, s1_tk=256, s1_wpe=4, s1_xcd=4, s1_kw=2,
            s2_tn=128, s2_tk=128, s2_bcm=2, s2_xcd=4, s2_persist=False,
        )
    if tokens <= 2:
        return dict(
            s1_tn=32, s1_tk=128, s1_wpe=4, s1_xcd=0, s1_kw=4,
            s2_tn=128, s2_tk=128, s2_bcm=2, s2_xcd=0, s2_persist=False,
        )
    if tokens <= 4:
        return dict(
            s1_tn=32, s1_tk=256, s1_wpe=4, s1_xcd=4, s1_kw=2,
            s1_rbcm=2, s1_sbcm=0,
            s1_stn=32, s1_stk=256, s1_skw=2,
            s2_tn=128, s2_tk=128, s2_bcm=2, s2_xcd=4, s2_persist=False,
            s2_rbcm=2, s2_sbcm=2, s2_stn=64, s2_stk=128,
        )
    if tokens <= 8:
        return dict(
            s1_tn=32, s1_tk=256, s1_wpe=4, s1_xcd=4, s1_kw=2,
            s1_rbcm=2, s1_sbcm=0,
            s1_stn=32, s1_stk=256, s1_skw=2,
            s2_tn=128, s2_tk=128, s2_bcm=3, s2_xcd=0, s2_persist=False,
            s2_rbcm=3, s2_sbcm=3, s2_stn=64, s2_stk=128,
        )
    if tokens <= 9:
        return dict(
            s1_tn=64, s1_tk=256, s1_wpe=3, s1_xcd=4, s1_kw=2,
            s1_rbcm=3, s1_sbcm=3,
            s1_stn=32, s1_stk=256, s1_skw=2,
            s2_tn=128, s2_tk=128, s2_bcm=2, s2_xcd=4, s2_persist=False,
            s2_rbcm=3, s2_sbcm=0, s2_stn=64, s2_stk=128,
        )
    if tokens <= 21:
        return dict(
            s1_tn=64, s1_tk=256, s1_wpe=4, s1_xcd=0, s1_kw=2,
            s1_rbcm=2, s1_sbcm=2,
            s1_stn=32, s1_stk=256, s1_skw=2,
            s2_tn=256, s2_tk=128, s2_bcm=2, s2_xcd=0, s2_persist=False,
            s2_rbcm=2, s2_sbcm=2, s2_stn=64, s2_stk=128,
        )
    return dict(
        s1_tn=128, s1_tk=256, s1_wpe=3, s1_xcd=4, s1_kw=1,
        s1_rbcm=2, s1_sbcm=3,
        s1_stn=32, s1_stk=256, s1_skw=2,
        s2_tn=256, s2_tk=128, s2_bcm=0, s2_xcd=4, s2_persist=False,
        s2_rbcm=3, s2_sbcm=3, s2_stn=64, s2_stk=128,
    )


@dataclass
class KimiK3FHMoEWorkspace:
    """Reusable intermediate storage for one fixed maximum sorting capacity."""

    routed_inter: torch.Tensor
    shared_inter: torch.Tensor
    shared_expert_ids: torch.Tensor
    shared_cumsum: torch.Tensor
    shared_m_indices: torch.Tensor
    max_tokens: int
    num_experts: int | None = None
    sorted_token_ids: torch.Tensor | None = None
    sorted_weights: torch.Tensor | None = None
    sorted_expert_ids: torch.Tensor | None = None
    cumsum_tensor: torch.Tensor | None = None
    routed_out: torch.Tensor | None = None
    sorting_workspace: torch.Tensor | None = None


def prepare_kimi_k3_fhmoe_shared_weights(
    shared_w1: torch.Tensor,
    shared_w2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preshuffle the two BF16 shared-expert matrices once at model load time."""
    _validate_shared_weights(shared_w1, shared_w2, preshuffled=False)
    return (
        shuffle_weight(shared_w1.contiguous(), layout=(16, 16)),
        shuffle_weight(shared_w2.contiguous(), layout=(16, 16)),
    )


def create_kimi_k3_fhmoe_workspace(
    *,
    max_sorted_tokens: int,
    max_tokens: int = KIMI_K3_MAX_DECODE_TOKENS,
    num_experts: int | None = None,
    device: torch.device | str | int = "cuda",
) -> KimiK3FHMoEWorkspace:
    """Allocate graph-reusable Stage 1 intermediates and shared route metadata."""
    if max_sorted_tokens <= 0:
        raise ValueError(f"max_sorted_tokens must be positive, got {max_sorted_tokens}")
    if not 1 <= max_tokens <= KIMI_K3_BLOCK_M:
        raise ValueError(
            f"max_tokens must be in [1, {KIMI_K3_BLOCK_M}], got {max_tokens}"
        )
    if num_experts is not None and num_experts < KIMI_K3_TOPK:
        raise ValueError(
            f"num_experts must be at least {KIMI_K3_TOPK}, got {num_experts}"
        )

    max_m_blocks = (max_sorted_tokens + KIMI_K3_BLOCK_M - 1) // KIMI_K3_BLOCK_M
    shared_m_indices = torch.full(
        (KIMI_K3_BLOCK_M,),
        max_tokens,
        dtype=torch.int32,
        device=device,
    )
    shared_m_indices[:max_tokens] = torch.arange(
        max_tokens, dtype=torch.int32, device=device
    )
    sorting_workspace = None
    if num_experts is not None:
        from aiter.ops.moe_sorting_opus import (
            moe_sorting_opus_get_workspace_size,
        )

        sorting_workspace_size = moe_sorting_opus_get_workspace_size(
            max_tokens,
            num_experts,
            KIMI_K3_TOPK,
            0,
        )
        if sorting_workspace_size > 0:
            sorting_workspace = torch.empty(
                sorting_workspace_size,
                dtype=torch.uint8,
                device=device,
            )

    shared_cumsum = torch.full(
        (2,), KIMI_K3_BLOCK_M, dtype=torch.int32, device=device
    )
    shared_cumsum[1].fill_(max_tokens)

    return KimiK3FHMoEWorkspace(
        routed_inter=torch.empty(
            (max_sorted_tokens, KIMI_K3_ROUTED_INTER),
            dtype=torch.bfloat16,
            device=device,
        ),
        shared_inter=torch.empty(
            (KIMI_K3_BLOCK_M, KIMI_K3_SHARED_INTER),
            dtype=torch.bfloat16,
            device=device,
        ),
        shared_expert_ids=torch.zeros(1, dtype=torch.int32, device=device),
        shared_cumsum=shared_cumsum,
        shared_m_indices=shared_m_indices,
        max_tokens=max_tokens,
        num_experts=num_experts,
        sorted_token_ids=(
            torch.empty(max_sorted_tokens, dtype=torch.int32, device=device)
            if num_experts is not None
            else None
        ),
        sorted_weights=(
            torch.empty(max_sorted_tokens, dtype=torch.float32, device=device)
            if num_experts is not None
            else None
        ),
        sorted_expert_ids=(
            torch.empty(max_m_blocks, dtype=torch.int32, device=device)
            if num_experts is not None
            else None
        ),
        cumsum_tensor=(
            torch.empty(2, dtype=torch.int32, device=device)
            if num_experts is not None
            else None
        ),
        routed_out=(
            torch.empty(
                (max_tokens, KIMI_K3_ROUTED_HIDDEN),
                dtype=torch.bfloat16,
                device=device,
            )
            if num_experts is not None
            else None
        ),
        sorting_workspace=sorting_workspace,
    )


def _validate_shared_weights(
    shared_w1: torch.Tensor,
    shared_w2: torch.Tensor,
    *,
    preshuffled: bool,
) -> None:
    expected_w1 = (2 * KIMI_K3_SHARED_INTER, KIMI_K3_SHARED_HIDDEN)
    expected_w2 = (KIMI_K3_SHARED_HIDDEN, KIMI_K3_SHARED_INTER)
    if tuple(shared_w1.shape) != expected_w1:
        raise ValueError(
            f"expected shared_w1 shape {expected_w1}, got {shared_w1.shape}"
        )
    if tuple(shared_w2.shape) != expected_w2:
        raise ValueError(
            f"expected shared_w2 shape {expected_w2}, got {shared_w2.shape}"
        )
    if shared_w1.dtype != torch.bfloat16 or shared_w2.dtype != torch.bfloat16:
        raise ValueError("Kimi shared weights must use BF16")
    if not shared_w1.is_cuda or not shared_w2.is_cuda:
        raise ValueError("Kimi shared weights must be CUDA/ROCm tensors")
    if shared_w1.device != shared_w2.device:
        raise ValueError("Kimi shared weights must be on the same device")
    if not shared_w1.is_contiguous() or not shared_w2.is_contiguous():
        raise ValueError("Kimi shared weights must be contiguous")
    if preshuffled and not (
        getattr(shared_w1, "is_shuffled", False)
        and getattr(shared_w2, "is_shuffled", False)
    ):
        # Tensor attributes can be lost across framework wrappers, so the low-level
        # launch API does not call this check.  Keep it for explicit preparation use.
        raise ValueError("shared BF16 weights must be preshuffled with shuffle_weight")


def _validate_launch_contract(
    routed_x: torch.Tensor,
    shared_x: torch.Tensor,
    routed_w1: torch.Tensor,
    routed_w2: torch.Tensor,
    routed_w1_scale: torch.Tensor,
    routed_w2_scale: torch.Tensor,
    shared_w1: torch.Tensor,
    shared_w2: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    cumsum_tensor: torch.Tensor,
    workspace: KimiK3FHMoEWorkspace,
    shared_weight_layout: str,
) -> tuple[int, int]:
    M = routed_x.shape[0] if routed_x.ndim == 2 else -1
    if not 1 <= M <= KIMI_K3_MAX_DECODE_TOKENS:
        raise ValueError(
            "Kimi FHMoE decode kernel supports "
            f"M=1..{KIMI_K3_MAX_DECODE_TOKENS}, got shape {routed_x.shape}"
        )
    if tuple(routed_x.shape) != (M, KIMI_K3_ROUTED_HIDDEN):
        raise ValueError(
            f"expected routed_x shape {(M, KIMI_K3_ROUTED_HIDDEN)}, "
            f"got {tuple(routed_x.shape)}"
        )
    if tuple(shared_x.shape) != (M, KIMI_K3_SHARED_HIDDEN):
        raise ValueError(
            f"expected shared_x shape {(M, KIMI_K3_SHARED_HIDDEN)}, "
            f"got {tuple(shared_x.shape)}"
        )
    if routed_x.dtype != torch.bfloat16 or shared_x.dtype != torch.bfloat16:
        raise ValueError("Kimi FHMoE activations must use BF16")
    if routed_x.device != shared_x.device:
        raise ValueError("routed_x and shared_x must be on the same device")

    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if (
        fp4_dtype is None
        or routed_w1.dtype != fp4_dtype
        or routed_w2.dtype != fp4_dtype
    ):
        raise ValueError("routed weights must use torch.float4_e2m1fn_x2")
    if routed_w1.ndim != 3 or routed_w2.ndim != 3:
        raise ValueError("routed weights must be rank-3 expert tensors")
    NE = routed_w1.shape[0]
    if tuple(routed_w1.shape) != (
        NE,
        2 * KIMI_K3_ROUTED_INTER,
        KIMI_K3_ROUTED_HIDDEN // 2,
    ):
        raise ValueError(f"unexpected routed_w1 shape {tuple(routed_w1.shape)}")
    if tuple(routed_w2.shape) != (
        NE,
        KIMI_K3_ROUTED_HIDDEN,
        KIMI_K3_ROUTED_INTER // 2,
    ):
        raise ValueError(f"unexpected routed_w2 shape {tuple(routed_w2.shape)}")
    if NE < KIMI_K3_TOPK:
        raise ValueError(f"Kimi top-{KIMI_K3_TOPK} requires at least 16 experts")
    scale_dtypes = (torch.uint8, getattr(torch, "float8_e8m0fnu", torch.uint8))
    if routed_w1_scale.dtype not in scale_dtypes:
        raise ValueError("routed_w1_scale must use E8M0/uint8 storage")
    if routed_w2_scale.dtype not in scale_dtypes:
        raise ValueError("routed_w2_scale must use E8M0/uint8 storage")

    _validate_shared_weights(shared_w1, shared_w2, preshuffled=False)
    if shared_weight_layout not in ("preshuffled", "rowmajor"):
        raise ValueError(
            "shared_weight_layout must be 'preshuffled' or 'rowmajor', got "
            f"{shared_weight_layout!r}"
        )
    tensors = (
        routed_w1,
        routed_w2,
        routed_w1_scale,
        routed_w2_scale,
        shared_w1,
        shared_w2,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        cumsum_tensor,
        workspace.routed_inter,
        workspace.shared_inter,
        workspace.shared_expert_ids,
        workspace.shared_cumsum,
        workspace.shared_m_indices,
    )
    if any(t.device != routed_x.device for t in tensors):
        raise ValueError("all Kimi FHMoE tensors must be on the same device")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("all Kimi FHMoE tensors must be contiguous")
    if workspace.routed_inter.shape[0] < sorted_token_ids.numel():
        raise ValueError(
            "workspace.routed_inter is too small: "
            f"need {sorted_token_ids.numel()} rows, got {workspace.routed_inter.shape[0]}"
        )
    if tuple(workspace.shared_inter.shape) != (
        KIMI_K3_BLOCK_M,
        KIMI_K3_SHARED_INTER,
    ):
        raise ValueError(
            "workspace.shared_inter must have shape "
            f"{(KIMI_K3_BLOCK_M, KIMI_K3_SHARED_INTER)}"
        )
    if workspace.max_tokens < M:
        raise ValueError(
            f"workspace supports at most {workspace.max_tokens} tokens, got M={M}"
        )
    return M, NE


def kimi_k3_fhmoe_a16w4_from_sorted(
    *,
    routed_x: torch.Tensor,
    shared_x: torch.Tensor,
    routed_w1: torch.Tensor,
    routed_w2: torch.Tensor,
    routed_w1_scale: torch.Tensor,
    routed_w2_scale: torch.Tensor,
    shared_w1: torch.Tensor,
    shared_w2: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    cumsum_tensor: torch.Tensor,
    routed_out: torch.Tensor,
    workspace: KimiK3FHMoEWorkspace,
    shared_out: torch.Tensor | None = None,
    situ_beta: float = 4.0,
    situ_linear_beta: float = 25.0,
    swiglu_limit: float = float("inf"),
    shared_weight_layout: str = "preshuffled",
    stream=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the two unified compute launches using precomputed routing metadata.

    ``routed_out`` must already be zeroed, normally by ``moe_sorting``.
    ``shared_w1`` and ``shared_w2`` must be BF16 weights preshuffled with
    :func:`prepare_kimi_k3_fhmoe_shared_weights`.
    """
    M, NE = _validate_launch_contract(
        routed_x,
        shared_x,
        routed_w1,
        routed_w2,
        routed_w1_scale,
        routed_w2_scale,
        shared_w1,
        shared_w2,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        cumsum_tensor,
        workspace,
        shared_weight_layout,
    )
    if tuple(routed_out.shape) != (M, KIMI_K3_ROUTED_HIDDEN):
        raise ValueError(
            f"expected routed_out shape {(M, KIMI_K3_ROUTED_HIDDEN)}, "
            f"got {tuple(routed_out.shape)}"
        )
    if shared_out is None:
        shared_out = torch.empty(
            (M, KIMI_K3_SHARED_HIDDEN),
            dtype=torch.bfloat16,
            device=routed_x.device,
        )
    elif tuple(shared_out.shape) != (M, KIMI_K3_SHARED_HIDDEN):
        raise ValueError(
            f"expected shared_out shape {(M, KIMI_K3_SHARED_HIDDEN)}, "
            f"got {tuple(shared_out.shape)}"
        )

    beta = float(situ_beta)
    linear_beta = float(situ_linear_beta)
    if beta <= 0.0 or linear_beta <= 0.0:
        raise ValueError(
            "situ_beta and situ_linear_beta must be positive, got "
            f"{beta}/{linear_beta}"
        )
    if stream is None:
        stream = torch.cuda.current_stream(routed_x.device)
    use_k16 = "gfx95" not in str(get_rocm_arch())
    profile = _decode_kernel_profile(M)
    if shared_weight_layout == "rowmajor":
        profile.update(
            s1_sbcm=1,
            s1_stn=16,
            s1_stk=128,
            s1_skw=4,
            s1_sfirst=True,
            s2_sbcm=1,
            s2_stn=64,
            s2_stk=128,
            s2_sfirst=True,
        )
    compute_bm = _decode_compute_block_m(M)

    routed_max_blocks = _routed_grid_block_upper_bound(
        tokens=M,
        num_experts=NE,
        capacity_blocks=int(sorted_expert_ids.numel()),
    )
    stage1_grid = (
        routed_max_blocks * (KIMI_K3_ROUTED_INTER // int(profile["s1_tn"]))
        + KIMI_K3_SHARED_INTER
        // int(profile.get("s1_stn", profile["s1_tn"]))
    )
    stage1 = compile_kimi_k3_fhmoe_stage1(
        NE=NE,
        BM=compute_bm,
        SORT_BM=KIMI_K3_SORT_BLOCK_M,
        TILE_N=int(profile["s1_tn"]),
        TILE_K=int(profile["s1_tk"]),
        waves_per_eu=profile["s1_wpe"],
        routed_b_cache_mod=int(profile.get("s1_rbcm", 2)),
        shared_b_cache_mod=int(profile.get("s1_sbcm", 0)),
        use_k16=use_k16,
        shared_weight_layout=shared_weight_layout,
        routed_xcd_swizzle=int(profile["s1_xcd"]),
        routed_k_wave=int(profile["s1_kw"]),
        shared_tile_n=int(profile.get("s1_stn", profile["s1_tn"])),
        shared_tile_k=int(profile.get("s1_stk", profile["s1_tk"])),
        shared_k_wave=int(profile.get("s1_skw", profile["s1_kw"])),
        shared_first=bool(profile.get("s1_sfirst", False)),
    )
    _run_compiled(
        stage1,
        routed_x.data_ptr(),
        shared_x.data_ptr(),
        routed_w1.data_ptr(),
        routed_w1_scale.data_ptr(),
        shared_w1.data_ptr(),
        sorted_expert_ids.data_ptr(),
        cumsum_tensor.data_ptr(),
        sorted_token_ids.data_ptr(),
        workspace.shared_expert_ids.data_ptr(),
        workspace.shared_cumsum.data_ptr(),
        workspace.shared_m_indices.data_ptr(),
        M,
        stage1_grid,
        beta,
        1.0 / beta,
        linear_beta,
        1.0 / linear_beta,
        float(swiglu_limit),
        workspace.routed_inter.data_ptr(),
        workspace.shared_inter.data_ptr(),
        stream,
    )

    stage2_routed_grid = routed_max_blocks * (
        KIMI_K3_ROUTED_HIDDEN // int(profile["s2_tn"])
    )
    stage2_shared_grid = KIMI_K3_SHARED_HIDDEN // int(
        profile.get("s2_stn", profile["s2_tn"])
    )
    stage2_grid = (
        max(min(stage2_routed_grid, _GFX950_CU_COUNT), stage2_shared_grid)
        if profile["s2_persist"]
        else stage2_routed_grid + stage2_shared_grid
    )
    stage2 = compile_kimi_k3_fhmoe_stage2(
        NE=NE,
        BM=compute_bm,
        SORT_BM=KIMI_K3_SORT_BLOCK_M,
        TILE_N=int(profile["s2_tn"]),
        TILE_K=int(profile["s2_tk"]),
        routed_b_cache_mod=int(profile.get("s2_rbcm", profile["s2_bcm"])),
        shared_b_cache_mod=int(profile.get("s2_sbcm", profile["s2_bcm"])),
        use_k16=use_k16,
        shared_weight_layout=shared_weight_layout,
        routed_xcd_swizzle=int(profile["s2_xcd"]),
        routed_persist=bool(profile["s2_persist"]),
        shared_tile_n=int(profile.get("s2_stn", profile["s2_tn"])),
        shared_tile_k=int(profile.get("s2_stk", profile["s2_tk"])),
        shared_rowmajor_b_to_lds=bool(profile.get("s2_sblds", False)),
        shared_first=bool(profile.get("s2_sfirst", False)),
    )
    _run_compiled(
        stage2,
        workspace.routed_inter.data_ptr(),
        workspace.shared_inter.data_ptr(),
        routed_w2.data_ptr(),
        routed_w2_scale.data_ptr(),
        shared_w2.data_ptr(),
        sorted_expert_ids.data_ptr(),
        cumsum_tensor.data_ptr(),
        sorted_token_ids.data_ptr(),
        sorted_weights.data_ptr(),
        M,
        stage2_grid,
        routed_out.data_ptr(),
        shared_out.data_ptr(),
        stream,
    )
    return routed_out, shared_out


def kimi_k3_fhmoe_a16w4(
    *,
    routed_x: torch.Tensor,
    shared_x: torch.Tensor,
    routed_w1: torch.Tensor,
    routed_w2: torch.Tensor,
    routed_w1_scale: torch.Tensor,
    routed_w2_scale: torch.Tensor,
    shared_w1: torch.Tensor,
    shared_w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    workspace: KimiK3FHMoEWorkspace | None = None,
    shared_out: torch.Tensor | None = None,
    situ_beta: float = 4.0,
    situ_linear_beta: float = 25.0,
    swiglu_limit: float = float("inf"),
    shared_weight_layout: str = "preshuffled",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience API: sort routes, then execute the complete two-stage FHMoE."""
    M = routed_x.shape[0]
    NE = routed_w1.shape[0]
    if tuple(topk_ids.shape) != (M, KIMI_K3_TOPK):
        raise ValueError(
            f"expected topk_ids shape {(M, KIMI_K3_TOPK)}, "
            f"got {tuple(topk_ids.shape)}"
        )
    if tuple(topk_weights.shape) != tuple(topk_ids.shape):
        raise ValueError("topk_weights and topk_ids must have the same shape")

    if workspace is None:
        max_sorted_tokens = M * KIMI_K3_TOPK + NE * KIMI_K3_BLOCK_M - KIMI_K3_TOPK
        workspace = create_kimi_k3_fhmoe_workspace(
            max_sorted_tokens=max_sorted_tokens,
            max_tokens=M,
            num_experts=NE,
            device=routed_x.device,
        )
    elif workspace.max_tokens < M:
        raise ValueError(
            f"workspace supports at most {workspace.max_tokens} tokens, got M={M}"
        )

    reusable_sort = (
        workspace.num_experts == NE
        and workspace.sorted_token_ids is not None
        and workspace.sorted_weights is not None
        and workspace.sorted_expert_ids is not None
        and workspace.cumsum_tensor is not None
        and workspace.routed_out is not None
    )
    if reusable_sort:
        from aiter.ops.moe_sorting_opus import moe_sorting_opus_fwd

        sorted_token_ids = workspace.sorted_token_ids
        sorted_weights = workspace.sorted_weights
        sorted_expert_ids = workspace.sorted_expert_ids
        cumsum_tensor = workspace.cumsum_tensor
        routed_out = workspace.routed_out[:M]
        moe_sorting_opus_fwd(
            topk_ids,
            topk_weights,
            sorted_token_ids,
            sorted_weights,
            sorted_expert_ids,
            cumsum_tensor,
            routed_out,
            NE,
            KIMI_K3_BLOCK_M,
            None,
            None,
            workspace.sorting_workspace,
            0,
            None,
            None,
            None,
        )
    else:
        from aiter.fused_moe import moe_sorting

        (
            sorted_token_ids,
            sorted_weights,
            sorted_expert_ids,
            cumsum_tensor,
            routed_out,
        ) = moe_sorting(
            topk_ids,
            topk_weights,
            NE,
            KIMI_K3_ROUTED_HIDDEN,
            torch.bfloat16,
            KIMI_K3_BLOCK_M,
            accumulate=True,
        )

    return kimi_k3_fhmoe_a16w4_from_sorted(
        routed_x=routed_x,
        shared_x=shared_x,
        routed_w1=routed_w1,
        routed_w2=routed_w2,
        routed_w1_scale=routed_w1_scale,
        routed_w2_scale=routed_w2_scale,
        shared_w1=shared_w1,
        shared_w2=shared_w2,
        sorted_token_ids=sorted_token_ids,
        sorted_weights=sorted_weights,
        sorted_expert_ids=sorted_expert_ids,
        cumsum_tensor=cumsum_tensor,
        routed_out=routed_out,
        workspace=workspace,
        shared_out=shared_out,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        swiglu_limit=swiglu_limit,
        shared_weight_layout=shared_weight_layout,
    )


__all__ = [
    "KimiK3FHMoEWorkspace",
    "create_kimi_k3_fhmoe_workspace",
    "kimi_k3_fhmoe_a16w4",
    "kimi_k3_fhmoe_a16w4_from_sorted",
    "prepare_kimi_k3_fhmoe_shared_weights",
]
