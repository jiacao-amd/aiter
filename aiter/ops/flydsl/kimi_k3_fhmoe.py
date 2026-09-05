# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Runtime wrapper for the Kimi-K3 heterogeneous two-stage FHMoE kernels."""

from __future__ import annotations

from dataclasses import dataclass
import os

import torch
from flydsl.runtime.device import get_rocm_arch

from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, ptr_arg
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.moe.kimi_k3_fhmoe_bf16 import (
    kimi_k3_shared_stage1_splitk_bf16_env_split_k,
    kimi_k3_shared_stage1_splitk_bf16_from_env,
    kimi_k3_shared_stage1_splitk_bf16_workspace_size,
)

from .moe_kernels import compile_flydsl_moe_stage1, flydsl_moe_stage1

from .kernels.kimi_k3_fhmoe_a16w4 import (
    KIMI_K3_ROUTED_HIDDEN,
    KIMI_K3_ROUTED_INTER,
    KIMI_K3_SHARED_HIDDEN,
    KIMI_K3_SHARED_INTER,
    KIMI_K3_TOPK,
    compile_kimi_k3_fhmoe_stage1,
    compile_kimi_k3_fhmoe_stage2,
    kimi_k3_stage2_routed_m8_epilogue_enabled,
    kimi_k3_stage2_routed_m8_meta_broadcast_enabled,
    kimi_k3_stage2_routed_m8_register_epilogue_enabled,
    kimi_k3_stage2_split_paths_enabled,
    kimi_k3_stage2_wide_n_8wave_enabled,
)

KIMI_K3_SORT_BLOCK_M = 32
# Backward-compatible public name: sorting/workspace rows remain fixed at 32.
KIMI_K3_BLOCK_M = KIMI_K3_SORT_BLOCK_M
KIMI_K3_MAX_DECODE_TOKENS = KIMI_K3_SORT_BLOCK_M
_KIMI_K3_ROUTED_EXPERTS = 896
_STAGE1_TILE_N = 128
_STAGE2_TILE_N = 128
_GFX950_CU_COUNT = 256
_FUSED_ROUTED_A_FP8_ENV = "AITER_KIMI_K3_FUSED_ROUTED_A_FP8"
_INTEGRATED_STAGE1_SPLITK_ENV = "AITER_KIMI_K3_INTEGRATED_STAGE1_SPLITK"
_INTEGRATED_STAGE1_SPLITK_CHOICES = (4, 7, 14)
_INTEGRATED_STAGE1_TILE_N_ENV = "AITER_KIMI_K3_INTEGRATED_STAGE1_TILE_N"
_INTEGRATED_STAGE1_N_MAJOR_ENV = "AITER_KIMI_K3_INTEGRATED_STAGE1_N_MAJOR"
_INTEGRATED_STAGE1_FUSED_REDUCE_ENV = "AITER_KIMI_K3_SHARED_FUSED_REDUCE"
_INTEGRATED_STAGE1_IDENTITY_M_INDICES_ENV = (
    "AITER_KIMI_K3_SHARED_IDENTITY_M_INDICES"
)
_INTEGRATED_STAGE1_A_LDS_SWIZZLE_ENV = "AITER_KIMI_K3_SHARED_A_LDS_SWIZZLE"
_INTEGRATED_STAGE1_VEC2_PARTIALS_ENV = "AITER_KIMI_K3_SHARED_VEC2_PARTIALS"
_SHARED_STAGE1_B_CACHE_MOD_ENV = "AITER_KIMI_K3_SHARED_B_CACHE_MOD"
_ROUTED_STAGE1_B_CACHE_MOD_ENV = "AITER_KIMI_K3_ROUTED_STAGE1_B_CACHE_MOD"
_UNIFIED_STAGE1_XCD_SWIZZLE_ENV = "AITER_KIMI_K3_UNIFIED_STAGE1_XCD_SWIZZLE"
_UNIFIED_STAGE1_BLOCK_WAVES_ENV = "AITER_KIMI_K3_UNIFIED_STAGE1_BLOCK_WAVES"
_STAGE2_ROUTE_REDUCE_ENV = "AITER_KIMI_K3_STAGE2_ROUTE_REDUCE"
_STAGE2_ROUTE_REDUCE_BACKEND_ENV = (
    "AITER_KIMI_K3_STAGE2_ROUTE_REDUCE_BACKEND"
)


def _stage2_route_reduce_enabled() -> bool:
    """Enable the gfx950 M<=8 token-slot Stage2 output path."""
    value = os.environ.get(_STAGE2_ROUTE_REDUCE_ENV, "0").strip().lower()
    if value in ("", "0", "false", "off", "no"):
        return False
    if value in ("1", "true", "on", "yes"):
        return True
    raise ValueError(
        f"{_STAGE2_ROUTE_REDUCE_ENV} must be a boolean, got {value!r}"
    )


def _stage2_route_reduce_backend() -> str:
    """Select the graph-safe top16 reducer implementation."""
    backend = os.environ.get(
        _STAGE2_ROUTE_REDUCE_BACKEND_ENV, "flydsl"
    ).strip().lower()
    if backend not in ("flydsl", "opus"):
        raise ValueError(
            f"{_STAGE2_ROUTE_REDUCE_BACKEND_ENV} must be 'flydsl' or "
            f"'opus', got {backend!r}"
        )
    return backend


def _run_stage2_route_reduce(
    route_partials: torch.Tensor,
    routed_out: torch.Tensor,
    tokens: int,
    stream,
) -> None:
    """Reduce weighted BF16 [token, slot, hidden] rows without allocations."""
    if _stage2_route_reduce_backend() == "opus":
        from aiter.ops.opus.moe_stage2_a8w4 import (
            opus_moe_stage2_reduce_token_slot_route_output_fwd,
        )

        with torch.cuda.stream(stream):
            opus_moe_stage2_reduce_token_slot_route_output_fwd(
                route_partials[:tokens],
                out=routed_out,
                topk=KIMI_K3_TOPK,
                block_n=2048,
            )
        return

    from .kernels.moe_reduce import compile_moe_reduction

    reduce_exe = compile_moe_reduction(
        topk=KIMI_K3_TOPK,
        model_dim=KIMI_K3_ROUTED_HIDDEN,
        dtype_str="bf16",
        use_mask=False,
        num_experts=0,
        out_dtype_str="bf16",
        use_weight=False,
    )
    # The benchmark harness intercepts ``_run_compiled`` to isolate logical
    # FHMoE stages.  Mark this extra dispatch explicitly so it is attributed to
    # Stage 2 without shifting the primary Stage1/Stage2 launch pattern.
    reduce_exe._aiter_fhmoe_stage_index = 1
    # The final three pointers are ignored by this specialization. Reusing the
    # persistent partial buffer keeps every captured argument address fixed.
    ignored = ptr_arg(route_partials)
    _run_compiled(
        reduce_exe,
        ptr_arg(route_partials),
        ptr_arg(routed_out),
        ignored,
        ignored,
        ignored,
        tokens,
        stream,
    )


def _fused_routed_a_fp8_enabled() -> bool:
    """Enable the sorting-fused routed BF16-to-FP8 prototype."""
    return os.environ.get(_FUSED_ROUTED_A_FP8_ENV, "0") == "1"


def _integrated_stage1_split_k() -> int | None:
    """Return the env-selected one-dispatch shared Stage1 split-K."""
    value = os.environ.get(_INTEGRATED_STAGE1_SPLITK_ENV, "0").strip().lower()
    if value in ("", "0", "false", "off", "no"):
        return None
    if value in ("1", "true", "on", "yes"):
        return 14
    try:
        split_k = int(value)
    except ValueError as error:
        raise ValueError(
            f"{_INTEGRATED_STAGE1_SPLITK_ENV} must be 0, 4, 7, or 14; "
            f"got {value!r}"
        ) from error
    if split_k not in _INTEGRATED_STAGE1_SPLITK_CHOICES:
        raise ValueError(
            f"{_INTEGRATED_STAGE1_SPLITK_ENV} must be 0, 4, 7, or 14; "
            f"got {split_k}"
        )
    return split_k


def _unified_stage1_xcd_swizzle() -> int:
    """Return the experimental routed-only XCD swizzle for unified Stage1."""
    value = os.environ.get(_UNIFIED_STAGE1_XCD_SWIZZLE_ENV, "0").strip()
    try:
        xcd_swizzle = int(value)
    except ValueError as error:
        raise ValueError(
            f"{_UNIFIED_STAGE1_XCD_SWIZZLE_ENV} must be 0 or 4; got {value!r}"
        ) from error
    if xcd_swizzle not in (0, 4):
        raise ValueError(
            f"{_UNIFIED_STAGE1_XCD_SWIZZLE_ENV} must be 0 or 4; "
            f"got {xcd_swizzle}"
        )
    return xcd_swizzle


def _unified_stage1_block_waves() -> int:
    """Return the opt-in Kimi unified Stage1 workgroup width."""
    value = os.environ.get(_UNIFIED_STAGE1_BLOCK_WAVES_ENV, "4").strip()
    try:
        block_waves = int(value)
    except ValueError as error:
        raise ValueError(
            f"{_UNIFIED_STAGE1_BLOCK_WAVES_ENV} must be 4 or 8; got {value!r}"
        ) from error
    if block_waves not in (4, 8):
        raise ValueError(
            f"{_UNIFIED_STAGE1_BLOCK_WAVES_ENV} must be 4 or 8; "
            f"got {block_waves}"
        )
    return block_waves


def _integrated_stage1_workspace_size(split_k: int) -> int:
    """Fixed BM8 FP32 gate/up partial bytes for the one-dispatch prototype."""
    if split_k not in _INTEGRATED_STAGE1_SPLITK_CHOICES:
        raise ValueError(f"unsupported integrated Stage1 split-K {split_k}")
    return 8 * 2 * split_k * KIMI_K3_SHARED_INTER * 4


def _integrated_stage1_tile_n() -> int:
    """Return the env-selected shared N tile for integrated Stage1."""
    value = os.environ.get(_INTEGRATED_STAGE1_TILE_N_ENV, "32").strip()
    try:
        tile_n = int(value)
    except ValueError as error:
        raise ValueError(
            f"{_INTEGRATED_STAGE1_TILE_N_ENV} must be 16, 32, or 64; "
            f"got {value!r}"
        ) from error
    if tile_n not in (16, 32, 64):
        raise ValueError(
            f"{_INTEGRATED_STAGE1_TILE_N_ENV} must be 16, 32, or 64; "
            f"got {tile_n}"
        )
    return tile_n


def _integrated_stage1_n_major() -> bool:
    """Return whether split workgroups are ordered by N tile before K split."""
    value = os.environ.get(_INTEGRATED_STAGE1_N_MAJOR_ENV, "0").strip().lower()
    if value in ("", "0", "false", "off", "no"):
        return False
    if value in ("1", "true", "on", "yes"):
        return True
    raise ValueError(
        f"{_INTEGRATED_STAGE1_N_MAJOR_ENV} must be a boolean, got {value!r}"
    )


def _integrated_stage1_fused_reduce() -> bool:
    """Return whether the sgsk7 shared path fuses its two LDS reductions."""
    value = os.environ.get(_INTEGRATED_STAGE1_FUSED_REDUCE_ENV, "0").strip().lower()
    if value in ("", "0", "false", "off", "no"):
        return False
    if value in ("1", "true", "on", "yes"):
        return True
    raise ValueError(
        f"{_INTEGRATED_STAGE1_FUSED_REDUCE_ENV} must be a boolean, got {value!r}"
    )


def _integrated_stage1_identity_m_indices() -> bool:
    """Skip shared token-index loads when the integrated M tile is identity."""
    value = os.environ.get(
        _INTEGRATED_STAGE1_IDENTITY_M_INDICES_ENV, "0"
    ).strip().lower()
    if value in ("", "0", "false", "off", "no"):
        return False
    if value in ("1", "true", "on", "yes"):
        return True
    raise ValueError(
        f"{_INTEGRATED_STAGE1_IDENTITY_M_INDICES_ENV} must be a boolean, "
        f"got {value!r}"
    )


def _integrated_stage1_a_lds_swizzle() -> bool:
    """Return whether integrated shared Stage1 XOR-swizzles its A-LDS tile."""
    value = os.environ.get(
        _INTEGRATED_STAGE1_A_LDS_SWIZZLE_ENV, "0"
    ).strip().lower()
    if value in ("", "0", "false", "off", "no"):
        return False
    if value in ("1", "true", "on", "yes"):
        return True
    raise ValueError(
        f"{_INTEGRATED_STAGE1_A_LDS_SWIZZLE_ENV} must be a boolean, "
        f"got {value!r}"
    )


def _integrated_stage1_vec2_partials() -> bool:
    """Return whether split gate/up partials use an adjacent vec2 layout."""
    value = os.environ.get(
        _INTEGRATED_STAGE1_VEC2_PARTIALS_ENV, "0"
    ).strip().lower()
    if value in ("", "0", "false", "off", "no"):
        return False
    if value in ("1", "true", "on", "yes"):
        return True
    raise ValueError(
        f"{_INTEGRATED_STAGE1_VEC2_PARTIALS_ENV} must be a boolean, "
        f"got {value!r}"
    )


def _shared_stage1_b_cache_mod(default: int) -> int:
    """Return the shared-weight cache modifier without changing routed loads."""
    value = os.environ.get(_SHARED_STAGE1_B_CACHE_MOD_ENV)
    if value is None or not value.strip():
        return default
    try:
        cache_mod = int(value)
    except ValueError as error:
        raise ValueError(
            f"{_SHARED_STAGE1_B_CACHE_MOD_ENV} must be 0, 1, 2, or 3; "
            f"got {value!r}"
        ) from error
    if cache_mod not in (0, 1, 2, 3):
        raise ValueError(
            f"{_SHARED_STAGE1_B_CACHE_MOD_ENV} must be 0, 1, 2, or 3; "
            f"got {cache_mod}"
        )
    return cache_mod


def _routed_stage1_b_cache_mod(default: int) -> int:
    """Return the routed-weight cache modifier for focused profiling."""
    value = os.environ.get(_ROUTED_STAGE1_B_CACHE_MOD_ENV)
    if value is None or not value.strip():
        return default
    try:
        cache_mod = int(value)
    except ValueError as error:
        raise ValueError(
            f"{_ROUTED_STAGE1_B_CACHE_MOD_ENV} must be 0, 1, 2, or 3; "
            f"got {value!r}"
        ) from error
    if cache_mod not in (0, 1, 2, 3):
        raise ValueError(
            f"{_ROUTED_STAGE1_B_CACHE_MOD_ENV} must be 0, 1, 2, or 3; "
            f"got {cache_mod}"
        )
    return cache_mod


def _create_overlap_stream(device: torch.device | str | int) -> torch.cuda.Stream:
    """Create the auxiliary stream at the configured priority."""
    get_priority_range = getattr(torch.cuda, "get_stream_priority_range", None)
    if get_priority_range:
        least_priority, greatest_priority = map(int, get_priority_range())
    else:
        # This ROCm PyTorch build accepts HIP's conventional high-priority
        # value (-1) but does not expose get_stream_priority_range().
        least_priority, greatest_priority = 0, -1
    priority_name = os.environ.get(
        "AITER_KIMI_K3_OVERLAP_STREAM_PRIORITY", "low"
    ).strip().lower()
    priorities = {
        "low": least_priority,
        "default": 0,
        "high": greatest_priority,
    }
    if priority_name not in priorities:
        raise ValueError(
            "AITER_KIMI_K3_OVERLAP_STREAM_PRIORITY must be one of "
            f"low/default/high, got {priority_name!r}"
        )
    return torch.cuda.Stream(device=device, priority=priorities[priority_name])


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
    shared_stage1_splitk_workspace: torch.Tensor | None
    shared_stage1_splitk_semaphore: torch.Tensor | None
    shared_expert_ids: torch.Tensor
    shared_cumsum: torch.Tensor
    shared_m_indices: torch.Tensor
    max_tokens: int
    num_experts: int | None = None
    routed_x_fp8: torch.Tensor | None = None
    routed_stage2_partials: torch.Tensor | None = None
    sorted_token_ids: torch.Tensor | None = None
    sorted_weights: torch.Tensor | None = None
    sorted_expert_ids: torch.Tensor | None = None
    cumsum_tensor: torch.Tensor | None = None
    routed_out: torch.Tensor | None = None
    sorting_workspace: torch.Tensor | None = None
    overlap_stream: torch.cuda.Stream | None = None
    overlap_start_event: torch.cuda.Event | None = None
    overlap_done_event: torch.cuda.Event | None = None


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

    fused_routed_a_fp8 = _fused_routed_a_fp8_enabled()
    if fused_routed_a_fp8:
        if not str(get_rocm_arch()).startswith("gfx950"):
            raise ValueError(f"{_FUSED_ROUTED_A_FP8_ENV} is supported only on gfx950")
        if num_experts != _KIMI_K3_ROUTED_EXPERTS:
            raise ValueError(
                f"{_FUSED_ROUTED_A_FP8_ENV} requires exactly "
                f"{_KIMI_K3_ROUTED_EXPERTS} routed experts"
            )
        if os.environ.get("AITER_KIMI_K3_SCALED_ROUTED_STAGE1", "0") != "1":
            raise ValueError(
                f"{_FUSED_ROUTED_A_FP8_ENV} requires "
                "AITER_KIMI_K3_SCALED_ROUTED_STAGE1=1"
            )
        if _integrated_stage1_split_k() != 7:
            raise ValueError(
                f"{_FUSED_ROUTED_A_FP8_ENV} requires "
                f"{_INTEGRATED_STAGE1_SPLITK_ENV}=7"
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

    profile_overrides = {
        item.strip()
        for item in os.environ.get("AITER_KIMI_K3_PROFILE_OVERRIDES", "").split(",")
        if item.strip()
    }
    integrated_shared_stage1_split_k = _integrated_stage1_split_k()
    integrated_shared_stage1_tile_n = (
        _integrated_stage1_tile_n()
        if integrated_shared_stage1_split_k is not None
        else 32
    )
    # Integrated Stage1 is a single-dispatch path.  Do not retain the auxiliary
    # overlap stream/events or size its scratch from the mutually exclusive
    # Triton split-K path even if stale environment variables remain set.
    overlap_paths = (
        "overlap_paths" in profile_overrides
        and integrated_shared_stage1_split_k is None
    )
    triton_shared_stage1_split_k = (
        kimi_k3_shared_stage1_splitk_bf16_env_split_k()
        if integrated_shared_stage1_split_k is None
        else None
    )
    # The integrated kernel uses a fixed BM8 partial layout so the semaphore
    # address and capture-time ABI do not depend on runtime M.
    shared_stage1_splitk_tokens = 8
    shared_stage1_splitk_workspace_bytes = max(
        (
            kimi_k3_shared_stage1_splitk_bf16_workspace_size(
                shared_stage1_splitk_tokens,
                triton_shared_stage1_split_k,
            )
            if triton_shared_stage1_split_k is not None
            else 0
        ),
        (
            _integrated_stage1_workspace_size(integrated_shared_stage1_split_k)
            if integrated_shared_stage1_split_k is not None
            else 0
        ),
    )
    routed_x_fp8 = None
    if fused_routed_a_fp8:
        from aiter.utility import dtypes

        routed_x_fp8 = torch.empty(
            (max_tokens, KIMI_K3_ROUTED_HIDDEN),
            dtype=dtypes.fp8,
            device=device,
        )
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
        shared_stage1_splitk_workspace=(
            torch.empty(
                shared_stage1_splitk_workspace_bytes,
                dtype=torch.uint8,
                device=device,
            )
            if shared_stage1_splitk_workspace_bytes > 0
            else None
        ),
        shared_stage1_splitk_semaphore=(
            torch.zeros(
                KIMI_K3_SHARED_INTER // integrated_shared_stage1_tile_n,
                dtype=torch.int32,
                device=device,
            )
            if integrated_shared_stage1_split_k is not None
            else None
        ),
        # The dense shared expert always has id 0.  Keep one entry per possible
        # 16-row compute tile so BM16/BM32 variants cannot read past the table.
        shared_expert_ids=torch.zeros(
            (max_tokens + 15) // 16,
            dtype=torch.int32,
            device=device,
        ),
        shared_cumsum=shared_cumsum,
        shared_m_indices=shared_m_indices,
        max_tokens=max_tokens,
        num_experts=num_experts,
        routed_x_fp8=routed_x_fp8,
        routed_stage2_partials=(
            torch.empty(
                (max_tokens, KIMI_K3_TOPK, KIMI_K3_ROUTED_HIDDEN),
                dtype=torch.bfloat16,
                device=device,
            )
            if _stage2_route_reduce_enabled()
            else None
        ),
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
        overlap_stream=(_create_overlap_stream(device) if overlap_paths else None),
        overlap_start_event=(torch.cuda.Event() if overlap_paths else None),
        overlap_done_event=(torch.cuda.Event() if overlap_paths else None),
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
    route_partials = workspace.routed_stage2_partials
    if route_partials is not None:
        expected_partials = (
            workspace.max_tokens,
            KIMI_K3_TOPK,
            KIMI_K3_ROUTED_HIDDEN,
        )
        if tuple(route_partials.shape) != expected_partials:
            raise ValueError(
                "workspace.routed_stage2_partials must have shape "
                f"{expected_partials}, got {tuple(route_partials.shape)}"
            )
        if route_partials.dtype != torch.bfloat16:
            raise ValueError("workspace.routed_stage2_partials must use BF16")
        if route_partials.device != routed_x.device:
            raise ValueError(
                "workspace.routed_stage2_partials must be on the input device"
            )
        if not route_partials.is_contiguous():
            raise ValueError("workspace.routed_stage2_partials must be contiguous")
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
    routed_x_fp8_ready: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the two unified compute launches using precomputed routing metadata.

    ``routed_out`` must already be zeroed, normally by ``moe_sorting``.
    ``shared_w1`` and ``shared_w2`` may use the original row-major layout or
    the optional preshuffled layout selected by ``shared_weight_layout``.
    ``routed_x_fp8_ready`` is an explicit promise that the optional reusable
    FP8 scratch contains the current ``routed_x``; ordinary callers must leave
    it false so an uninitialized scratch can never be consumed.
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
    rocm_arch = str(get_rocm_arch())
    use_k16 = "gfx95" not in rocm_arch
    stage2_route_reduce = (
        _stage2_route_reduce_enabled()
        and rocm_arch.startswith("gfx950")
        and M <= 8
    )
    if stage2_route_reduce and workspace.routed_stage2_partials is None:
        raise ValueError(
            f"{_STAGE2_ROUTE_REDUCE_ENV}=1 requires a workspace created "
            "while the switch is enabled"
        )
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
    shared_stage1_b_cache_mod = _shared_stage1_b_cache_mod(
        int(profile.get("s1_sbcm", 0))
    )
    routed_stage1_b_cache_mod = _routed_stage1_b_cache_mod(
        int(profile.get("s1_rbcm", 2))
    )
    compute_bm = _decode_compute_block_m(M)

    routed_max_blocks = _routed_grid_block_upper_bound(
        tokens=M,
        num_experts=NE,
        capacity_blocks=int(sorted_expert_ids.numel()),
    )
    stage1_routed_grid = routed_max_blocks * (
        KIMI_K3_ROUTED_INTER // int(profile["s1_tn"])
    )
    stage1_shared_grid = KIMI_K3_SHARED_INTER // int(
        profile.get("s1_stn", profile["s1_tn"])
    )
    use_scaled_routed_stage1 = (
        os.environ.get("AITER_KIMI_K3_SCALED_ROUTED_STAGE1", "0") == "1"
    )
    integrated_stage1_split_k = _integrated_stage1_split_k()
    integrated_stage1_tile_n = (
        _integrated_stage1_tile_n()
        if integrated_stage1_split_k is not None
        else 32
    )
    if integrated_stage1_split_k is not None:
        if not use_scaled_routed_stage1:
            raise ValueError(
                f"{_INTEGRATED_STAGE1_SPLITK_ENV} requires "
                "AITER_KIMI_K3_SCALED_ROUTED_STAGE1=1"
            )
        if shared_weight_layout != "rowmajor":
            raise ValueError(
                f"{_INTEGRATED_STAGE1_SPLITK_ENV} requires row-major shared weights"
            )
        if not str(get_rocm_arch()).startswith("gfx950"):
            raise ValueError(
                f"{_INTEGRATED_STAGE1_SPLITK_ENV} is supported only on gfx950"
            )
    use_integrated_stage1_splitk = (
        integrated_stage1_split_k is not None and M <= 8
    )
    requested_stage1_block_waves = _unified_stage1_block_waves()
    stage1_block_waves = (
        requested_stage1_block_waves if use_integrated_stage1_splitk else 4
    )
    if stage1_block_waves == 8:
        integrated_stage1_tile_n = 64
    use_unified_scaled_stage1 = use_scaled_routed_stage1 and (
        os.environ.get("AITER_KIMI_K3_UNIFIED_SCALED_STAGE1", "0") == "1"
        or use_integrated_stage1_splitk
    )
    profile_overrides = {
        item.strip()
        for item in os.environ.get("AITER_KIMI_K3_PROFILE_OVERRIDES", "").split(",")
        if item.strip()
    }
    # The integrated path has one Stage1 launch and therefore must not create
    # the auxiliary-stream/event chain used by the two-launch overlap path.
    overlap_paths = (
        "overlap_paths" in profile_overrides
        and integrated_stage1_split_k is None
    )
    use_fused_routed_a_fp8 = (
        _fused_routed_a_fp8_enabled() and use_integrated_stage1_splitk
    )
    if use_fused_routed_a_fp8:
        if not (
            str(get_rocm_arch()).startswith("gfx950")
            and NE == _KIMI_K3_ROUTED_EXPERTS
            and 1 <= M <= KIMI_K3_MAX_DECODE_TOKENS
            and shared_weight_layout == "rowmajor"
            and integrated_stage1_split_k == 7
            and use_scaled_routed_stage1
            and use_unified_scaled_stage1
            and "overlap_paths" not in profile_overrides
        ):
            raise ValueError(
                f"{_FUSED_ROUTED_A_FP8_ENV} requires gfx950, Kimi-K3 E896, "
                "M=1..8, row-major shared weights, and the serial unified "
                "integrated Stage1 split-K=7 path"
            )
        if not routed_x_fp8_ready:
            raise ValueError(
                f"{_FUSED_ROUTED_A_FP8_ENV} requires routed_x_fp8_ready=True "
                "after the current routed_x was converted by Opus sorting"
            )
        routed_x_fp8 = workspace.routed_x_fp8
        if routed_x_fp8 is None:
            raise ValueError(
                f"{_FUSED_ROUTED_A_FP8_ENV} requires a workspace created "
                "while the switch is enabled"
            )
        from aiter.utility import dtypes

        if (
            tuple(routed_x_fp8.shape)
            != (workspace.max_tokens, KIMI_K3_ROUTED_HIDDEN)
            or routed_x_fp8.dtype != dtypes.fp8
            or routed_x_fp8.device != routed_x.device
            or not routed_x_fp8.is_contiguous()
        ):
            raise ValueError(
                "routed_x_fp8 must be one contiguous "
                f"[{workspace.max_tokens}, {KIMI_K3_ROUTED_HIDDEN}] FP8 tensor "
                "on the routed input device"
            )
    if stage1_block_waves == 8 and not use_fused_routed_a_fp8:
        raise ValueError(
            f"{_UNIFIED_STAGE1_BLOCK_WAVES_ENV}=8 requires the sorting-fused "
            "native-FP8 routed input path"
        )
    overlap_shared_first = (
        overlap_paths and "shared_first_enqueue" in profile_overrides
    )
    mask_padded_a = (
        "mask_padded_a" in profile_overrides or use_fused_routed_a_fp8
    )
    shared_stage1_split_k = kimi_k3_shared_stage1_splitk_bf16_env_split_k()
    use_shared_stage1_splitk = (
        shared_stage1_split_k is not None
        and M <= 8
        and shared_weight_layout == "rowmajor"
        and integrated_stage1_split_k is None
    )
    if use_integrated_stage1_splitk:
        assert integrated_stage1_split_k is not None
        required_workspace_bytes = _integrated_stage1_workspace_size(
            integrated_stage1_split_k
        )
        if (
            workspace.shared_stage1_splitk_workspace is None
            or workspace.shared_stage1_splitk_workspace.dtype != torch.uint8
            or not workspace.shared_stage1_splitk_workspace.is_contiguous()
            or workspace.shared_stage1_splitk_workspace.device != routed_x.device
            or workspace.shared_stage1_splitk_workspace.numel()
            < required_workspace_bytes
        ):
            raise ValueError(
                "integrated shared split-K requires an 8-token partial workspace "
                "created while its env switch is enabled"
            )
        if (
            workspace.shared_stage1_splitk_semaphore is None
            or workspace.shared_stage1_splitk_semaphore.dtype != torch.int32
            or not workspace.shared_stage1_splitk_semaphore.is_contiguous()
            or workspace.shared_stage1_splitk_semaphore.device != routed_x.device
            or workspace.shared_stage1_splitk_semaphore.numel()
            < KIMI_K3_SHARED_INTER // integrated_stage1_tile_n
        ):
            raise ValueError(
                "integrated shared split-K requires a contiguous "
                f"{KIMI_K3_SHARED_INTER // integrated_stage1_tile_n}-element "
                "int32 semaphore on the input device"
            )
    if use_shared_stage1_splitk and (
        not overlap_paths or not use_scaled_routed_stage1 or use_unified_scaled_stage1
    ):
        raise ValueError(
            "AITER_KIMI_K3_SHARED_STAGE1_TRITON_SPLITK requires "
            "overlap_paths, scaled routed stage1, and unified stage1 disabled"
        )
    if use_shared_stage1_splitk and workspace.shared_stage1_splitk_workspace is None:
        raise ValueError(
            "shared split-K requires a workspace created while "
            "AITER_KIMI_K3_SHARED_STAGE1_TRITON_SPLITK is enabled"
        )
    if overlap_paths and (not use_scaled_routed_stage1 or use_unified_scaled_stage1):
        raise ValueError(
            "overlap_paths requires scaled routed stage1 with unified stage1 disabled"
        )
    if overlap_paths and (
        workspace.overlap_stream is None
        or workspace.overlap_start_event is None
        or workspace.overlap_done_event is None
    ):
        raise ValueError(
            "overlap_paths requires a workspace created while the override is enabled"
        )

    overlap_stream = workspace.overlap_stream if overlap_paths else None
    if overlap_paths:
        assert overlap_stream is not None
        assert workspace.overlap_start_event is not None
        workspace.overlap_start_event.record(stream)
        overlap_stream.wait_event(workspace.overlap_start_event)

    def launch_custom_stage1(path: str, grid: int, launch_stream) -> None:
        stage1 = compile_kimi_k3_fhmoe_stage1(
            NE=NE,
            BM=compute_bm,
            SORT_BM=KIMI_K3_SORT_BLOCK_M,
            TILE_N=int(profile["s1_tn"]),
            TILE_K=int(profile["s1_tk"]),
            waves_per_eu=profile["s1_wpe"],
            routed_b_cache_mod=routed_stage1_b_cache_mod,
            shared_b_cache_mod=shared_stage1_b_cache_mod,
            use_k16=use_k16,
            shared_weight_layout=shared_weight_layout,
            routed_xcd_swizzle=int(profile["s1_xcd"]),
            routed_k_wave=int(profile["s1_kw"]),
            shared_tile_n=int(profile.get("s1_stn", profile["s1_tn"])),
            shared_tile_k=int(profile.get("s1_stk", profile["s1_tk"])),
            shared_k_wave=int(profile.get("s1_skw", profile["s1_kw"])),
            shared_first=bool(profile.get("s1_sfirst", False)),
            path=path,
        )
        _run_compiled(
            stage1,
            routed_x.data_ptr(),
            shared_x.data_ptr(),
            routed_w1.data_ptr(),
            routed_w1_scale.data_ptr(),
            shared_w1.data_ptr(),
            sorted_expert_ids.data_ptr(),
            (
                cumsum_tensor.data_ptr()
                if path != "shared"
                else workspace.shared_expert_ids.data_ptr()
            ),
            sorted_token_ids.data_ptr(),
            workspace.shared_expert_ids.data_ptr(),
            workspace.shared_cumsum.data_ptr(),
            workspace.shared_m_indices.data_ptr(),
            M,
            grid,
            beta,
            1.0 / beta,
            linear_beta,
            1.0 / linear_beta,
            float(swiglu_limit),
            workspace.routed_inter.data_ptr(),
            workspace.shared_inter.data_ptr(),
            launch_stream,
        )

    if use_unified_scaled_stage1:
        # One launch: reserved grid-y rows cover the dense BF16 shared tiles
        # and the other rows execute the scaled-MFMA routed kernel.
        unified_stage1_wpe = int(
            os.environ.get("AITER_KIMI_K3_UNIFIED_STAGE1_WPE", "5")
        )
        if not 1 <= unified_stage1_wpe <= 10:
            raise ValueError(
                "AITER_KIMI_K3_UNIFIED_STAGE1_WPE must be in [1, 10], "
                f"got {unified_stage1_wpe}"
            )
        unified_stage1_xcd_swizzle = _unified_stage1_xcd_swizzle()
        stage1 = compile_flydsl_moe_stage1(
            model_dim=KIMI_K3_ROUTED_HIDDEN,
            inter_dim=KIMI_K3_ROUTED_INTER,
            experts=NE,
            topk=KIMI_K3_TOPK,
            tile_m=compute_bm,
            tile_n=32 * stage1_block_waves,
            tile_k=256,
            doweight_stage1=False,
            a_dtype="fp8",
            b_dtype="fp4",
            out_dtype="bf16",
            act="situv2",
            persist_m=1,
            use_async_copy=False,
            waves_per_eu=unified_stage1_wpe,
            b_nt=routed_stage1_b_cache_mod,
            gate_mode="interleave",
            a_scale_one=True,
            xcd_swizzle=unified_stage1_xcd_swizzle,
            k_wave=1,
            block_waves=stage1_block_waves,
            v2_output_layout=True,
            a_source_bf16=not use_fused_routed_a_fp8,
            bf16_load_ahead=(
                not use_fused_routed_a_fp8
                and os.environ.get("AITER_KIMI_K3_BF16_LOAD_AHEAD", "0") == "1"
            ),
            mask_padded_a=mask_padded_a,
            sort_block_m=KIMI_K3_SORT_BLOCK_M,
            kimi_shared_bf16=True,
            kimi_shared_weight_layout=shared_weight_layout,
            kimi_shared_b_cache_mod=shared_stage1_b_cache_mod,
            kimi_shared_wave_local_wait=(
                not use_integrated_stage1_splitk
                and shared_weight_layout == "rowmajor"
                and os.environ.get(
                    "AITER_KIMI_K3_SHARED_WAVE_LOCAL_WAIT", "0"
                )
                == "1"
            ),
            kimi_shared_pipeline_wait=(
                os.environ.get(
                    "AITER_KIMI_K3_SHARED_PIPELINE_WAIT", "default"
                )
                if shared_weight_layout == "rowmajor"
                else "default"
            ),
            # Only the A-load schedules depend on the BF16 source conversion.
            # B-pipeline, priority, and shared-workgroup schedules remain valid
            # when Opus supplies the routed input as native FP8.
            kimi_shared_start_sleep=int(
                os.environ.get("AITER_KIMI_K3_SHARED_START_SLEEP", "0")
            ),
            kimi_routed_late_a1=(
                not use_fused_routed_a_fp8
                and os.environ.get("AITER_KIMI_K3_ROUTED_LATE_A1", "0") == "1"
            ),
            kimi_routed_a_ring2=(
                not use_fused_routed_a_fp8
                and os.environ.get("AITER_KIMI_K3_ROUTED_A_RING2", "0") == "1"
            ),
            kimi_routed_b_ring4=(
                os.environ.get("AITER_KIMI_K3_ROUTED_B_RING4", "0") == "1"
            ),
            kimi_routed_b_early4=(
                os.environ.get("AITER_KIMI_K3_ROUTED_B_EARLY4", "0") == "1"
            ),
            kimi_routed_b_half_carry=(
                os.environ.get("AITER_KIMI_K3_ROUTED_B_HALF_CARRY", "0") == "1"
            ),
            kimi_routed_priority3=(
                os.environ.get("AITER_KIMI_K3_ROUTED_PRIORITY3", "0") == "1"
            ),
            kimi_shared_wg_schedule=os.environ.get(
                "AITER_KIMI_K3_SHARED_WG_SCHEDULE",
                "prefix",
            ),
            kimi_shared_grid_split_k=(
                integrated_stage1_split_k
                if use_integrated_stage1_splitk
                else 1
            ),
            kimi_shared_grid_tile_n=integrated_stage1_tile_n,
            kimi_shared_grid_n_major=(
                _integrated_stage1_n_major()
                if use_integrated_stage1_splitk
                else False
            ),
            kimi_shared_fused_reduce=(
                _integrated_stage1_fused_reduce()
                if use_integrated_stage1_splitk
                else False
            ),
            kimi_shared_identity_m_indices=(
                _integrated_stage1_identity_m_indices()
                if use_integrated_stage1_splitk
                else False
            ),
            kimi_shared_a_lds_swizzle=(
                _integrated_stage1_a_lds_swizzle()
                if use_integrated_stage1_splitk
                else False
            ),
            kimi_shared_vec2_partials=(
                _integrated_stage1_vec2_partials()
                if use_integrated_stage1_splitk
                else False
            ),
        )
        if use_integrated_stage1_splitk:
            assert workspace.shared_stage1_splitk_workspace is not None
            assert workspace.shared_stage1_splitk_semaphore is not None
        _run_compiled(
            stage1,
            ptr_arg(workspace.routed_inter),
            ptr_arg(workspace.routed_x_fp8 if use_fused_routed_a_fp8 else routed_x),
            ptr_arg(routed_w1),
            ptr_arg(
                workspace.shared_stage1_splitk_workspace
                if use_integrated_stage1_splitk
                else workspace.shared_expert_ids
            ),
            ptr_arg(routed_w1_scale),
            ptr_arg(shared_w1),
            ptr_arg(shared_x),
            ptr_arg(sorted_token_ids),
            ptr_arg(sorted_expert_ids),
            ptr_arg(workspace.shared_m_indices),
            ptr_arg(cumsum_tensor),
            ptr_arg(
                workspace.shared_stage1_splitk_semaphore
                if use_integrated_stage1_splitk
                else workspace.shared_cumsum
            ),
            ptr_arg(workspace.shared_inter),
            M,
            2 * KIMI_K3_ROUTED_INTER,
            KIMI_K3_ROUTED_HIDDEN,
            routed_max_blocks,
            beta,
            1.0 / beta,
            linear_beta,
            1.0 / linear_beta,
            float(swiglu_limit),
            stream,
        )
    elif use_scaled_routed_stage1:
        routed_stage1_wpe = int(
            os.environ.get("AITER_KIMI_K3_ROUTED_STAGE1_WPE", "5")
        )
        if not 1 <= routed_stage1_wpe <= 10:
            raise ValueError(
                "AITER_KIMI_K3_ROUTED_STAGE1_WPE must be in [1, 10], "
                f"got {routed_stage1_wpe}"
            )
        if overlap_shared_first and not use_shared_stage1_splitk:
            assert overlap_stream is not None
            launch_custom_stage1("shared", stage1_shared_grid, overlap_stream)

        if overlap_shared_first and use_shared_stage1_splitk:
            assert overlap_stream is not None
            assert workspace.shared_stage1_splitk_workspace is not None
            kimi_k3_shared_stage1_splitk_bf16_from_env(
                shared_x,
                shared_w1,
                shared_intermediate_out=workspace.shared_inter[:M],
                workspace=workspace.shared_stage1_splitk_workspace,
                stream=overlap_stream,
                done_event=workspace.overlap_done_event,
            )

        # The generic scaled-MFMA kernel writes BF16 directly in sorted-row
        # layout for the existing custom Stage2.  Its BF16-source mode performs
        # the BF16->FP8 conversion in registers, so no persistent activation or
        # scale workspace is needed.
        with torch.cuda.stream(stream):
            flydsl_moe_stage1(
                a=routed_x,
                w1=routed_w1,
                sorted_token_ids=sorted_token_ids,
                sorted_expert_ids=sorted_expert_ids,
                num_valid_ids=cumsum_tensor,
                out=workspace.routed_inter,
                topk=KIMI_K3_TOPK,
                tile_m=compute_bm,
                tile_n=128,
                tile_k=256,
                a_dtype="fp8",
                b_dtype="fp4",
                out_dtype="bf16",
                act="situv2",
                situ_beta=beta,
                situ_linear_beta=linear_beta,
                w1_scale=routed_w1_scale,
                a1_scale=workspace.shared_expert_ids,
                sorted_weights=None,
                persist_m=1,
                use_async_copy=False,
                waves_per_eu=routed_stage1_wpe,
                b_nt=2,
                gate_mode="interleave",
                a_scale_one=True,
                xcd_swizzle=0,
                swiglu_limit=swiglu_limit,
                k_wave=1,
                v2_output_layout=True,
                a_source_bf16=True,
                bf16_load_ahead=(
                    os.environ.get("AITER_KIMI_K3_BF16_LOAD_AHEAD", "0") == "1"
                ),
                mask_padded_a=mask_padded_a,
                sort_block_m=KIMI_K3_SORT_BLOCK_M,
            )

        # Reuse the custom unified kernel as a shared-only launch by presenting
        # a zero routed cumsum.  This keeps its proven BF16 shared-expert path
        # without duplicating the large routed weight reads.
        stage1_grid = stage1_shared_grid

        if not overlap_shared_first and use_shared_stage1_splitk:
            assert overlap_stream is not None
            assert workspace.shared_stage1_splitk_workspace is not None
            kimi_k3_shared_stage1_splitk_bf16_from_env(
                shared_x,
                shared_w1,
                shared_intermediate_out=workspace.shared_inter[:M],
                workspace=workspace.shared_stage1_splitk_workspace,
                stream=overlap_stream,
                done_event=workspace.overlap_done_event,
            )
    else:
        stage1_grid = stage1_routed_grid + stage1_shared_grid

    if not use_unified_scaled_stage1:
        if not overlap_shared_first and not use_shared_stage1_splitk:
            launch_custom_stage1(
                "shared" if use_scaled_routed_stage1 else "both",
                stage1_grid,
                overlap_stream if overlap_paths else stream,
            )

    stage2_split_paths = kimi_k3_stage2_split_paths_enabled()
    if stage2_split_paths and overlap_paths and not use_shared_stage1_splitk:
        assert overlap_stream is not None
        assert workspace.overlap_done_event is not None
        # Preserve the existing Stage1 overlap, then join before the serial
        # path-specialized Stage2 microbenchmark candidate.
        workspace.overlap_done_event.record(overlap_stream)
        stream.wait_event(workspace.overlap_done_event)

    stage2_routed_tile_n = int(profile["s2_tn"])
    stage2_shared_tile_n = int(profile.get("s2_stn", profile["s2_tn"]))
    if kimi_k3_stage2_wide_n_8wave_enabled():
        if use_k16:
            raise ValueError(
                "AITER_KIMI_K3_STAGE2_WIDE_N_8WAVE is supported only on gfx950"
            )
        stage2_routed_tile_n = 256
        stage2_shared_tile_n = 128
    stage2_routed_grid = routed_max_blocks * (
        KIMI_K3_ROUTED_HIDDEN // stage2_routed_tile_n
    )
    stage2_shared_grid = KIMI_K3_SHARED_HIDDEN // stage2_shared_tile_n
    # Model top-k IDs are unique within each token, so for M<=8 no routed
    # expert can own more than eight valid rows.  Keep M>8 on the full epilogue
    # even when the experiment switch remains enabled for an AgentX run.
    stage2_routed_m8_meta_broadcast = (
        kimi_k3_stage2_routed_m8_meta_broadcast_enabled() and M <= 8
    )
    stage2_routed_m8_register_epilogue = (
        not stage2_route_reduce
        and kimi_k3_stage2_routed_m8_register_epilogue_enabled()
        and M <= 8
    )
    if stage2_routed_m8_register_epilogue:
        if not str(get_rocm_arch()).startswith("gfx950"):
            raise ValueError(
                "AITER_KIMI_K3_STAGE2_ROUTED_M8_REGISTER_EPILOGUE is "
                "supported only on gfx950"
            )
        if not (
            compute_bm == 16
            and stage2_routed_tile_n == 128
            and int(profile["s2_tk"]) == 128
            and shared_weight_layout == "rowmajor"
            and stage2_shared_tile_n == 64
            and int(profile.get("s2_stk", profile["s2_tk"])) == 128
            and not bool(profile.get("s2_sblds", False))
            and bool(profile.get("s2_sfirst", False))
        ):
            raise ValueError(
                "AITER_KIMI_K3_STAGE2_ROUTED_M8_REGISTER_EPILOGUE requires "
                "the Kimi gfx950 M<=8 BM16 routed-TN128/TK128 four-wave "
                "row-major shared-weight profile"
            )
    stage2_routed_m8_epilogue = M <= 8 and (
        stage2_route_reduce
        or kimi_k3_stage2_routed_m8_epilogue_enabled()
        or stage2_routed_m8_meta_broadcast
        or stage2_routed_m8_register_epilogue
    )
    stage2_grid = (
        max(min(stage2_routed_grid, _GFX950_CU_COUNT), stage2_shared_grid)
        if profile["s2_persist"]
        else stage2_routed_grid + stage2_shared_grid
    )
    def launch_stage2(path: str, grid: int, launch_stream) -> None:
        stage2 = compile_kimi_k3_fhmoe_stage2(
            NE=NE,
            BM=compute_bm,
            SORT_BM=KIMI_K3_SORT_BLOCK_M,
            TILE_N=stage2_routed_tile_n,
            TILE_K=int(profile["s2_tk"]),
            routed_b_cache_mod=int(profile.get("s2_rbcm", profile["s2_bcm"])),
            shared_b_cache_mod=int(profile.get("s2_sbcm", profile["s2_bcm"])),
            use_k16=use_k16,
            shared_weight_layout=shared_weight_layout,
            routed_xcd_swizzle=int(profile["s2_xcd"]),
            routed_persist=bool(profile["s2_persist"]),
            routed_valid_m_cap=(
                8 if stage2_routed_m8_epilogue and path != "shared" else 0
            ),
            shared_tile_n=stage2_shared_tile_n,
            shared_tile_k=int(profile.get("s2_stk", profile["s2_tk"])),
            shared_rowmajor_b_to_lds=bool(profile.get("s2_sblds", False)),
            shared_first=bool(profile.get("s2_sfirst", False)),
            path=path,
            routed_route_output=(stage2_route_reduce and path != "shared"),
        )
        routed_stage2_out = (
            workspace.routed_stage2_partials
            if stage2_route_reduce
            else routed_out
        )
        assert routed_stage2_out is not None
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
            grid,
            routed_stage2_out.data_ptr(),
            shared_out.data_ptr(),
            launch_stream,
        )

    def launch_stage2_split_serially() -> None:
        if bool(profile.get("s2_sfirst", False)):
            launch_stage2("shared", stage2_shared_grid, stream)
            launch_stage2("routed", stage2_routed_grid, stream)
        else:
            launch_stage2("routed", stage2_routed_grid, stream)
            launch_stage2("shared", stage2_shared_grid, stream)

    if use_shared_stage1_splitk:
        assert workspace.overlap_done_event is not None
        # Split-K shared Stage 1 runs on the auxiliary stream.  Join once both
        # Stage 1 paths are complete, then use the faster unified Stage 2.
        stream.wait_event(workspace.overlap_done_event)
        if stage2_split_paths:
            launch_stage2_split_serially()
        else:
            launch_stage2("both", stage2_grid, stream)
    elif overlap_paths and not stage2_split_paths:
        assert overlap_stream is not None
        assert workspace.overlap_done_event is not None
        # Same-stream ordering forms two independent chains:
        # routed S1 -> routed S2 and shared S1 -> shared S2.  Join only after
        # both chains have completed so CUDA Graph can retain the overlap.
        if overlap_shared_first:
            launch_stage2("shared", stage2_shared_grid, overlap_stream)
            launch_stage2("routed", stage2_routed_grid, stream)
        else:
            launch_stage2("routed", stage2_routed_grid, stream)
            launch_stage2("shared", stage2_shared_grid, overlap_stream)
        workspace.overlap_done_event.record(overlap_stream)
        stream.wait_event(workspace.overlap_done_event)
    elif stage2_split_paths:
        launch_stage2_split_serially()
    else:
        launch_stage2("both", stage2_grid, stream)
    if stage2_route_reduce:
        assert workspace.routed_stage2_partials is not None
        _run_stage2_route_reduce(
            workspace.routed_stage2_partials,
            routed_out,
            M,
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
    # The integrated shared split-K prototype currently owns an eight-row
    # partial buffer.  Keep M=9..32 on the existing BF16-source path.
    prepare_routed_x_fp8 = _fused_routed_a_fp8_enabled() and M <= 8
    if prepare_routed_x_fp8 and not reusable_sort:
        raise ValueError(
            f"{_FUSED_ROUTED_A_FP8_ENV} requires the reusable Opus sorting path"
        )
    if prepare_routed_x_fp8 and workspace.routed_x_fp8 is None:
        raise ValueError(
            f"{_FUSED_ROUTED_A_FP8_ENV} requires a workspace created while "
            "the switch is enabled"
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
            routed_x if prepare_routed_x_fp8 else None,
            workspace.routed_x_fp8 if prepare_routed_x_fp8 else None,
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
        routed_x_fp8_ready=prepare_routed_x_fp8,
    )


__all__ = [
    "KimiK3FHMoEWorkspace",
    "create_kimi_k3_fhmoe_workspace",
    "kimi_k3_fhmoe_a16w4",
    "kimi_k3_fhmoe_a16w4_from_sorted",
    "prepare_kimi_k3_fhmoe_shared_weights",
]
