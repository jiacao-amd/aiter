// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Opus-based MOE sorting torch-free binding.
// Self-contained: no CK header dependency.

#define MOE_SORTING_OPUS_IMPL
#include "moe_sorting_opus.h"

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "aiter_tensor.h"

void moe_sorting_opus_fwd(aiter_tensor_t& topk_ids,
                          aiter_tensor_t& topk_weights,
                          aiter_tensor_t& sorted_token_ids,
                          aiter_tensor_t& sorted_weights,
                          aiter_tensor_t& sorted_expert_ids,
                          aiter_tensor_t& num_valid_ids,
                          aiter_tensor_t& moe_buf,
                          int num_experts,
                          int unit_size,
                          std::optional<aiter_tensor_t> local_expert_mask,
                          std::optional<aiter_tensor_t> num_local_tokens,
                          std::optional<aiter_tensor_t> workspace,
                          int dispatch_policy,
                          std::optional<aiter_tensor_t> local_topk_ids,
                          std::optional<aiter_tensor_t> m_indices,
                          std::optional<aiter_tensor_t> reverse_sorted,
                          std::optional<aiter_tensor_t> fused_routed_x,
                          std::optional<aiter_tensor_t> fused_routed_x_fp8)
{
    AITER_CHECK(topk_weights.dtype() == AITER_DTYPE_fp32,
                "topk_weights must be FP32 (float32)");

    auto dtype_str = AiterDtype_to_str(topk_ids.dtype());
    int num_tokens = topk_ids.size(0);
    int topk       = topk_ids.size(1);

    if(local_topk_ids.has_value())
    {
        auto& ids_out = local_topk_ids.value();
        AITER_CHECK(ids_out.dim() == 2 && ids_out.size(0) == topk_ids.size(0) &&
                        ids_out.size(1) == topk_ids.size(1),
                    "local_topk_ids must have the same [tokens, topk] shape as topk_ids");
        AITER_CHECK(ids_out.dtype() == topk_ids.dtype(),
                    "local_topk_ids dtype must match topk_ids");
        AITER_CHECK(ids_out.device_id == topk_ids.device_id,
                    "local_topk_ids must be on the same device as topk_ids");
        AITER_CHECK(ids_out.is_contiguous(), "local_topk_ids must be contiguous");
    }

    const bool fuse_routed_x = fused_routed_x.has_value();
    AITER_CHECK(
        fuse_routed_x == fused_routed_x_fp8.has_value(),
        "fused_routed_x and fused_routed_x_fp8 must either both be set or both be null");

    HipDeviceGuard device_guard(topk_ids.device_id);
    if(fuse_routed_x)
    {
        auto& routed_x      = fused_routed_x.value();
        auto& routed_x_fp8  = fused_routed_x_fp8.value();
        const int ws_bytes  = moe_sorting_opus_get_workspace_size(
            num_tokens, num_experts, topk, dispatch_policy);

        AITER_CHECK(get_gpu_arch() == "gfx950",
                    "sorting-fused routed BF16-to-FP8 is supported only on gfx950");
        AITER_CHECK(num_tokens >= 1 && num_tokens <= 8,
                    "sorting-fused routed BF16-to-FP8 prototype requires M in [1, 8]");
        AITER_CHECK(num_experts == 896 && topk == 16 && unit_size == 32,
                    "sorting-fused routed BF16-to-FP8 requires Kimi-K3 "
                    "E896/topk16/unit32");
        AITER_CHECK(dispatch_policy == 0 && ws_bytes > 0 && workspace.has_value(),
                    "sorting-fused routed BF16-to-FP8 requires automatic "
                    "multi-phase Opus sorting with reusable workspace");
        AITER_CHECK(!local_expert_mask.has_value() && !num_local_tokens.has_value() &&
                        !local_topk_ids.has_value() && !m_indices.has_value() &&
                        !reverse_sorted.has_value(),
                    "sorting-fused routed BF16-to-FP8 does not support local or "
                    "auxiliary sorting outputs");
        AITER_CHECK(topk_ids.dtype() == AITER_DTYPE_i32,
                    "sorting-fused routed BF16-to-FP8 requires int32 topk_ids");
        AITER_CHECK(routed_x.dim() == 2 && routed_x.size(0) == num_tokens &&
                        routed_x.size(1) == 3584 && routed_x.dtype() == AITER_DTYPE_bf16,
                    "fused_routed_x must be contiguous BF16 [M, 3584]");
        AITER_CHECK(routed_x_fp8.dim() == 2 && routed_x_fp8.size(0) >= num_tokens &&
                        routed_x_fp8.size(1) == 3584 &&
                        routed_x_fp8.dtype() == AITER_DTYPE_fp8,
                    "fused_routed_x_fp8 must be contiguous FP8 [>=M, 3584]");
        AITER_CHECK(moe_buf.dim() == 2 && moe_buf.size(0) == num_tokens &&
                        moe_buf.size(1) == 3584 && moe_buf.dtype() == AITER_DTYPE_bf16,
                    "sorting-fused routed BF16-to-FP8 requires BF16 moe_buf [M, 3584]");
        AITER_CHECK(routed_x.is_contiguous() && routed_x_fp8.is_contiguous() &&
                        moe_buf.is_contiguous(),
                    "sorting-fused routed BF16-to-FP8 tensors must be contiguous");
        AITER_CHECK(routed_x.device_id == topk_ids.device_id &&
                        routed_x_fp8.device_id == topk_ids.device_id &&
                        moe_buf.device_id == topk_ids.device_id,
                    "sorting-fused routed BF16-to-FP8 tensors must share a device");
        auto& ws = workspace.value();
        AITER_CHECK(ws.dtype() == AITER_DTYPE_u8 && ws.is_contiguous() &&
                        ws.device_id == topk_ids.device_id &&
                        ws.numel() >= static_cast<size_t>(ws_bytes),
                    "sorting-fused routed BF16-to-FP8 requires a valid reusable "
                    "Opus uint8 workspace");
    }
    const hipStream_t stream = aiter::getCurrentHIPStream();

    void* ws_ptr = workspace.has_value() ? workspace.value().data_ptr() : nullptr;

    moe_sorting_opus(
        {
            dtype_str,
            "fp32",
            local_expert_mask.has_value(),
            true,
            dispatch_policy
        },
        {topk_ids.data_ptr(),
         topk_weights.data_ptr(),
         local_expert_mask.has_value() ? local_expert_mask.value().data_ptr() : nullptr,
         num_local_tokens.has_value() ? num_local_tokens.value().data_ptr() : nullptr,
         sorted_token_ids.data_ptr(),
         sorted_weights.data_ptr(),
         sorted_expert_ids.data_ptr(),
         num_valid_ids.data_ptr(),
         moe_buf.data_ptr(),
         ws_ptr,
         local_topk_ids.has_value() ? local_topk_ids.value().data_ptr() : nullptr,
         m_indices.has_value() ? m_indices.value().data_ptr() : nullptr,
         reverse_sorted.has_value() ? reverse_sorted.value().data_ptr() : nullptr,
         fuse_routed_x ? fused_routed_x.value().data_ptr() : nullptr,
         fuse_routed_x ? fused_routed_x_fp8.value().data_ptr() : nullptr,
         num_tokens,
         unit_size,
         num_experts,
         topk,
         static_cast<int>(moe_buf.size(-1)),
         static_cast<int>(moe_buf.element_size()),
         fuse_routed_x ? static_cast<int>(fused_routed_x.value().size(1)) : 0},
        {stream});
}
