# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from aiter.fused_moe import get_cktile_pad_zeros, get_cktile_stage_pads


def test_get_cktile_pad_zeros_prefers_supported_unpadded_alignment():
    assert get_cktile_pad_zeros(7168, 192) == 192
    assert get_cktile_pad_zeros(7168, 128) == 128
    assert get_cktile_pad_zeros(8192, 256) == 256
    assert get_cktile_pad_zeros(8192, 512) == 512
    assert get_cktile_pad_zeros(4096, 0) == 0


def test_get_cktile_stage_pads_handles_g1u1_padding_symmetrically():
    assert get_cktile_stage_pads(7168, 256, 192, 128, True) == (
        256,
        192,
        192,
        128,
    )
    assert get_cktile_stage_pads(7168, 256, 192, 128, False) == (
        128,
        192,
        192,
        128,
    )
