# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the original and optimized Kimi-K3 BF16 FHMoE prototypes."""

from __future__ import annotations

import argparse
import importlib
import statistics
from collections.abc import Callable

import torch
import triton

kimi = importlib.import_module("aiter.ops.triton.moe.kimi_k3_fhmoe_bf16")

ROUTED_HIDDEN = 3584
SHARED_HIDDEN = 7168
INTERMEDIATE = 384
TOPK = 16
SHARED_EXPERTS = 2
SHARED_INTERMEDIATE = SHARED_EXPERTS * INTERMEDIATE
EXPERTS = TOPK


def _inputs(seed: int):
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
    topk_ids = torch.randperm(EXPERTS, generator=generator, device="cuda")
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


def _measure(
    operation: Callable[[], None],
    *,
    operations_per_graph: int,
    replays: int,
    trials: int,
) -> float:
    for _ in range(3):
        operation()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(operations_per_graph):
            operation()

    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()

    samples = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(
            start.elapsed_time(end) * 1000.0 / (replays * operations_per_graph)
        )
    return statistics.median(samples)


def _rotating(operations: list[Callable[[], None]]) -> Callable[[], None]:
    index = 0

    def operation():
        nonlocal index
        operations[index % len(operations)]()
        index += 1

    return operation


def _serial_operation(inputs, workspaces):
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
    routed_intermediate, shared_intermediate, routed_out, shared_out = workspaces

    def operation():
        stage1_block_n = 32
        stage1_block_k = 64
        kimi._kimi_k3_fhmoe_stage1_bf16[
            (TOPK + SHARED_EXPERTS, triton.cdiv(INTERMEDIATE, stage1_block_n))
        ](
            routed_x,
            shared_x,
            routed_w1,
            shared_w1,
            topk_ids,
            routed_intermediate,
            shared_intermediate,
            ROUTED_HIDDEN=ROUTED_HIDDEN,
            SHARED_HIDDEN=SHARED_HIDDEN,
            INTERMEDIATE=INTERMEDIATE,
            TOPK=TOPK,
            SHARED_EXPERTS=SHARED_EXPERTS,
            SITU_BETA=4.0,
            SITU_LINEAR_BETA=25.0,
            BLOCK_N=stage1_block_n,
            BLOCK_K=stage1_block_k,
            num_warps=4,
        )

        stage2_block_n = 32
        stage2_block_k = 64
        routed_blocks = triton.cdiv(ROUTED_HIDDEN, stage2_block_n)
        kimi._kimi_k3_fhmoe_stage2_bf16[
            (routed_blocks + triton.cdiv(SHARED_HIDDEN, stage2_block_n),)
        ](
            routed_intermediate,
            shared_intermediate,
            routed_w2,
            shared_w2,
            topk_ids,
            topk_weights,
            routed_out,
            shared_out,
            ROUTED_HIDDEN=ROUTED_HIDDEN,
            SHARED_HIDDEN=SHARED_HIDDEN,
            INTERMEDIATE=INTERMEDIATE,
            TOPK=TOPK,
            SHARED_EXPERTS=SHARED_EXPERTS,
            ROUTED_BLOCKS=routed_blocks,
            BLOCK_N=stage2_block_n,
            BLOCK_K=stage2_block_k,
            num_warps=4,
        )

    return operation


def _optimized_operation(inputs, outputs, workspace):
    routed_out, shared_out = outputs

    def operation():
        kimi.kimi_k3_fhmoe_bf16(
            *inputs,
            routed_out=routed_out,
            shared_out=shared_out,
            workspace=workspace,
        )

    return operation


def _relative_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    error = (actual.float() - expected.float()).square().mean().sqrt()
    scale = expected.float().square().mean().sqrt().clamp_min(1.0e-12)
    return float(error / scale)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--rotations", type=int, default=4)
    parser.add_argument("--operations-per-graph", type=int, default=8)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--trials", type=int, default=15)
    args = parser.parse_args()

    if (
        min(
            args.rotations,
            args.operations_per_graph,
            args.replays,
            args.trials,
        )
        <= 0
    ):
        raise ValueError("benchmark counts must be positive")
    if args.operations_per_graph % args.rotations:
        raise ValueError("operations-per-graph must be divisible by rotations")

    cases = [_inputs(args.seed + index) for index in range(args.rotations)]
    serial_workspaces = [
        (
            torch.empty((TOPK, INTERMEDIATE), dtype=torch.bfloat16, device="cuda"),
            torch.empty(
                (SHARED_EXPERTS, INTERMEDIATE),
                dtype=torch.bfloat16,
                device="cuda",
            ),
            torch.empty((1, ROUTED_HIDDEN), dtype=torch.bfloat16, device="cuda"),
            torch.empty((1, SHARED_HIDDEN), dtype=torch.bfloat16, device="cuda"),
        )
        for _ in cases
    ]
    optimized_outputs = [
        (
            torch.empty((1, ROUTED_HIDDEN), dtype=torch.bfloat16, device="cuda"),
            torch.empty((1, SHARED_HIDDEN), dtype=torch.bfloat16, device="cuda"),
        )
        for _ in cases
    ]
    optimized_workspaces = [
        torch.empty(
            kimi.kimi_k3_fhmoe_bf16_workspace_size(),
            dtype=torch.uint8,
            device="cuda",
        )
        for _ in cases
    ]

    serial_operations = [
        _serial_operation(inputs, workspaces)
        for inputs, workspaces in zip(cases, serial_workspaces, strict=True)
    ]
    optimized_operations = [
        _optimized_operation(inputs, outputs, workspace)
        for inputs, outputs, workspace in zip(
            cases,
            optimized_outputs,
            optimized_workspaces,
            strict=True,
        )
    ]

    serial_operations[0]()
    optimized_operations[0]()
    torch.cuda.synchronize()
    routed_error = _relative_rmse(
        optimized_outputs[0][0],
        serial_workspaces[0][2],
    )
    shared_error = _relative_rmse(
        optimized_outputs[0][1],
        serial_workspaces[0][3],
    )
    if max(routed_error, shared_error) >= 0.01:
        raise AssertionError(
            f"relative RMSE exceeds 0.01: routed={routed_error}, "
            f"shared={shared_error}"
        )

    serial_us = _measure(
        _rotating(serial_operations),
        operations_per_graph=args.operations_per_graph,
        replays=args.replays,
        trials=args.trials,
    )
    optimized_us = _measure(
        _rotating(optimized_operations),
        operations_per_graph=args.operations_per_graph,
        replays=args.replays,
        trials=args.trials,
    )
    weight_bytes = sum(
        tensor.numel() * tensor.element_size()
        for inputs in cases
        for tensor in inputs[2:6]
    )

    print("shape: Kimi-K3 B1 TP8-local BF16 expert body")
    print(f"rotating weight set: {weight_bytes / 2**20:.1f} MiB")
    print(f"serial prototype: {serial_us:.3f} us")
    print(f"optimized public API: {optimized_us:.3f} us")
    print(f"speedup: {serial_us / optimized_us:.3f}x")
    print(f"routed RRMSE: {routed_error:.6f}")
    print(f"shared RRMSE: {shared_error:.6f}")


if __name__ == "__main__":
    main()
