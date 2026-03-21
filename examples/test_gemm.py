from __future__ import annotations

import torch
import triton
import triton.language as tl
from helion.runtime import default_launcher as _default_launcher

_BLOCK_SIZE_1 = tl.constexpr(128)
_BLOCK_SIZE_0 = tl.constexpr(64)
_BLOCK_SIZE_2 = tl.constexpr(256)

@triton.jit
def _helion_scaled_mm(a, b, c, a_stride_0, a_stride_1, b_stride_0, b_stride_1, c_stride_0, c_stride_1, N, M, K):
    # src[test_gemm.py:26]: for tile_m, tile_n in hl.tile([M, N]):
    num_blocks_0 = tl.cdiv(N, _BLOCK_SIZE_1)
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_1 = pid_0 * _BLOCK_SIZE_1
    indices_1 = (offset_1 + tl.arange(0, _BLOCK_SIZE_1)).to(tl.int32)
    mask_1 = indices_1 < N
    offset_0 = pid_1 * _BLOCK_SIZE_0
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    mask_0 = indices_0 < M
    # src[test_gemm.py:27]: accumulator = hl.zeros([tile_m, tile_n], accumulator_dtype)
    accumulator = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1], 0.0, tl.float32)
    # src[test_gemm.py:28]: for tile_k in hl.tile(K):
    # src[test_gemm.py:29]:     a_blk = a[tile_m, tile_k]
    # src[test_gemm.py:30]:     b_blk = b[tile_k, tile_n]
    # src[test_gemm.py:28-35]: ...
    for offset_2 in tl.range(0, tl.cast(K, tl.int32), _BLOCK_SIZE_2, loop_unroll_factor=1, num_stages=3, flatten=True):
        indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
        mask_2 = indices_2 < K
        accumulator_copy = accumulator
        accumulator_copy_0 = accumulator_copy
        # src[test_gemm.py:29]: a_blk = a[tile_m, tile_k]
        a_blk = tl.load(a + (indices_0[:, None] * a_stride_0 + indices_2[None, :] * a_stride_1), mask_0[:, None] & mask_2[None, :], other=0.0, eviction_policy='evict_last')
        # src[test_gemm.py:30]: b_blk = b[tile_k, tile_n]
        b_blk = tl.load(b + (indices_2[:, None] * b_stride_0 + indices_1[None, :] * b_stride_1), mask_2[:, None] & mask_1[None, :], other=0.0, eviction_policy='evict_first')
        # src[test_gemm.py:31]: accumulator = hl.dot(
        # src[test_gemm.py:32]:     a_blk,
        # src[test_gemm.py:33]:     b_blk,
        # src[test_gemm.py:31-35]: ...
        accumulator = tl.dot(tl.cast(a_blk, tl.float8e4nv), tl.cast(b_blk, tl.float8e4nv), acc=accumulator_copy_0, input_precision='tf32', out_dtype=tl.float32)
    # src[test_gemm.py:37]: c_blk = accumulator.to(out_dtype)
    v_0 = tl.cast(accumulator, tl.bfloat16)
    # src[test_gemm.py:39]: c[tile_m, tile_n] = c_blk
    tl.store(c + (indices_0[:, None] * c_stride_0 + indices_1[None, :] * c_stride_1), v_0, mask_0[:, None] & mask_1[None, :])

def test_scaled_mm(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor, out_dtype: torch.dtype, bias: torch.Tensor | None=None, *, _launcher=_default_launcher):
    # src[test_gemm.py:20]: M, K = a.shape
    M, K = a.shape
    # src[test_gemm.py:21]: N = b.shape[1]
    N = b.shape[1]
    # src[test_gemm.py:23]: c = torch.empty((M, N), dtype=out_dtype, device=a.device)
    c = torch.empty((M, N), dtype=out_dtype, device=a.device)
    # src[test_gemm.py:24]: accumulator_dtype = torch.float32 if a.is_floating_point() else torch.int32
    accumulator_dtype = torch.float32 if a.is_floating_point() else torch.int32
    # src[test_gemm.py:26]: for tile_m, tile_n in hl.tile([M, N]):
    _BLOCK_SIZE_1 = 128
    _BLOCK_SIZE_0 = 64
    # src[test_gemm.py:26]: for tile_m, tile_n in hl.tile([M, N]):
    # src[test_gemm.py:27]:     accumulator = hl.zeros([tile_m, tile_n], accumulator_dtype)
    # src[test_gemm.py:28]:     for tile_k in hl.tile(K):
    # src[test_gemm.py:26-39]: ...
    _launcher(_helion_scaled_mm, (triton.cdiv(N, _BLOCK_SIZE_1) * triton.cdiv(M, _BLOCK_SIZE_0),), a, b, c, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), N, M, K, num_warps=2, num_stages=6)

    # experiement 1: manually override num_warps to 4 and num_stages to 2
    # _launcher(_helion_scaled_mm, (triton.cdiv(N, _BLOCK_SIZE_1) * triton.cdiv(M, _BLOCK_SIZE_0),), a, b, c, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), N, M, K, num_warps=4, num_stages=2)

    # experiement 2: let triton do jit autotuning to find "best" num_warps and num_stages
    # grid = lambda META: (
    #     triton.cdiv(N, _BLOCK_SIZE_1) * triton.cdiv(M, _BLOCK_SIZE_0),
    # )
    # _helion_scaled_mm[grid](a, b, c, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), N, M, K)

    # src[test_gemm.py:41]: return c
    return c


import helion
import helion.language as hl
import torch

@helion.kernel(
    autotune_effort="full",
    static_shapes=False,
    autotune_ignore_errors=True,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    autotune_config_overrides={"block_sizes": [64, 128, 256], "l2_groupings": None, "indexing": ["pointer", "pointer", "pointer"]},
)
def scaled_mm(
    a: torch.Tensor,  # [M, K]
    b: torch.Tensor,  # [K, N]
    scale_a: torch.Tensor,  # [1]/[1, 1]/[M]/[M, 1]
    scale_b: torch.Tensor,  # [1]/[1, 1]/[N]/[N, 1]
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,  # [N]
) -> torch.Tensor:
    M, K = a.shape
    N = b.shape[1]

    c = torch.empty((M, N), dtype=out_dtype, device=a.device)
    accumulator_dtype = torch.float32 if a.is_floating_point() else torch.int32

    for tile_m, tile_n in hl.tile([M, N]):
        accumulator = hl.zeros([tile_m, tile_n], accumulator_dtype)
        for tile_k in hl.tile(K):
            a_blk = a[tile_m, tile_k]
            b_blk = b[tile_k, tile_n]
            accumulator = hl.dot(
                a_blk,
                b_blk,
                acc=accumulator,
            )

        c_blk = accumulator.to(out_dtype)

        c[tile_m, tile_n] = c_blk

    return c

def generate_input():
    num_tokens = 256
    hidden_size = 2048
    in_dtype: torch.dtype = torch.float8_e4m3fn
    scale_dtype: torch.dtype = torch.float32
    out_dtype: torch.dtype = torch.bfloat16

    feature_size = hidden_size * 4
    a = torch.randn(num_tokens, hidden_size, dtype=torch.float32, device="cuda").to(
        in_dtype
    )
    b = torch.randn(
        feature_size, hidden_size, dtype=torch.float32, device="cuda"
    ).to(in_dtype)
    b = b.t()
    scale_a = torch.randn(num_tokens, 1, dtype=scale_dtype, device="cuda")
    scale_b = torch.randn(feature_size, 1, dtype=scale_dtype, device="cuda")
    bias = torch.randn(feature_size, dtype=out_dtype, device="cuda")
    input = (a, b, scale_a, scale_b, out_dtype, bias)

    return input

def autotune():
    input = generate_input()
    config = scaled_mm.autotune(input)
    config.save("test_config.json")

import copy
from dataclasses import dataclass

import triton

@dataclass
class Row:
    case: str
    baseline_ms: float
    kernel_ms: float
    speedup_x: float

    baseline_peak_mb: float
    kernel_peak_mb: float
    mem_improve_x: float


def print_table(rows: list[Row]) -> None:
    headers = [
        "case",
        "baseline_ms",
        "kernel_ms",
        "speedup(x)",
        "baseline_peak(MB)",
        "kernel_peak(MB)",
        "mem_improve(x)",
    ]

    data = [
        [
            r.case,
            f"{r.baseline_ms:.3f}",
            f"{r.kernel_ms:.3f}",
            f"{r.speedup_x:.3f}",
            f"{r.baseline_peak_mb:.2f}",
            f"{r.kernel_peak_mb:.2f}",
            f"{r.mem_improve_x:.3f}",
        ]
        for r in rows
    ]

    cols = list(zip(*([headers] + data)))
    widths = [max(len(cell) for cell in col) for col in cols]

    def fmt(row: list[str]) -> str:
        return " | ".join(cell.ljust(w) for cell, w in zip(row, widths))

    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in data:
        print(fmt(row))


def cleanup_gpu_resources():
    import gc

    try:
        if torch.cuda.is_available():
            # Clear GPU memory cache
            torch.cuda.empty_cache()

            # Force garbage collection
            gc.collect()

            # Clear torch compilation cache
            if hasattr(torch, "_dynamo"):
                torch._dynamo.reset()

            # Synchronize all CUDA streams
            torch.cuda.synchronize()

            # Reset peak memory stats for clean measurements
            torch.cuda.reset_peak_memory_stats()

            print("GPU resources cleaned up successfully")

    except Exception as e:
        print(f"Failed to cleanup GPU resources: {e}")


@torch.inference_mode()
def benchmark(fn, baseline, repeat=1000, cudagraph=True):
    rows: list[Row] = []
    benchmark_fn = (
        triton.testing.do_bench_cudagraph if cudagraph else triton.testing.do_bench
    )

    inputs_list = [generate_input()]

    for inputs in inputs_list:
        try:
            print(f"Start benchmarking")

            inputs_clone = copy.deepcopy(inputs)

            helion_kernel = lambda: fn(*inputs)
            baseline_kernel = lambda: baseline(*inputs_clone)

            torch.cuda.reset_peak_memory_stats()
            helion_latency = benchmark_fn(helion_kernel, rep=repeat, return_mode="mean")
            helion_peak_mem = torch.cuda.max_memory_allocated() / 1e6

            torch.cuda.reset_peak_memory_stats()
            baseline_latency = benchmark_fn(
                baseline_kernel, rep=repeat, return_mode="mean"
            )
            baseline_peak_mem = torch.cuda.max_memory_allocated() / 1e6

            speedup = baseline_latency / helion_latency
            mem_improve = baseline_peak_mem / helion_peak_mem

            rows.append(
                Row(
                    case="gemm",
                    baseline_ms=baseline_latency,
                    kernel_ms=helion_latency,
                    speedup_x=speedup,
                    baseline_peak_mb=baseline_peak_mem,
                    kernel_peak_mb=helion_peak_mem,
                    mem_improve_x=mem_improve,
                )
            )

            cleanup_gpu_resources()

        except Exception as e:
            print(f"Benchmarking failed: {e}")
            continue

    print_table(rows)


def run_benchmark():
    input = generate_input()
    bound = scaled_mm.bind(input)
    # test_config.json is autotuned with autotune_config_overrides={"block_sizes": [64, 128, 256], "l2_groupings": None, "indexing": ["pointer", "pointer", "pointer"]}
    # test_config_2.json is autotuned without autotune_config_overrides
    config = helion.Config.load("test_config_2.json")
    compiled = bound.compile_config(config)

    benchmark(compiled, test_scaled_mm)

if __name__ == "__main__":
    # autotune()
    run_benchmark()
