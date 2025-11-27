"""
Benchmark attention implementations.

Compare:
1. Full attention (Triton) vs PyTorch reference
2. Sparse attention (once implemented) vs Full attention
3. Different sequence lengths and sparsity levels
"""

import torch
import time
import numpy as np
from typing import Callable


def benchmark_function(
    func: Callable,
    *args,
    num_warmup: int = 10,
    num_iters: int = 100,
    **kwargs
) -> dict:
    """
    Benchmark a function.

    Args:
        func: Function to benchmark
        args, kwargs: Arguments to pass to function
        num_warmup: Number of warmup iterations
        num_iters: Number of benchmark iterations

    Returns:
        dict with timing statistics
    """

    # Warmup
    for _ in range(num_warmup):
        _ = func(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_iters):
        start = time.perf_counter()
        _ = func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    times = np.array(times)

    return {
        'mean': np.mean(times) * 1000,  # ms
        'std': np.std(times) * 1000,
        'min': np.min(times) * 1000,
        'max': np.max(times) * 1000,
        'median': np.median(times) * 1000,
    }


def benchmark_full_attention():
    """
    Benchmark full attention at different sequence lengths.
    """
    from attention import full_attention_forward
    from tests.test_attention import reference_attention

    print("\n" + "="*70)
    print("Benchmarking Full Attention")
    print("="*70)

    batch_size = 4
    num_heads = 8
    head_dim = 64

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    seq_lengths = [256, 512, 1024, 2048]

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num heads: {num_heads}")
    print(f"  Head dim: {head_dim}")
    print(f"  Device: {device}")

    print(f"\n{'Seq Len':<10} {'Triton (ms)':<15} {'PyTorch (ms)':<15} {'Speedup':<10}")
    print("-" * 60)

    for seq_len in seq_lengths:
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

        # Benchmark Triton
        triton_stats = benchmark_function(full_attention_forward, q, k, v)

        # Benchmark PyTorch reference
        pytorch_stats = benchmark_function(reference_attention, q, k, v)

        speedup = pytorch_stats['mean'] / triton_stats['mean']

        print(f"{seq_len:<10} {triton_stats['mean']:<15.3f} {pytorch_stats['mean']:<15.3f} {speedup:<10.2f}x")


def benchmark_sparse_attention():
    """
    Benchmark sparse attention patterns.

    TODO: Implement once sparse attention kernel is ready!
    """
    print("\n" + "="*70)
    print("Benchmarking Sparse Attention")
    print("="*70)
    print("\nTODO: Implement sparse attention benchmarks!")
    print("This will compare:")
    print("  1. Sparse attention (Triton) vs Full attention")
    print("  2. Different sparsity patterns (local, block, random)")
    print("  3. Different sparsity levels (50%, 75%, 90%)")


def benchmark_memory_usage():
    """
    Measure memory usage of different attention implementations.
    """
    print("\n" + "="*70)
    print("Memory Usage Analysis")
    print("="*70)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory benchmark")
        return

    from attention import full_attention_forward

    batch_size = 4
    num_heads = 8
    head_dim = 64

    seq_lengths = [512, 1024, 2048, 4096]

    print(f"\n{'Seq Len':<10} {'Memory (MB)':<15} {'Theoretical':<15}")
    print("-" * 45)

    for seq_len in seq_lengths:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')

        mem_before = torch.cuda.memory_allocated() / 1024**2

        _ = full_attention_forward(q, k, v)

        mem_after = torch.cuda.memory_allocated() / 1024**2
        mem_peak = torch.cuda.max_memory_allocated() / 1024**2

        # Theoretical memory for QK^T matrix: batch * num_heads * seq_len^2 * 4 bytes
        theoretical_mb = (batch_size * num_heads * seq_len * seq_len * 4) / 1024**2

        print(f"{seq_len:<10} {mem_peak:<15.2f} {theoretical_mb:<15.2f}")


def print_flops_analysis():
    """
    Print theoretical FLOPS analysis.
    """
    print("\n" + "="*70)
    print("FLOPS Analysis")
    print("="*70)

    print("\nFull Attention:")
    print("  QK^T: O(N^2 * D) multiply-adds = 2 * N^2 * D FLOPs")
    print("  Softmax: O(N^2) operations")
    print("  (QK^T)V: O(N^2 * D) multiply-adds = 2 * N^2 * D FLOPs")
    print("  Total: ~4 * N^2 * D FLOPs")

    print("\nSparse Attention (with sparsity s):")
    print("  QK^T: O(s * N^2 * D) FLOPs")
    print("  (QK^T)V: O(s * N^2 * D) FLOPs")
    print("  Total: ~4 * s * N^2 * D FLOPs")
    print("  Speedup: 1/s (theoretical)")

    print("\nExample (N=2048, D=64, s=0.25):")
    full_flops = 4 * 2048**2 * 64
    sparse_flops = full_flops * 0.25
    print(f"  Full: {full_flops / 1e9:.2f} GFLOPs")
    print(f"  Sparse: {sparse_flops / 1e9:.2f} GFLOPs")
    print(f"  Theoretical speedup: {1/0.25:.1f}x")


if __name__ == '__main__':
    benchmark_full_attention()
    benchmark_memory_usage()
    print_flops_analysis()
    benchmark_sparse_attention()

    print("\n" + "="*70)
    print("Benchmark complete!")
    print("="*70)
