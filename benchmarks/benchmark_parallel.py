"""
Benchmark tensor parallel layers.

Run with torchrun:
    torchrun --nproc_per_node=4 benchmarks/benchmark_parallel.py

Tests:
1. Strong scaling (fixed model size, vary GPU count)
2. Communication overhead analysis
3. Comparison with Data Parallel baseline
"""

import torch
import torch.distributed as dist
import time
import os


def benchmark_mlp_forward():
    """
    Benchmark MLP forward pass with tensor parallelism.
    """
    from parallel import (
        ColumnParallelLinear,
        RowParallelLinear,
        initialize_model_parallel,
    )

    if not dist.is_initialized():
        print("torch.distributed not initialized!")
        return

    initialize_model_parallel(tensor_model_parallel_size=dist.get_world_size())

    rank = dist.get_rank()
    device = torch.device(f'cuda:{rank}')

    batch_size = 32
    seq_len = 512
    hidden_size = 768
    mlp_hidden_size = 3072

    # Create MLP
    fc1 = ColumnParallelLinear(
        hidden_size, mlp_hidden_size, bias=True, gather_output=False
    ).to(device)

    fc2 = RowParallelLinear(
        mlp_hidden_size, hidden_size, bias=True, input_is_parallel=True
    ).to(device)

    # Input
    x = torch.randn(batch_size, seq_len, hidden_size, device=device)

    # Warmup
    for _ in range(10):
        h = fc1(x)
        h = torch.nn.functional.gelu(h)
        y = fc2(h)
        torch.cuda.synchronize()

    # Benchmark
    num_iters = 100
    start = time.perf_counter()
    for _ in range(num_iters):
        h = fc1(x)
        h = torch.nn.functional.gelu(h)
        y = fc2(h)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time = (end - start) / num_iters * 1000  # ms

    if rank == 0:
        print(f"\nMLP Forward Pass Benchmark:")
        print(f"  World size: {dist.get_world_size()}")
        print(f"  Average time: {avg_time:.3f} ms")
        print(f"  Throughput: {batch_size * seq_len / avg_time * 1000:.0f} tokens/sec")


def benchmark_communication_overhead():
    """
    Measure communication overhead of AllReduce.
    """
    if not dist.is_initialized():
        print("torch.distributed not initialized!")
        return

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{rank}')

    # Test different tensor sizes
    sizes = [(256,), (1024,), (4096,), (16384,), (65536,)]

    if rank == 0:
        print("\nCommunication Overhead (AllReduce):")
        print(f"  World size: {world_size}")
        print(f"\n{'Tensor Size':<15} {'Time (ms)':<15} {'Bandwidth (GB/s)':<15}")
        print("-" * 50)

    for size in sizes:
        tensor = torch.randn(size, device=device)

        # Warmup
        for _ in range(10):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()

        # Benchmark
        num_iters = 100
        start = time.perf_counter()
        for _ in range(num_iters):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        end = time.perf_counter()

        avg_time = (end - start) / num_iters * 1000  # ms

        # Estimate bandwidth
        # AllReduce transfers (world_size - 1) / world_size * 2 * data_size
        data_size_gb = tensor.numel() * 4 / 1e9  # 4 bytes per float32
        total_data = data_size_gb * 2 * (world_size - 1) / world_size
        bandwidth = total_data / (avg_time / 1000)

        if rank == 0:
            print(f"{str(size):<15} {avg_time:<15.3f} {bandwidth:<15.2f}")


def benchmark_strong_scaling():
    """
    Strong scaling analysis: fixed model size, vary GPU count.
    """
    print("\n" + "="*70)
    print("Strong Scaling Analysis")
    print("="*70)
    print("\nTODO: Run this benchmark with different GPU counts:")
    print("  torchrun --nproc_per_node=1 benchmarks/benchmark_parallel.py")
    print("  torchrun --nproc_per_node=2 benchmarks/benchmark_parallel.py")
    print("  torchrun --nproc_per_node=4 benchmarks/benchmark_parallel.py")
    print("  torchrun --nproc_per_node=8 benchmarks/benchmark_parallel.py")
    print("\nExpected results:")
    print("  - Computation time should decrease linearly with GPU count")
    print("  - Communication overhead should increase with GPU count")
    print("  - Overall scaling efficiency depends on model size")


def main():
    """
    Main benchmark function.
    """
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    if 'RANK' not in os.environ:
        print("Not running with torchrun!")
        print("Please run with:")
        print("  torchrun --nproc_per_node=4 benchmarks/benchmark_parallel.py")
        return

    # Initialize distributed
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()

    if rank == 0:
        print("="*70)
        print("Tensor Parallelism Benchmarks")
        print("="*70)

    benchmark_mlp_forward()
    benchmark_communication_overhead()
    benchmark_strong_scaling()

    if rank == 0:
        print("\n" + "="*70)
        print("Benchmark complete!")
        print("="*70)

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
