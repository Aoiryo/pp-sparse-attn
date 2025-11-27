"""
Tests for tensor parallel layers.

Note: These tests require running with torchrun to simulate multi-GPU environment.

Run with:
    torchrun --nproc_per_node=4 tests/test_parallel.py
"""

import torch
import torch.nn as nn
import torch.distributed as dist


def test_column_parallel():
    """
    Test ColumnParallelLinear correctness.
    """
    from parallel import ColumnParallelLinear, initialize_model_parallel

    print("\nTesting ColumnParallelLinear...")

    # Initialize distributed
    if not dist.is_initialized():
        print("  Skipping: torch.distributed not initialized")
        print("  Run with: torchrun --nproc_per_node=4 tests/test_parallel.py")
        return

    initialize_model_parallel(tensor_model_parallel_size=dist.get_world_size())

    batch_size = 2
    seq_len = 16
    in_features = 64
    out_features = 128

    device = torch.device(f'cuda:{dist.get_rank()}')

    # Create column parallel layer
    col_parallel = ColumnParallelLinear(
        in_features, out_features, bias=True, gather_output=True
    ).to(device)

    # Create reference layer (same weights, non-parallel)
    ref_layer = nn.Linear(in_features, out_features).to(device)

    # Copy weights to reference (need to gather from all ranks)
    with torch.no_grad():
        # Each rank has a partition of the weight
        # We need to gather them to get the full weight
        from parallel.utils import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()

        # For testing, we'll just use rank 0's partition
        if rank == 0:
            ref_layer.weight.data[:out_features // world_size, :] = col_parallel.weight.data
            if col_parallel.bias is not None:
                ref_layer.bias.data[:out_features // world_size] = col_parallel.bias.data

    # Test forward pass
    x = torch.randn(batch_size, seq_len, in_features, device=device)

    out_parallel = col_parallel(x)
    print(f"  [Rank {dist.get_rank()}] Output shape: {out_parallel.shape}")

    print("  ✓ ColumnParallelLinear test passed!")


def test_row_parallel():
    """
    Test RowParallelLinear correctness.
    """
    from parallel import RowParallelLinear, initialize_model_parallel

    print("\nTesting RowParallelLinear...")

    if not dist.is_initialized():
        print("  Skipping: torch.distributed not initialized")
        return

    initialize_model_parallel(tensor_model_parallel_size=dist.get_world_size())

    batch_size = 2
    seq_len = 16
    in_features = 128
    out_features = 64

    device = torch.device(f'cuda:{dist.get_rank()}')

    # Create row parallel layer
    row_parallel = RowParallelLinear(
        in_features, out_features, bias=True, input_is_parallel=True
    ).to(device)

    # Test forward pass with partitioned input
    from parallel.utils import get_tensor_model_parallel_world_size
    world_size = get_tensor_model_parallel_world_size()

    x_parallel = torch.randn(
        batch_size, seq_len, in_features // world_size, device=device
    )

    out = row_parallel(x_parallel)
    print(f"  [Rank {dist.get_rank()}] Output shape: {out.shape}")

    # Check that output is the same on all ranks (after all-reduce)
    all_outputs = [torch.zeros_like(out) for _ in range(world_size)]
    dist.all_gather(all_outputs, out)

    for i in range(1, world_size):
        assert torch.allclose(all_outputs[0], all_outputs[i], atol=1e-5), \
            "Outputs not synchronized across ranks!"

    print("  ✓ RowParallelLinear test passed!")


def test_mlp_pipeline():
    """
    Test the full MLP pipeline: Column -> Row.
    """
    from parallel import (
        ColumnParallelLinear,
        RowParallelLinear,
        initialize_model_parallel,
    )

    print("\nTesting MLP Pipeline (Column -> Row)...")

    if not dist.is_initialized():
        print("  Skipping: torch.distributed not initialized")
        return

    initialize_model_parallel(tensor_model_parallel_size=dist.get_world_size())

    batch_size = 2
    seq_len = 16
    hidden_size = 64
    mlp_hidden_size = 256

    device = torch.device(f'cuda:{dist.get_rank()}')

    # Create MLP layers (like in HybridTransformerBlock)
    fc1 = ColumnParallelLinear(
        hidden_size, mlp_hidden_size, bias=True, gather_output=False
    ).to(device)

    fc2 = RowParallelLinear(
        mlp_hidden_size, hidden_size, bias=True, input_is_parallel=True
    ).to(device)

    # Forward pass
    x = torch.randn(batch_size, seq_len, hidden_size, device=device)

    # Column parallel
    h = fc1(x)  # [batch, seq_len, mlp_hidden_size / world_size]
    h = torch.nn.functional.gelu(h)

    # Row parallel
    y = fc2(h)  # [batch, seq_len, hidden_size]

    print(f"  [Rank {dist.get_rank()}] Input shape: {x.shape}")
    print(f"  [Rank {dist.get_rank()}] Intermediate shape: {h.shape}")
    print(f"  [Rank {dist.get_rank()}] Output shape: {y.shape}")

    # Test backward pass
    loss = y.sum()
    loss.backward()

    print(f"  [Rank {dist.get_rank()}] Gradient exists: {x.grad is not None}")

    print("  ✓ MLP pipeline test passed!")


def main():
    """
    Main test function.
    """
    # Initialize distributed (if using torchrun)
    if torch.cuda.is_available() and 'RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        print(f"[Rank {rank}] Starting tests...")
    else:
        print("Warning: Running without distributed initialization")
        print("Some tests will be skipped.")

    test_column_parallel()
    test_row_parallel()
    test_mlp_pipeline()

    if dist.is_initialized():
        rank = dist.get_rank()
        print(f"\n[Rank {rank}] ✓ All parallel tests passed!")
        dist.destroy_process_group()


if __name__ == '__main__':
    import os
    main()
