"""
Benchmark a parallel Transformer block that combines:

- Triton full attention kernel
- Tensor model-parallel QKV / Out projections
- Tensor model-parallel MLP (feedforward)

Run with:
    torchrun --nproc_per_node=4 tests/test_transformer_block.py --batch-size 8 --seq-len 1024 --hidden-size 1024 --num-heads 16

Then repeat with nproc_per_node = 1,2,4,8 and compare forward times. 
"""

import os
import time
import argparse
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# from attention import full_attention_forward  # Triton attention kernel
from attention import sparse_attention_forward
from parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    initialize_model_parallel,
)
from parallel.utils import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

@contextmanager
def _timed(label: str, sync_cuda: bool = True):
    """Simple timing context manager with optional CUDA sync."""
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        yield
    finally:
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"  [{label}] {elapsed_ms:.2f} ms")


def _get_device():
    """Get the local CUDA device for this rank."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    rank = dist.get_rank() if dist.is_initialized() else 0
    return torch.device(f"cuda:{rank}")


# ---------------------------------------------------------------------------
# Parallel Self-Attention using Triton kernel
# ---------------------------------------------------------------------------

class ParallelSelfAttention(nn.Module):
    """
    Tensor-parallel self-attention:

    - QKV projection: ColumnParallelLinear(hidden, 3*hidden, gather_output=False)
      -> split heads across tensor-parallel ranks

    - Attention: Triton full_attention_forward on local heads only
      Q,K,V shape on each rank: [B, H_local, L, D_head]

    - Output projection: RowParallelLinear(hidden, hidden, input_is_parallel=True)
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout_p: float = 0.0, mask: torch.Tensor = None, 
                 prune_empty_blocks: bool = False,
                 block_indices: torch.Tensor = None,
                 block_counts: torch.Tensor = None,
                 max_blocks_per_q: int = None,
                 num_q_blocks: int = None,):
        super().__init__()
        assert hidden_size % num_heads == 0, \
            "hidden_size must be divisible by num_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        world_size = get_tensor_model_parallel_world_size()
        assert num_heads % world_size == 0, \
            "num_heads must be divisible by tensor_model_parallel_world_size"
        self.world_size = world_size
        self.num_heads_per_partition = num_heads // world_size
        
        self.prune_empty_blocks = prune_empty_blocks
        self.block_indices = block_indices
        self.block_counts = block_counts
        self.max_blocks_per_q = max_blocks_per_q
        self.num_q_blocks = num_q_blocks
        
        self.mask = mask
        # QKV projection (sharded on output features)
        self.qkv = ColumnParallelLinear(
            hidden_size,
            3 * hidden_size,
            bias=True,
            gather_output=False,  # keep [B, L, 3 * hidden/world_size] on each rank
        )

        # Output projection (sharded on input features; result is full hidden_size)
        self.out_proj = RowParallelLinear(
            hidden_size,
            hidden_size,
            bias=True,
            input_is_parallel=True,
        )

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: [batch, seq_len, hidden_size] (replicated across tensor-parallel ranks)
        attn_mask: currently unused (your Triton kernel is unmasked)
        """
        bsz, seq_len, _ = x.size()

        # 1) QKV projection (parallel on last dim: 3 * hidden_size / world_size)
        #    qkv_parallel: [B, L, 3 * local_hidden]
        qkv_parallel = self.qkv(x)
        local_hidden = self.hidden_size // self.world_size

        # 2) Reshape to separate heads and split Q/K/V
        #    local_hidden = num_heads_per_partition * head_dim
        #    qkv_parallel: [B, L, num_heads_local, 3 * head_dim]
        qkv_parallel = qkv_parallel.view(
            bsz,
            seq_len,
            self.num_heads_per_partition,
            3 * self.head_dim,
        )
        # [B, num_heads_local, L, 3 * head_dim]
        qkv_parallel = qkv_parallel.permute(0, 2, 1, 3).contiguous()

        q, k, v = torch.split(qkv_parallel, self.head_dim, dim=-1)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # 3) Triton full attention on local heads
        #    q,k,v: [B, H_local, L, D_head]
        #    out:   [B, H_local, L, D_head]
        # NOTE: full_attention_forward does scaling + softmax internally
        # context = full_attention_forward(q, k, v)
               
        context = sparse_attention_forward(q, k, v, self.mask, 
                                          prune_empty_blocks=self.prune_empty_blocks,
                                          block_indices=self.block_indices,
                                          block_counts=self.block_counts,
                                          max_blocks_per_q=self.max_blocks_per_q,
                                          num_q_blocks=self.num_q_blocks,)

        # 4) Merge heads back to sharded hidden dimension
        #    context: [B, L, H_local * D_head] = [B, L, local_hidden]
        context = context.permute(0, 2, 1, 3).contiguous()
        context_parallel = context.view(bsz, seq_len, local_hidden)
        context_parallel = self.dropout(context_parallel)

        # 5) Output projection (RowParallelLinear expects parallel input)
        out = self.out_proj(context_parallel)  # [B, L, hidden_size] replicated
        return out


# ---------------------------------------------------------------------------
# Parallel Feedforward (MLP)
# ---------------------------------------------------------------------------

class ParallelFeedForward(nn.Module):
    """
    Tensor-parallel MLP:

      hidden_size -> mlp_hidden_size -> hidden_size

    implemented as ColumnParallelLinear + RowParallelLinear.
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_hidden_size: int,
        dropout_p: float = 0.0,
        activation=F.gelu,
    ):
        super().__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout_p)

        self.fc1 = ColumnParallelLinear(
            hidden_size,
            mlp_hidden_size,
            bias=True,
            gather_output=False,  # [B, L, mlp_hidden/world_size]
        )
        self.fc2 = RowParallelLinear(
            mlp_hidden_size,
            hidden_size,
            bias=True,
            input_is_parallel=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, hidden_size] (replicated)
        h = self.fc1(x)                # [B, L, mlp_hidden / world_size]
        h = self.activation(h)
        h = self.dropout(h)
        out = self.fc2(h)              # [B, L, hidden_size] replicated
        return out


# ---------------------------------------------------------------------------
# Full Parallel Transformer Block
# ---------------------------------------------------------------------------

class ParallelTransformerBlock(nn.Module):
    """
    Basic Transformer block with:

    - LayerNorm
    - Parallel self-attention (Triton + tensor model parallel)
    - Parallel MLP (tensor model parallel)
    - Residual connections

    Everything that's a Linear (QKV, out_proj, MLP) is column/row-parallel.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_hidden_size: int,
        dropout_p: float = 0.0,
        mask: torch.Tensor = None,
        prune_empty_blocks: bool = False,
        block_indices: torch.Tensor = None,
        block_counts: torch.Tensor = None,
        max_blocks_per_q: int = None,
        num_q_blocks: int = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = ParallelSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_p=dropout_p,
            mask=mask,
            prune_empty_blocks=False,
            block_indices=block_indices,
            block_counts=block_counts,
            max_blocks_per_q=max_blocks_per_q,
            num_q_blocks=num_q_blocks,
        )
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = ParallelFeedForward(
            hidden_size=hidden_size,
            mlp_hidden_size=mlp_hidden_size,
            dropout_p=dropout_p,
            activation=F.gelu,
        )
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, H]
        # Self-attention sub-layer
        residual = x
        x = self.ln1(x)
        x = self.attn(x)       # [B, L, H]
        x = self.dropout(x)
        x = x + residual

        # Feedforward sub-layer
        residual = x
        x = self.ln2(x)
        x = self.mlp(x)        # [B, L, H]
        x = self.dropout(x)
        x = x + residual
        return x


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def _build_block_metadata(mask, block_m, block_n, prune_empty_blocks=True):
    """
    Build block-sparse metadata from a dense mask.

    Returns:
        block_indices: [total_q_blocks, max_blocks_per_q] int32 key block ids
        block_counts: [total_q_blocks] int32 number of valid key blocks per q block
        max_blocks_per_q: int
        num_q_blocks: int
    """
    batch_size, num_heads, seq_len, _ = mask.shape
    num_q_blocks = (seq_len + block_m - 1) // block_m
    num_k_blocks = (seq_len + block_n - 1) // block_n
    total_q_blocks = batch_size * num_heads * num_q_blocks
    device = mask.device

    block_counts = torch.zeros(total_q_blocks, device=device, dtype=torch.int32)

    if prune_empty_blocks:
        block_lists = []
        max_blocks_per_q = 0
        idx = 0
        for b in range(batch_size):
            for h in range(num_heads):
                for qb in range(num_q_blocks):
                    q_start = qb * block_m
                    q_end = min((qb + 1) * block_m, seq_len)
                    key_blocks = []
                    for kb in range(num_k_blocks):
                        k_start = kb * block_n
                        k_end = min((kb + 1) * block_n, seq_len)
                        if mask[b, h, q_start:q_end, k_start:k_end].any():
                            key_blocks.append(kb)
                    block_counts[idx] = len(key_blocks)
                    max_blocks_per_q = max(max_blocks_per_q, len(key_blocks))
                    block_lists.append(key_blocks)
                    idx += 1

        if max_blocks_per_q == 0:
            max_blocks_per_q = 1  # avoid zero-sized tensors

        block_indices = torch.zeros(
            (total_q_blocks, max_blocks_per_q), device=device, dtype=torch.int32
        )
        for i, key_blocks in enumerate(block_lists):
            if key_blocks:
                block_indices[i, :len(key_blocks)] = torch.tensor(
                    key_blocks, device=device, dtype=torch.int32
                )
    else:
        # No pruning: all query blocks attend to all key blocks
        max_blocks_per_q = num_k_blocks
        # Create indices: each row is [0, 1, 2, ..., num_k_blocks-1]
        block_indices = torch.arange(
            num_k_blocks, device=device, dtype=torch.int32
        ).unsqueeze(0).repeat(total_q_blocks, 1)
        block_counts.fill_(num_k_blocks)

    return block_indices, block_counts, max_blocks_per_q, num_q_blocks


def benchmark_block(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_heads: int,
    mlp_hidden_size: int,
    warmup_iters: int = 5,
    bench_iters: int = 20,
):
    device = _get_device()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(
            f"\n=== Benchmark ParallelTransformerBlock ===\n"
            f"  world_size       : {world_size}\n"
            f"  batch_size       : {batch_size}\n"
            f"  seq_len          : {seq_len}\n"
            f"  hidden_size      : {hidden_size}\n"
            f"  num_heads        : {num_heads}\n"
            f"  mlp_hidden_size  : {mlp_hidden_size}\n"
        )

    # Build block
    mask = torch.zeros(batch_size, num_heads // world_size, seq_len, seq_len, dtype=torch.bool, device=device)
    window_size = 32
    for i in range(seq_len):
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2)
        mask[:, :, i, start:end] = True
        
    def _sync():
        if device.startswith('cuda'):
            torch.cuda.synchronize()

    # Precompute metadata to exclude CPU-side building from timing
    block_indices_dense, block_counts_dense, max_blocks_per_q_dense, num_q_blocks_dense = \
        _build_block_metadata(mask, block_m=64, block_n=64, prune_empty_blocks=False)
    block_indices_pruned, block_counts_pruned, max_blocks_per_q_pruned, num_q_blocks_pruned = \
        _build_block_metadata(mask, block_m=64, block_n=64, prune_empty_blocks=True)
    _sync()

    block = ParallelTransformerBlock(
        hidden_size=hidden_size,
        num_heads=num_heads,
        mlp_hidden_size=mlp_hidden_size,
        dropout_p=0.0,  # disable dropout for clean timing
        mask=mask,
        prune_empty_blocks=True,
        block_indices=block_indices_pruned,
        block_counts=block_counts_pruned,
        max_blocks_per_q=max_blocks_per_q_pruned,
        num_q_blocks=num_q_blocks_pruned,        
    ).to(device)
    block.eval()

    # Dummy input (replicated on all ranks)
    x = torch.randn(batch_size, seq_len, hidden_size, device=device)

    # Warm-up (to trigger Triton/JIT compilation etc.)
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = block(x)

    # Benchmark forward pass
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(bench_iters):
            _ = block(x)
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / bench_iters

    # We care about the max time across ranks (bottleneck)
    elapsed_tensor = torch.tensor([elapsed_ms], device=device)
    dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
    max_elapsed_ms = elapsed_tensor.item()

    if rank == 0:
        print(f"  Avg forward time per iter (max over ranks): {max_elapsed_ms:.2f} ms\n")
        print("  >>> To get speedup, run this script with nproc_per_node=1,2,4,8")
        print("  >>> and compare the reported times from world_size=1 as baseline.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark parallel Transformer block (Triton + tensor parallel)"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--mlp-hidden-size", type=int, default=4096)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--bench-iters", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    # Require torchrun (so that dist is properly initialized)
    if "RANK" not in os.environ:
        raise RuntimeError(
            "This script is meant to be run with torchrun, e.g.:\n"
            "  torchrun --nproc_per_node=4 tests/test_parallel_transformer_block.py"
        )

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Set device for this rank
    torch.cuda.set_device(rank)

    if rank == 0:
        print(f"[Rank {rank}] Distributed initialized with world_size={world_size}")

    # Initialize tensor model parallel using all ranks
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    benchmark_block(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        mlp_hidden_size=args.mlp_hidden_size,
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
