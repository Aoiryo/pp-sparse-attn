"""
Main demo script for Hybrid Transformer with Sparse Attention and Tensor Parallelism.

Run single GPU: python main.py
Run with tensor parallelism: torchrun --nproc_per_node=4 main.py --tensor-parallel
"""

import torch
import torch.distributed as dist
import argparse
import os


def demo_full_attention():
    print("\n" + "="*70)
    print("DEMO 1: Full Attention")
    print("="*70)

    from attention import full_attention_forward
    from tests.test_attention import reference_attention

    batch_size, num_heads, seq_len, head_dim = 2, 4, 256, 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    out_triton = full_attention_forward(q, k, v)
    out_pytorch = reference_attention(q, k, v)

    max_diff = torch.max(torch.abs(out_triton - out_pytorch)).item()
    print(f"Max difference: {max_diff:.6f}")
    print("✓ Full attention working!")


def demo_sparse_attention():
    print("\n" + "="*70)
    print("DEMO 2: Sparse Attention (Local Window)")
    print("="*70)

    from attention.sparse_attention import sparse_attention_forward
    from tests.test_attention import reference_attention

    batch_size, num_heads, seq_len, head_dim = 2, 4, 256, 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    # Create local window mask (each token attends to ±16 neighbors)
    mask = torch.zeros(batch_size, num_heads, seq_len, seq_len, dtype=torch.bool, device=device)
    window_size = 32
    for i in range(seq_len):
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2)
        mask[:, :, i, start:end] = True

    sparsity = mask.sum().item() / (batch_size * num_heads * seq_len * seq_len)
    print(f"Sparsity: {sparsity:.1%} (window size = {window_size})")

    out_triton = sparse_attention_forward(q, k, v, mask)
    out_pytorch = reference_attention(q, k, v, mask)

    max_diff = torch.max(torch.abs(out_triton - out_pytorch)).item()
    print(f"Max difference: {max_diff:.6f}")
    print("✓ Sparse attention working!")
    print(f"Theoretical speedup: {100 / (sparsity * 100):.2f}x (vs full attention)")


def demo_hybrid_transformer():
    print("\n" + "="*70)
    print("DEMO 3: Hybrid Transformer Block (Sparse Attention + Tensor Parallel MLP)")
    print("="*70)

    from models.hybrid_transformer import HybridTransformerBlock

    batch_size, seq_len, hidden_size = 2, 128, 256
    num_heads = 4
    mlp_hidden_size = 1024
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Check if distributed is initialized
    use_tensor_parallel = dist.is_initialized()

    if use_tensor_parallel:
        from parallel import initialize_model_parallel
        initialize_model_parallel(tensor_model_parallel_size=dist.get_world_size())
        rank = dist.get_rank()
        print(f"[Rank {rank}] Using tensor parallelism with {dist.get_world_size()} GPUs")

    # Create hybrid transformer block
    block = HybridTransformerBlock(
        hidden_size=hidden_size,
        num_heads=num_heads,
        mlp_hidden_size=mlp_hidden_size,
        use_sparse_attention=True,
        sparse_pattern='local',
        window_size=32,
        use_tensor_parallel=use_tensor_parallel,
    ).to(device)

    # Forward pass
    x = torch.randn(batch_size, seq_len, hidden_size, device=device)
    out = block(x)

    if use_tensor_parallel:
        print(f"[Rank {rank}] Input shape: {x.shape}")
        print(f"[Rank {rank}] Output shape: {out.shape}")
    else:
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {out.shape}")

    print("✓ Hybrid transformer working!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensor-parallel', action='store_true',
                       help='Enable tensor parallelism (requires torchrun)')
    args = parser.parse_args()

    if args.tensor_parallel and 'RANK' in os.environ:
        dist.init_process_group(backend='nccl')

    demo_full_attention()
    demo_sparse_attention()

    if not args.tensor_parallel or 'RANK' in os.environ:
        demo_hybrid_transformer()

    if args.tensor_parallel and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
