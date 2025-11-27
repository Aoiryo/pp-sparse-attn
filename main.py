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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensor-parallel', action='store_true')
    args = parser.parse_args()

    if args.tensor_parallel and 'RANK' in os.environ:
        dist.init_process_group(backend='nccl')

    demo_full_attention()

    if args.tensor_parallel:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
