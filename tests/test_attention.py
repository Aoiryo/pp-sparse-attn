"""
Tests for attention modules.

Tests:
1. Full attention correctness vs PyTorch reference
2. Sparse attention correctness (once implemented)
3. Numerical stability tests
"""

import time
from contextlib import contextmanager

import torch
import torch.nn.functional as F


@contextmanager
def _timed(label: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"  [{label}] {elapsed_ms:.2f} ms")


def reference_attention(q, k, v, mask=None):
    """
    Reference PyTorch implementation of attention.

    This is slow but correct - use for validation only!
    """
    scale = 1.0 / (q.size(-1) ** 0.5)
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale

    if mask is not None:
        attn = attn.masked_fill(~mask, float('-inf'))

    attn = F.softmax(attn, dim=-1)
    attn = attn.masked_fill(torch.isnan(attn), 0.0)

    out = torch.matmul(attn, v)
    return out


def test_full_attention_correctness():
    """
    Test that our Triton full attention matches PyTorch reference.
    """
    from attention import full_attention_forward

    print("Testing Full Attention Correctness...")

    batch_size = 2
    num_heads = 4
    seq_len = 128
    head_dim = 64

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with _timed("full_attention_correctness"):
        # Generate random Q, K, V
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

        # Compute with Triton
        out_triton = full_attention_forward(q, k, v)

        # Compute with PyTorch reference
        out_ref = reference_attention(q, k, v)

        # Check correctness
        max_diff = torch.max(torch.abs(out_triton - out_ref)).item()
        mean_diff = torch.mean(torch.abs(out_triton - out_ref)).item()

        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")

        # Assert correctness (allow small numerical error)
        # Note: Triton uses float32, so we allow slightly larger tolerance
        assert max_diff < 5e-3, f"Max difference too large: {max_diff}"
        assert mean_diff < 1e-3, f"Mean difference too large: {mean_diff}"

    print("  ✓ Full attention correctness test passed!")


def test_sparse_attention_patterns():
    """
    Test that sparse attention mask generation works correctly.
    """
    from attention.sparse_attention import (
        create_local_mask,
        create_block_diagonal_mask,
        create_random_sparse_mask,
    )

    print("\nTesting Sparse Attention Mask Generation...")

    seq_len = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with _timed("sparse_attention_patterns"):
        # Test local window mask
        local_mask = create_local_mask(seq_len, window_size=32, device=device)
        assert local_mask.shape == (seq_len, seq_len)
        print(f"  ✓ Local mask: {local_mask.sum().item()} / {seq_len * seq_len} entries")

        # Test block diagonal mask
        block_mask = create_block_diagonal_mask(seq_len, block_size=32, device=device)
        assert block_mask.shape == (seq_len, seq_len)
        print(f"  ✓ Block mask: {block_mask.sum().item()} / {seq_len * seq_len} entries")

        # Test random sparse mask
        random_mask = create_random_sparse_mask(seq_len, sparsity=0.25, device=device)
        assert random_mask.shape == (seq_len, seq_len)
        sparsity_ratio = random_mask.sum().item() / (seq_len * seq_len)
        print(f"  ✓ Random mask: {random_mask.sum().item()} / {seq_len * seq_len} entries "
              f"(sparsity: {sparsity_ratio:.2%})")


def test_sparse_attention_correctness():
    """
    Test that Triton sparse attention matches PyTorch reference.
    """
    from attention.sparse_attention import sparse_attention_forward

    print("\nTesting Sparse Attention Correctness...")

    batch_size = 2
    num_heads = 4
    seq_len = 128
    head_dim = 64

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with _timed("sparse_attention_correctness"):
        # Generate random Q, K, V
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

        # Create a local window mask (25% sparsity)
        mask = torch.zeros(batch_size, num_heads, seq_len, seq_len, dtype=torch.bool, device=device)
        window_size = 32
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2)
            mask[:, :, i, start:end] = True

        sparsity = mask.sum().item() / (batch_size * num_heads * seq_len * seq_len)
        print(f"  Mask sparsity: {sparsity:.1%}")

        # Compute with Triton
        out_triton = sparse_attention_forward(q, k, v, mask)

        # Compute with PyTorch reference
        out_ref = reference_attention(q, k, v, mask)

        # Check correctness
        max_diff = torch.max(torch.abs(out_triton - out_ref)).item()
        mean_diff = torch.mean(torch.abs(out_triton - out_ref)).item()

        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")

        # Assert correctness (same tolerance as full attention)
        assert max_diff < 5e-3, f"Max difference too large: {max_diff}"
        assert mean_diff < 1e-3, f"Mean difference too large: {mean_diff}"

    print("  ✓ Sparse attention correctness test passed!")


def test_attention_backward():
    """
    Test that gradients flow correctly through attention.
    """
    from attention import full_attention_forward

    print("\nTesting Full Attention Backward Pass...")

    batch_size = 2
    num_heads = 4
    seq_len = 64
    head_dim = 32

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with _timed("full_attention_backward"):
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, requires_grad=True)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, requires_grad=True)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, requires_grad=True)

        # Forward pass
        out = full_attention_forward(q, k, v)

        # Backward pass
        loss = out.sum()
        loss.backward()

        # Check that gradients exist
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None

        print(f"  Gradient norms: q={q.grad.norm().item():.4f}, "
              f"k={k.grad.norm().item():.4f}, v={v.grad.norm().item():.4f}")
    print("  ✓ Full attention backward pass test passed!")


def test_sparse_attention_backward():
    """
    Test that gradients flow correctly through sparse attention.
    """
    from attention.sparse_attention import sparse_attention_forward

    print("\nTesting Sparse Attention Backward Pass...")

    batch_size = 2
    num_heads = 4
    seq_len = 64
    head_dim = 32

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with _timed("sparse_attention_backward"):
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, requires_grad=True)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, requires_grad=True)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, requires_grad=True)

        # Create local window mask
        mask = torch.zeros(batch_size, num_heads, seq_len, seq_len, dtype=torch.bool, device=device)
        window_size = 16
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2)
            mask[:, :, i, start:end] = True

        # Forward pass
        out = sparse_attention_forward(q, k, v, mask)

        # Backward pass
        loss = out.sum()
        loss.backward()

        # Check that gradients exist
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None

        print(f"  Gradient norms: q={q.grad.norm().item():.4f}, "
              f"k={k.grad.norm().item():.4f}, v={v.grad.norm().item():.4f}")
    print("  ✓ Sparse attention backward pass test passed!")


def test_sparse_attention_perf_compare():
    """
    Compare runtime before/after skipping fully masked tiles.
    """
    from attention.sparse_attention import sparse_attention_forward, _build_block_metadata

    print("\nBenchmarking Sparse Attention (dense blocks vs pruned)...")

    batch_size = 2
    num_heads = 4
    seq_len = 256
    head_dim = 64
    window_size = 32

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Inputs and local mask
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    mask = torch.zeros(batch_size, num_heads, seq_len, seq_len, dtype=torch.bool, device=device)
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

    # Warmup to exclude compilation from timing
    with torch.no_grad():
        sparse_attention_forward(
            q, k, v, mask,
            prune_empty_blocks=False,
            block_indices=block_indices_dense,
            block_counts=block_counts_dense,
            max_blocks_per_q=max_blocks_per_q_dense,
            num_q_blocks=num_q_blocks_dense,
        )
        sparse_attention_forward(
            q, k, v, mask,
            prune_empty_blocks=True,
            block_indices=block_indices_pruned,
            block_counts=block_counts_pruned,
            max_blocks_per_q=max_blocks_per_q_pruned,
            num_q_blocks=num_q_blocks_pruned,
        )
        _sync()

    def run_and_time(label, prune, block_indices, block_counts, max_blocks_per_q, num_q_blocks):
        _sync()
        with _timed(label):
            out = sparse_attention_forward(
                q, k, v, mask,
                prune_empty_blocks=prune,
                block_indices=block_indices,
                block_counts=block_counts,
                max_blocks_per_q=max_blocks_per_q,
                num_q_blocks=num_q_blocks,
            )
            _sync()
        return out

    out_dense = run_and_time(
        "sparse_dense_blocks",
        prune=False,
        block_indices=block_indices_dense,
        block_counts=block_counts_dense,
        max_blocks_per_q=max_blocks_per_q_dense,
        num_q_blocks=num_q_blocks_dense,
    )
    out_pruned = run_and_time(
        "sparse_pruned_blocks",
        prune=True,
        block_indices=block_indices_pruned,
        block_counts=block_counts_pruned,
        max_blocks_per_q=max_blocks_per_q_pruned,
        num_q_blocks=num_q_blocks_pruned,
    )

    diff = torch.max(torch.abs(out_dense - out_pruned)).item()
    print(f"  Output diff between modes: {diff:.6f}")
    assert diff < 5e-3, "Outputs diverged between dense/pruned block paths"

    print("  ✓ Sparse attention perf comparison complete!")


if __name__ == '__main__':
    test_full_attention_correctness()
    test_sparse_attention_patterns()
    test_sparse_attention_correctness()
    test_attention_backward()
    test_sparse_attention_backward()
    test_sparse_attention_perf_compare()
    print("\n✓ All attention tests passed!")
