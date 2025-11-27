"""
Tests for attention modules.

Tests:
1. Full attention correctness vs PyTorch reference
2. Sparse attention correctness (once implemented)
3. Numerical stability tests
"""

import torch
import torch.nn.functional as F


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
    assert max_diff < 1e-3, f"Max difference too large: {max_diff}"
    assert mean_diff < 1e-4, f"Mean difference too large: {mean_diff}"

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


def test_attention_backward():
    """
    Test that gradients flow correctly through attention.
    """
    from attention import full_attention_forward

    print("\nTesting Attention Backward Pass...")

    batch_size = 2
    num_heads = 4
    seq_len = 64
    head_dim = 32

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    print("  ✓ Backward pass test passed!")


if __name__ == '__main__':
    test_full_attention_correctness()
    test_sparse_attention_patterns()
    test_attention_backward()
    print("\n✓ All attention tests passed!")
