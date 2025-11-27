"""
Sparse Attention Implementation using Triton.

This module implements various sparse attention patterns:
1. Block-diagonal attention
2. Local window attention (Longformer-style)
3. Random sparse attention

Key challenges:
- Irregular memory access patterns
- Load imbalance across GPU threads
- Efficient mask/index representation
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# TODO: Implement Sparse Attention Kernels
# ============================================================================
#
# Your task is to implement the sparse attention kernel that only computes
# attention for a subset of (query, key) pairs defined by a sparsity pattern.
#
# Key differences from full attention:
# 1. Not all queries attend to all keys - use sparse mask/indices
# 2. Need to handle irregular workload distribution
# 3. Memory access patterns are non-contiguous
#
# Recommended approach:
# 1. Start with a simple block-diagonal pattern (each query only attends to
#    keys in the same block)
# 2. Use a mask tensor to indicate which (q, k) pairs to compute
# 3. Focus on correctness first, then optimize for performance
#
# Hints:
# - You can pass a mask tensor to the kernel and check it before computing QK^T
# - Consider using block-sparse representation for efficiency
# - The online softmax algorithm still applies, but only over valid keys
# ============================================================================


@triton.jit
def _sparse_attention_forward_kernel(
    Q, K, V, Mask, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_mb, stride_mh, stride_mm, stride_mn,
    stride_ob, stride_oh, stride_om, stride_od,
    batch_size, num_heads, seq_len, head_dim,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    """
    Sparse attention forward kernel.

    TODO: Implement this kernel!

    Key ideas:
    1. Similar structure to full attention, but skip blocks where mask is all False
    2. Use the Mask tensor to determine which (query, key) pairs to compute
    3. Apply mask BEFORE computing softmax to avoid numerical issues

    Mask format: [batch, num_heads, seq_len, seq_len]
        - Mask[b, h, i, j] = 1 if query i should attend to key j
        - Mask[b, h, i, j] = 0 otherwise

    Optimization ideas (for later):
    - Pre-compute which blocks are non-empty to skip entire blocks
    - Use block-sparse format instead of dense mask
    - Load balance by sorting blocks by sparsity
    """

    # TODO: Implement sparse attention kernel
    # For now, this is a placeholder that raises an error
    pass


def sparse_attention_forward(q, k, v, mask):
    """
    Sparse attention forward pass.

    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_heads, seq_len, head_dim]
        v: Value tensor [batch, num_heads, seq_len, head_dim]
        mask: Sparse attention mask [batch, num_heads, seq_len, seq_len]
              1 = attend, 0 = ignore

    Returns:
        out: Output tensor [batch, num_heads, seq_len, head_dim]

    TODO: Implement this function!
    """

    # TODO: Implement sparse attention
    # Hints:
    # 1. Check mask shape and ensure it matches q, k, v
    # 2. Allocate output tensor
    # 3. Choose appropriate BLOCK_M, BLOCK_N sizes
    # 4. Launch the sparse attention kernel
    # 5. Return the output

    raise NotImplementedError(
        "Sparse attention not yet implemented. "
        "This is your TODO! Implement the sparse attention kernel above."
    )


class SparseAttention(torch.nn.Module):
    """
    Sparse Attention module.

    Supports multiple sparsity patterns:
    - 'local': Local window attention (attend to nearby tokens)
    - 'block': Block-diagonal attention
    - 'random': Random sparse attention
    """

    def __init__(self, pattern='local', window_size=256):
        super().__init__()
        self.pattern = pattern
        self.window_size = window_size

    def create_mask(self, seq_len, batch_size, num_heads, device):
        """
        Create sparse attention mask based on pattern.

        Returns:
            mask: [batch, num_heads, seq_len, seq_len] boolean tensor
        """

        if self.pattern == 'local':
            # Local window: each token attends to tokens within window_size
            mask = torch.zeros(batch_size, num_heads, seq_len, seq_len,
                             dtype=torch.bool, device=device)
            for i in range(seq_len):
                start = max(0, i - self.window_size // 2)
                end = min(seq_len, i + self.window_size // 2)
                mask[:, :, i, start:end] = True

        elif self.pattern == 'block':
            # Block-diagonal: divide sequence into blocks
            block_size = self.window_size
            mask = torch.zeros(batch_size, num_heads, seq_len, seq_len,
                             dtype=torch.bool, device=device)
            num_blocks = (seq_len + block_size - 1) // block_size
            for b in range(num_blocks):
                start = b * block_size
                end = min((b + 1) * block_size, seq_len)
                mask[:, :, start:end, start:end] = True

        elif self.pattern == 'random':
            # Random sparse: each query attends to random subset of keys
            sparsity = self.window_size / seq_len  # sparsity ratio
            mask = torch.rand(batch_size, num_heads, seq_len, seq_len,
                            device=device) < sparsity

        else:
            raise ValueError(f"Unknown sparsity pattern: {self.pattern}")

        return mask

    def forward(self, q, k, v):
        """
        Forward pass with sparse attention.

        Args:
            q, k, v: [batch, num_heads, seq_len, head_dim]
        Returns:
            out: [batch, num_heads, seq_len, head_dim]
        """
        batch_size, num_heads, seq_len, head_dim = q.shape

        # Create sparse mask
        mask = self.create_mask(seq_len, batch_size, num_heads, q.device)

        # TODO: Call sparse_attention_forward once implemented
        # return sparse_attention_forward(q, k, v, mask)

        # Temporary: use PyTorch implementation for testing
        return self._pytorch_sparse_attention(q, k, v, mask)

    def _pytorch_sparse_attention(self, q, k, v, mask):
        """
        Reference PyTorch implementation for correctness testing.

        This is SLOW but correct - use for validation only!
        """
        scale = 1.0 / (q.size(-1) ** 0.5)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply mask: set masked positions to -inf before softmax
        attn = attn.masked_fill(~mask, float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        attn = attn.masked_fill(torch.isnan(attn), 0.0)  # Handle all -inf rows

        out = torch.matmul(attn, v)
        return out


# ============================================================================
# Utility functions for creating different sparsity patterns
# ============================================================================

def create_local_mask(seq_len, window_size, device='cuda'):
    """
    Create a local window attention mask (Longformer-style).

    Args:
        seq_len: Sequence length
        window_size: Size of local attention window
        device: Device to create mask on

    Returns:
        mask: [seq_len, seq_len] boolean tensor
    """
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    for i in range(seq_len):
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2)
        mask[i, start:end] = True
    return mask


def create_block_diagonal_mask(seq_len, block_size, device='cuda'):
    """
    Create a block-diagonal attention mask.

    Args:
        seq_len: Sequence length
        block_size: Size of each diagonal block
        device: Device to create mask on

    Returns:
        mask: [seq_len, seq_len] boolean tensor
    """
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    num_blocks = (seq_len + block_size - 1) // block_size
    for b in range(num_blocks):
        start = b * block_size
        end = min((b + 1) * block_size, seq_len)
        mask[start:end, start:end] = True
    return mask


def create_random_sparse_mask(seq_len, sparsity, device='cuda'):
    """
    Create a random sparse attention mask.

    Args:
        seq_len: Sequence length
        sparsity: Fraction of entries to keep (0 to 1)
        device: Device to create mask on

    Returns:
        mask: [seq_len, seq_len] boolean tensor
    """
    mask = torch.rand(seq_len, seq_len, device=device) < sparsity
    return mask
