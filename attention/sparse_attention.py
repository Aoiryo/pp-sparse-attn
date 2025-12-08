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
    Q, K, V, Mask, BlockCounts, BlockIndices, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_mb, stride_mh, stride_mm, stride_mn,
    stride_ob, stride_oh, stride_om, stride_od,
    batch_size, num_heads, num_q_blocks, seq_len, head_dim,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    MAX_BLOCKS_PER_Q: tl.constexpr,
):
    """
    Sparse attention forward kernel with block-level early exit optimization.

    Key changes from full attention:
    1. Added Mask parameter and strides
    2. Load mask block FIRST in main loop
    3. Block-level early exit: if mask_block is all False, skip K/V loading and computation
    4. Apply mask to QK^T scores (set masked positions to -inf)

    Optimization:
    - For sparse patterns, many blocks have mask=all False
    - We check this BEFORE loading K/V to save memory bandwidth
    - This can provide 2-8x speedup depending on sparsity

    Mask format: [batch, num_heads, seq_len, seq_len]
        - Mask[b, h, i, j] = True if query i should attend to key j
        - Mask[b, h, i, j] = False otherwise

    The beauty: exp(-inf) = 0, so masked positions automatically get zero weight!
    """

    # Get program IDs
    pid_m = tl.program_id(0)    # Which query block
    pid_bh = tl.program_id(1)   # Which (batch, head) combination

    # Decompose batch and head indices
    batch_idx = pid_bh // num_heads
    head_idx = pid_bh % num_heads

    # Compute offsets for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # Query positions [BLOCK_M]
    offs_n = tl.arange(0, BLOCK_N)                     # Key positions [BLOCK_N]
    offs_d = tl.arange(0, BLOCK_DMODEL)                # Feature dimensions [BLOCK_DMODEL]

    # Base pointers for this (batch, head)
    q_offset = batch_idx * stride_qb + head_idx * stride_qh
    k_offset = batch_idx * stride_kb + head_idx * stride_kh
    v_offset = batch_idx * stride_vb + head_idx * stride_vh
    mask_offset = batch_idx * stride_mb + head_idx * stride_mh  # ← 🆕 Mask offset!

    # 加载 Q（与 full attention 完全相同）
    # Load Q block: [BLOCK_M, BLOCK_DMODEL]
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seq_len, other=0.0)

    # Online Softmax 初始化（与 full attention 完全相同）
    # Accumulators for online softmax
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')  # Running max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                 # Running sum of exp

    # Scale factor for attention scores
    scale = 1.0 / tl.sqrt(head_dim * 1.0)

    # Load block metadata for this (b,h,q_block)
    bhq_offset = pid_bh * num_q_blocks + pid_m
    blocks_per_q = tl.load(BlockCounts + bhq_offset)

    for idx in range(MAX_BLOCKS_PER_Q):
        # Only process valid entries (tensor mask)
        process_block = idx < blocks_per_q
        key_block_id = tl.load(
            BlockIndices + bhq_offset * MAX_BLOCKS_PER_Q + idx,
            mask=process_block,
            other=0,
        )
        start_n = key_block_id * BLOCK_N
        offs_n_cur = start_n + offs_n  # Current key positions

        # Mask block: [BLOCK_M, BLOCK_N]
        mask_ptrs = Mask + mask_offset + \
                    offs_m[:, None] * stride_mm + \
                    offs_n_cur[None, :] * stride_mn
        mask_block = tl.load(
            mask_ptrs,
            mask=process_block & (offs_m[:, None] < seq_len) & (offs_n_cur[None, :] < seq_len),
            other=False
        )

        # Load K block: [BLOCK_N, BLOCK_DMODEL]
        k_ptrs = K + k_offset + offs_n_cur[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=process_block & (offs_n_cur[:, None] < seq_len), other=0.0)

        # Compute QK^T: [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, tl.trans(k)) * scale

        qk = tl.where(process_block, qk, float('-inf'))
        qk = tl.where(mask_block, qk, float('-inf'))
        qk = tl.where(offs_n_cur[None, :] < seq_len, qk, float('-inf'))

        # Online Softmax
        # Update running maximum
        m_ij = tl.max(qk, axis=1)         # Max of current block
        m_new = tl.maximum(m_i, m_ij)     # Update global max

        # Compute exponentials (numerically stable)
        # Handle case where qk is all -inf: exp(-inf - (-inf)) = exp(nan) = nan
        # We set p = 0 when qk is -inf to avoid nan
        p = tl.exp(qk - m_new[:, None])
        # Replace nan with 0 (happens when qk is all -inf for this block)
        p = tl.where(qk > float('-inf'), p, 0.0)

        # Update normalization factor
        alpha = tl.exp(m_i - m_new)       # Correction factor
        # If m_new is still -inf (all blocks so far are masked), alpha will be nan
        # In this case, we should keep l_i = 0 and acc = 0
        alpha = tl.where(m_new > float('-inf'), alpha, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)

        # Load V block: [BLOCK_N, BLOCK_DMODEL]
        v_ptrs = V + v_offset + offs_n_cur[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=process_block & (offs_n_cur[:, None] < seq_len), other=0.0)

        # Update accumulator
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    acc = tl.where(l_i[:, None] > 0, acc / l_i[:, None], 0.0)

    # Write output
    o_offset = batch_idx * stride_ob + head_idx * stride_oh
    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seq_len)


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


class _SparseAttentionFunction(torch.autograd.Function):
    """
    Autograd wrapper for sparse attention Triton kernel.

    Forward: Uses Triton kernel for efficiency
    Backward: Uses PyTorch fallback for gradient computation (like full attention)
    """

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        mask,
        prune_empty_blocks=True,
        block_indices=None,
        block_counts=None,
        max_blocks_per_q=None,
        num_q_blocks=None,
    ):
        """
        Forward pass using Triton kernel.

        Args:
            q, k, v: [batch, num_heads, seq_len, head_dim]
            mask: [batch, num_heads, seq_len, seq_len] boolean tensor
            prune_empty_blocks: precompute and skip fully masked tiles for bandwidth savings
            block_indices/block_counts: precomputed block metadata (optional)
            max_blocks_per_q/num_q_blocks: metadata dimensions (required if providing block_indices)
        """
        # Validate shapes
        batch_size, num_heads, seq_len, head_dim = q.shape
        assert k.shape == q.shape, "K shape must match Q"
        assert v.shape == q.shape, "V shape must match Q"
        assert mask.shape == (batch_size, num_heads, seq_len, seq_len), \
            f"Mask shape {mask.shape} doesn't match expected {(batch_size, num_heads, seq_len, seq_len)}"

        # Ensure contiguous
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        mask = mask.contiguous()

        # Allocate output
        out = torch.empty_like(q)

        # Choose block sizes adaptively based on GPU capability
        from attention.full_attention import _get_block_sizes
        # BLOCK_M, BLOCK_N, BLOCK_DMODEL = _get_block_sizes(head_dim, q.device)
        BLOCK_M, BLOCK_N, BLOCK_DMODEL = 64, 64, head_dim

        # Build block metadata on the fly if not provided
        if block_indices is None or block_counts is None:
            block_indices, block_counts, max_blocks_per_q, num_q_blocks = _build_block_metadata(
                mask, BLOCK_M, BLOCK_N, prune_empty_blocks=prune_empty_blocks
            )
        else:
            # Use precomputed metadata
            assert max_blocks_per_q is not None and num_q_blocks is not None, \
                "max_blocks_per_q and num_q_blocks must be provided with precomputed metadata"

        block_indices = block_indices.contiguous()
        block_counts = block_counts.contiguous()

        # Grid: (num_query_blocks, batch * num_heads)
        grid = (
            num_q_blocks,
            batch_size * num_heads,
        )

        # Launch kernel
        _sparse_attention_forward_kernel[grid](
            q, k, v, mask, block_counts, block_indices, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            mask.stride(0), mask.stride(1), mask.stride(2), mask.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            batch_size, num_heads, num_q_blocks, seq_len, head_dim,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
            MAX_BLOCKS_PER_Q=max_blocks_per_q,
        )

        # Save for backward
        ctx.save_for_backward(q, k, v, mask)
        ctx.prune_empty_blocks = prune_empty_blocks

        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using PyTorch fallback.

        Like full attention, we use PyTorch's autograd for gradient computation.
        This is simpler than implementing custom backward kernels.
        """
        q, k, v, mask = ctx.saved_tensors

        # Recompute forward with PyTorch to get gradients
        with torch.enable_grad():
            q_temp = q.detach().requires_grad_(True)
            k_temp = k.detach().requires_grad_(True)
            v_temp = v.detach().requires_grad_(True)

            # Use PyTorch reference implementation
            scale = 1.0 / (q_temp.size(-1) ** 0.5)
            attn = torch.matmul(q_temp, k_temp.transpose(-2, -1)) * scale
            attn = attn.masked_fill(~mask, float('-inf'))
            attn = torch.softmax(attn, dim=-1)
            attn = attn.masked_fill(torch.isnan(attn), 0.0)
            out_temp = torch.matmul(attn, v_temp)

            # Compute gradients
            torch.autograd.backward(out_temp, grad_output)

        # Return gradients for all forward inputs: q, k, v, mask, prune flag, and metadata placeholders
        return (
            q_temp.grad,  # q
            k_temp.grad,  # k
            v_temp.grad,  # v
            None,         # mask
            None,         # prune_empty_blocks
            None,         # block_indices
            None,         # block_counts
            None,         # max_blocks_per_q
            None,         # num_q_blocks
        )


def sparse_attention_forward(
    q,
    k,
    v,
    mask,
    prune_empty_blocks=True,
    block_indices=None,
    block_counts=None,
    max_blocks_per_q=None,
    num_q_blocks=None,
):
    """
    Sparse attention forward pass.

    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_heads, seq_len, head_dim]
        v: Value tensor [batch, num_heads, seq_len, head_dim]
        mask: Sparse attention mask [batch, num_heads, seq_len, seq_len]
              True = attend, False = ignore
        prune_empty_blocks: If True, precompute and skip fully masked tiles (performance opt)
        block_indices/block_counts: optional precomputed block metadata to skip rebuild
        max_blocks_per_q/num_q_blocks: required if passing precomputed metadata

    Returns:
        out: Output tensor [batch, num_heads, seq_len, head_dim]
    """
    return _SparseAttentionFunction.apply(
        q,
        k,
        v,
        mask,
        prune_empty_blocks,
        block_indices,
        block_counts,
        max_blocks_per_q,
        num_q_blocks,
    )


class SparseAttention(torch.nn.Module):
    """
    Sparse Attention module.

    Supports multiple sparsity patterns:
    - 'local': Local window attention (attend to nearby tokens)
    - 'block': Block-diagonal attention
    - 'random': Random sparse attention
    """

    def __init__(self, pattern='local', window_size=256, prune_empty_blocks=True):
        super().__init__()
        self.pattern = pattern
        self.window_size = window_size
        self.prune_empty_blocks = prune_empty_blocks

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

        # Use Triton implementation
        return sparse_attention_forward(
            q, k, v, mask,
            prune_empty_blocks=self.prune_empty_blocks,
        )

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
