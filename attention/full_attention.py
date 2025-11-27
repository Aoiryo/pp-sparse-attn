"""
Full (Dense) Attention Implementation using Triton.

This serves as:
1. A baseline for correctness testing
2. A reference implementation to understand the attention kernel structure
3. A performance comparison baseline

Standard attention: O = softmax(Q @ K^T / sqrt(d)) @ V
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _full_attention_forward_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    batch_size, num_heads, seq_len, head_dim,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    """
    Full attention forward kernel using Triton.

    This kernel computes standard scaled dot-product attention:
    - For each query token, attend to ALL key tokens (dense attention)
    - Use online softmax algorithm for numerical stability
    - Write output directly to global memory

    Parallelization strategy:
    - Each program instance handles BLOCK_M query tokens
    - Iterate over all key tokens in chunks of BLOCK_N
    - Each thread block processes one (batch, head, query_block) combination
    """

    # Get program IDs for parallelization
    pid_m = tl.program_id(0)  # Which block of queries
    pid_bh = tl.program_id(1)  # Which (batch, head) combination

    # Decompose batch-head index
    batch_idx = pid_bh // num_heads
    head_idx = pid_bh % num_heads

    # Compute offsets for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M) # 列表形式的BLOCK_M个位置的seq
    offs_n = tl.arange(0, BLOCK_N) # for key & value
    offs_d = tl.arange(0, BLOCK_DMODEL) # which dim

    # Pointers to Q, K, V for this (batch, head)
    q_offset = batch_idx * stride_qb + head_idx * stride_qh
    k_offset = batch_idx * stride_kb + head_idx * stride_kh
    v_offset = batch_idx * stride_vb + head_idx * stride_vh

    # Load Q block for this program (BLOCK_M x head_dim)
    # BLOCK_M个seq位置的所有head_dim维度的数据 <- 指针
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    # seq len可能不能被BLOCK整除
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seq_len, other=0.0)

    # Initialize accumulator and normalization factor for online softmax
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')  # max score
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # sum of exp

    # Scale factor for attention
    scale = 1.0 / tl.sqrt(float(head_dim))

    # Iterate over all key blocks (this is the O(N^2) loop)
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n_cur = start_n + offs_n

        # Load K block (BLOCK_N x head_dim)
        k_ptrs = K + k_offset + offs_n_cur[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=offs_n_cur[:, None] < seq_len, other=0.0)

        # Compute attention scores: qk = Q @ K^T (BLOCK_M x BLOCK_N)
        qk = tl.dot(q, tl.trans(k)) * scale

        # Apply mask for out-of-bounds keys
        qk = tl.where(offs_n_cur[None, :] < seq_len, qk, float('-inf'))

        # Online softmax: update running max and sum
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        # Compute exponentials with numerical stability
        p = tl.exp(qk - m_new[:, None])

        # Update normalization factor
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)

        # Load V block (BLOCK_N x head_dim)
        v_ptrs = V + v_offset + offs_n_cur[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=offs_n_cur[:, None] < seq_len, other=0.0)

        # Update accumulator: acc = acc * alpha + p @ V
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # Final normalization
    acc = acc / l_i[:, None]

    # Write output
    o_offset = batch_idx * stride_ob + head_idx * stride_oh
    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seq_len)


def full_attention_forward(q, k, v):
    """
    Full attention forward pass using Triton kernel.
    Now it is simply self-attention. 

    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_heads, seq_len, head_dim]
        v: Value tensor [batch, num_heads, seq_len, head_dim]

    Returns:
        out: Output tensor [batch, num_heads, seq_len, head_dim]
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    assert k.shape == v.shape == q.shape, "Q, K, V must have the same shape"

    # Allocate output
    out = torch.empty_like(q)

    # Launch parameters
    BLOCK_M = 64 # num of queries per node
    BLOCK_N = 64 # num of keys
    BLOCK_DMODEL = head_dim

    # Grid dimensions: (num_query_blocks, batch * num_heads)
    grid = (triton.cdiv(seq_len, BLOCK_M), batch_size * num_heads)

    # Launch kernel
    _full_attention_forward_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        batch_size, num_heads, seq_len, head_dim,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
    )

    return out


class FullAttention(torch.nn.Module):
    """
    Full Attention module wrapper.

    This is a simple wrapper around the Triton kernel for use in nn.Module.
    """

    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        """
        Args:
            q, k, v: [batch, num_heads, seq_len, head_dim]
        Returns:
            out: [batch, num_heads, seq_len, head_dim]
        """
        return full_attention_forward(q, k, v)
