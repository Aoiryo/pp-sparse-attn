"""
Hybrid Transformer Block combining:
1. Sparse Attention (or Full Attention for baseline)
2. Tensor Parallel MLP layers

This is the main deliverable of the project!
"""

import torch
import torch.nn as nn

from attention import FullAttention, SparseAttention
from parallel import ColumnParallelLinear, RowParallelLinear


class HybridTransformerBlock(nn.Module):
    """
    A single Transformer block with:
    1. Multi-head attention (Full or Sparse)
    2. Feed-forward network with Tensor Parallelism

    Architecture:
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))

    The MLP uses tensor parallelism:
        MLP(x) = GELU(x @ W1) @ W2
               = GELU(x @ [W1_1 | W1_2]) @ [W2_1; W2_2]
               = [GELU(x @ W1_1) | GELU(x @ W1_2)] @ [W2_1; W2_2]

    ColumnParallel computes: [Y1 | Y2] = [x @ W1_1 | x @ W1_2]
    RowParallel computes: Z = Y1 @ W2_1 + Y2 @ W2_2 (with all-reduce)
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_hidden_size=None,
        attention_type='full',
        attention_window_size=256,
        dropout=0.1,
    ):
        """
        Args:
            hidden_size: Hidden dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            mlp_hidden_size: MLP intermediate size (default: 4 * hidden_size)
            attention_type: 'full', 'sparse-local', 'sparse-block', 'sparse-random'
            attention_window_size: Window size for sparse attention
            dropout: Dropout probability
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_hidden_size = mlp_hidden_size or 4 * hidden_size

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.head_dim = hidden_size // num_heads

        # Layer normalization
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        # Attention projection layers (these can also be parallelized, but
        # we keep them simple for now to focus on MLP parallelism)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        # Attention module
        if attention_type == 'full':
            self.attention = FullAttention()
        elif attention_type.startswith('sparse'):
            pattern = attention_type.split('-')[1]  # 'local', 'block', or 'random'
            self.attention = SparseAttention(
                pattern=pattern,
                window_size=attention_window_size
            )
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")

        # MLP with Tensor Parallelism
        # This is where the magic happens!
        # W1: [hidden_size, mlp_hidden_size] - split column-wise
        # W2: [mlp_hidden_size, hidden_size] - split row-wise
        self.mlp_fc1 = ColumnParallelLinear(
            hidden_size,
            self.mlp_hidden_size,
            bias=True,
            gather_output=False,  # Keep output partitioned for next layer
        )

        self.mlp_fc2 = RowParallelLinear(
            self.mlp_hidden_size,
            hidden_size,
            bias=True,
            input_is_parallel=True,  # Input comes from ColumnParallel
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask

        Returns:
            output: [batch, seq_len, hidden_size]
        """

        # ============================================================
        # 1. Multi-Head Attention
        # ============================================================

        residual = x
        x = self.ln1(x)

        batch_size, seq_len, hidden_size = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # [batch, seq_len, hidden_size]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        # [batch, seq_len, hidden_size] -> [batch, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply attention (Full or Sparse)
        attn_output = self.attention(q, k, v)  # [batch, num_heads, seq_len, head_dim]

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, hidden_size)

        # Output projection and residual
        x = self.o_proj(attn_output)
        x = self.dropout(x)
        x = x + residual

        # ============================================================
        # 2. Feed-Forward Network (MLP) with Tensor Parallelism
        # ============================================================

        residual = x
        x = self.ln2(x)

        # MLP with GELU activation
        # fc1: [hidden_size, mlp_hidden_size] - Column Parallel
        x = self.mlp_fc1(x)  # [batch, seq_len, mlp_hidden_size / world_size]
        x = torch.nn.functional.gelu(x)
        x = self.dropout(x)

        # fc2: [mlp_hidden_size, hidden_size] - Row Parallel (with all-reduce)
        x = self.mlp_fc2(x)  # [batch, seq_len, hidden_size]
        x = self.dropout(x)

        # Residual connection
        x = x + residual

        return x


class HybridTransformer(nn.Module):
    """
    Full Transformer model with multiple HybridTransformerBlocks.

    This stacks multiple blocks together to form a complete Transformer.
    """

    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_layers,
        num_heads,
        mlp_hidden_size=None,
        max_seq_length=2048,
        attention_type='full',
        attention_window_size=256,
        dropout=0.1,
    ):
        """
        Args:
            vocab_size: Vocabulary size
            hidden_size: Hidden dimension
            num_layers: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_hidden_size: MLP intermediate size (default: 4 * hidden_size)
            max_seq_length: Maximum sequence length
            attention_type: 'full' or 'sparse-{local,block,random}'
            attention_window_size: Window size for sparse attention
            dropout: Dropout probability
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)

        # Position embeddings
        self.position_embedding = nn.Embedding(max_seq_length, hidden_size)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            HybridTransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_hidden_size=mlp_hidden_size,
                attention_type=attention_type,
                attention_window_size=attention_window_size,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(hidden_size)

        # Output projection (language modeling head)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Tie weights between token embedding and output projection
        self.lm_head.weight = self.token_embedding.weight

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass.

        Args:
            input_ids: [batch, seq_len] token indices
            attention_mask: Optional attention mask

        Returns:
            logits: [batch, seq_len, vocab_size]
        """

        batch_size, seq_len = input_ids.shape

        # Token embeddings
        x = self.token_embedding(input_ids)  # [batch, seq_len, hidden_size]

        # Position embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = x + self.position_embedding(positions)

        x = self.dropout(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)

        # Final layer norm
        x = self.ln_f(x)

        # Output projection
        logits = self.lm_head(x)

        return logits

    def get_num_params(self):
        """Get the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
