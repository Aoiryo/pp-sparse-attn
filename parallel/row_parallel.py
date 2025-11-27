"""
Row-wise Parallel Linear Layer (Megatron-LM style).

In row parallelism, the weight matrix W [in_features, out_features] is split
along the input dimension (rows):

    Y = X @ W = X @ [W_1; W_2; ...; W_n]
              = [X_1; X_2; ...; X_n] @ [W_1; W_2; ...; W_n]
              = X_1 @ W_1 + X_2 @ W_2 + ... + X_n @ W_n

Each GPU computes Y_i = X_i @ W_i, then we all-reduce to get final Y.

This layer typically follows a ColumnParallelLinear layer:
    ColumnParallel: X @ [W1 | W2] = [Y1 | Y2] (partitioned output)
    RowParallel: [Y1 | Y2] @ [W1; W2] requires all-reduce

Gradient flow:
    dL/dY is same on all GPUs (after all-reduce)
    dL/dX_i = dL/dY @ W_i^T (each GPU computes its partition independently)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    reduce_from_model_parallel_region,
    split_tensor_along_dim,
)


class RowParallelLinear(nn.Module):
    """
    Linear layer with row parallelism.

    The weight matrix is partitioned row-wise across GPUs:
        W = [W_1; W_2; ...; W_n] where n = tensor_parallel_size

    Forward:
        Input is expected to be partitioned along last dim: X = [X_1 | X_2 | ... | X_n]
        Each GPU computes: Y_i = X_i @ W_i
        All-reduce to get final output: Y = sum_i Y_i

    Backward:
        Gradient of input: dL/dX_i = dL/dY @ W_i^T (no communication)

    Args:
        in_features: Input feature dimension (will be divided across GPUs)
        out_features: Output feature dimension
        bias: Whether to use bias
        input_is_parallel: If True, input is already partitioned (default: False)
        init_method: Weight initialization function
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        input_is_parallel=False,
        init_method=None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel

        # Get tensor parallel info
        world_size = get_tensor_model_parallel_world_size()
        self.world_size = world_size

        # Input features are divided across GPUs
        if in_features % world_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by "
                f"tensor_parallel_world_size ({world_size})"
            )

        self.in_features_per_partition = in_features // world_size

        # Each GPU holds a partition of the weight matrix
        # Shape: [out_features, in_features_per_partition]
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition)
        )

        if bias:
            # Note: only one GPU needs to have the bias (we'll put it on rank 0)
            # After all-reduce, the bias will be added only once
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize weights
        if init_method is None:
            nn.init.kaiming_uniform_(self.weight, a=5**0.5)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
        else:
            init_method(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(self, input):
        """
        Forward pass.

        Args:
            input: [..., in_features] if input_is_parallel=False
                   [..., in_features_per_partition] if input_is_parallel=True

        Returns:
            output: [..., out_features] (same on all GPUs after all-reduce)
        """

        if not self.input_is_parallel:
            # Split input across tensor parallel dimension
            rank = get_tensor_model_parallel_rank()
            input_list = split_tensor_along_dim(
                input, dim=-1, num_partitions=self.world_size
            )
            input_parallel = input_list[rank]
        else:
            # Input is already partitioned
            input_parallel = input

        # Matrix multiplication: each GPU computes its partition
        # input_parallel: [..., in_features_per_partition]
        # weight: [out_features, in_features_per_partition]
        # output: [..., out_features]
        output_parallel = F.linear(input_parallel, self.weight)

        # All-reduce across tensor parallel group
        output = reduce_from_model_parallel_region(output_parallel)

        # Add bias (only on rank 0 to avoid adding it multiple times)
        if self.bias is not None:
            rank = get_tensor_model_parallel_rank()
            if rank == 0:
                output = output + self.bias

        return output

    def extra_repr(self):
        return (
            f'in_features={self.in_features}, '
            f'out_features={self.out_features}, '
            f'in_features_per_partition={self.in_features_per_partition}, '
            f'bias={self.bias is not None}, '
            f'input_is_parallel={self.input_is_parallel}, '
            f'world_size={self.world_size}'
        )
