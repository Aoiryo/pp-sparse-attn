"""
Column-wise Parallel Linear Layer (Megatron-LM style).

In column parallelism, the weight matrix W [in_features, out_features] is split
along the output dimension (columns):

    Y = X @ W = X @ [W_1 | W_2 | ... | W_n]
              = [X @ W_1 | X @ W_2 | ... | X @ W_n]

Each GPU holds a partition W_i and computes Y_i = X @ W_i independently.
No communication is needed in the forward pass!

Gradient flow:
    dL/dX = dL/dY @ W^T = sum_i (dL/dY_i @ W_i^T)
    This requires an all-reduce in the backward pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    copy_to_model_parallel_region,
    reduce_from_model_parallel_region,
)


class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column parallelism.

    The weight matrix is partitioned column-wise across GPUs:
        W = [W_1, W_2, ..., W_n] where n = tensor_parallel_size

    Forward:
        Each GPU computes: Y_i = X @ W_i (no communication)
        Output is partitioned: [Y_1, Y_2, ..., Y_n]

    Backward:
        Gradient of input requires all-reduce: dL/dX = sum_i (dL/dY_i @ W_i^T)

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension (will be divided across GPUs)
        bias: Whether to use bias
        gather_output: If True, gather the output from all GPUs (default: False)
        init_method: Weight initialization function
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        gather_output=False,
        init_method=None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output

        # Get tensor parallel info
        world_size = get_tensor_model_parallel_world_size()
        self.world_size = world_size

        # Output features are divided across GPUs
        if out_features % world_size != 0:
            raise ValueError(
                f"out_features ({out_features}) must be divisible by "
                f"tensor_parallel_world_size ({world_size})"
            )

        self.out_features_per_partition = out_features // world_size

        # Each GPU holds a partition of the weight matrix
        # Shape: [in_features, out_features_per_partition]
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features_per_partition)
            )
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
            input: [..., in_features]

        Returns:
            output: [..., out_features] if gather_output=True
                    [..., out_features_per_partition] if gather_output=False
        """

        # Copy input to all GPUs (handles gradient splitting in backward)
        input_parallel = copy_to_model_parallel_region(input)

        # Matrix multiplication: each GPU computes its partition
        # input_parallel: [..., in_features]
        # weight: [out_features_per_partition, in_features]
        # output: [..., out_features_per_partition]
        output = F.linear(input_parallel, self.weight, self.bias)

        if self.gather_output:
            # Gather outputs from all GPUs: [..., out_features]
            from .utils import gather_from_model_parallel_region
            output = gather_from_model_parallel_region(output)

        return output

    def extra_repr(self):
        return (
            f'in_features={self.in_features}, '
            f'out_features={self.out_features}, '
            f'out_features_per_partition={self.out_features_per_partition}, '
            f'bias={self.bias is not None}, '
            f'gather_output={self.gather_output}, '
            f'world_size={self.world_size}'
        )
