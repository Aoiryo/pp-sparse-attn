"""
Utility functions for tensor parallelism.

This module provides helper functions for:
1. Initializing distributed process groups
2. Getting rank and world size information
3. Splitting tensors across GPUs
"""

import torch
import torch.distributed as dist


# Global state for tensor model parallel group
_TENSOR_MODEL_PARALLEL_GROUP = None


def initialize_model_parallel(tensor_model_parallel_size=1):
    """
    Initialize model parallelism.

    This should be called AFTER torch.distributed.init_process_group().

    Args:
        tensor_model_parallel_size: Number of GPUs to use for tensor parallelism
    """
    global _TENSOR_MODEL_PARALLEL_GROUP

    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed is not initialized. "
            "Please call torch.distributed.init_process_group() first."
        )

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size % tensor_model_parallel_size != 0:
        raise ValueError(
            f"World size ({world_size}) must be divisible by "
            f"tensor_model_parallel_size ({tensor_model_parallel_size})"
        )

    # Create tensor model parallel group
    # All processes in the same group will do tensor parallelism together
    num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size

    for i in range(num_tensor_model_parallel_groups):
        ranks = list(range(i * tensor_model_parallel_size,
                          (i + 1) * tensor_model_parallel_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group

    print(f"[Rank {rank}] Initialized tensor model parallel with size {tensor_model_parallel_size}")


def get_tensor_model_parallel_group():
    """Get the tensor model parallel group."""
    if _TENSOR_MODEL_PARALLEL_GROUP is None:
        raise RuntimeError("Tensor model parallel group is not initialized")
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_tensor_model_parallel_world_size():
    """Get the size of the tensor model parallel group."""
    if _TENSOR_MODEL_PARALLEL_GROUP is None:
        return 1
    return dist.get_world_size(group=_TENSOR_MODEL_PARALLEL_GROUP)


def get_tensor_model_parallel_rank():
    """Get the rank within the tensor model parallel group."""
    if _TENSOR_MODEL_PARALLEL_GROUP is None:
        return 0
    return dist.get_rank(group=_TENSOR_MODEL_PARALLEL_GROUP)


def split_tensor_along_dim(tensor, dim, num_partitions, contiguous=True):
    """
    Split a tensor along a dimension into equal chunks.

    Args:
        tensor: Input tensor to split
        dim: Dimension to split along
        num_partitions: Number of partitions
        contiguous: Whether to make each partition contiguous

    Returns:
        List of tensor partitions
    """
    if tensor.size(dim) % num_partitions != 0:
        raise ValueError(
            f"Tensor size along dim {dim} ({tensor.size(dim)}) "
            f"must be divisible by num_partitions ({num_partitions})"
        )

    chunk_size = tensor.size(dim) // num_partitions
    tensor_list = torch.split(tensor, chunk_size, dim=dim)

    if contiguous:
        tensor_list = [chunk.contiguous() for chunk in tensor_list]

    return tensor_list


def gather_tensor_along_dim(tensor, dim, group=None):
    """
    Gather tensor from all ranks along a dimension.

    Args:
        tensor: Input tensor (partition from this rank)
        dim: Dimension to gather along
        group: Process group (default: tensor model parallel group)

    Returns:
        Gathered tensor (same on all ranks)
    """
    if group is None:
        group = get_tensor_model_parallel_group()

    world_size = dist.get_world_size(group=group)

    # Gather tensors from all ranks
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor, group=group)

    # Concatenate along the specified dimension
    output = torch.cat(tensor_list, dim=dim)
    return output


class _CopyToModelParallelRegion(torch.autograd.Function):
    """
    Copy input tensor to all ranks in the tensor model parallel group.

    This is an identity operation in the forward pass, but splits gradients
    in the backward pass.
    """

    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # Split gradient across tensor parallel group
        world_size = get_tensor_model_parallel_world_size()
        if world_size == 1:
            return grad_output

        # Each rank only keeps its portion of the gradient
        rank = get_tensor_model_parallel_rank()
        grad_chunks = split_tensor_along_dim(grad_output, dim=-1, num_partitions=world_size)
        return grad_chunks[rank]


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """
    All-reduce across the tensor model parallel group.

    This is an all-reduce in the forward pass, and identity in the backward pass.
    """

    @staticmethod
    def forward(ctx, input):
        # All-reduce across tensor parallel group
        if get_tensor_model_parallel_world_size() == 1:
            return input

        output = input.clone()
        dist.all_reduce(output, group=get_tensor_model_parallel_group())
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Identity: each rank keeps its gradient
        return grad_output


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """
    Gather tensors from all ranks and concatenate along the last dimension.

    This is a gather in the forward pass, and split in the backward pass.
    """

    @staticmethod
    def forward(ctx, input):
        if get_tensor_model_parallel_world_size() == 1:
            return input

        # Gather along last dimension
        output = gather_tensor_along_dim(input, dim=-1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Split gradient across ranks
        world_size = get_tensor_model_parallel_world_size()
        if world_size == 1:
            return grad_output

        rank = get_tensor_model_parallel_rank()
        grad_chunks = split_tensor_along_dim(grad_output, dim=-1, num_partitions=world_size)
        return grad_chunks[rank]


# Public API
def copy_to_model_parallel_region(input):
    """Copy input to all ranks (forward), split gradient (backward)."""
    return _CopyToModelParallelRegion.apply(input)


def reduce_from_model_parallel_region(input):
    """All-reduce across ranks (forward), identity (backward)."""
    return _ReduceFromModelParallelRegion.apply(input)


def gather_from_model_parallel_region(input):
    """Gather from all ranks (forward), split gradient (backward)."""
    return _GatherFromModelParallelRegion.apply(input)
