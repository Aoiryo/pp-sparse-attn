"""
Tensor Parallelism modules (Megatron-LM style).
"""

from .column_parallel import ColumnParallelLinear
from .row_parallel import RowParallelLinear
from .utils import initialize_model_parallel, get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank

__all__ = [
    'ColumnParallelLinear',
    'RowParallelLinear',
    'initialize_model_parallel',
    'get_tensor_model_parallel_world_size',
    'get_tensor_model_parallel_rank',
]
