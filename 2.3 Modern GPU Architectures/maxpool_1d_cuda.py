from pathlib import Path

import torch


def _load():
    import warnings

    warnings.filterwarnings("ignore")
    from torch.utils.cpp_extension import load

    return load("maxpool_1d_cuda", sources=[Path(__file__).parent / "maxpool_1d_cuda.cu"], verbose=False)


def maxpool_1d_cuda_brute(x: torch.Tensor, window_size: int) -> torch.Tensor:
    return _load().maxpool_1d_brute(x, window_size)


def maxpool_1d_cuda_memory(x: torch.Tensor, window_size: int) -> torch.Tensor:
    return _load().maxpool_1d_memory(x, window_size)


def maxpool_1d_cuda_smart(x: torch.Tensor, window_size: int) -> torch.Tensor:
    return _load().maxpool_1d_smart(x, window_size)
