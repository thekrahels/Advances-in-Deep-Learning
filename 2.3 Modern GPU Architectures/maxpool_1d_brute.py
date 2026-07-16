import torch


def maxpool_1d_brute(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """A windowed maximum pooling operation for 1D tensors."""
    output = x.new_zeros(x.size(0) - window_size + 1)
    for i in range(output.size(0)):
        for j in range(window_size):
            output[i] = max(output[i], x[i + j])
    return output
