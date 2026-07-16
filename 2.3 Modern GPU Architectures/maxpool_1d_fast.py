import torch


def maxpool_1d_fast(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """A windowed maximum pooling operation for 1D tensors."""
    if x.shape[0] % window_size != 0:
        y = x.new_full([(x.shape[0] // window_size + 1) * window_size], -float("inf"))
        y[: x.shape[0]] = x
        y = y.view(-1, window_size)
    else:
        y = x.view(-1, window_size)

    max_1 = y.cummax(dim=1)[0].view(-1)
    max_2 = y.flip(1).cummax(dim=1)[0].flip(1).view(-1)
    return torch.maximum(max_2[: x.shape[0] - window_size + 1], max_1[window_size - 1 : x.shape[0]])


if __name__ == "__main__":
    x = torch.randn(10)
    x[0] = 10
    x[5] = 11
    x[-1] = 12
    print(x)
    print(maxpool_1d_fast(x, 3))
