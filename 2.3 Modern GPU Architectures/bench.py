from time import time

import torch
from maxpool_1d_fast import maxpool_1d_fast


def maxpool_1d_torch(x: torch.Tensor, window_size: int) -> torch.Tensor:
    return torch.nn.functional.max_pool1d(x[None, None], kernel_size=window_size, stride=1).squeeze()


def bench(f, tensor_size=2**10, window_size=128, device="cpu"):
    data = torch.randn(tensor_size, device=device)
    print(f"Running benchmark for {f.__name__:20s} ", end="...")
    f(data, window_size)  # warmup
    start = time()
    for _ in range(10):
        y = f(data, window_size)
        # Force a GPU synchronization
        y[0].item()
    end = time()
    print(f"  {end - start:.2f} seconds")

    y_hat = maxpool_1d_torch(data, window_size)
    print(f"  Error  {abs(y - y_hat.squeeze()).max().item():0.2f}")
    print(f"  Error  {abs(y - y_hat.squeeze()).argmax().item()}")


if __name__ == "__main__":
    # tensor_size = 2**15
    # window_size = 2**8
    # bench(maxpool_1d_brute, tensor_size, window_size)
    # bench(maxpool_1d_heap, tensor_size, window_size)

    tensor_size = 2**25
    window_size = 2**15
    # bench(maxpool_1d_torch, tensor_size, window_size, device="cuda")
    # bench(maxpool_1d_cuda_brute, tensor_size, window_size, device="cuda")
    # bench(maxpool_1d_cuda_smart, tensor_size, window_size, device="cuda")
    # bench(maxpool_1d_cuda_memory, tensor_size, window_size, device="cuda")

    bench(maxpool_1d_fast, tensor_size, window_size, device="cuda")
