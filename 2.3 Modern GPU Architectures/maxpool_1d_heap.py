import heapq

import torch


def maxpool_1d_heap(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """A windowed maximum pooling operation for 1D tensors."""
    output = x.new_zeros(x.size(0) - window_size + 1)

    h = []
    for i in range(x.size(0)):
        heapq.heappush(h, (-x[i].item(), i))
        if i >= window_size - 1:
            while h[0][1] <= i - window_size:
                heapq.heappop(h)
            output[i - window_size + 1] = -h[0][0]
    return output
