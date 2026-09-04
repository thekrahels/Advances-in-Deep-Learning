from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401

def load(path: Path | None):
    # TODO (extra credit): Implement a BigNet that uses in
    # average less than 4 bits per parameter (<9MB)
    # Make sure the network retains some decent accuracy
    return None

def block_quantize_3bit(x: torch.Tensor, group_size: int = 32,) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 1
    assert x.size(0) % group_size == 0
    assert group_size % 8 == 0

    x = x.view(-1, group_size)
    normalization = x.abs().max(dim=1, keepdim=True).values
    safe_normalization = torch.where(normalization == 0, torch.torch.ones_like(normalization), normalization)

    x_norm = (x + safe_normalization) / (2 * normalization)

    x_quant = x_norm.mul(7).round().clamp(0, 7).to(torch.int32)
    x_quant = x_quant.view(x_quant.size(0), group_size // 8, 8)

    packed_24 = x_quant[:, :, 0] + (x_quant[:, :, 1] << 3) + (x_quant[:, :, 2] << 6) + (x_quant[:, :, 3] << 9) + \
               (x_quant[:, :, 4] << 12) + (x_quant[:, :, 5] << 15) + (x_quant[:, :, 6] << 18) + (x_quant[:, :, 7] << 21)

    byte_0 = packed_24 & 0xFF
    byte_1 = (packed_24 >> 8) & 0xFF
    byte_2 = (packed_24 >> 16) & 0xFF
    byte_3 = (packed_24 >> 24) & 0xFF

    packed = torch.stack((byte_0, byte_1, byte_2, byte_3), dim=1)
    packed = packed.view(packed.size(0), group_size * 3 // 8)

    return packed, normalization.to(torch.float16)


def block_dequantize_3bit( packed: torch.Tensor, normalization: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    assert packed.dim() == 2
    assert group_size % 8 == 0

    number_of_groups = packed.size(0)

    packed_bytes = packed.view(number_of_groups, group_size // 8, 3).to(torch.int32)

    packed_24 = packed[:, :, 0] + (packed[:, :, 1] << 8) + (packed[:, :, 2] << 16)
   
    unpacked = torch.stack(
        [
            (packed_24 >> 0) & 0x7F,
            (packed_24 >> 3) & 0x7F,
            (packed_24 >> 6) & 0x7F,
            (packed_24 >> 9) & 0x7F,
            (packed_24 >> 12) & 0x7F,
            (packed_24 >> 15) & 0x7F,
            (packed_24 >> 18) & 0x7F,
            (packed_24 >> 21) & 0x7F,
        ],
        dim=1,
    )

    unpacked = unpacked.view(number_of_groups, group_size).to(torch.float32)

    normalization = normalization.to(torch.float32)
    x_norm = unpacked / 7
    x = (unpacked * 2 * normalization) - normalization
    return x.view(-1)

class Linear3bit(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, group_size: int = 32,) -> None:
        super().__init__()
        

