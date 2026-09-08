from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401


def block_quantize_3bit(x: torch.Tensor, group_size: int = 32,) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 1
    assert x.size(0) % group_size == 0
    assert group_size % 8 == 0

    x = x.view(-1, group_size)
    normalization = x.abs().max(dim=1, keepdim=True).values
    safe_normalization = torch.where(normalization == 0, torch.ones_like(normalization), normalization,)

    #x_norm = (x + safe_normalization) / (2 * safe_normalization)

    #x_quant = x_norm.mul(7).round().clamp(0, 7).to(torch.int32)

    scale = safe_normalization / 3.0

    x_quant = (
        (x / scale)
        .round()
        .clamp(-3, 3)
        .to(torch.int32)
    )

    # Shift signed levels -3 through 3 into stored values 0 through 6.
    x_quant = x_quant + 3
    
    x_quant = x_quant.view(x_quant.size(0), group_size // 8, 8)

    packed_24 = x_quant[:, :, 0] + (x_quant[:, :, 1] << 3) + (x_quant[:, :, 2] << 6) + (x_quant[:, :, 3] << 9) + \
               (x_quant[:, :, 4] << 12) + (x_quant[:, :, 5] << 15) + (x_quant[:, :, 6] << 18) + (x_quant[:, :, 7] << 21)

    byte_0 = packed_24 & 0xFF
    byte_1 = (packed_24 >> 8) & 0xFF
    byte_2 = (packed_24 >> 16) & 0xFF

    packed = torch.stack((byte_0, byte_1, byte_2), dim=-1).to(torch.uint8)
    packed = packed.reshape(packed.size(0), group_size * 3 // 8)

    return packed, normalization.to(torch.float16)


def block_dequantize_3bit( packed: torch.Tensor, normalization: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    assert packed.dim() == 2
    assert group_size % 8 == 0

    number_of_groups = packed.size(0)

    packed_bytes = packed.view(number_of_groups, group_size // 8, 3).to(torch.int32)

    packed_24 = (packed_bytes[:, :, 0] | (packed_bytes[:, :, 1] << 8) | (packed_bytes[:, :, 2] << 16))
   
    unpacked = torch.stack(
        (
            (packed_24 >> 0) & 0x7,
            (packed_24 >> 3) & 0x7,
            (packed_24 >> 6) & 0x7,
            (packed_24 >> 9) & 0x7,
            (packed_24 >> 12) & 0x7,
            (packed_24 >> 15) & 0x7,
            (packed_24 >> 18) & 0x7,
            (packed_24 >> 21) & 0x7,
        ),
        dim=-1,
    )

    unpacked = unpacked.view(number_of_groups, group_size).to(torch.float32)

    normalization = normalization.to(torch.float32)
    x_norm = unpacked / 7.0
    x = (x_norm * 2 * normalization) - normalization
    return x.view(-1)

class Linear3Bit(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, group_size: int = 32,) -> None:
        super().__init__()
        
        assert group_size % 8 == 0
        assert out_features * in_features % group_size == 0
        
        self._shape = (out_features, in_features)
        self._group_size = group_size

        number_of_weights = out_features * in_features
        number_of_groups = number_of_weights // group_size

        self.register_buffer(
            "weight_q3",
            torch.zeros(number_of_groups, group_size * 3 // 8, dtype=torch.uint8),
            persistent=False,
        )
        self.register_buffer(
            "weight_norm",
            torch.zeros(number_of_groups, 1, dtype=torch.float16),
            persistent=False,
        )

        self._register_load_state_dict_pre_hook(Linear3Bit._load_state_dict_pre_hook, with_module=True)

        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32,))

    def _load_state_dict_pre_hook(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        ):
            weight_key = f"{prefix}weight"
            if weight_key in state_dict:
                # Load the original weights and remove them from the state_dict (mark them as loaded)
                weight = state_dict[weight_key]  # noqa: F841
                del state_dict[weight_key]
                    

                weight_q3, weight_norm = block_quantize_3bit(weight.flatten(), self._group_size,)
                self.weight_q3.copy_(weight_q3)
                self.weight_norm.copy_(weight_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():

            weight = block_dequantize_3bit(self.weight_q3, self.weight_norm, self._group_size)
            weight = weight.view(self._shape)
            return torch.nn.functional.linear(x, weight, self.bias)

class LowerPrecisionBigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels: int, group_size: int):
            super().__init__()
            self.model = torch.nn.Sequential(
                Linear3Bit(channels, channels, group_size=group_size),
                torch.nn.ReLU(),
                Linear3Bit(channels, channels, group_size=group_size),
                torch.nn.ReLU(),
                Linear3Bit(channels, channels, group_size=group_size),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self, group_size: int = 32):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, group_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def load(path: Path | None) -> LowerPrecisionBigNet:
    net = LowerPrecisionBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net
    
