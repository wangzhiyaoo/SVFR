from typing import List, Optional, Tuple, Union
import torch

from diffusers.utils.torch_utils import randn_tensor

def random_noise(
    tensor: torch.Tensor = None,
    shape: Tuple[int] = None,
    dtype: torch.dtype = None,
    device: torch.device = None,
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    noise_offset: Optional[float] = None,  # typical value is 0.1
) -> torch.Tensor:
    if tensor is not None:
        shape = tensor.shape
        device = tensor.device
        dtype = tensor.dtype
    if isinstance(device, str):
        device = torch.device(device)
    noise = randn_tensor(shape, dtype=dtype, device=device, generator=generator)
    if noise_offset is not None:
        noise += noise_offset * torch.randn(
            (tensor.shape[0], tensor.shape[1], 1, 1, 1), device
        )
    return noise
