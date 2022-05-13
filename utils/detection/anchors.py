import torch
from typing import List, Tuple, Union

def generate_anchor_base(base_size: int, ratios: list, scales: list):
    """
    Generate base anchors given the aspect ratios and scales.
    
    Args:
        base_size (int): Base size of single pixel in feature map
        ratios (list[N]): Ratios of anchors
        scales (list[M]): Scales of anchors
    
    Return:
        Tensor[N x M, 4]: Base anchors in the format (x1, y1, x2, y2)
    """
    assert isinstance(ratios, List) and isinstance(scales, List)

    cx = base_size / 2
    cy = base_size / 2

    ratios = torch.tensor(ratios, dtype=torch.float32) 
    scales = torch.tensor(scales, dtype=torch.float32)
    ratios, scales = torch.meshgrid(ratios, scales)
    ratios, scales = ratios.flatten(), scales.flatten()
    h = base_size * torch.sqrt(ratios) * scales
    w = base_size * 1 / torch.sqrt(ratios) * scales

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    
    anchor_base = torch.stack((x1, y1, x2, y2), dim=-1)

    return anchor_base

def generate_anchor_cell(anchor_base: torch.Tensor, stride: int, size: Union[Tuple, int]):
    """
    Generate anchors at all cell locations

    Args:
        anchor_base (Tensor[K, 4]): Base anchors with format (x1, y1, x2, y2)
        stride (int): Strides
        size (Tuple | int): Size for height and width of the feature map
    Return:
        Tensor[K * H * W, 4]: K * H * W anchors with format (x1, y1, x2, y2)
    """
    if isinstance(size, Tuple):
        height = size[0]
        width = size[1]
    elif isinstance(size, int):
        height, width = size, size      
    else:
        raise TypeError(f"Expected tuple or int, but get {type(size)}")
    
    shift_x = torch.arange(0, width) * stride
    shift_y = torch.arange(0, height) * stride
    shift_x, shift_y = torch.meshgrid(shift_x, shift_y)

    shift = torch.stack((shift_x.flatten(), shift_y.flatten(), shift_x.flatten(), shift_y.flatten()), dim=-1) # (w x h), 4
    anchors = (anchor_base[None, :, :] + shift[:, None, :]).view(anchor_base.shape[0] * height * width, -1)
    return anchors

    