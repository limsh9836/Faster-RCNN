import torch

def cxcywh_to_xyxy(boxes):
    """
    Convert bounding boxes from (cx, cy, w, h) to (x1, y1, x2, y2)

    Args:
        boxes (Tensor[N, 4])

    Returns:
        Tensor[N, 4]
    """
    x1 = boxes[:, 0] - boxes[:, 2] / 2 
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2 
    y2 = boxes[:, 1] + boxes[:, 3] / 2 

    boxes = torch.stack((x1, y1, x2, y2), dim=-1)

    return boxes

def xyxy_to_cxcywh(boxes):
    """
    Convert bounding boxes from (x1, y1, x2, y2) to (cx, cy, w, h)

    Args:
        boxes (Tensor[N, 4])

    Returns:
        Tensor[N, 4]
    """
    cx = (boxes[:, 2] + boxes[:, 0]) / 2
    cy = (boxes[:, 3] + boxes[:, 1]) / 2
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    boxes = torch.stack((cx, cy, w, h), dim=-1)

    return boxes

def xywh_to_xyxy(boxes):
    """
    Convert bounding boxes from (x, y, w, h) to (x1, y1, x2, y2)

    Args:
        boxes (Tensor[N, 4])

    Returns:
        Tensor[N, 4]
    """
    x1 = boxes[:, 0]
    x2 = boxes[:, 0] + boxes[:, 2]
    y1 = boxes[:, 1]
    y2 = boxes[:, 1] + boxes[:, 3]

    boxes = torch.stack((x1, y1, x2, y2), dim=-1)
    
    return boxes

def xyxy_to_xywh(boxes):
    """
    Convert bounding boxes from (x1, y1, x2, y2) to (x, y, w, h)

    Args:
        boxes (Tensor[N, 4])

    Returns:
        Tensor[N, 4]
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    boxes = torch.stack((x, y, w, h), dim=-1)

    return boxes