import torch
# import numpy as np

def nms(boxes, scores, iou_threshold, return_index=False):
    """
    Perform Non-max suppression to remove redundant boxes
    Args:
        boxes (Tensor[N, 4]): Bounding boxes in format (x1, y1, x2, y2)
        scores (Tensor[N])
        iou_threshold (int)
    Return:
        Tensor[K, 4]
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = torch.sort(scores, descending=True).indices

    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        xx1 = torch.max(x1[i], x1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        w = torch.clamp(xx2 - xx1 + 1, min=0)
        h = torch.clamp(yy2 - yy1 + 1, min=0)
        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)

        inds = torch.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    if return_index:
        return keep
        
    return boxes[keep]
        

def iou(boxes1, boxes2):
    """
    Computer intersection over union (IoU) for two set of bounding boxes

    Args:
        boxes1 (Tensor[N, 4]): Bounding boxes in format (x1, y1, x2, y2)
        boxes2 (Tensor[M, 4]): Bounding boxes in format (x1, y1, x2, y2)
    Return:
        Tensor[N, M]: The N x M tensor containing the pairwise IoU for the cartesian product between boxes_1 and boxes_2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    # N, 1, 2
    #    M, 2
    # N, M, 2
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    intersection = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - intersection

    return intersection / union


def intersection_union(boxes1, boxes2):
    """
    Compute intersection and union

    Args:
        boxes1 (Tensor[N, 4]): Bounding boxes in format (x1, y1, x2, y2)
        boxes2 (Tensor[M, 4]): Bounding boxes in format (x1, y1, x2, y2)

    Return:
        Tuple(Tensor[N, M], Tensor[N, M])
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    intersection = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - intersection
    return intersection, union
    
def box_area(boxes):
    """
    Computes area of bounding boxes
    
    Args:
        boxes (Tensor[N, 4]): Bounding boxes in the format (x1, y1, x2, y2) to compute area.
    Returns:
        Tensor[N]: Area for each bounding box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])



    
    