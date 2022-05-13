from utils.detection.format import xyxy_to_xywh
import torch

def clip_box(boxes, max, min=0):
    """
    Clip box so that only valid boxes remain

    Args:
        boxes (Tensor[N, 4]): Boxes in format (x1, y1, y1, y2) to be clipped
        max (int): Maximum value allowed for the boxes
        min (int): Minimum value allowed for the boxes
    Return:
        Tensor[M, 4]: Clipped boxes
    """
    index = torch.where(
        (boxes[:, 0] >= min) and
        (boxes[:, 1] >= min) and
        (boxes[:, 2] <= max) and
        (boxes[:, 3] <= max)
    )[0]

    return boxes[index]

def box_to_loc(boxes, anchors):
    """
    Convert boxes to regions proposal locations

    Args:
        boxes (Tensor[N, 4]): Bounding boxes in format (x1, y1, x2, y2)
        anchors (Tensor[N, 4]): Anchors in format (x1, y1, x2, y2)
    Return:
        Tensor[N, 4]: Target locations in format (tx, ty, tw, th)
    """

    x = (boxes[:, 0] + boxes[:, 2]) / 2
    y = (boxes[:, 1] + boxes[:, 3]) / 2
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    ax = (anchors[:, 0] + anchors[:, 2]) / 2
    ay = (anchors[:, 1] + anchors[:, 3]) / 2
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]

    tx = (x - ax) / aw
    ty = (y - ay) / ah
    tw = torch.log(w / aw)
    th = torch.log(h / ah) 

    locs = torch.stack((tx, ty, tw, th), dim=-1)
    
    return locs

    # tx = (boxes[:, 0] - anchors[:, 0]) / anchors[:, 2]
    # ty = (boxes[:, 1] - anchors[:, 1]) / anchors[:, 3]
    # tw = torch.log(boxes[:, 2] / anchors[:, 2])
    # th = torch.log(boxes[:, 3] / anchors[:, 3])

    # locs = torch.stack((tx, ty, tw, th), dim=-1)
    
    # return locs

def loc_to_box(locs, anchors):
    """
    Convert region proposal location to boxes

    Args:
        anchors (Tensor[N, 4]): Anchors in format (x1, y1, x2, y2)
        locs (Tensor[N, 4]): Locations in format (tx, ty, tw, th)
    Return:
        Tensor[N, 4]: Bounding boxes in format (x1, y1, x2, y2)
    """

    x = (anchors[:, 0] + anchors[:, 2]) / 2
    y = (anchors[:, 1] + anchors[:, 3]) / 2
    w = anchors[:, 2] - anchors[:, 0]
    h = anchors[:, 3] - anchors[:, 1]

    tx = locs[:, 0]
    ty = locs[:, 1]
    tw = locs[:, 2]
    th = locs[:, 3]

    x = w * tx + x
    y = h * ty + y
    w = torch.exp(tw) * w
    h = torch.exp(th) * h

    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2

    # x = anchors[:, 2] * tx + anchors[:, 0]
    # y = anchors[:, 3] * ty + anchors[:, 1]
    # w = torch.exp(tw) * anchors[:, 2]
    # h = torch.exp(th) * anchors[:, 3]
    
    boxes = torch.stack((x1, y1, x2, y2), dim=-1)

    return boxes