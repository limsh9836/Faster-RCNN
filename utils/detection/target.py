from typing import Tuple
import torch
import math
from utils.detection.nms import nms, iou
from utils.detection.box import loc_to_box, box_to_loc
# from utils.detection.format import xywh_to_xyxy, xyxy_to_xywh

class AnchorTarget:
    def __init__(self, n_sample=256, pos_threshold=0.7, neg_threshold=0.3, positive_ratio=0.5):
        self.n_sample = n_sample
        self.positive_threshold = pos_threshold
        self.negative_threshold = neg_threshold
        self.positive_ratio = positive_ratio
    
    def get_valid_index(self, anchors, size):
        if isinstance(size, tuple):
            H, W = size
        elif isinstance(size, int):
            H = size
            W = H
        else:
            raise TypeError("Expected tuple of int, but get {} instead".format(type(size)))

        H, W = size
        valid_index = torch.where(
            (anchors[:, 0] >= 0) &
            (anchors[:, 1] >= 0) &
            (anchors[:, 2] <= H) &
            (anchors[:, 3] <= W)
        )[0]

        return valid_index

    def unmap(self, data, count, index, fill=0):
        if len(data.shape) == 1:
            unmapped = torch.full((count,), fill, dtype=data.dtype)
            unmapped[index] = data
        else:
            unmapped = torch.full(((count,) + data.shape[1:]), fill, dtype=data.dtype)
            unmapped[index, :] = data
        return unmapped

    def __call__(self, anchors, boxes, size):
        """
        Assign ground truth target to anchors

        Args:
            anchors (Tensor[M, 4]): Anchor boxes (x1, y1, x2, y2) # Check
            boxes (Tensor[N, 4]): Ground truth boxes (x1, y1, x2, y2)
            size (tuple): Height and width of image
        Return:
            Tensor[M, 4]: Target locations
            Tensor[M,]: Target labels
        """

        # boxes, anchors = xyxy_to_xywh(boxes), xyxy_to_xywh(anchors)

        n_anchors = anchors.shape[0]
        valid_index = self.get_valid_index(anchors, size)
        anchors = anchors[valid_index]


        labels = torch.zeros((anchors.shape[0], ), dtype=torch.float32)
        labels = labels - 1

        ious = iou(anchors, boxes) # (N, M)
        box_arg_max = torch.max(ious, dim=0).indices # (M)
        anchor_max, anchor_arg_max = torch.max(ious, dim=1) #(N)

        labels[anchor_max < self.negative_threshold] = 0

        labels[box_arg_max] = 1

        labels[anchor_max >= self.positive_threshold] = 1

        n_positive = math.ceil(self.positive_ratio * self.n_sample)

        positive_indices = torch.where(labels == 1)[0]

        if positive_indices.shape[0] > n_positive:
            replace = torch.randperm(positive_indices.shape[0])[:positive_indices.shape[0] - n_positive]
            labels[positive_indices[replace]] = -1
        
        n_negative = self.n_sample - torch.where(labels == 1)[0].shape[0]


        negative_indices = torch.where(labels == 0)[0]

        if negative_indices.shape[0] > n_negative:
            replace = torch.randperm(negative_indices.shape[0])[:negative_indices.shape[0] - n_negative]
            labels[negative_indices[replace]] = -1

        locs = box_to_loc(boxes[anchor_arg_max], anchors)

        labels = self.unmap(labels, n_anchors, valid_index, fill=-1)
        locs = self.unmap(locs, n_anchors, valid_index, fill=0)

        # labels, anchors_max = self.create_target(anchors, boxes)
        # locs = box_to_loc(boxes[anchors_max], anchors)

        return locs, labels

    # def create_target(self, anchors, boxes):
    #     labels = torch.zeros((anchors.shape[0], ), dtype=torch.float32)
    #     labels = labels - 1

    #     ious = iou(anchors, boxes) # (N, M)
    #     box_arg_max = torch.max(ious, dim=0).indices # (M)
    #     anchor_max = torch.max(ious, dim=1).values #(N)

        
    #     labels[anchor_max < self.negative_threshold] = 0

    #     labels[box_arg_max] = 1

    #     labels[anchor_max >= self.positive_threshold] = 1

    #     n_positive = math.ceil(self.positive_ratio * self.n_sample)

    #     positive_indices = torch.where(labels == 1)[0]

    #     if positive_indices.shape[0] > n_positive:
    #         replace = torch.randperm(positive_indices.shape[0])[:positive_indices.shape[0] - n_positive]
    #         labels[positive_indices[replace]] = -1
        
    #     n_negative = self.n_sample - torch.where(labels == 1)[0].shape[0]


    #     negative_indices = torch.where(labels == 0)[0]

    #     if negative_indices.shape[0] > n_negative:
    #         replace = torch.randperm(negative_indices.shape[0])[:negative_indices.shape[0] - n_negative]
    #         labels[negative_indices[replace]] = -1
        
    #     return labels, anchor_max
        

class ProposalTarget:
    def __init__(self, 
                n_sample=128, 
                positive_ratio=0.25, 
                positive_threshold=0.5, 
                negative_threshold=(0.1, 0.5)):
        self.n_sample = n_sample
        self.positive_ratio = positive_ratio
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold

    def __call__(self, rois, boxes, labels):
        """
        Generate proposal targets

        Args:
            rois (Tensor[N, 4]): Region of interests in format (x1, y1, x2, y2)
            boxes (Tensor[M, 4]): Ground truth boxes in format (x1, y1, x2, y2)
            labels (Tensor[M]): Ground truth labels
        Return:
            Tensor(K, 4): Target locations
            Tensor(K,) : Target labels
        """
        ious = iou(rois, boxes) # N, M

        roi_max, roi_arg_max = torch.max(ious, dim=1) # N
        # print("Labels: ", labels)
        # print("Arg Max: ", roi_arg_max)

        gt_roi_labels = labels[roi_arg_max] + 1 # Reserve 0 for background

        positive_indices = torch.where(roi_max >= self.positive_threshold)[0]

        n_positive = math.ceil(self.n_sample * self.positive_ratio)
        
        if positive_indices.shape[0] > n_positive:
            keep = torch.randperm(positive_indices.shape[0])[:n_positive]
            positive_indices = positive_indices[keep]
        
        negative_indices = torch.where(
            (roi_max < self.negative_threshold[1]) &
            (roi_max >= self.negative_threshold[0])
        )[0]

        n_negative = self.n_sample - n_positive

        if negative_indices.shape[0] > n_negative:
            keep = torch.randperm(negative_indices.shape[0])[:n_negative]
            negative_indices = negative_indices[keep]
        
        keep = torch.cat((positive_indices, negative_indices))
        gt_roi_labels = gt_roi_labels[keep]
        gt_roi_labels[n_positive:] = 0
        
        sampled_rois = rois[keep]
        gt_roi_locs = box_to_loc(boxes[roi_arg_max[keep]], sampled_rois)

        return sampled_rois, gt_roi_locs, gt_roi_labels

    # def create_target(self, rois, boxes, labels):
    #     ious = iou(rois, boxes) # N, M

    #     roi_max, roi_arg_max = torch.max(ious, dim=1)

    #     gt_roi_labels = labels[roi_arg_max] + 1 # Reserve 0 for background

    #     positive_indices = torch.where(roi_max >= self.positive_threshold)[0]

    #     n_positive = self.n_sample * self.positive_ratio
        
    #     if positive_indices.shape[0] > n_positive:
    #         keep = torch.randperm(positive_indices.shape[0])[:n_positive]
    #         positive_indices = positive_indices[keep]
        
    #     negative_indices = torch.where(
    #         (roi_max < self.negative_threshold[1]) &
    #         (roi_max >= self.negative_threshold[0])
    #     )

    #     n_negative = self.n_sample - n_positive

    #     if negative_indices.shape[0] > n_negative:
    #         keep = torch.randperm(negative_indices.shape[0])[:n_negative]
    #         negative_indices = negative_indices[keep]
        
    #     keep = torch.cat((positive_indices, negative_indices))
    #     gt_roi_labels = gt_roi_labels[keep]
    #     gt_roi_labels[n_positive:] = 0
        
    #     sample = rois[keep]
    #     gt_roi_locs = box_to_loc(boxes[roi_arg_max[keep]], sample)

    #     return gt_roi_locs, gt_roi_labels

        # pass



class Proposal:
    def __init__(self, 
                training=False, 
                threshold = 0.7,
                n_train_pre_nms=12000,
                n_train_post_nms=2000,
                n_test_pre_nms=6000,
                n_test_post_nms=300,
                min_size=32):
        self.training = training
        self.threshold = threshold
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, locs, scores, anchors, size, scale=1.):
        """
        Generate proposal
        Args:
            locs (Tensor[N, 4]): Locations (tx, ty, tw, th)
            scores (Tensor[N]): Scores
            anchors (Tenshor[N, 4]): Anchors (x1, y1, x2, y2)
            size (Tuple(int, int)): Size
            scale (float): Scale
        Return:
            Tensor[M, 4]: Proposals (x1, y1, x2, y2)
        """
        if isinstance(size, int):
            size = (size, size)
            
        n_pre_nms, n_post_nms = (self.n_train_pre_nms, self.n_train_post_nms) if self.training else (self.n_test_pre_nms, self.n_test_post_nms)
        rois = loc_to_box(locs, anchors)
        # rois = xywh_to_xyxy(rois)

        rois[:, ::2] = torch.clamp(rois[:, ::2].clone(), min=0, max=size[0])
        rois[:, 1::2] = torch.clamp(rois[:, 1::2].clone(), min=0, max=size[1])

        min_size = self.min_size * scale

        w = rois[:, 2] - rois[:, 0]
        h = rois[:, 3] - rois[:, 1]

        keep = torch.where(
            (w >= min_size) & (h >= min_size)
        )[0]

        rois = rois[keep].clone()
        scores = scores[keep].clone()

        order = torch.sort(scores, dim=-1, descending=True).indices

        rois = rois[order[:n_pre_nms]].clone()
        scores = scores[order[:n_pre_nms]].clone()
        rois = nms(rois, scores, self.threshold).clone()
        rois = rois[:n_post_nms].clone()
                                                                        
        return rois
        

        

        

        