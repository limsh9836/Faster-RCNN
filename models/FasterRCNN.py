import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from models.RegionProposalNetwork import RegionProposalNetwork
from models.RoI import RoIHead
from models.Backbone import VGG16
from utils.detection.box import loc_to_box
from utils.detection.nms import nms

# from models import RegionProposalNetwork, RoI, Backbone

class FasterRCNN(nn.Module):
    def __init__(self, backbone, rpn, roi_head):
        super(FasterRCNN, self).__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.roi_head = roi_head

    @property
    def n_class(self):
        return self.roi_head.n_class

    def forward(self, x, scale=1.):
        size = x.shape[-2:]
        feature = self.backbone(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchors = self.rpn(feature, size, scale)
        # rois = self.rpn(feature, img_size, scale)
        roi_locs, roi_scores = self.roi_head(feature, rois)
        return roi_locs, roi_scores
        # rpn_locs, rpn_scores, rois, roi_indices, anchors = self.rpn(feature, img_size, scale)
        # roi_locs, roi_scores = self.head(feature, rois, roi_indices)
        # return roi_locs, roi_scores, rois, roi_indices

    def suppress(self, raw_boxes, raw_probs):
        boxes = list()
        labels = list()
        scores = list()
        for i in range(1, self.n_class):
            cls_boxes = raw_boxes.view(-1, self.n_class, 4)[:, i]
            cls_probs = raw_probs[i]
            mask = cls_probs > 0.5
            cls_boxes = cls_boxes[mask]
            cls_probs = cls_probs[mask]
            mask = nms(cls_boxes, cls_probs, 0.5, return_index=True)
            boxes.append(cls_boxes[mask])
            labels.append((i - 1) * torch.ones((len(mask),)))
            scores.append(cls_probs[mask])

        boxes = torch.cat(boxes, dim=0)
        labels = torch.cat(labels, dim=0)
        scores = torch.cat(scores, dim=0)

        return boxes, labels, scores
            
    def predict(self, image, scale):
        self.eval()
        boxes = list()
        labels = list()
        scores = list()

        size = image.shape[-2:]
        feature = self.backbone(image)
        rpn_locs, rpn_scores, rois, roi_indices, anchors = self.rpn(feature, size, scale)
        roi_locs, roi_scores = self.roi_head(feature, rois)

        rois = rois[1:] # Remove indices
        rois = rois / scale
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])

        roi_locs = roi_locs * std + mean
        roi_locs = roi_locs.view(-1, self.n_class, 4)
        rois = rois.view(-1, 1, 4).expand_as(roi_locs)
        boxes = loc_to_box(roi_locs.view(-1,4), rois.view(-1,4))
        boxes = boxes.view(-1, self.n_class * 4)
        boxes[:, ::2] = (boxes[:, ::2]).clamp(min=0, max=size[0] * scale)
        boxes[:, 1::2] = (boxes[:, 1::2]).clamp(min=0, max=size[1] * scale)

        probs = F.softmax(roi_scores, dim=1)
        
        boxes, labels, scores = self.suppress(boxes, probs)

        self.train()
        return boxes, labels, scores


class FasterRCNNVGG16(FasterRCNN):
    def __init__(self, n_class, ratios=[0.5, 1, 2], scales=[8, 16, 32]):
        # self.backbone = VGG16()
        backbone = torchvision.models.vgg16(pretrained=True).features
        # TODO: Check 
        rpn = RegionProposalNetwork(
            in_channels=512,
            mid_channels=512,
            ratios=ratios,
            scales=scales,
            stride=32
        )
        roi_head = RoIHead(
            in_channels=512,
            n_class=n_class + 1,
            roi_size=7,
            spatial_scale=(1./32)
        )
        super(FasterRCNNVGG16, self).__init__(backbone, rpn, roi_head)

    