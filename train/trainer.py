import torch
import torch.nn as nn
from torch.optim import Adam


import cv2
import os
import time

from utils.detection.target import AnchorTarget, ProposalTarget
from loss.loss import RPNLoss, FastRCNNLoss

class FasterRCNNTrainer(nn.Module):
    def __init__(self, faster_rcnn, optimizer=None, rpn_loss=None, fastrcnn_loss=None):
        super(FasterRCNNTrainer, self).__init__()
        self.device = torch.device("cpu") # Temp
        self.faster_rcnn = faster_rcnn
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = Adam(self.faster_rcnn.parameters())
        self.anchor_target = AnchorTarget()
        self.proposal_target = ProposalTarget()
        self.rpn_loss = rpn_loss if rpn_loss is not None else RPNLoss()
        self.fastrcnn_loss = fastrcnn_loss if fastrcnn_loss is not None else FastRCNNLoss()
         
    
    def forward(self, images, boxes, labels, scale):
        assert images.shape[0] == 1

        _, _, H, W = images.shape
        features = self.faster_rcnn.backbone(images)
        rpn_locs, rpn_scores, rois, _, anchors = self.faster_rcnn.rpn(features, (H, W), scale)
        
        
        boxes = torch.squeeze(boxes, dim=0) # M, 4
        rpn_locs = torch.squeeze(rpn_locs, dim=0) # K, 4
        rpn_scores = torch.squeeze(rpn_scores, dim=0) # K, 2
        rois = rois[:, 1:]
        
        gt_rpn_locs, gt_rpn_labels = self.anchor_target(anchors, boxes, (H, W))
        
        sampled_rois, gt_roi_locs, gt_roi_labels = self.proposal_target(rois, boxes, labels)
        sampled_rois_indices = torch.zeros((len(sampled_rois),1), dtype=sampled_rois.dtype)
        sampled_rois = torch.cat((sampled_rois_indices, sampled_rois), dim=-1)
        
        roi_locs, roi_scores = self.faster_rcnn.roi_head(features, sampled_rois)
        roi_locs = roi_locs.view(roi_locs.shape[0], -1, 4).contiguous()
        
        roi_cls_locs = torch.zeros_like(gt_roi_locs)
        n_sample = gt_roi_locs.shape[0]
        for i in range(n_sample):
            loc = roi_locs[i, gt_roi_labels[i].item(), :]
            label = gt_roi_labels[i]
            loc = roi_locs[i, label.item()]
            roi_cls_locs[i] = loc
           
        roi_locs = roi_cls_locs
        mask = torch.where(gt_rpn_labels >= 0)[0]
        rpn_scores = rpn_scores[mask]
        rpn_locs = rpn_locs[mask]
        gt_rpn_labels = gt_rpn_labels[mask]
        gt_rpn_locs = gt_rpn_locs[mask]
        
        rpn_loss = self.rpn_loss(rpn_scores, gt_rpn_labels.long(), rpn_locs, gt_rpn_locs)
        fastrcnn_loss = self.fastrcnn_loss(roi_scores, gt_roi_labels.long(), roi_locs, gt_roi_locs)
        total_loss = rpn_loss + fastrcnn_loss

        loss = dict()
        loss.update(
            {
                "rpn": rpn_loss,
                "fastrcnn": fastrcnn_loss,
                "total": total_loss
            }
        )
        
        
        return loss
    
    def step(self, images, boxes, labels, scale):
        self.optimizer.zero_grad()
        loss = self(images, boxes, labels, scale)
        loss['total'].backward()
        self.optimizer.step()
        return loss
    
    def load(self, path=None):
        if os.path.exists(path):
            self.faster_rcnn.load_state_dict(torch.load(path))
            return True
        
        return False
        
    
    def save(self, path=None):
        suffix = time.strftime("%m%d%H%M")
        if path is not None:
            path = path.strip(".pt") + suffix + ".pt"
        else:
            path = os.path.join("assets","fasterrcnn".format(suffix + ".pt"))
        
        model_dir = os.path.dirname(path)
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        torch.save(self.faster_rcnn.state_dict(), path)