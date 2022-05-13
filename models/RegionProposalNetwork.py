import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.detection.anchors import generate_anchor_base, generate_anchor_cell
from utils.detection.target import Proposal

class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels, mid_channels, ratios=[0.5, 1, 2], scales=[8, 16, 32], stride=32, **kwargs):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_base = generate_anchor_base(stride, ratios=ratios, scales=scales)
        self.propose = Proposal(self.training, **kwargs)
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, padding=1, stride=1)
        self.cls = nn.Conv2d(in_channels=mid_channels, out_channels=self.anchor_base.shape[0] * 2, kernel_size=1)
        self.reg = nn.Conv2d(in_channels=mid_channels, out_channels=self.anchor_base.shape[0] * 4, kernel_size=1)
        self.normal_init(self.children(), mean=0, std=0.01)
    
    # TODO: Training Mode, Testing Mode
    def forward(self, x, img_size, scale=1.):
        """
        Generate region proposals

        Args:
            x (Tensor[N, C, H, W]): Feature maps with input format (C, H, W)
        Return:

        """
        anchors = generate_anchor_cell(self.anchor_base, self.stride, size=(x.shape[-2:])) # K * H * W, 4

        feature = F.relu(self.conv1(x)) # N, M, H, W
        rpn_locs = self.reg(feature) # N, K * 4, H, W
        rpn_scores = self.cls(feature) # N, K * 2, H, W
        
        rpn_fg_scores = F.softmax(rpn_scores.view(x.shape[0], self.anchor_base.shape[0], x.shape[2], x.shape[3], 2), dim=-1)
        rpn_fg_scores = rpn_fg_scores[:, :, :, :, 1].view(x.shape[0], -1)
        rpn_scores = rpn_scores.view(x.shape[0], -1, 2)
        rpn_locs = rpn_locs.view(x.shape[0], -1 ,4)
        
        rois = []
        roi_indices = []

        for i in range(x.shape[0]):
            roi = self.propose(rpn_locs[i], rpn_fg_scores[i], anchors, img_size, scale=scale)
            index = torch.ones((len(roi),), dtype=torch.int32) * i

            rois.append(roi)
            roi_indices.append(index)

        rois = torch.cat(rois, dim=0)  
        roi_indices = torch.cat(roi_indices, dim=0)

        rois = torch.cat((roi_indices[:, None].to(torch.float32), roi), dim=-1)
        
        return rpn_locs, rpn_scores, rois, roi_indices, anchors
                
    def normal_init(self, layers, mean=0, std=1):
        for layer in layers:
            nn.init.normal_(layer.weight, std=std)
            nn.init.constant_(layer.bias, val=mean)
