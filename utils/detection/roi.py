import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class RoI(Function):
    def __init__(self):
        pass

    def forward(self, x, rois):
        x = x.contiguous()
        rois = rois.contiguous()
        

    def backward(self, grad):
        pass


class RoIPooling2d(nn.Module):
    def __init__(self, height, width, scale):
        super(RoIPooling2d, self).__init__()
        self.RoI = RoI(height, width, scale)

    def forward(self, x, rois):
        return self.RoI(x, rois)