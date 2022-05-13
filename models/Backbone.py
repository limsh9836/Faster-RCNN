import torch
import torch.nn as nn
from torchvision.models import vgg16

class VGG16(nn.Module):
    def __init__(self, freeze=False):
        super(VGG16, self).__init__()
        self.backbone = vgg16(pretrained=True).features
        if freeze:
            self.freeze()

    def forward(self, x):
        return self.backbone(x)

    def freeze(self):
        for child in self.children():
            for params in child.parameters():
                params.requires_grad = False    