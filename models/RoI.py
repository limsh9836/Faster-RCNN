import torch
import torch.nn as nn
from torchvision.ops import RoIPool, roi_pool

class RoIHead(nn.Module):
    def __init__(self, in_channels, n_class, roi_size, spatial_scale):
        super(RoIHead, self).__init__()
        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi_pool = RoIPool(roi_size, spatial_scale)
        self.fc = nn.Sequential(
            nn.Linear(roi_size * roi_size * in_channels, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
        )
        self.cls = nn.Linear(4096, n_class)
        self.reg = nn.Linear(4096, n_class * 4)
        self.normal_init(self.modules(), mean=0, std=.001)

    def forward(self, features, rois):
        pool = self.roi_pool(features, rois)
        # feature = roi_pool(x, rois, output_size=self.roi_size, spatial_scale=self.spatial_scale)
        pool = pool.view(pool.shape[0], -1)
        # print(feature.shape)
        output = self.fc(pool)
        # print(feature.shape)
        roi_locs = self.reg(output)
        roi_scores = self.cls(output)
        return roi_locs, roi_scores

    def normal_init(self, layers, mean=0, std=1.):
        for layer in layers:
            if isinstance(layer, nn.modules.linear.Linear):
                nn.init.normal_(layer.weight, std=std)
                nn.init.constant_(layer.bias, val=mean)