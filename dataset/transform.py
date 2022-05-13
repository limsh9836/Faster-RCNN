import torchvision.transforms as transforms
import numpy as np
import cv2
from dataset.augmentation import random_horizontal_flip, resize_box

class Transform(object):
    def __init__(self, training=False, min_size=600, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.training = training
        self.min_size = min_size
        self.mean = mean
        self.std = std
                                                                                      
        self.transform = transforms.Compose([
            transforms.Lambda(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
    def __call__(self, image, boxes, labels):
        if isinstance(boxes, list):
            boxes = np.array(boxes)
        
        H, W, C = image.shape
        
        if self.training:
            image, boxes = random_horizontal_flip(image, boxes)
        
        image = self.transform(image)
        _, nH, nW = image.shape
        scale = nH / H
        boxes = resize_box(boxes, (H, W), (nH, nW))
        
            
        return image, boxes, labels, scale
    
    def resize(self, image):
        H, W, C = image.shape
        scale = self.min_size / min(H, W)
        image = cv2.resize(image, (int(W * scale), int(H * scale)))
        return image