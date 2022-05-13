import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import cv2
import os

VOC_LABEL = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'
)

class PascalVOC(Dataset):
    def __init__(self, data_dir, transform, split="trainval", difficult=False):
        self.data_dir = data_dir
        self.split = split
        self.difficult = difficult
        self.transform = transform
        image_list_dir = os.path.join(data_dir, "ImageSets", "Main", '{}.txt'.format(split))
        with open(image_list_dir) as f:
            self.image_list = [image.strip() for image in f]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        """
        Return item in ith index

        Args:
            index (int): Index of item
        Return:
            tuple: Tuple of image, bouding boxes, labels
        """
        name = self.image_list[index]
        image_path = os.path.join(self.data_dir, "JPEGImages", "{}.jpg".format(name))
        annotation_path = os.path.join(self.data_dir, "Annotations", "{}.xml".format(name))

        image = cv2.imread(image_path) # in np.ndarray
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes, labels, difficulties = self.parse(annotation_path)

        # if self.transform:
        image, boxes, labels, scale = self.transform(image, boxes, labels)

        return image, boxes, labels, difficulties, scale 
    
    def parse(self, path):
        """
        Parse XML annotation file
        
        Args:
            path (string): XML path to be parsed
        Return:
            list[tuple]: List of bounding boxes in (x1, y1, x2, y2)
            list[int]: List of labels
        """
        tree = ET.parse(path)
        root = tree.getroot()
        boxes = []
        labels = []
        difficulties = []

        for obj in root.findall("object"):
            if self.difficult and int(obj.find("difficult").text) != 1:
                continue
            
            box = obj.find("bndbox")
            label = VOC_LABEL.index(obj.find("name").text)
            x1, y1, x2, y2 = int(box.find("xmin").text), int(box.find("ymin").text), int(box.find("xmax").text), int(box.find("ymax").text)
            difficulty = int(box.find("difficulty").text)
            boxes.append((x1, y1, x2, y2))
            labels.append(label)
            difficulties.append(difficulty)
        
        return boxes, labels, difficulties