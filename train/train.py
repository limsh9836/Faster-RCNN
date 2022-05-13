import torch
from torch.utils.data import DataLoader
from models.FasterRCNN import FasterRCNNVGG16
from dataset.transform import Transform
from dataset.VOC import PascalVOC
from train.trainer import FasterRCNNTrainer
from utils.detection.metrics import calc_map

def train(data_dir, model_path, batch_size=1, n_class=20, epochs=10):
    train_transform = Transform(training=True)
    train_dataset = PascalVOC(data_dir, transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    
    model = FasterRCNNVGG16(n_class)
    trainer = FasterRCNNTrainer(model)
    trainer.load(model_path)
    for epoch in range(epochs):
        for i, (image, boxes, labels, difficulties, scale) in enumerate(train_dataloader):
            labels = torch.tensor(labels, dtype=torch.uint8)
            loss = trainer.step(image, boxes, labels, scale)

def eval(dataloader, model, iteration):
    pred_boxes, pred_labels, pred_scores = list(), list(), list()
    gt_boxes, gt_labels, gt_difficulties = list(), list()
    
    for i, (image, boxes, labels, difficulties, scale) in enumerate(dataloader):
        if i == iteration: 
            break
        _pred_boxes, _pred_labels, _pred_scores = model.predict(image, scale)
        gt_boxes.append(boxes)
        gt_labels.append(labels)
        gt_difficulties.append(difficulties)
        pred_boxes.append(_pred_boxes)
        pred_labels.append(_pred_labels)
        pred_scores.append(_pred_scores)
    
    result = calc_map(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, gt_difficulties)
    
    return result