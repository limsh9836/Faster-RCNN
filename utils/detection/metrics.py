
import torch
from utils.detection.nms import iou

def calc_map(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, difficulties, n_classes=20, iou_threshold=0.5):
    gt_images = list()
    for i in range(len(gt_labels)):
        gt_images.extend([i] * gt_labels[i].shape[0])
    gt_images = torch.LongTensor(gt_images)
    gt_boxes = torch.cat(gt_boxes, dim=0)
    gt_labels = torch.cat(gt_labels, dim=0)
    difficulties = torch.cat(difficulties, dim=0)
    
    pred_images = list()
    for i in range(len(pred_labels)):
        pred_images.extend([i] * pred_labels[i].shape[0])
    pred_images = torch.LongTensor(pred_images)
    pred_boxes = torch.cat(pred_boxes, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    pred_scores = torch.cat(pred_scores, dim=0)
    
    ap = torch.zeros((n_classes-1, ), dtype=torch.float32)
    
    for i in range(1, n_classes):
        gt_class_images = gt_images[gt_labels == i]
        gt_class_boxes = gt_boxes[gt_labels == i]
        class_difficulties = difficulties[gt_labels == i]
        n_easy_class_objects = (1 - class_difficulties).sum().item()
        
        gt_class_boxes_detected = torch.zeros((gt_class_boxes.shape[0]), dtype=torch.uint8)
        
        pred_class_images = pred_images[pred_labels == i]
        pred_class_boxes = pred_boxes[pred_labels == i]
        pred_class_scores = pred_scores[pred_labels == i]
        n_class_pred = pred_class_boxes.shape[0]
        if n_class_pred:
            continue
        
        pred_class_scores, sort_index = torch.sort(pred_class_scores, dim=0, descending=True)
        pred_class_images = pred_class_images[sort_index]
        pred_class_boxes = pred_class_boxes[sort_index]
        
        tp = torch.zeros((n_class_pred), dtype=torch.float32)
        fp = torch.zeros((n_class_pred), dtype=torch.float32)
        
        for p in range(n_class_pred):
            pred_box = pred_class_boxes[p].unsqueeze(0)
            pred_image = pred_class_images[p]
            
            gt_box = gt_class_boxes[gt_class_images == pred_image]
            object_difficulties = class_difficulties[gt_class_images == pred_image]
            
            if gt_box.shape[0] == 0:
                fp[p] = 1
                continue
            
            overlap = iou(pred_box, gt_box)
            max_overlap, max_index = torch.max(overlap.squeeze(0), dim=0)
            
            original_index = torch.LongTensor(range(gt_class_boxes.shape[0]))[gt_class_images == pred_image][max_index]
            
            if max_overlap.item() > iou_threshold:
                if object_difficulties[max_index] == 0:
                    if gt_class_boxes_detected[original_index] == 0:
                        tp[p] = 1
                        gt_class_boxes_detected[original_index] = 1
                    else:
                        fp[p] = 1
            else:
                fp[p] = 1
                
        tp_cum = torch.cumsum(tp, dim=0)
        fp_cum = torch.cumsum(fp, dim=0)
        precision_cum = tp_cum / (tp_cum + fp_cum + 1e-10)
        recall_cum = tp_cum / n_easy_class_objects
        
        recall_thresholds = torch.arange(0., 1.1, .1).tolist()
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float32)
        for i, threshold in enumerate(recall_thresholds):
            mask = recall_cum >= threshold
            if mask.any():
                precisions[i] = torch.max(precision_cum[mask])
            else:
                precisions[i] = 0
        ap[i-1] = precisions.mean()
        
    mAp = ap.mean().item()
    
    return {"average_precision": ap, "mean_average_precision": mAp}