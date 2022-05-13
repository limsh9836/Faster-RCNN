import torch
import torch.nn as nn
import torch.nn.functional as F

class FastRCNNLoss(nn.Module):
    def __init__(self, lmbd=1):
        super(FastRCNNLoss, self).__init__()
        self.lmbd = lmbd

    def forward(self, p, u, t, v):
        """
        Compute Multi-task Loss for Detection Network as defined in Fast R-CNN
        
        Args:
            Note: Omit N for now
            p (Tensor[N, M, K]) => (Tensor[M, K]): Predicted probabilities
            u (Tensor[N, M) => (Tensor[M]): Ground truth labels
            t (Tensor[N, M, 4]) => (Tensor[M, 4]): Predicted bounding boxes in format (tx, ty, tw, th)
            v (Tensor[N, M, 4]) => (Tensor[M, 4]): Target bounding boxes (vx, vy, vw, vh)
        Return:
            Tensor[N, M]: Multi-task Loss
        """
        return self.classification_loss(p, u) + self.lmbd * self.smooth_l1_loss(u, t, v) 

    def classification_loss(self, p, u):
        return F.cross_entropy(p, u)
    
    def smooth_l1_loss(self, u, t, v):
        weightage = torch.zeros_like(v)
        mask = torch.where(u >= 1)[0]
        weightage[mask] = 1
        
        diff = torch.abs(weightage * (t - v))
        mask = (diff < 1).to(torch.float32)
        losses = mask * 0.5 * (diff ** 2) + (1 - mask) * (diff - 0.5)
        loss = losses.sum() / losses.shape[0]
        
        return loss        

class RPNLoss(nn.Module):
    """
    Multi-task Loss for Region Proposal Network as defined in Faster R-CNN

    Args:
        lmbd: Regularization hyperparameter that regularize between classification and regression loss
    """
    def __init__(self, lmbd=10):
        super(RPNLoss, self).__init__()
        self.lmbd = lmbd
    
    def forward(self, p, u, t, v):
        """
        Compute Multi-task Loss
        
        Args:
            Omit N for now
            p (Tensor[N, M, 2]) => (Tensor[M, 2]): Predicted probabilities
            u (Tensor[N, M) => (Tensor[M]): Ground truth labels
            t (Tensor[N, M, 4]) => (Tensor[M, 4]): Predicted anchor boxes in format (tx, ty, tw, th)
            v (Tensor[N, M, 4]) => (Tensor[M, 4]): Target anchor boxes (vx, vy, vw, vh)
        Return:
            Tensor[N, M]: Multi-task Loss
        """
        return self.classification_loss(p, u) + self.smooth_l1_loss(u, t, v)
        # return self.classification_loss(p, u) + self.lmbd * self.smooth_l1_loss(u, t, v)

    def classification_loss(self, p, u):
        # Default 1 / N_mini_batch
        # n_anchors = p.shape[0]
        # return F.cross_entropy(p, u) * n_anchors
        return F.cross_entropy(p, u)

    
    def smooth_l1_loss(self, u, t, v):
        # n_anchors = t.shape[0]
        weightage = torch.zeros_like(v)
        mask = torch.where(u >= 1)[0]
        weightage[mask] = 1

        diff = torch.abs(weightage * (t - v))
        mask = (diff < 1).to(torch.float32)
        losses = mask * 0.5 * (diff ** 2) + (1 - mask) * (diff - 0.5)
        loss = losses.sum() / (u >= 0).sum().to(torch.float32)
        # loss = losses.sum() /  n_anchors

        return loss



        # diff = torch.abs(t - v)
        # loss = torch.zeros_like(diff)
        # loss[diff < 1] = 0.5 * (diff[diff < 1] ** 2)
        # loss[diff >= 1] = diff[diff >= 1] - 0.5
    
        # return loss


# def multitask_loss(p, u, t, v, lmd=1):
#     """
#     Multi-task Loss
    
#     Args:
#         p (Tensor[N, M]): Predicted probabilities
#         u (Tensor[N, M): Ground truth labels
#         t (Tensor[N, M, 4]): Predicted bounding boxes in format (tx, ty, tw, th)
#         v (Tensor[N, M, 4]): Ground truth bounding boxes (vx, vy, vw, vh)
#     """
#     return classification_loss(p, u) + lmd * u[u >= 1] * smooth_l1_loss(t, v) 
#     # pass

# def classification_loss(p, u):
#     return F.cross_entropy(p, u)
#     # pass

# def smooth_l1_loss(t, v):
#     diff = torch.abs(t - v)
#     loss = torch.zeros_like(diff)
#     loss[diff < 1] = 0.5 * (diff[diff < 1] ** 2)
#     loss[diff >= 1] = diff[diff >= 1] - 0.5

#     return loss
    
#     # pass
