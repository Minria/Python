import torch
import torch.nn as nn

def loss_cross_entropy(scores, labels):
    cross_entropy = -torch.sum(labels * torch.log(scores + 1e-10), dim=1)
    loss = torch.div(torch.sum(cross_entropy), torch.sum(labels)+1e-10)
    return loss

def loss_Rotation(pred_R, gt_R, label, model):
    device = pred_R.device
    models_pcd = model[label - 1].to(device)
    models_pcd = models_pcd.float()
    gt_points = models_pcd @ gt_R
    pred_points = models_pcd @ pred_R
    loss = ((pred_points - gt_points) ** 2).sum(dim=2).sqrt().mean()
    return loss

