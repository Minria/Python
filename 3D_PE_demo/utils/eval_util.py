import random

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

from torchvision.ops import box_iou
import sys, os
import trimesh
import pyrender
import tqdm
import torch
import torch.nn as nn



# 将RT进行合并成为(3,4)
def merge_rt(R, T):
    sub_tensor1 = R[:, :, :, :3]
    sub_tensor2 = T[:, :, :3]
    merged_tensor = torch.cat((sub_tensor1, sub_tensor2.unsqueeze(-1)), dim=-1)
    return merged_tensor


# 从预测的标签中计算出bbx
def get_pred_bbx(num_classes, label):
    bbx = []
    bs, h, w = label.shape
    device = label.device
    label_repeat = label.view(bs, 1, h, w).repeat(1, num_classes, 1, 1).to(device)
    label_target = torch.linspace(0, num_classes - 1, steps=num_classes).view(1, -1, 1, 1).repeat(bs, 1, h, w).to(
        device)
    mask = (label_repeat == label_target)
    for batch_id in range(mask.shape[0]):
        for cls_id in range(mask.shape[1]):
            if cls_id != 0:
                # cls_id == 0 is the background
                y, x = torch.where(mask[batch_id, cls_id] != 0)
                if y.numel() >= 500:
                    bbx.append([batch_id, torch.min(x).item(), torch.min(y).item(),
                                torch.max(x).item(), torch.max(y).item(), cls_id])
    bbx = torch.tensor(bbx).to(device)
    return bbx

def get_center_from_label(num_classes, label):
    center = []
    bs, h, w = label.shape
    device = label.device
    label = label.to('cpu').numpy()
    for batch_id in range(bs):
        label_tmp = label[batch_id]
        tmp = np.zeros((num_classes, 2))
        for cls_id in range(num_classes+1):
            if cls_id != 0:
                mask = (label_tmp == cls_id)
                indices = np.argwhere(mask)
                min_x, min_y = np.min(indices, axis=0)
                max_x, max_y = np.max(indices, axis=0)
                tmp[cls_id - 1, 0] = (max_y + min_y)/2
                tmp[cls_id - 1, 1] = (max_x + min_x)/2
        center.append(tmp)
    center = torch.tensor(center).to(device)
    return center

def get_tran_tz(translation, bbx, label):
    size = bbx.shape[0]
    pred_tran = torch.zeros(size, 3)
    for idx, bbx in enumerate(bbx):
        batch_id = int(bbx[0].item())
        cls = int(bbx[5].item())
        trans_map = translation[batch_id, (cls - 1) * 3: cls * 3, :]
        tmp = (label[batch_id] == cls).detach()
        pred_t = trans_map[:, tmp].mean(dim=1)
        pred_tran[idx] = pred_t
    return pred_tran


def get_rot_from_quaternion(quaternions):
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def get_rotation_from_quaternions(quaternions, bbx, bs=2):
    pred_rot = torch.zeros(bs, 3, 3, 3)
    for idx, bbx in enumerate(bbx):
        batch_id = int(bbx[0].item())
        cls = int(bbx[5].item())
        quaternion = quaternions[idx, (cls - 1) * 4: cls * 4]
        quaternion = nn.functional.normalize(quaternion, dim=0)
        pred_rot[batch_id, cls-1] = get_rot_from_quaternion(quaternion)
    return pred_rot

def compute_tran(pred_tran, pred_centers, bboxes, cam_intrinsic, bs=2):
    tran = np.zeros((bs, 3, 3))
    for idx, bbx in enumerate(bboxes):
        bs, _, _, _, _, obj_id = bbx
        center = pred_centers[bs.long(), obj_id.long()-1].to('cpu').numpy()
        depth = pred_tran[idx, 2].item()
        if (center**2).sum() != 0:
            T = np.linalg.inv(cam_intrinsic) @ np.array([center[0], center[1], 1]) * depth
            tran[bs.long(), obj_id.long()-1] = T
    return torch.tensor(tran)

def HoughVoting(label, centermaps, num_classes=3):
    bs, _, _ = label.shape
    device = label.device
    x = np.linspace(0, 639, 640)
    y = np.linspace(0, 479, 480)
    xv, yv = np.meshgrid(x, y)
    xy = torch.from_numpy(np.array((xv, yv))).to(device=device, dtype=torch.int32)
    x = torch.from_numpy(x).to(device=device, dtype=torch.int32)

    centers = torch.zeros(bs, num_classes, 2)
    for b in range(bs):
        for cls in range(1, num_classes+1):
            if (label[b] == cls).sum() > 100:
                target_index = xy[:2, label[b] == cls]
                centermap = centermaps[b, (cls-1)*3:cls*3][:2, label[b] == cls]
                c0_sub_x = x.unsqueeze(dim=0) - target_index[0].unsqueeze(dim=1)
                pred_y = torch.round(target_index[1].unsqueeze(dim=1) + (centermap[1]/centermap[0]).unsqueeze(dim=1) * c0_sub_x)\
                    .to(device=device, dtype=torch.int32)
                mask = (pred_y >= 0) * (pred_y <= 480)
                count = pred_y * 640 + x.unsqueeze(dim=0)
                center, num = torch.bincount(count[mask]).argmax(), torch.bincount(count[mask]).max()
                center_x, center_y = center % 640, torch.div(center, 640, rounding_mode='trunc')
                centers[b, cls - 1, 0], centers[b, cls - 1, 1] = center_x, center_y
    return centers


# def HoughVoting(label, centermap, num_classes=3):
#     batches, H, W = label.shape
#     x = np.linspace(0, W - 1, W)
#     y = np.linspace(0, H - 1, H)
#     xv, yv = np.meshgrid(x, y)
#     xy = torch.from_numpy(np.array((xv, yv))).to(device=label.device, dtype=torch.float32)
#     x_index = torch.from_numpy(x).to(device=label.device, dtype=torch.int32)
#     centers = torch.zeros(batches, num_classes, 2)
#     depths = torch.zeros(batches, num_classes)
#     for bs in range(batches):
#         for cls in range(1, num_classes + 1):
#             if (label[bs] == cls).sum() >= 500:
#                 pixel_location = xy[:2, label[bs] == cls]
#                 pixel_direction = centermap[bs, (cls-1)*3:cls*3][:2, label[bs] == cls]
#                 y_index = x_index.unsqueeze(dim=0) - pixel_location[0].unsqueeze(dim=1)
#                 y_index = torch.round(pixel_location[1].unsqueeze(dim=1) + (pixel_direction[1]/pixel_direction[0]).unsqueeze(dim=1) * y_index).to(torch.int32)
#                 mask = (y_index >= 0) * (y_index < H)
#                 count = y_index * W + x_index.unsqueeze(dim=0)
#                 center, inlier_num = torch.bincount(count[mask]).argmax(), torch.bincount(count[mask]).max()
#                 center_x, center_y = center % W, torch.div(center, W, rounding_mode='trunc')
#                 if inlier_num > 500:
#                     centers[bs, cls - 1, 0], centers[bs, cls - 1, 1] = center_x, center_y
#                     xyplane_dis = xy - torch.tensor([center_x, center_y])[:, None, None].to(device = label.device)
#                     xyplane_direction = xyplane_dis/(xyplane_dis**2).sum(dim=0).sqrt()[None, :, :]
#                     predict_direction = centermap[bs, (cls-1)*3:cls*3][:2]
#                     inlier_mask = ((xyplane_direction * predict_direction).sum(dim=0).abs() >= 0.9) * label[bs] == cls
#                     depths[bs, cls - 1] = centermap[bs, (cls-1)*3:cls*3][2, inlier_mask].mean()
#     return centers, depths


def compute_overlap_ratio(x1_1, y1_1, x2_1, y2_1, x1_2, y1_2, x2_2, y2_2):
    # 计算交集面积
    intersection_area = max(0, min(x2_1, x2_2) - max(x1_1, x1_2)) * max(0, min(y2_1, y2_2) - max(y1_1, y1_2))

    # 计算并集面积
    union_area = (x2_1 - x1_1) * (y2_1 - y1_1) + (x2_2 - x1_2) * (y2_2 - y1_2) - intersection_area

    # 计算交占比
    overlap_ratio = intersection_area / union_area

    return overlap_ratio

print(compute_overlap_ratio(0,0,100,100,50,50,200,200))



