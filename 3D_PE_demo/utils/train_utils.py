
import torch
import torch.nn as nn
from utils.eval_util import *


def get_rot_from_gt(bbx, RT):
    n = bbx.shape[0]
    gt_rs = torch.zeros(n, 3, 3)
    for idx, bbx in enumerate(bbx):
        batch_id = int(bbx[0].item())
        cls = int(bbx[5].item())
        gt_rs[idx] = RT[batch_id][cls - 1][:3, :3]
    return gt_rs

def get_rotation(quaternions, bbx):
    size = bbx.shape[0]
    pred_rot = torch.zeros(size, 3, 3)
    label = []
    for idx, bbx in enumerate(bbx):
        batch_id = int(bbx[0].item())
        cls = int(bbx[5].item())
        quaternion = quaternions[idx, (cls - 1) * 4: cls * 4]
        quaternion = nn.functional.normalize(quaternion, dim=0)
        pred_rot[idx] = get_rot_from_quaternion(quaternion)
        label.append(cls)
    label = torch.tensor(label)
    return pred_rot, label


# 将(bs, num, size)的bbx转换为(bs*num,size+1)的bbx,多出来的维度作为批次信息
def bbx_change(tensor):
    bs, num, size = tensor.shape
    bbx = torch.zeros(bs, num, size+1)
    for i in range(bs):
        bbx[i, :, 0] = i
        for j in range(num):
            bbx[i, j, 1:] = tensor[i, j, :]
    return torch.reshape(bbx, (bs*num, size+1))

