import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.dataset import YCBDataSet
from model.model import *
import torchvision.models as models
from Logger import Logger
from model.loss import loss_cross_entropy, loss_Rotation
from utils.dataset_utils import *
from utils.eval_util import *
from utils.train_utils import *
import torch.nn as nn
from model.loss import *
from utils.vision_util import visualize_dataset, visualize_output
import numpy as np

def compute_loss(tensor1, tensor2):
    arr1 = np.array(tensor1)
    arr2 = np.array(tensor2)
    loss = np.mean(np.square(arr1 - arr2))
    return loss


def extract_columns(tensor):
    # 将张量转换为 NumPy 数组进行操作
    arr = np.array(tensor)
    # 提取�? 3 列组成新�? (1, 3, 3, 3) 张量
    tensor_3d = arr[:, :, :, :3]

    # 提取最后一列组成新�? (1, 3, 3) 张量
    tensor_2d = arr[:, :, :, 3]
    return np.squeeze(tensor_3d, axis=0), np.squeeze(tensor_2d, axis=0)
    # return tensor_3d, tensor_2d

batch_size = 1

start = time.time()
print(f'cuda is available: {torch.cuda.is_available()}')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device is {device}')


test_dataset = YCBDataSet(file_name='data/0000test', data_path='E:/Archive files/dataset/YCB_Video_Dataset',
                          mode='test', train=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f'dataset length: {len(test_loader)}')
print(f'finsh data loading, time is: {time.time() - start}')

# 初始化模型和优化�?
vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
# model1 = BaseModulePlus(pre_model=vgg16, is_init_pre=True).to(device)
model1 = BaseModule(pre_model=vgg16, is_init_pre=True).to(device)
model2 = ClassModule(num_classes=3).to(device)
model3 = TranslationModule(num_classes=3).to(device)
model4 = RotationModule(num_classes=3).to(device)

criterion = nn.L1Loss()

print(f'load model and optimizer, time is: {time.time() - start}')

criterion2 = nn.CrossEntropyLoss()

# model1.load_state_dict(torch.load('BaseModule.pth'))
model2.load_state_dict(torch.load('ClassModule50.pth'))
model3.load_state_dict(torch.load('TranslationModule.pth'))
model4.load_state_dict(torch.load('RotationModule.pth'))
model1.eval()
model2.eval()
model3.eval()
model4.eval()
loss_list = [0, 0, 0]
st_time = time.time()
for batch_idx, input_dict in enumerate(test_loader):
    data = input_dict['rgb'].to(device)
    data = data.float()
    f1, f2 = model1(data)
    output1, label = model2(f1, f2)
    # _, label = torch.max(input_dict['label'], dim=1)
    # label = label.to(device)
    output2 = model3(f1, f2)
    bbx = get_pred_bbx(4, label)
    # t_bbx = bbx_change(input_dict['bbx']).to(device)
    # for i in range(3):
    #     print(compute_overlap_ratio(bbx[i][1], bbx[i][2], bbx[i][3], bbx[i][4],
    #                                 t_bbx[i][1], t_bbx[i][2], t_bbx[i][3], t_bbx[i][4]))
    output3 = model4(f1, f2, bbx[:, :5])
    pred_rotation = get_rotation_from_quaternions(output3, bbx, bs=batch_size)
    pred_tran = get_tran_tz(output2, bbx, label)
    center1 = get_center_from_label(3, label)
    center3 = HoughVoting(label, output2).to(device)
    pred_translation = compute_tran(pred_tran=pred_tran, pred_centers=center1, bboxes=bbx,
                                    cam_intrinsic=test_dataset.cam_intrinsic, bs=batch_size)
    RT = merge_rt(pred_rotation, pred_translation)
    a, b = extract_columns(RT.detach().numpy())
    a1, b1 = extract_columns(input_dict['RTs'].detach().numpy())
    # print((compute_loss(a1[0], a[0])+compute_loss(a1[1], a[1])+compute_loss(a1[2], a[2]))/3)
    print(compute_loss(b,b1))
    output = {'rgb': data.to('cpu').numpy(), 'RTs': input_dict['RTs'].to('cpu').numpy(), 'RTs2': RT.detach().numpy()}
    grid_vis = visualize_output(test_dataset, alpha=0.25, sample=output)
    plt.axis('off')
    plt.imshow(grid_vis)
    plt.title(f'tBbx-votingCenter-{batch_idx + 1}', y=-0.2)
    # plt.savefig(f'./answer/tBbx-midCenter-{batch_idx+1}.png')
    plt.tight_layout()
    plt.show()


print(time.time()-st_time)

