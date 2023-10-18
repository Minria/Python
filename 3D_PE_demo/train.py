import time
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.dataset import YCBDataSet
from model.model import *
import torchvision.models as models
from Logger import Logger
from model.loss import *
from utils.train_utils import *
import torch.nn as nn


start = time.time()


print(f'cuda is available: {torch.cuda.is_available()}')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device is {device}')




train_dataset = YCBDataSet(file_name='data/0000', data_path='E:/Archive files/dataset/YCB_Video_Dataset',
                           mode='train', train=1)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

print(f'dataset length: {len(train_loader)}')
print(f'finsh data loading, time is: {time.time() - start}')


# 初始化模型和优化器
vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
model1 = BaseModule(pre_model=vgg16, is_init_pre=True).to(device)

# model1 = BaseModulePlus(pre_model=vgg16, is_init_pre=True, req_grad=False).to(device)
model2 = ClassModule(num_classes=3).to(device)
model3 = TranslationModule(num_classes=3).to(device)
model4 = RotationModule(num_classes=3).to(device)

criterion = nn.L1Loss()

# optimizer1 = optim.Adam(model1.parameters(), lr=0.01)
optimizer2 = optim.Adam(model2.parameters(), lr=0.01)
optimizer3 = optim.Adam(model3.parameters(), lr=0.01)
optimizer4 = optim.Adam(model4.parameters(), lr=0.01)

print(f'load model and optimizer, time is: {time.time() - start}')


# model2.load_state_dict(torch.load('ClassModule.pth'))

# model1.train()
model2.train()
model3.train()
model4.train()

epochs = 10

for epoch in range(epochs):
    sum_loss_lost = [0, 0, 0]
    loss_list = [0, 0, 0]
    for batch_idx, input_dict in enumerate(train_loader):
        data = input_dict['rgb'].to(device)
        data = data.float()
        target = input_dict['label'].to(device)
        centermaps = input_dict['centermaps'].to(device)

        # optimizer1.zero_grad()
        optimizer2.zero_grad()
        # optimizer3.zero_grad()
        # optimizer4.zero_grad()

        f1, f2 = model1(data)
        output1, _ = model2(f1, f2)
        output2 = model3(f1, f2)

        # x1,x2,y1,y2,id
        bbx = bbx_change(input_dict['bbx']).to(device)
        # [bs,.... id]
        quaternion = model4(f1, f2, bbx[:, :5])
        # (N,12)
        # [0,1,2,3] [4,5,6,7]
        gt_rotation = get_rot_from_gt(bbx, input_dict['RTs'])
        pred_rotation, label2 = get_rotation(quaternion, bbx)
        label2 = label2.long()

        loss_list[0] = loss_cross_entropy(output1, target)
        loss_list[1] = criterion(output2, centermaps)
        loss_list[2] = loss_Rotation(pred_rotation, gt_rotation, label2, torch.tensor(train_dataset.model))

        loss_list[0].backward()
        # loss_list[1].backward()
        # loss_list[2].backward()
        # loss = loss_list[1] + loss_list[2] + loss_list[0]
        # loss.backward()

        # optimizer1.step()
        optimizer2.step()
        # optimizer3.step()
        # optimizer4.step()

        for i in range(3):
            sum_loss_lost[i] = sum_loss_lost[i] + loss_list[i]

    print(f'epoch = {epoch + 1}, loss1 = {sum_loss_lost[0]/len(train_loader)},'
          f' loss2 = {sum_loss_lost[1]/len(train_loader)}, loss3 = {sum_loss_lost[2]/len(train_loader)}')


# torch.save(model1.state_dict(),'BaseModule.pth')
# torch.save(model2.state_dict(), 'ClassModule.pth')
# torch.save(model3.state_dict(), 'TranslationModule.pth')
# torch.save(model4.state_dict(), 'RotationModule.pth')


