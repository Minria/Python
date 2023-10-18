import torch.nn.functional as F
import torch.nn as nn
import torch
from torchvision.ops import RoIPool

def up_sample(input_tensor, size=(480, 640)):
    return F.interpolate(input_tensor, size=size, mode='bilinear')


def conv(in_planes=512, out_planes=512, kernel_size=3, stride=1, relu=True, bias=False):
    if relu:
        conv_layer = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=(kernel_size - 1) // 2, bias=bias),
            nn.ReLU(inplace=True))
    else:
        conv_layer = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                               padding=(kernel_size - 1) // 2, bias=bias)
    if relu:
        nn.init.kaiming_normal_(conv_layer[0].weight, mode='fan_in', nonlinearity='relu')
        if bias:
            conv_layer[0].bias.data = torch.zeros(out_planes)
    else:
        nn.init.kaiming_normal_(conv_layer.weight, mode='fan_in', nonlinearity='relu')
        if bias:
            conv_layer.bias.data = torch.zeros(out_planes)

    return conv_layer

def fc(in_planes=25088, out_planes=4096, relu=True, bias=True):
    if relu:
        fc_layer = nn.Sequential(
            nn.Linear(in_planes, out_planes, bias=bias),
            nn.ReLU())
    else:
        fc_layer = nn.Linear(in_planes, out_planes, bias=bias)

    if relu:
        nn.init.kaiming_normal_(fc_layer[0].weight, mode='fan_in', nonlinearity='relu')
        if bias:
            fc_layer[0].bias.data = torch.zeros(out_planes)
    else:
        nn.init.kaiming_normal_(fc_layer.weight, mode='fan_in', nonlinearity='relu')
        if bias:
            fc_layer.bias.data = torch.zeros(out_planes)
    return fc_layer


class BaseModule(nn.Module):
    def __init__(self, num_classes=4, pre_model=None, is_init_pre=False):
        super(BaseModule, self).__init__()
        self.num_classes = num_classes

        self.conv1 = conv(in_planes=3, out_planes=64, bias=True)
        self.conv2 = conv(in_planes=64, out_planes=64, bias=True)

        self.conv3 = conv(in_planes=64, out_planes=128, bias=True)
        self.conv4 = conv(in_planes=128, out_planes=128, bias=True)

        self.conv5 = conv(in_planes=128, out_planes=256, bias=True)
        self.conv6 = conv(in_planes=256, out_planes=256, bias=True)
        self.conv7 = conv(in_planes=256, out_planes=256, bias=True)

        self.conv8 = conv(in_planes=256, out_planes=512, bias=True)
        self.conv9 = conv(in_planes=512, out_planes=512, bias=True)
        self.conv10 = conv(in_planes=512, out_planes=512, bias=True)

        self.conv11 = conv(in_planes=512, out_planes=512, bias=True)
        self.conv12 = conv(in_planes=512, out_planes=512, bias=True)
        self.conv13 = conv(in_planes=512, out_planes=512, bias=True)
        self.down_sample = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.pre_model = pre_model
        if is_init_pre:
            self.init_model()


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.down_sample(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.down_sample(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.down_sample(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        f1 = x
        x = self.down_sample(x)
        x = self.conv11(x)
        x = self.conv12(x)
        f2 = self.conv13(x)
        return f1, f2

    def init_model(self):
        embedding_layers = list(self.pre_model.features)[:30]
        self.conv1[0].weight.data = embedding_layers[0].weight.data
        self.conv1[0].bias.data = embedding_layers[0].bias.data

        self.conv2[0].weight.data = embedding_layers[2].weight.data
        self.conv2[0].bias.data = embedding_layers[2].bias.data

        self.conv3[0].weight.data = embedding_layers[5].weight.data
        self.conv3[0].bias.data = embedding_layers[5].bias.data

        self.conv4[0].weight.data = embedding_layers[7].weight.data
        self.conv4[0].bias.data = embedding_layers[7].bias.data

        self.conv5[0].weight.data = embedding_layers[10].weight.data
        self.conv5[0].bias.data = embedding_layers[10].bias.data

        self.conv6[0].weight.data = embedding_layers[12].weight.data
        self.conv6[0].bias.data = embedding_layers[12].bias.data

        self.conv7[0].weight.data = embedding_layers[14].weight.data
        self.conv7[0].bias.data = embedding_layers[14].bias.data

        self.conv8[0].weight.data = embedding_layers[17].weight.data
        self.conv8[0].bias.data = embedding_layers[17].bias.data

        self.conv9[0].weight.data = embedding_layers[19].weight.data
        self.conv9[0].bias.data = embedding_layers[19].bias.data

        self.conv10[0].weight.data = embedding_layers[21].weight.data
        self.conv10[0].bias.data = embedding_layers[21].bias.data

        self.conv11[0].weight.data = embedding_layers[24].weight.data
        self.conv11[0].bias.data = embedding_layers[24].bias.data

        self.conv12[0].weight.data = embedding_layers[26].weight.data
        self.conv12[0].bias.data = embedding_layers[26].bias.data

        self.conv13[0].weight.data = embedding_layers[28].weight.data
        self.conv13[0].bias.data = embedding_layers[28].bias.data

class BaseModulePlus(nn.Module):
    def __init__(self,  pre_model=None, is_init_pre=False, req_grad=False):
        super(BaseModulePlus, self).__init__()
        self.req_grad = req_grad
        # self.conv_list = []
        # self.conv_list.append(conv(in_planes=3, out_planes=64, bias=True))
        # self.conv_list.append(conv(in_planes=64, out_planes=64, bias=True))
        #
        # self.conv_list.append(conv(in_planes=64, out_planes=128, bias=True))
        # self.conv_list.append(conv(in_planes=128, out_planes=128, bias=True))
        #
        # self.conv_list.append(conv(in_planes=128, out_planes=256, bias=True))
        # self.conv_list.append(conv(in_planes=256, out_planes=256, bias=True))
        # self.conv_list.append(conv(in_planes=256, out_planes=256, bias=True))
        #
        # self.conv_list.append(conv(in_planes=256, out_planes=512, bias=True))
        # self.conv_list.append(conv(in_planes=512, out_planes=512, bias=True))
        # self.conv_list.append(conv(in_planes=512, out_planes=512, bias=True))
        #
        # self.conv_list.append(conv(in_planes=512, out_planes=512, bias=True))
        # self.conv_list.append(conv(in_planes=512, out_planes=512, bias=True))
        # self.conv_list.append(conv(in_planes=512, out_planes=512, bias=True))

        self.conv_list = nn.ModuleList([
            conv(in_planes=3, out_planes=64, bias=True),
            conv(in_planes=64, out_planes=64, bias=True),
            conv(in_planes=64, out_planes=128, bias=True),
            conv(in_planes=128, out_planes=128, bias=True),
            conv(in_planes=128, out_planes=256, bias=True),
            conv(in_planes=256, out_planes=256, bias=True),
            conv(in_planes=256, out_planes=256, bias=True),
            conv(in_planes=256, out_planes=512, bias=True),
            conv(in_planes=512, out_planes=512, bias=True),
            conv(in_planes=512, out_planes=512, bias=True),
            conv(in_planes=512, out_planes=512, bias=True),
            conv(in_planes=512, out_planes=512, bias=True),
            conv(in_planes=512, out_planes=512, bias=True)
        ])
        self.down_sample = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.pre_model = pre_model
        if is_init_pre:
            self.init_model()

        self.make_grad()


    def forward(self, x):
        x = self.conv_list[0](x)
        x = self.conv_list[1](x)
        x = self.down_sample(x)

        x = self.conv_list[2](x)
        x = self.conv_list[3](x)
        x = self.down_sample(x)

        x = self.conv_list[4](x)
        residual1 = x
        x = self.conv_list[5](x)
        x = self.conv_list[6](x)
        x = x + residual1
        x = self.down_sample(x)

        x = self.conv_list[7](x)
        residual2 = x
        x = self.conv_list[8](x)
        x = x + residual2
        x = self.conv_list[9](x)
        f1 = x
        x = self.down_sample(x)
        x = self.conv_list[10](x)
        x = self.conv_list[11](x)
        f2 = self.conv_list[12](x)
        return f1, f2

    def init_model(self):
        embedding_layers = list(self.pre_model.features)[:30]
        pre_model_idx = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
        for i in range(len(pre_model_idx)):
            self.conv_list[i][0].weight.data = embedding_layers[pre_model_idx[i]].weight.data
            self.conv_list[i][0].bias.data = embedding_layers[pre_model_idx[i]].bias.data

    def make_grad(self):
        for i in range(len(self.conv_list)):
            self.conv_list[i][0].weight.requires_grad = self.req_grad
            self.conv_list[i][0].bias.requires_grad = self.req_grad


class ClassModule(nn.Module):
    def __init__(self, num_classes=3):
        super(ClassModule, self).__init__()
        self.num_classes = num_classes
        self.conv1 = conv(in_planes=512, out_planes=64, kernel_size=1, bias=True)
        self.conv2 = conv(in_planes=512, out_planes=64, kernel_size=1, bias=True)
        self.conv3 = conv(in_planes=64, out_planes=self.num_classes+1, kernel_size=1, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, f1, f2):
        f1 = self.conv1(f1)
        f2 = self.conv2(f2)
        f2 = up_sample(f2, size=(60, 80))
        x = f1 + f2
        x = up_sample(x, size=(480, 640))
        x = self.conv3(x)
        x = self.softmax(x)
        _, label = torch.max(x, dim=1)
        return x, label


class TranslationModule(nn.Module):
    def __init__(self, num_classes=3):
        super(TranslationModule, self).__init__()
        self.num_classes = num_classes
        self.conv3_1 = conv(in_planes=512, out_planes=128, kernel_size=1, bias=True)
        self.conv3_2 = conv(in_planes=512, out_planes=128, kernel_size=1, bias=True)
        self.conv3_3 = conv(in_planes=128, out_planes=3 * self.num_classes, kernel_size=1, relu=False)


    def forward(self, f1, f2):
        f1 = self.conv3_1(f1)
        f2 = self.conv3_2(f2)
        f2 = up_sample(f2, size=(60, 80))
        x = f1 + f2
        x = up_sample(x, size=(480, 640))
        x = self.conv3_3(x)
        return x


class RotationModule(nn.Module):
    def __init__(self, num_classes=3):
        super(RotationModule, self).__init__()

        self.num_classes = num_classes
        self.roi_pool1 = RoIPool(output_size=7, spatial_scale=1 / 8)
        self.roi_pool2 = RoIPool(output_size=7, spatial_scale=1 / 16)
        self.fc1 = fc(25088, 4096)
        self.fc2 = fc(4096, 4096)
        self.fc3 = fc(4096, 4 * self.num_classes, relu=False)

    def forward(self, f1, f2, bbx):
        bbx_copy = bbx.to(dtype=torch.float32)
        f1 = self.roi_pool1(f1, bbx_copy)
        f2 = self.roi_pool2(f2, bbx_copy)

        x = f1 + f2
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        # (N,12)
        return x



'''
    --------------------------------------------------------------------------------------------------------------------
    --------------------------------------------------------------------------------------------------------------------
                                           以下版本代码网络收敛效果不好
    --------------------------------------------------------------------------------------------------------------------
    --------------------------------------------------------------------------------------------------------------------
'''
'''
def up_sample(input_tensor, scale_factor=2):
    return F.interpolate(input_tensor, scale_factor=scale_factor, mode='bilinear', align_corners=False)

def down_sample(input_tensor, scale_factor=2):
    return F.max_pool2d(input_tensor, kernel_size=scale_factor, stride=scale_factor)

# def conv(in_planes=512, out_planes=512, kernel_size=3, stride=1, relu=True,bias=False):
#     if relu:
#         return nn.Sequential(
#             nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
#                       padding=(kernel_size - 1) // 2, bias=bias),
#             nn.ReLU(inplace=True))
#     else:
#         return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
#                          padding=(kernel_size - 1) // 2, bias=bias)

def conv(in_planes=512, out_planes=512, kernel_size=3, stride=1, relu=True, bias=False):
    if relu:
        conv_layer = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=(kernel_size - 1) // 2, bias=bias),
            nn.ReLU(inplace=True))
    else:
        conv_layer = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                               padding=(kernel_size - 1) // 2, bias=bias)
    # 相较于原代码新增kaiming初始化
    nn.init.kaiming_uniform_(conv_layer[0].weight, mode='fan_in', nonlinearity='relu')
    return conv_layer


def deconv(in_planes=512, out_planes=512, kernel_size=2, stride=2, relu=True, bias=False):
    if relu:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                               padding=(kernel_size - 1) // 2, bias=bias),
            nn.ReLU(inplace=True))
    else:
        return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                  padding=(kernel_size - 1) // 2, bias=bias)




class BaseModule(nn.Module):
    def __init__(self, num_classes=4):
        super(BaseModule, self).__init__()
        self.num_classes = num_classes
        self.conv1_1 = conv(in_planes=3, out_planes=64)
        self.conv1_2 = conv(in_planes=64, out_planes=64)
        self.conv1_3 = conv(in_planes=64, out_planes=128)


        self.conv1_4 = conv(in_planes=128, out_planes=128)
        self.conv1_5 = conv(in_planes=128, out_planes=256)
        self.conv1_6 = conv(in_planes=256, out_planes=256)
        self.conv1_7 = conv(in_planes=256, out_planes=256)

        self.conv1_8 = conv(in_planes=256, out_planes=512)
        self.conv1_9 = conv(in_planes=512, out_planes=512)
        self.conv1_10 = conv(in_planes=512, out_planes=512)

        self.conv1_11 = conv(in_planes=512, out_planes=512)
        self.conv1_12 = conv(in_planes=512, out_planes=512)
        self.conv1_13 = conv(in_planes=512, out_planes=512)



    def forward(self, x):

        x = self.conv1_1(x)
        # x = self.conv1_2(x)
        x = down_sample(x)
        x = self.conv1_3(x)
        x = self.conv1_4(x)
        x = down_sample(x)
        x = self.conv1_5(x)
        # x = self.conv1_6(x)
        x = self.conv1_7(x)
        x = down_sample(x)
        x = self.conv1_8(x)
        # x = self.conv1_9(x)
        x = self.conv1_10(x)
        f1 = x
        x = down_sample(x)
        x = self.conv1_11(x)
        # x = self.conv1_12(x)
        f2 = self.conv1_13(x)
        return f1, f2




class ClassModule(nn.Module):
    def __init__(self, num_classes=4):
        super(ClassModule, self).__init__()
        self.num_classes = num_classes
        self.conv2_1 = conv(in_planes=512, out_planes=64)
        self.conv2_2 = conv(in_planes=512, out_planes=64)
        self.conv2_3 = conv(in_planes=64, out_planes=self.num_classes)

        self.deconv2_1 = deconv(in_planes=64, out_planes=64)
        self.deconv2_2 = deconv(in_planes=64, out_planes=64)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, f1, f2):
        f1 = self.conv2_1(f1)
        f2 = self.conv2_2(f2)
        f2 = up_sample(f2)
        f2 = self.deconv2_1(f2)
        x = f1 + f2
        x = up_sample(x, 8)
        x = self.deconv2_2(x)
        x = self.conv2_3(x)
        x = self.softmax(x)
        return x


class TranslationModule(nn.Module):
    def __init__(self, num_classes=4):
        super(TranslationModule, self).__init__()
        self.num_classes = num_classes
        self.conv3_1 = conv(in_planes=512, out_planes=128)
        self.conv3_2 = conv(in_planes=512, out_planes=128)
        self.conv3_3 = conv(in_planes=128, out_planes=3 * self.num_classes)
        self.deconv3_1 = deconv(in_planes=128, out_planes=128)
        self.deconv3_2 = deconv(in_planes=128, out_planes=128)
    def forward(self, f1, f2):
        f1 = self.conv3_1(f1)
        f2 = self.conv3_2(f2)
        f2 = up_sample(f2)
        f2 = self.deconv3_1(f2)
        x = f1 + f2
        x = up_sample(x, 8)
        # x = self.deconv3_2(x)
        x = self.conv3_3(x)
        return x


class RotationModule(nn.Module):
    def __init__(self, num_classes=4):
        super(RotationModule, self).__init__()
        self.num_classes = num_classes
        self.roi_pool1 = RoIPool(output_size=7, spatial_scale=1 / 8)
        self.roi_pool2 = RoIPool(output_size=7, spatial_scale=1 / 16)

        self.fc1 = nn.Linear(25088, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4 * self.num_classes)
    def forward(self, f1, f2, bbx):
        bbx_copy = bbx.to(dtype=torch.float32)
        f1 = self.roi_pool1(f1, bbx_copy)
        f2 = self.roi_pool2(f2, bbx_copy)
        x = f1 + f2
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


import torch
import torch.nn as nn


if __name__ == '__main__':
    # kaiming初始化
    # conv1 = nn.Conv2d(3, 1, kernel_size=1)
    # print(conv1.weight)
    # nn.init.kaiming_uniform_(conv1.weight, mode='fan_in', nonlinearity='relu')
    # print(conv1.weight)
    data = torch.randn(1, 3, 30, 60)
    upsample_layer = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=2, stride=2)
    data2 = upsample_layer(data)
    print(1)

'''



'''
    --------------------------------------------------------------------------------------------------------------------
    --------------------------------------------------------------------------------------------------------------------
                                              以下版本的代码不能正常进行反向传播
    --------------------------------------------------------------------------------------------------------------------
    --------------------------------------------------------------------------------------------------------------------
'''

'''

def up_sample(input_tensor, scale_factor=2):
    return F.interpolate(input_tensor, scale_factor=scale_factor, mode='bilinear', align_corners=False)

def down_sample(input_tensor, scale_factor=2):
    return F.max_pool2d(input_tensor, kernel_size=scale_factor, stride=scale_factor)

def conv(in_planes=512, out_planes=512, kernel_size=3, stride=1, relu=True, bias=False):
    if relu:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=(kernel_size - 1) // 2, bias=bias),
            nn.ReLU(inplace=True))
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                         padding=(kernel_size - 1) // 2, bias=bias)

def deconv(in_planes=512, out_planes=512, kernel_size=3, stride=1, relu=True, bias=False):
    if relu:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                               padding=(kernel_size - 1) // 2, bias=bias),
            nn.ReLU(inplace=True))
    else:
        return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                  padding=(kernel_size - 1) // 2, bias=bias)


class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.num_classes = 3
        self.conv1_1 = conv(in_planes=3, out_planes=64)
        self.conv1_2 = conv(in_planes=64, out_planes=64)

        self.conv1_3 = conv(in_planes=64, out_planes=128)
        self.conv1_4 = conv(in_planes=128, out_planes=128)

        self.conv1_5 = conv(in_planes=128, out_planes=256)
        self.conv1_6 = conv(in_planes=256, out_planes=256)
        self.conv1_7 = conv(in_planes=256, out_planes=256)

        self.conv1_8 = conv(in_planes=256, out_planes=512)
        self.conv1_9 = conv(in_planes=512, out_planes=512)
        self.conv1_10 = conv(in_planes=512, out_planes=512)

        self.conv1_11 = conv(in_planes=512, out_planes=512)
        self.conv1_12 = conv(in_planes=512, out_planes=512)
        self.conv1_13 = conv(in_planes=512, out_planes=512)

        self.conv2_1 = conv(in_planes=512, out_planes=64)
        self.conv2_2 = conv(in_planes=512, out_planes=64)
        self.conv2_3 = conv(in_planes=64, out_planes=self.num_classes+1)
        self.deconv2_1 = deconv(in_planes=64, out_planes=64)
        self.deconv2_2 = deconv(in_planes=64, out_planes=64)

        self.conv3_1 = conv(in_planes=512, out_planes=128)
        self.conv3_2 = conv(in_planes=512, out_planes=128)
        self.conv3_3 = conv(in_planes=128, out_planes=3*self.num_classes)
        self.deconv3_1 = deconv(in_planes=128, out_planes=128)
        self.deconv3_2 = deconv(in_planes=128, out_planes=128)

        self.roi_pool1 = RoIPool(output_size=7, spatial_scale=1/8)
        self.roi_pool2 = RoIPool(output_size=7, spatial_scale=1/16)

        self.fc1 = nn.Linear(25088, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4*self.num_classes)


    def feature_extraction(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = down_sample(x)
        x = self.conv1_3(x)
        x = self.conv1_4(x)
        x = down_sample(x)
        x = self.conv1_5(x)
        x = self.conv1_6(x)
        x = self.conv1_7(x)
        x = down_sample(x)
        x = self.conv1_8(x)
        x = self.conv1_9(x)
        x = self.conv1_10(x)
        f1 = x
        x = down_sample(x)
        x = self.conv1_11(x)
        x = self.conv1_12(x)
        f2 = self.conv1_13(x)
        return f1, f2

    def segmentation(self, f1, f2):
        f1 = self.conv2_1(f1)
        f2 = self.conv2_2(f2)
        f2 = up_sample(f2)
        f2 = self.deconv2_1(f2)
        x = f1 + f2
        x = up_sample(x, 8)
        x = self.deconv2_2(x)
        x = self.conv2_3(x)
        x = nn.Softmax(dim=1)(x)
        _, label = torch.max(x, dim=1)
        return x, label

    def translation(self, f1, f2):
        f1 = self.conv3_1(f1)
        f2 = self.conv3_2(f2)
        f2 = up_sample(f2)
        f2 = self.deconv3_1(f2)
        x = f1 + f2
        x = up_sample(x, 8)
        x = self.deconv3_2(x)
        x = self.conv3_3(x)
        return x


    def rotation(self, f1, f2, bbx):
        bbx_copy = bbx.to(dtype=torch.float32)
        f1 = self.roi_pool1(f1, bbx_copy)
        f2 = self.roi_pool2(f2, bbx_copy)
        x = f1 + f2
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def forward(self, x):
        f1, f2 = self.feature_extraction(x)
        labels = self.segmentation(f1, f2)
        tran = self.translation(f1, f2)
        bbx = self.label2bbx(labels)
        bbx = bbx.narrow(1, 0, 5)
        r = self.rotation(f1, f2, bbx)
        return labels, tran, r

'''

# if __name__ == '__main__':
#     c = MyCNN()
#     data = torch.rand(16, 3, 480, 640)
#     f1, f2 = c.feature_extraction(data)
#     t = c.translation(f1, f2)
#     labels = c.segmentation(f1, f2)
#     bbx = c.label2bbx(labels)
#     bbx = bbx.narrow(1, 0, 5)
#     r = c.rotation(f1, f2, bbx)
#     print(1)


# def fun(pre, tar):
#     return ((pre-tar)**2).mean()

'''
if __name__ == '__main__':
    data = torch.randn(3, 3, 480, 640)
    target1 = torch.randn(3, 480, 640)
    target2 = torch.randn(3, 9, 480, 640)
    target3 = torch.randn(3, 12)

    model = MyCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for i in range(10):
        ans1, ans2, ans3 = model(data)
        loss1 = fun(ans1, target1)
        loss2 = fun(ans2, target2)
        loss3 = fun(ans3, 0)
        loss = loss3 + loss2 +loss1
        print(loss1, loss2, loss3, loss)
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()

'''