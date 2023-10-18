from PIL import Image
import numpy as np
import torch
from PIL import ImageDraw
from torchvision.ops import box_iou
import torch.nn as nn

# 使用部分数据，将标签的值进行重新映射
def label_change(label):
    data = np.zeros((480, 640))
    data[label == 8] = 1
    data[label == 14] = 2
    data[label == 21] = 3
    return data

# 将label数值进行重新编码
# label_encoder 用于最初的dataset
# new_label_encoder 用于追钟的dataset
# 两者的区别就是编码时机不同而区分torch和numpy格式
def label_encoder(num_classes, label):
    new_tensor = torch.zeros(num_classes, 480, 640)
    for i in range(num_classes):
        # 将原始标签张量中值为0的位置置为1
        new_tensor[i] = (label == i).float()
    return new_tensor

def new_label_encoder(num_classes, label):
    new_array = np.zeros((num_classes, 480, 640), dtype=int)
    for i in range(num_classes):
        channel_mask = label == i
        new_array[i, ...] = channel_mask.astype(int)

    return new_array

# 从xxxxx-box.txt中读取信息
def process_box(file_path=None, mapping_dict=None, is_mapping=False):
    file = open(file_path, 'r')
    lines = file.readlines()
    data = []
    for line in lines:
        # 去除行末的换行符，并使用空格进行分割
        row = line.rstrip().split(' ')
        # 将每行的数据转换为整数，并添加到二维列表中
        row_data = [x for x in row]
        if is_mapping:
            row_data.append(mapping_dict[row_data[0]])
        data.append(row_data)
    file.close()
    delete_first = np.delete(data, 0, axis=1)
    last_data = delete_first.astype(float)
    return last_data



'''

# 旋转矩阵转四元数
def rot_to_quaternion(R):
    r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
    r21, r22, r23 = R[1, 0], R[1, 1], R[1, 2]
    r31, r32, r33 = R[2, 0], R[2, 1], R[2, 2]
    # 计算四元数的实部
    w = np.sqrt(1 + r11 + r22 + r33) / 2
    # 计算四元数的虚部分量
    x = (r32 - r23) / (4 * w)
    y = (r13 - r31) / (4 * w)
    z = (r21 - r12) / (4 * w)
    return w, x, y, z

# 四元数转旋转矩阵
def quaternion_to_rot(quaternion):
    w, x, y, z = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    # 计算旋转矩阵的元素
    r11 = 1 - 2 * y ** 2 - 2 * z ** 2
    r12 = 2 * x * y - 2 * w * z
    r13 = 2 * x * z + 2 * w * y
    r21 = 2 * x * y + 2 * w * z
    r22 = 1 - 2 * x ** 2 - 2 * z ** 2
    r23 = 2 * y * z - 2 * w * x
    r31 = 2 * x * z - 2 * w * y
    r32 = 2 * y * z + 2 * w * x
    r33 = 1 - 2 * x ** 2 - 2 * y ** 2
    # 构建旋转矩阵
    rotation_matrix = np.array([[r11, r12, r13],
                                [r21, r22, r23],
                                [r31, r32, r33]])
    return rotation_matrix
'''