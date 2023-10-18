import numpy as np
import torch

# # 创建一个tensor
# tensor = torch.tensor([1, 2, 3])
#
# # 将tensor保存到本地文件
# torch.save(tensor, './tensor.pt')

# # 从本地文件加载tensor
# loaded_tensor = torch.load('./tensor.pt')
#
# # 打印加载的tensor
# print(loaded_tensor)
#
#
# import pickle

# # 创建一个list
# my_list = [1, 2, 3, 4, 5]
#
# # 将list保存到本地文件
# with open('my_list.pkl', 'wb') as f:
#     pickle.dump(my_list, f)
#
# # 从本地文件加载list
# with open('my_list.pkl', 'rb') as f:
#     loaded_list = pickle.load(f)
#
# # 打印加载的list
# print(loaded_list)
#
# import os
#
# # 指定文件路径
# file_path = '/path/to/my/file.txt'
#
# # 判断文件是否存在
# if os.path.exists(file_path):
#     print('File exists!')
# else:
#     print('File does not exist.')

# import torch
# list = torch.rand(2, 1, 2, 3)
# print(list)
# list = list.squeeze()
# print(list)

x = np.array([3, 2, 3, 4, 5])
y = np.array([1, 2, 3, 4, 5])
ans = ((x - y) ** 2).mean()
print(ans)