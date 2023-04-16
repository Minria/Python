import itertools
from datetime import time

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from hid.UFarray import UFarray

# 读取csv文件
data_path = 'D:\wangfuming\Desktop\毕设\data and answer\data2/'
data_name = 'nn'
df = pd.read_csv(data_path+data_name+'.csv', header=None)

data = df.values
a, b = data.shape
# 将数据转换为灰度图
# plt.imshow(data, cmap='gray')
plt.imshow(data)
plt.savefig(data_path+data_name+'.jpg')
plt.show()