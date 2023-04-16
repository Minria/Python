

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy.io as sio

v = 150
# 1.读取.mat文件中的数据并reshape
data = sio.loadmat('d:/wangfuming/Desktop/CNND_v1/input/abu-airport-2.mat')['data']
data = np.reshape(data, (10000, 205))
# 2.标准化
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
# 3.PCA降维
pca = PCA(n_components=v)
data_pca = pca.fit_transform(data)
# 4.降维后数据reshape
data_pca = np.reshape(data_pca, (100, 100, v))
data = np.reshape(data, (100, 100, 205))
# 5.可视化特征图
fig, ax = plt.subplots(1, 2)
ax[0].imshow(data[:, :, 0], cmap='gray')
ax[0].set_title('Original Feature Map')
ax[1].imshow(data_pca[:, :, 0], cmap='gray')
ax[1].set_title('PCA Feature Map')
plt.show()
