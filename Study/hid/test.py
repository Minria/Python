import scipy.io as sio
import numpy as np


img = sio.loadmat('D:\wangfuming\Desktop\毕设\Code\CNND\input\Salinas.mat')['salinas']
gt = sio.loadmat('D:\wangfuming\Desktop\毕设\Code\CNND\input\Salinas_gt.mat')['salinas_gt']
d = img.shape[2]
class_num = len(np.unique(gt) ) -1
dis_samples = 1
sim_samples = 1
img = img.astype(np.float32)
gt = gt.astype(np.float32)

img = img.reshape(-1, d)
gt = gt.flatten()
class_num = len(np.unique(gt) ) -1 # except anomaly pixel

training_data = np.zeros((1, d))
training_labels = np.zeros(1)

# dissimilar pixel pair
for i in range(class_num -1):
    for j in range(i+1, class_num):

        index_i = np.where(gt == i + 1)
        index_j = np.where(gt == j + 1)
        data_i = img[index_i][:1]
        data_j = img[index_j][:1]

# similar pixel pair
for i in range(class_num):
    data_similar = np.zeros((sim_samples, sim_samples, d))
    index = np.where(gt == i + 1)
    data = img[index][:sim_samples]
    for j in range(sim_samples):
        data_similar[j] = np.abs(data - data[j, None])
    data_similar = data_similar[np.triu_indices(sim_samples, k=1)]
    training_data = np.concatenate((training_data, data_similar), axis=0)

dis_a = img[48911]
dis_b = img[39633]
sim_c = img[49126]

y1 = dis_a - dis_b
y2 = dis_a - sim_c
x = []
for i in range(224):
    x.append(i)

print(x)

import matplotlib.pyplot as plt


# 使用matplotlib库绘制曲线
plt.plot(x, y2)

# 添加x和y轴标签
plt.xlabel('x')
plt.ylabel('y')

# 显示图形
plt.savefig("./sim_s.png")
plt.show()

