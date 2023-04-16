

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

san_diego = sio.loadmat('D:\wangfuming\Desktop\CNND-tf/train_data/San_Diego.mat')['map']
ad = sio.loadmat('D:\wangfuming\Desktop\CNND-tf/train_data/tests/cnndRES.mat')['data']
# TP = 0
# FP = 0
# TN = 0
# FN = 0
m,n = ad.shape
# rate = 1
# for i in range(m):
#     for j in range(n):
#         if ad[i][j] < rate and san_diego[i][j] == 1:
#             TP += 1
#         if ad[i][j] >= rate and san_diego[i][j] == 1:
#             FN += 1
#         if ad[i][j] < rate and san_diego[i][j] == 0:
#             FP += 1
#         if ad[i][j] >= rate and san_diego[i][j] == 0:
#             TN += 1

# print(TP/(TP+FN))
plt.imshow(ad, cmap='gray')
# plt.savefig('E:/1.jpg')
plt.show()
# plt.imshow(san_diego,cmap='gray')
# plt.savefig('E:/2.jpg')
# plt.show()