import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import chi2
import spectral as spy


import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import math
import time


def RXD(mat):
    XX, YY, band = mat.shape  # XX为图像的长度 YY为图像的高度 band为图像的波段数
    r = np.reshape(mat, (XX * YY, band))
    pxl_no = XX * YY
    K = np.cov(r.T)  # 得到矩阵r的协方差矩阵K，K为band乘band的矩阵
    IK = np.linalg.inv(K)  # IK为K矩阵的逆矩阵
    u = np.expand_dims(np.mean(r.T, axis=1), 0).T  # 求r矩阵即每一个波段所有像素的均值，u为均值列向量
    if XX >= 80:
        st = 200
        k = math.ceil(pxl_no / st)
        A = []
        for i in range(1, k + 1):
            replng = len(range(st * (i - 1) + 1, min(st * i, pxl_no) + 1))  # 实现RX算子所用的的列数
            A += [np.diag(np.dot(np.dot((r[st * (i - 1): min(st * i, pxl_no), :].T - np.repeat(u, replng, axis=1)).T, IK), (r[st * (i - 1): min(st * i, pxl_no), :].T - np.ones((1, replng)) * u)).T)]
    else:
        A = np.diag((r - np.ones((1, pxl_no)) * u).T * IK * (r - np.ones((1, pxl_no)) * u))
    result = np.reshape(A, (XX, YY))
    result = abs(result)
    return result


if __name__ == '__main__':
    t1 = time.time()
    dataFile = 'D:\wangfuming\Desktop\毕设\data and answer\data3\data.mat'
    data = scio.loadmat(dataFile)
    result = RXD(data['data'])
    t2 = time.time()
    print('time:', round(t2 - t1, 3), 's')
    np.savetxt('./result.csv', result, delimiter=',')
    plt.imshow(result)
    plt.show()

