import numpy.matlib
import numpy as np

# 计算内积
# a : ndarray 数组
# b : ndarray 数组
# out : ndarray, 可选，用来保存dot()的计算结果

# a = np.array([[1, 2], [3, 4]])
# b = np.array([[11, 12], [13, 14]])
# print(np.dot(a, b))

# 两个向量的点积

# a = np.array([[1, 2], [3, 4]])
# b = np.array([[11, 12], [13, 14]])
# print(np.vdot(a, b))

# 计算向量外积
# print (np.inner(np.array([1,2,3]),np.array([0,1,0])))

# 多维度计算外积
# a = np.array([[1, 2], [3, 4]])
# print('数组 a：')
# print(a)
# b = np.array([[11, 12], [13, 14]])
# print('数组 b：')
# print(b)
# print('内积：')
# print(np.inner(a, b))

# 矩阵乘积

# a = [[1, 0], [0, 1]]
# b = [[4, 1], [2, 2]]
# print(np.matmul(a, b))

# a = [[1, 0], [0, 1]]
# b = [1, 2]
# print(np.matmul(a, b))
# print(np.matmul(b, a))

# a = np.arange(8).reshape(2,2,2)
# b = np.arange(4).reshape(2,2)
# print (np.matmul(a,b))


# 计算行列式
# a = np.array([[1, 2], [3, 4]])
# print(np.linalg.det(a))

# 逆矩阵
# print(np.linalg.inv(a))
