import numpy as np

# 三行两列 随机的整数
# x = np.empty([3, 2], dtype=int)
# print(x)

# 默认为浮点数 一行五列
# x = np.zeros(5)
# print(x)

# 修改为整数
# y = np.zeros(5, dtype=int)
# print(y)

# 自定义类型
# z = np.zeros((2, 2), dtype=[('x', 'i4'), ('y', 'i4')])
# print(z)
# 初始化为1
# x = np.ones(5)
# print(x)
#
# x = np.ones([2,2], dtype = int)
# print(x)
# 将已有的数组进行初始化
# x = [1, 2, 3]
# a = np.asarray(x)
# print(a)

# x = [(1, 2, 3), (4, 5)]
# a = np.asarray(x)
# print(x)

# s = b'Hello World'
# a = np.frombuffer(s,dtype='S1')
# print (a)

# list = range(5)
# it = iter(list)
# # 使用迭代器创建 ndarray
# x = np.fromiter(it, dtype=float)
# print(x)