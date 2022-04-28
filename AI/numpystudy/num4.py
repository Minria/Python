import numpy as np

# a = np.arange(8)
# print('原始数组：')
# print(a)
# print('\n')
# 修改数组的类型
# b = a.reshape(4, 2)
# print('修改后的数组：')
# print(b)

# a = np.arange(9).reshape(3, 3)
# print('原始数组：')
# for row in a:
#     print(row)
# 迭代打印数组
# # 对数组中每个元素都进行处理，可以使用flat属性，该属性是一个数组元素迭代器：
# print('迭代后的数组：')
# for element in a.flat:
#     print(element)

# 数组的深拷贝
# a = np.arange(8).reshape(2, 4)
#
# print('原数组：')
# print(a)
# print('\n')
# # 默认按行
#
# print('展开的数组：')
# print(a.flatten())
# print('\n')
#
# print('以 F 风格顺序展开的数组：')
# print(a.flatten(order='F'))

# a = np.arange(8).reshape(2, 4)

# print('原数组：')
# print(a)
# print('\n')
#
# print('调用 ravel 函数之后：')
# print(a.ravel())
# print('\n')
#
# print('以 F 风格顺序调用 ravel 函数之后：')
# print(a.ravel(order='F'))


# a = np.arange(12).reshape(3, 4)
#
# print('原数组：')
# print(a)
# print('\n')
#
# print('对换数组：')
# print(np.transpose(a))
# # 等效于矩阵的转置
# print(a.T)

# 创建了三维的 ndarray
a = np.arange(8).reshape(2, 2, 2)

print('原数组：')
print(a)
print('获取数组中一个值：')
print(np.where(a == 6))
print(a[1, 1, 0])  # 为 6
print('\n')

# 将轴 2 滚动到轴 0（宽度到深度）

print('调用 rollaxis 函数：')
b = np.rollaxis(a, 2, 0)
print(b)
# 查看元素 a[1,1,0]，即 6 的坐标，变成 [0, 1, 1]
# 最后一个 0 移动到最前面
print(np.where(b == 6))
print('\n')

# 将轴 2 滚动到轴 1：（宽度到高度）

print('调用 rollaxis 函数：')
c = np.rollaxis(a, 2, 1)
print(c)
# 查看元素 a[1,1,0]，即 6 的坐标，变成 [1, 0, 1]
# 最后的 0 和 它前面的 1 对换位置
print(np.where(c == 6))
print('\n')