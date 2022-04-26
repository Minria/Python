import numpy as np

# a = np.arange(10)
# s = slice(2, 7, 2)  # 从索引 2 开始到索引 7 停止，间隔为2
# print(a[s])

# a = np.arange(10)
# b = a[2:7:2]   # 从索引 2 开始到索引 7 停止，间隔为 2
# print(b)

# a = np.arange(10)  # [0 1 2 3 4 5 6 7 8 9]
# print(a[5])  # 索引为5
# print(a[2:]) # 索引2以后
# print(a[2:4]) # 索引[2,4)
# print(a[2:5:2]) #索引[2,5) 且间隔未2

# 以下实例获取数组中(0,0)，(1,1)和(2,0)位置处的元素。
# x = np.array([[1,  2],  [3,  4],  [5,  6]])
# y = x[[0,1,2],  [0,1,0]]
# print (y)


# x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
# print('我们的数组是：')
# print(x)
# print('\n')
# rows = np.array([[0, 0], [3, 3]])
# cols = np.array([[0, 2], [0, 2]])
# y = x[rows, cols]
# print('这个数组的四个角元素是：')
# print(y)

# a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# b = a[1:3, 1:3]
# c = a[1:3, [1, 2]]
# d = a[..., 1:]
# print(b)
# print(c)
# print(d)


# 布尔索引
# x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
# print('我们的数组是：')
# print(x)
# print('\n')
# # 现在我们会打印出大于 5 的元素
# print('大于 5 的元素是：')
# print(x[x > 5])

# a = np.array([np.nan, 1, 2, np.nan, 3, 4, 5])
# print(a)
# print(a[~np.isnan(a)])

# a = np.array([1, 2 + 6j, 5, 3.5 + 5j])
# print(a[np.iscomplex(a)])

# 花式索引
# x = np.arange(32).reshape((8, 4))
# print("x:\n")
# print(x)
# print("\n")
# print(x[[4, 2, 1, 7]])  # x[4],x[2],x[1],x[7]

# x = np.arange(32).reshape((8, 4))
# print(x[[-4, -2, -1, -7]]) # 倒数第几个

# x = np.arange(32).reshape((8, 4))
# y = np.array([[x[1, 0], x[1, 3], x[1, 1], x[1, 2]],
#               [x[5, 0], x[5, 3], x[5, 1], x[5, 2]],
#               [x[7, 0], x[7, 3], x[7, 1], x[7, 2]],
#               [x[2, 0], x[2, 3], x[2, 1], x[2, 2]]])
# print(x[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])])
