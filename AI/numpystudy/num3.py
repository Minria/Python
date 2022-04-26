import numpy as np

# a = np.arange(6).reshape(2, 3)
# print('原始数组是：')
# print(a)
# print('\n')
# print('迭代输出元素：')
# for x in np.nditer(a):
#     print(x, end=", ")
# print('\n')

# [[0,1,2],[3,4,5]]
# a = np.arange(6).reshape(2, 3)
# for x in np.nditer(a.T):
#     print(x, end=", ")
# print('\n')
#
# for x in np.nditer(a.T.copy(order='C')):
#     print(x, end=", ")
# print('\n')

# a = np.arange(0, 60, 5)
# a = a.reshape(3, 4)
# print('原始数组是：')
# print(a)
# print('\n')
# print('原始数组的转置是：')
# b = a.T
# print(b)
# print('\n')
# print('以 C 风格顺序排序：')
# c = b.copy(order='C')
# print(c)
# for x in np.nditer(c):
#     print(x, end=", ")
# print('\n')
# print('以 F 风格顺序排序：')
# c = b.copy(order='F')
# print(c)
# for x in np.nditer(c):
#     print(x, end=", ")


# a = np.arange(0, 60, 5)
# a = a.reshape(3, 4)
# print('原始数组是：')
# print(a)
# print('\n')
# print('以 C 风格顺序排序：')
# for x in np.nditer(a, order='C'):
#     print(x, end=", ")
# print('\n')
# print('以 F 风格顺序排序：')
# for x in np.nditer(a, order='F'):
#     print(x, end=", ")

# a = np.arange(0, 60, 5)
# a = a.reshape(3, 4)
# print('原始数组是：')
# print(a)
# print('\n')
# for x in np.nditer(a, op_flags=['readwrite']):
#     x[...] = 2 * x
# print('修改后的数组是：')
# print(a)


a = np.arange(0,60,5)
a = a.reshape(3,4)
print  ('第一个数组为：')
print (a)
print  ('\n')
print ('第二个数组为：')
b = np.array([1,  2,  3,  4], dtype =  int)
print (b)
print ('\n')
print ('修改后的数组为：')
for x,y in np.nditer([a,b]):
    print ("%d:%d"  %  (x,y), end=", " )