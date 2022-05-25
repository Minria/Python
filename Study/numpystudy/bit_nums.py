# 位操作
import numpy as np

print('13 和 17 的二进制形式：')
a, b = 13, 17
print(bin(a), bin(b))
print('\n')

print('13 和 17 的位与：')
print(np.bitwise_and(13, 17))
print(13 & 17)

print('13 和 17 的位或：')
print(np.bitwise_or(13, 17))
print(a | b)

print('13 的位反转，其中 ndarray 的 dtype 是 uint8：')
print(np.invert(np.array([13], dtype=np.uint8)))
print('\n')
# 比较 13 和 242 的二进制表示，我们发现了位的反转

print('13 的二进制表示：')
print(np.binary_repr(13, width=8))
print('\n')

print('242 的二进制表示：')
print(np.binary_repr(242, width=8))

print(~13)
