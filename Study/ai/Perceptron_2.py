# 算法
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def perceptron(X, Y, a=np.zeros((3, 1)), b=0, r=1):  # 格式化输入：X为n×p维，Y为n×1维，w为2×1维，b为常数
    g = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            g[i][j] = np.dot(X[i], X[j])
    i = 0
    while True:
        ans = 0
        for j in range(3):
            ans += a[j] * Y[j] * g[i][j]
        ans += b
        if Y[i] * ans > 0:
            i += 1
            if i == X.shape[0]:
                break
        else:
            a[i] += 1
            b += Y[i]
            i = 0

    return a, b


# 测试
X = np.array([[3, 3], [4, 3], [1, 1]])
Y = np.array([[1, 1, -1]]).reshape(3, 1)
a, b = perceptron(X, Y)
w = np.zeros((2, 1))
for i in range(3):
    tmp = np.array(X[i]).reshape(2,1)
    w += a[i]*Y[i]*tmp
print(w)
print(a)
print(b)


# 可视化


Y = pd.DataFrame(Y)  # 便于索引Y中的值(注意！list（Y）中的每一个元素都是array，所以不要用)
color = {1: 'red', -1: 'blue'}
x = np.linspace(0, 4)
y = 5 - x
plt.scatter(X[:, 0], X[:, 1], color=[color[i] for i in Y[0]])  # 注意！索引数据帧的列，哪怕只有一列也要有对应索引值
plt.plot(x, y, 'pink')
plt.show()
