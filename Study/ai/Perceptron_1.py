# 算法
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def perceptron(X, Y, w=np.zeros((2, 1)), b=0, r=1):  # 格式化输入：X为n×p维，Y为n×1维，w为2×1维，b为常数
    i = 0
    while True:
        if int(Y[i]) == np.sign(float(np.dot(X[i],w)) + b):
            i += 1
            # 所有点都分类成功，可以退出
            if i == X.shape[0]:
                break
        else:
            # 随机梯度下降
            w += np.mat((r * Y[i] * X[i])).T  # 注意！一维array数组转置无效
            b += int(r * Y[i])
            print(f'误分类点X{i + 1}，\n{w, b}',
                  '\n-----------------------')
            i = 0
    return w, b


# 测试
X = np.array([[3, 3], [4, 3], [1, 1],[2,2]])
Y = np.array([[1, 1, -1, -1]]).reshape(4, 1)

w, b = perceptron(X, Y)
print(w)
print(b)
# 可视化
Y = pd.DataFrame(Y)  # 便于索引Y中的值(注意！list（Y）中的每一个元素都是array，所以不要用)
color = {1: 'red', -1: 'blue'}
x = np.linspace(0, 4)
y = 5 - x
plt.scatter(X[:, 0], X[:, 1], color=[color[i] for i in Y[0]])  # 注意！索引数据帧的列，哪怕只有一列也要有对应索引值
plt.plot(x, y, 'pink')
plt.show()