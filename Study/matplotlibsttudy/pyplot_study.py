import matplotlib.pyplot as plt
import numpy as np

# x = np.array([1, 6])
# y = np.array([1, 100])
# 绘制直线
# plt.plot(x, y)
# plt.show()
# 绘制两个点
# plt.plot(x, y, 'o')
# plt.show()

x = np.arange(0, 4 * np.pi, 0.1)  # start,stop,step
y = np.sin(x)
z = np.cos(x)
plt.plot(x, y, x, z)
plt.show()
