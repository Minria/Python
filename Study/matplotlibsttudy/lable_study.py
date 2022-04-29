import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4])
y = np.array([1, 4, 9, 16])
plt.plot(x, y)
# 设置标题
plt.title("RUNOOB TEST TITLE")
# 设置x轴名称,可选择位置,默认中间
plt.xlabel("x - label")
# 设置y轴名称,可选择位置,默认中间
plt.ylabel("y - label")

plt.show()