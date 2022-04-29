import matplotlib.pyplot as plt
import numpy as np
# 行数,列数,位置
# subplot(nrows, ncols, index, **kwargs)
# subplot(pos, **kwargs)
# subplot(**kwargs)
# subplot(ax)

#plot 1:
# xpoints = np.array([0, 6])
# ypoints = np.array([0, 100])
#
# plt.subplot(1, 2, 1)
# plt.plot(xpoints,ypoints)
# plt.title("plot 1")

#plot 2:
# x = np.array([1, 2, 3, 4])
# y = np.array([1, 4, 9, 16])
#
# plt.subplot(1, 2, 2)
# plt.plot(x,y)
# plt.title("plot 2")
#
# plt.suptitle("RUNOOB subplot Test")
# plt.show()


#plot 1:
# x = np.array([0, 6])
# y = np.array([0, 100])
#
# plt.subplot(2, 2, 1)
# plt.plot(x,y)
# plt.title("plot 1")

#plot 2:
# x = np.array([1, 2, 3, 4])
# y = np.array([1, 4, 9, 16])
#
# plt.subplot(2, 2, 2)
# plt.plot(x,y)
# plt.title("plot 2")

#plot 3:
# x = np.array([1, 2, 3, 4])
# y = np.array([3, 5, 7, 9])
#
# plt.subplot(2, 2, 3)
# plt.plot(x,y)
# plt.title("plot 3")

#plot 4:
# x = np.array([1, 2, 3, 4])
# y = np.array([4, 5, 6, 7])
#
# plt.subplot(2, 2, 4)
# plt.plot(x,y)
# plt.title("plot 4")
#
# plt.suptitle("RUNOOB subplot Test")
# plt.show()


# ############################################################################################

# matplotlib.pyplot.subplots(nrows=1, ncols=1, *,
#                            sharex=False, sharey=False, squeeze=True, subplot_kw=None,
#                            gridspec_kw=None, **fig_kw)


# 创建一些测试数据 -- 图1
x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)

# 创建一个画像和子图 -- 图2
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title('Simple plot')

# 创建两个子图 -- 图3
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(x, y)
ax1.set_title('Sharing Y axis')
ax2.scatter(x, y)

# 创建四个子图 -- 图4
fig, axs = plt.subplots(2, 2, subplot_kw=dict(projection="polar"))
axs[0, 0].plot(x, y)
axs[1, 1].scatter(x, y)

# 共享 x 轴
plt.subplots(2, 2, sharex='col')

# 共享 y 轴
plt.subplots(2, 2, sharey='row')

# 共享 x 轴和 y 轴
plt.subplots(2, 2, sharex='all', sharey='all')

# 这个也是共享 x 轴和 y 轴
plt.subplots(2, 2, sharex=True, sharey=True)

# 创建10 张图，已经存在的则删除
fig, ax = plt.subplots(num=10, clear=True)

plt.show()