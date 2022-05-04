from sklearn.neighbors import KNeighborsClassifier

# # 构造数据
# x = [[0], [1], [2], [3]]
# y = [0, 0, 1, 1]
# # 实例化对象，参考几个邻居
# estimator = KNeighborsClassifier(n_neighbors=1)
# # 进行训练
# estimator.fit(x, y)
# # 进行预测
# ret = estimator.predict([[4]])
# print(ret)

x = [[39, 0, 31], [3, 2, 65], [2, 3, 55], [9, 38, 2], [8, 34, 17], [5, 2, 57], [21, 17, 5], [45, 2, 9]]
y = [1, 2, 3, 3, 3, 2, 1, 1]
estimator = KNeighborsClassifier(n_neighbors=1)
estimator.fit(x, y)
ret = estimator.predict([[23, 3, 17]])
print(ret)
