from sklearn.neighbors import KNeighborsClassifier

from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

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

# x = [[39, 0, 31], [3, 2, 65], [2, 3, 55], [9, 38, 2], [8, 34, 17], [5, 2, 57], [21, 17, 5], [45, 2, 9]]
# y = [1, 2, 3, 3, 3, 2, 1, 1]
# estimator = KNeighborsClassifier(n_neighbors=1)
# estimator.fit(x, y)
# ret = estimator.predict([[23, 3, 17]])
# print(ret)


iris = load_iris()
print(iris)
print("鸢尾花的特征值:\n", iris["data"])
print("鸢尾花的⽬标值：\n", iris.target)
print("鸢尾花特征的名字：\n", iris.feature_names)
print("鸢尾花⽬标值的名字：\n", iris.target_names)
print("鸢尾花的描述：\n", iris.DESCR)
iris_d = pd.DataFrame(iris['data'], columns=['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'])
iris_d['Species'] = iris.target


def plot_iris(iris, col1, col2):
    sns.lmplot(x=col1, y=col2, data=iris, hue="Species", fit_reg=False)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title('鸢尾花种类分布图')
    plt.show()


plot_iris(iris_d, 'Petal_Width', 'Sepal_Length')