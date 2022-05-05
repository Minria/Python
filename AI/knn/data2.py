from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
iris = load_iris()
# print(iris)
# print("鸢尾花的特征值:\n", iris["data"])
# print("鸢尾花的⽬标值：\n", iris.target)
# print("鸢尾花特征的名字：\n", iris.feature_names)
# print("鸢尾花⽬标值的名字：\n", iris.target_names)
# print("鸢尾花的描述：\n", iris.DESCR)
iris_d = pd.DataFrame(iris['data'], columns=['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'])
iris_d['Species'] = iris.target


def plot_iris(iris, col1, col2):
    sns.lmplot(x=col1, y=col2, data=iris, hue="Species", fit_reg=False)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title('鸢尾花种类分布图')
    plt.show()


plot_iris(iris_d, 'Petal_Width', 'Sepal_Length')
