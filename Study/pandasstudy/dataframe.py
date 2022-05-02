import pandas as pd

# pandas.DataFrame( data, index, columns, dtype, copy)
# data：一组数据(ndarray、series, map, lists, dict 等类型)。
# index：索引值，或者可以称为行标签。
# columns：列标签，默认为 RangeIndex (0, 1, 2, …, n) 。
# dtype：数据类型。
# copy：拷贝数据，默认为 False。

# 使用列表创建
# data = [['Google', 10], ['Runoob', 12], ['Wiki', 13]]
# df = pd.DataFrame(data, columns=['Site', 'Age'], dtype=float)
# print(df)

# 使用ndarrays创建
# data = {'Site':['Google', 'Runoob', 'Wiki'], 'Age':[10, 12, 13]}
# df = pd.DataFrame(data)
# print (df)

# 使用字典创建

# data = [{'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20}]
# df = pd.DataFrame(data)
# print(df)

data = {
    "calories": [420, 380, 390],
    "duration": [50, 40, 45]
}
# 数据载入到 DataFrame 对象
df = pd.DataFrame(data)
print("返回第一行")
print(df.loc[0])
print("返回第二行")
print(df.loc[1])
print("返回第一行和第二行")
print(df.loc[[0, 1]])
