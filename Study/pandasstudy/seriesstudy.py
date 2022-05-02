import pandas as pd

# pandas.Series( data, index, dtype, name, copy)
# data：一组数据(ndarray 类型)。
# index：数据索引标签，如果不指定，默认从 0 开始。
# dtype：数据类型，默认会自己判断。
# name：设置名称。
# copy：拷贝数据，默认为 False。

import pandas as pd

# a = [1, 2, 3]
# myvar = pd.Series(a)
# 索引默认是从0开始
# print(myvar)
# print(myvar[1])

# 自己设定索引值
# a = ["Google", "Runoob", "Wiki"]
# myvar = pd.Series(a, index = ["x", "y", "z"])
# print(myvar)

# key-value设置索引值
# sites = {1: "Google", 2: "Runoob", 3: "Wiki"}
# myvar = pd.Series(sites)
# print(myvar)

# 只需要部分数据
sites = {1: "Google", 2: "Runoob", 3: "Wiki"}

myvar = pd.Series(sites, index = [1, 2])

print(myvar)