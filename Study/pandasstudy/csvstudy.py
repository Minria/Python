import pandas as pd

# 读取csv文件
# df = pd.read_csv('D:/wangfuming/Desktop/nba.csv')
# print(df.to_string())

# 写入csv文件
# nme = ["Google", "Runoob", "Taobao", "Wiki"]
# st = ["www.google.com", "www.runoob.com", "www.taobao.com", "www.wikipedia.org"]
# ag = [90, 40, 80, 98]
# 字典
# dict = {'name': nme, 'site': st, 'age': ag}
# df = pd.DataFrame(dict)
# 保存 dataframe
# df.to_csv('D:/wangfuming/Desktop/site.csv')

# 读行数
df = pd.read_csv('D:/wangfuming/Desktop/nba.csv')
# 默认5行
print(df.head())
print(df.tail())
# 打印一些信息
print(df.info())
