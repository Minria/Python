# 字符串

单引号或者双引号引出

```python
str = "hello world"
str.title() # Hello World
str.upper() # 所有都大写
str.lower() # 所有都小写
str2 = "python"
str + str2 	# 进行字符串拼接
# 使用/n来换行提高代码可读性
str3 = str.rstrip() # 删除尾空白
str4 = str.rstrip() # 删除前空白
str5 = str.strip() # 删除前后空白
# 使用str指定类型
age = 23 
message = "Happy " + str(age) + "rd Birthday!"
print(message)
```

# 列表

列表 由一系列按特定顺序排列的元素组成。你可以创建包含字母表中所有字母、数字0~9或所有家庭成员姓名的列表；也可以将任何东西加入列表中，其中的元素之间可以没有 任何关系。

参考数据结构的线性表

在Python中，用方括号（[] ）来表示列表，并用逗号来分隔其中的元素。

```python
bicycles = ['trek', 'cannondale', 'redline', 'specialized']
print(bicycles[0]) # 通过下标进行访问
print(bicycles[-1]) # 访问最后一个

motorcycles = ['honda', 'yamaha', 'suzuki']
motorcycles[0] = 'ducati'  # 修改元素
motorcycles.append('ducati') # 尾插元素
motorcycles.insert(0, 'ducati') # 在指定位置插入元素
del motorcycles[0] # 删除元素，前提是知道索引
motorcycles.pop() # 删除表尾元素，相当于栈的删除
motorcycles.pop(0) # 删除指定位置元素
motorcycles.remove('ducati') # 删除指定值的元素，只能删除第一个值

cars = ['bmw', 'audi', 'toyota', 'subaru']
cars.sort() # 永久性排序
sorted(cars) # 临时排序
cars.reverse() # 反转列表
len(cars) # 获取表长度

magicians = ['alice', 'david', 'carolina'] 
# 循环遍历整个表
for magician in magicians:
    print(magician)
    
# 创建数值表 左闭右开
for value in range(1,5):
    print(value)
    
players = ['charles', 'martina', 'michael', 'florence', 'eli'] 
print(players[0:3]) # 切边 0，1，2
print(players[:4]) # 默认从0开始
print(players[2:]) # 下标2开始
print(players[-3:]) # 倒数第三个开始
# 遍历切边
for player in players[:3]: 
    print(player.title())
    
my_foods = ['pizza', 'falafel', 'carrot cake'] 
friend_foods = my_foods # 两个指向同一个列表
# 赋值一个列表
friend_foods = my_foods[:]

```

# 元组

元组看起来犹如列表，但使用圆括号而不是方括号来标识。定义元组后，就可以使用索引来访问其元素，就像访问列表元素一样

```python
dimensions = (200, 50)
# 元组禁止修改
# 遍历元组
for dimension in dimensions: 
    print(dimension)
    
# 重新赋值元组    
dimensions = (400, 100)
```

# 字典

在Python中，字典 是一系列键—值对 。每个键 都与一个值相关联，你可以使用键来访问与之相关联的值。与键相关联的值可以是数字、字符串、列表乃至字典。事实上，可将 任何Python对象用作字典中的值。
在Python中，字典用放在花括号{} 中的一系列键—值对表示

```python
alien_0 = {'color': 'green', 'points': 5}
print(alien_0['color']) # 访问
alien_0['x_position'] = 0 # 添加键值对
alien_0 = {} # 创建一个空字典
alien_0['color'] = 'yellow' # 修改字典元素
del alien_0['points'] # 删除键值对

# 遍历键值对
for key, value in user_0.items():
    print("\nKey: " + key)
    print("Value: " + value)
 
# 访问所有键
for name in favorite_languages.keys():
    print(name.title())
       
# 排序访问
for name in sorted(favorite_languages.keys()):
    print(name.title() + ", thank you for taking the poll.")
# 访问所有的值
for language in favorite_languages.values(): 
    print(language.title())
    
# 嵌套 字典列表    
alien_0 = {'color': 'green', 'points': 5} 
alien_1 = {'color': 'yellow', 'points': 10}
alien_2 = {'color': 'red', 'points': 15}
aliens = [alien_0, alien_1, alien_2]
for alien in aliens:
    print(alien)
```



# 条件和循环

```python
cars = ['audi', 'bmw', 'subaru', 'toyota'] 
for car in cars:
    if car == 'bmw':
        print(car.upper())
    else:
        print(car.title())
        
  

banned_users = ['andrew', 'carolina', 'david'] 
user = 'marie'
# 检查某个元素是否在列表
if user not in banned_users:
    print(user.title() + ", you can post a response if you wish.")
```

# 输入输出

```python
message = input("Tell me something, and I will repeat it back to you: ") print(message)
age = int(input()) # s
```



# Numpy

## 数组属性

| 属性             | 说明                                                         |
| :--------------- | :----------------------------------------------------------- |
| ndarray.ndim     | 秩，即轴的数量或维度的数量                                   |
| ndarray.shape    | 数组的维度，对于矩阵，n 行 m 列                              |
| ndarray.size     | 数组元素的总个数，相当于 .shape 中 n*m 的值                  |
| ndarray.dtype    | ndarray 对象的元素类型                                       |
| ndarray.itemsize | ndarray 对象中每个元素的大小，以字节为单位                   |
| ndarray.flags    | ndarray 对象的内存信息                                       |
| ndarray.real     | ndarray元素的实部                                            |
| ndarray.imag     | ndarray 元素的虚部                                           |
| ndarray.data     | 包含实际数组元素的缓冲区，由于一般通过数组的索引获取元素，所以通常不需要使用这个属性。 |

## 创建数组

```python
numpy.empty(shape, dtype = float, order = 'C')
numpy.zeros(shape, dtype = float, order = 'C')
numpy.ones(shape, dtype = None, order = 'C')
```

| 参数  | 描述                                                         |
| :---- | :----------------------------------------------------------- |
| shape | 数组形状                                                     |
| dtype | 数据类型，可选                                               |
| order | 有"C"和"F"两个选项,分别代表，行优先和列优先，在计算机内存中的存储元素的顺序。 |

## 位运算

| 函数          | 描述                   |
| :------------ | :--------------------- |
| `bitwise_and` | 对数组元素执行位与操作 |
| `bitwise_or`  | 对数组元素执行位或操作 |
| `invert`      | 按位取反               |
| `left_shift`  | 向左移动二进制表示的位 |
| `right_shift` | 向右移动二进制表示的位 |

**注：**也可以使用 "&"、 "~"、 "|" 和 "^" 等操作符进行计算。
