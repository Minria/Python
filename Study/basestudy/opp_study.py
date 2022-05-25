class Employee:
    empCount = 0  # 类变量
    __a = ""

    def __init__(self, name, salary, age):
        self.name = name  # 实例变量
        self.salary = salary
        self.age = age
        Employee.a = name
        Employee.empCount += 1

    # 实例方法
    def displayEmployee(self):
        print("Name : ", self.name, ", Salary: ", self.salary)


e1 = Employee("wfm", 123333, 12)
e2 = Employee("wang", 22222, 19)
print("e1.a:", e1.a)
print("e2.a:", e2.a)
print("empCount:", e1.empCount)
print("empCount:", e2.empCount)
print(e1.age)
del e1.age  # 删除属性
e1.age = 13  # 新建一个属性
print(getattr(e1, 'name'))  # 通过getattr方式访问对象属性
print(hasattr(e1, 'age'))  # 检查是否存在这个属性
setattr(e1, 'age', 12)  # 设置字段的值为xx，如果不存在就创建一个字段
delattr(e1, 'age')  # 删除一个字段

print("Employee.__doc__:", Employee.__doc__)  # 类的文档字符串
print("Employee.__name__:", Employee.__name__)  # 类的名字
print("Employee.__module__:", Employee.__module__)  # 类定义所在的模块
print("Employee.__bases__:", Employee.__bases__)  # 类的所有父类构成元素
print("Employee.__dict__:", Employee.__dict__)  # 类的属性
