# def greet_user():
#     print("hello!")
# greet_user()
#
#
# def greet_user(name):
#     print("Hello", name.title(), "!")
#
# greet_user('wangfuming')
#
# 传递实参
# def describe_pet(animal_type, pet_name):
#     """显示宠物的信息"""
#     print("\nI have a " + animal_type + ".")
#     print("My " + animal_type + "'s name is " + pet_name.title() + ".")
#
#
# describe_pet('hamster', 'harry')
#
#
# 多次调用函数
# def describe_pet(animal_type, pet_name):
#     """显示宠物的信息"""
#     print("\nI have a " + animal_type + ".")
#     print("My " + animal_type + "'s name is " + pet_name.title() + ".")
#
#
# describe_pet('hamster', 'harry')
# describe_pet('dog', 'willie')
#
# 注意函数参数顺序
# def describe_pet(animal_type, pet_name):
#     """显示宠物的信息"""
#     print("\nI have a " + animal_type + ".")
#     print("My " + animal_type + "'s name is " + pet_name.title() + ".")
#
#
# describe_pet('harry', 'hamster')
#
# 关键字指定实参
# def describe_pet(animal_type, pet_name):
#     """显示宠物的信息"""
#     print("\nI have a " + animal_type + ".")
#     print("My " + animal_type + "'s name is " + pet_name.title() + ".")
#
# describe_pet(animal_type='hamster', pet_name='harry')
# describe_pet(pet_name='harry', animal_type='hamster')
#
# 函数函数指定默认值
# def describe_pet(pet_name, animal_type='dog'):
#     """显示宠物的信息"""
#     print("\nI have a " + animal_type + ".")
#     print("My " + animal_type + "'s name is " + pet_name.title() + ".")
#
#
# describe_pet(pet_name='willie')
#
# 提示错误
# def describe_pet(animal_type, pet_name):
#     """显示宠物的信息"""
#     print("\nI have a " + animal_type + ".")
#     print("My " + animal_type + "'s name is " + pet_name.title() + ".")
#
# describe_pet()