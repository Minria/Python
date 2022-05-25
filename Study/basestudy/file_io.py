from pip._vendor.distlib.compat import raw_input

# print("hello world")
#
# a = raw_input("输入:")
# print(a)
# b = input("请输入:")
# print(b)

f = open("D:/wangfuming/Desktop/code/test.txt", "r+")

f.write("hello world")
f.write("wfm")
f.write("12343\n")
f.write("a+b=3")

a = f.readline()
print(a)
