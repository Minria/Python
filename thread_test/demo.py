
import threading

event1 = threading.Event()
event2 = threading.Event()
name_list = []
s = ''
a = 0
def func():
    global a, s, name_list
    while True:
        event1.wait()
        event1.clear()
        print(a)
        print(s)
        print(name_list)
        a = 2
        event2.set()

thread_a = threading.Thread(target=func)
thread_a.start()
def fun():
    global a
    global name_list
    global s
    s = ''
    name_list = []
    s = 'hello'
    name_list.append('hello')
    a = 1
    event1.set()
    event2.wait()
    event2.clear()
    print(a)

fun()
fun()
fun()
