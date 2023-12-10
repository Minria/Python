import threading
import queue
import time

# 创建一个队列作为生产者和消费者之间的缓冲区
buffer = queue.Queue()
# 创建一个Event对象用于线程通知
event1 = threading.Event()
event2 = threading.Event()
# 线程A函数
def thread_a():
    while True:
        event1.wait()  # 等待通知
        event1.clear()  # 清除通知状态
        print("线程A：开始执行")
        while not buffer.empty():
            item = buffer.get()
            print("线程A：从队列中取出元素", item)
        # 执行任务
        print("线程A：执行完毕")
        event2.set()

# 创建并启动线程A
thread_a = threading.Thread(target=thread_a)
thread_a.start()

def func(a, b, c):
    buffer.put(a)
    buffer.put(b)
    buffer.put(c)
    event1.set()  # 发送通知
    print('发送通知A执行')
    event2.wait()
    event2.clear()
    print("收到线程A的通知，继续执行")

if __name__ == "__main__":
    func(1, 2, 3)
    func(4, 5, 6)
    func(7, 8, 9)


