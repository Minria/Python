import scipy.io as sio



map = sio.loadmat('D:\wangfuming\Desktop\CNND_v1\input\Salinas_gt.mat')['salinas_gt']
a,b = map.shape
sum = 0
num = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for i in range(a):
    for j in range(b):
       num[map[i][j]]+=1


print(num)


