import scipy.io as sio
import matplotlib.pyplot as plt


data_path = 'D:\wangfuming\Desktop\毕设\data and answer\data3/'
data_name = 'data'
mat_contents = sio.loadmat(data_path+data_name+'.mat')

# 获取数据和gt
# data = mat_contents['data']
gt = mat_contents['gt']

# 将数据转换为RGB图像格式

# plt.imshow(gt, cmap='gray')
plt.imshow(gt)
plt.savefig(data_path+data_name+'_gt.png')
plt.show()