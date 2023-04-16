
import scipy.io as sio
import matplotlib.pyplot as plt


data_path = 'D:\wangfuming\Desktop\毕设\Code\Hyperspectral-Anomaly-Detection/'
data_name = 'PlaneGT'
# 读取A.mat文件
mat_contents = sio.loadmat(data_path + data_name + '.mat')

# 获取数据和gt
data = mat_contents['data']
# gt = mat_contents['salinas_gt']

# 将数据转换为RGB图像格式
rgb = data[:, :, [29, 19, 9]]  # 选择三个波段作为RGB通道
rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())  # 归一化

# 打印彩图
plt.imshow(rgb)
plt.axis('off')
plt.savefig(data_path + data_name + '.png')
plt.show()
