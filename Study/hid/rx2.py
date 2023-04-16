
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import hyperspy.api as hs
# 加载数据
data = sio.loadmat('D:\wangfuming\Desktop\毕设\data and answer\data4\Beach.mat')['data']


# 将数据转换为Hyperspy对象
hs_data = hs.signals.Signal2D(data)

# 对数据进行预处理，包括背景减法和去噪
hs_data.subtract_bg(1, method='median')
hs_data.remove_noise('wavelet', sigma=3)

# 使用PCA降维
hs_data.decomposition('PCA', output_dimension=10)

# 对降维后的数据进行异常检测
scores = hs_data.get_decomposition_loadings()[0]
threshold = np.percentile(scores, 95)
pred = scores > threshold

# 将预测结果转换为灰度图像
pred_image = pred.reshape(data.shape[0], data.shape[1])

np.savetxt('./result.csv', pred_image, delimiter=',')
# 显示灰度图像
plt.imshow(pred_image, cmap='gray')
plt.show()