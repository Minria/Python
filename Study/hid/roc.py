

import numpy as np
import scipy.io
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
data_path = 'D:\wangfuming\Desktop\毕设\data and answer\data3/'
y1 = pd.read_csv(data_path+'nn.csv', header=None)
y2 = pd.read_csv(data_path+'vi.csv', header=None)
y3 = pd.read_csv(data_path+'rx.csv', header=None)
y1 = y1.values
y2 = y2.values
y3 = y3.values
y = scipy.io.loadmat(data_path+'data.mat')['gt']
# y = y.astype(np.float32)
# 计算模型1的ROC曲线和AUC
print(y1.shape)
print(y2.shape)
print(y.shape)
y1 = y1.flatten()
y2 = y2.flatten()
y3 = y3.flatten()
a,b = y.shape
y = y.reshape((a*b, 1))
fpr1, tpr1, _ = roc_curve(y, y1)
roc_auc1 = auc(fpr1, tpr1)

# 计算模型2的ROC曲线和AUC
fpr2, tpr2, _ = roc_curve(y, y2)
roc_auc2 = auc(fpr2, tpr2)

fpr3, tpr3, _ = roc_curve(y, y3)
roc_auc3 = auc(fpr3, tpr3)
# 绘制ROC曲线
plt.figure()
lw = 2
plt.plot(fpr1, tpr1, color='darkorange',
         lw=lw, label='NN (AUC = %0.4f)' % roc_auc1)
plt.plot(fpr2, tpr2, color='blue',
         lw=lw, label='BP (AUC = %0.4f)' % roc_auc2)
plt.plot(fpr3, tpr3, color='red',
         lw=lw, label='RX (AUC = %0.4f)' % roc_auc3)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig(data_path+"roc.png")
plt.show()