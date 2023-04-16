

import numpy as np
data = 0
gt = 0
num_classes = len(np.unique(gt))
class_data = []
for i in range(num_classes):
    class_data.append(data[gt==i])
# 生成相似的样本对
similar_pairs = []
for i in range(num_classes):
    n_samples = class_data[i].shape[0]
    for j in range(n_samples):
        for k in range( j +1, n_samples):
            pair = (class_data[i][j], class_data[i][k], 0)  # 0表示相似
            similar_pairs.append(pair)
# 生成不相似的样本对
dissimilar_pairs = []
for i in range(num_classes):
    n_samples_i = class_data[i].shape[0]
    for j in range(i+1, num_classes):
        n_samples_j = class_data[j].shape[0]
        for k in range(n_samples_i):
            for l in range(n_samples_j):
                pair = (class_data[i][k], class_data[j][l], 1)  # 1表示不相似
                dissimilar_pairs.append(pair)

# 将相似和不相似的样本对合并成一个列表
pairs = similar_pairs + dissimilar_pairs
# 打乱样本对的顺序
np.random.shuffle(pairs)
# 将样本对分成输入和标签
X = np.array([(p[0], p[1]) for p in pairs])
y = np.array([p[2] for p in pairs])


