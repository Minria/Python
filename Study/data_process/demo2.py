import math
from PIL import Image
import os
import numpy as np

def imgToArray(url):
    image = Image.open(url)
    mat = np.array(image)
    return mat

# file_dir = 'E:\Downloads\Compressed/flower_photos'
# daisy = []
# label_daisy = []
# dandelion = []
# label_dandelion = []
# roses = []
# label_roses = []
# sunflowers = []
# label_sunflowers = []
# tulips = []
# label_tulips = []
# print(file_dir+'/daisy')
#
# for file in os.listdir(file_dir + "/daisy"):
#     daisy.append(file_dir + "/daisy" + "/" + file)
#     label_daisy.append(0)
#
# for file in os.listdir(file_dir + "/dandelion"):
#     dandelion.append(file_dir + "/dandelion" + "/" + file)
#     label_dandelion.append(1)
#
# for file in os.listdir(file_dir + "/roses"):
#     roses.append(file_dir + "/roses" + "/" + file)
#     label_roses.append(2)
#
# for file in os.listdir(file_dir + "/sunflowers"):
#     sunflowers.append(file_dir + "/sunflowers" + "/" + file)
#     label_sunflowers.append(3)
#
# for file in os.listdir(file_dir + "/tulips"):
#     tulips.append(file_dir + "/tulips" + "/" + file)
#     label_tulips.append(4)
#
#
# data = np.hstack([daisy, dandelion, roses, sunflowers, tulips])
# labels = np.hstack([label_daisy, label_dandelion, label_roses, label_sunflowers, label_tulips])


file_dir = 'E:\Downloads\Compressed/flower_photos'

data = []
labels = []

for file in os.listdir(file_dir + "/daisy"):
    data.append(file_dir + "/daisy" + "/" + file)
    labels.append(0)

for file in os.listdir(file_dir + "/dandelion"):
    data.append(file_dir + "/dandelion" + "/" + file)
    labels.append(1)

for file in os.listdir(file_dir + "/roses"):
    data.append(file_dir + "/roses" + "/" + file)
    labels.append(2)

for file in os.listdir(file_dir + "/sunflowers"):
    data.append(file_dir + "/sunflowers" + "/" + file)
    labels.append(3)

for file in os.listdir(file_dir + "/tulips"):
    data.append(file_dir + "/tulips" + "/" + file)
    labels.append(4)


data = np.array(data)
labels = np.array(labels)

# 利用shuffle打乱顺序
temp = np.array([data, labels]).transpose()
np.random.shuffle(temp)

# 分离
data = list(temp[:, 0])
labels = list(temp[:, 1])
# labels中的数字变成字符串形式
labels = [int(i) for i in labels]


ratio = 0.2
n_sample = len(labels)
n_val = int(math.ceil(n_sample * ratio))
n_train = n_sample - n_val

train_data = data[0:n_train]
train_labels = labels[0:n_train]

test_data = data[n_train:-1]
test_labels = labels[n_train:-1]

x_train = []
x_test = []

for i in range(len(train_data)):
    mat = imgToArray(train_data[i])
    x_train.append(mat)

for i in range(len(test_data)):
    mat = imgToArray(test_data[i])
    x_test.append(mat)


print(1)
