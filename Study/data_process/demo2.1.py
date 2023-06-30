import os
import random
file_path = 'E:/Downloads/Compressed/flower_photos'

cls_list = os.listdir(file_path)
img_list = []
for cls in cls_list:
    file_name = os.listdir(os.path.join(file_path, cls))
    for name in file_name:
        img_list.append({'img_path': os.path.join(file_path, cls + '/' + name), 'img_class': cls})

random.shuffle(img_list)
rate = 0.2
num = len(img_list) - (int) (len(img_list) *rate)
train_data = img_list[:num]
test_data = img_list[num:]
bachsize = 10
train_batch = {'img_path': [], 'img_class': []}
for i in range(0, len(img_list), bachsize):
    x = img_list[i: min((i+bachsize), len(img_list))]
    for img_pair in x:
        train_batch['img_path'].append(img_pair['img_path'])
        train_batch['img_class'].append(img_pair['img_class'])


print(1)