import os
import pickle
import random
import time
from PIL import Image
import numpy as np
from torch import tensor


class DataSets:
    def __init__(self, path, train=0.8, mode='train'):
        self.test_list = None
        self.train_list = None
        self.dict = {}
        self.path = path
        self.mode = mode
        self.train = train
        self.img_list = []
        self.init_img()

    def init_img(self):
        file_path = './data_file/img_list.pkl'
        start = time.time()
        if os.path.exists(file_path):
            print('数据集预处理完成')
            with open(file_path, 'rb') as f:
                self.img_list = pickle.load(f)
        else:
            print('数据集预处理中')
            cls_list = os.listdir(self.path)
            start = time.time()
            for cls in cls_list:
                self.dict[cls] = len(self.dict)
                print(time.time()-start)
                img_name = os.listdir(os.path.join(self.path, cls))
                self.img_list.extend([{'img_path': self.get_img(os.path.join(self.path, cls + '/' + img)),
                                       'img_cls': self.dict[cls]} for img in img_name])
            with open(file_path, 'wb') as f:
                pickle.dump(self.img_list, f)
        random.shuffle(self.img_list)
        print('数据加载完成,用时', time.time()-start)
        self.train_list = self.img_list[:int(len(self.img_list) * self.train)]
        self.test_list = self.img_list[int(len(self.img_list) * self.train):]

    # 返回测试数据、训练数据
    def __getitem__(self, item):
        if self.mode == 'train':
            return self.train_list[item]
        else:
            return self.test_list[item]['img_path'], self.test_list[item]['img_cls']
    # 数据长度
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_list)
        else:
            return len(self.test_list)

    @staticmethod
    def collate_fn(batch):
        train_batch = {'img_path': [], 'img_cls': []}
        for img_pair in batch:
            train_batch['img_path'].append(img_pair['img_path'])
            train_batch['img_cls'].append(img_pair['img_cls'])
        return np.asarray(train_batch['img_path']), np.asarray(train_batch['img_cls'])

    @staticmethod
    def get_img(path):
        img = Image.open(path).resize((100, 100))
        img = np.asarray(img)/255
        img = img.astype(np.float32)
        return img

    @staticmethod
    def map_tensor(data):
        return tensor(data).reshape(len(data), -1).squeeze()

if __name__ == '__main__':
    ds = DataSets(path='E:/Downloads/Compressed/flower_photos')
    for i in range(0, len(ds), 21):
        x = ds[i:min(i+21, len(ds))]
        x, y = ds.collate_fn(x)
        # print(1)
