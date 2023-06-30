import os
import random


class Datasets:

    def __init__(self, path, rate=0.8, mode='train'):
        self.test_list = []
        self.train_list = []
        self.path = path
        self.mode = mode
        self.rate = rate
        self.img_list = []
        self.init_img()

    def init_img(self):
        cls_list = os.listdir(self.path)
        for cls in cls_list:
            file_name = os.listdir(os.path.join(self.path, cls))
            for name in file_name:
                self.img_list.append({'img_path': os.path.join(self.path, cls + '/' + name), 'img_class': cls})

        random.shuffle(self.img_list)
        rate = 0.2
        num = len(self.img_list) - int(len(self.img_list) * rate)
        self.train_list = self.img_list[:num]
        self.test_list = self.img_list[num:]

    def __getitem__(self, item):
        if self.mode == 'train':
            return self.train_list[item]
        else:
            return self.test_list[item]['img_path'], self.test_list[item]['img_class']
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_list)
        else:
            return len(self.test_list)

    @staticmethod
    def coffate_fn(batch):
        train_batch = {'img_path': [], 'img_class': []}
        for img_pair in batch:
            train_batch['img_path'].append(img_pair['img_path'])
            train_batch['img_class'].append(img_pair['img_class'])
        return train_batch['img_path'], train_batch['img_class']