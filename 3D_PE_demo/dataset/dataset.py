import random
import numpy as np
import scipy.io
import os
from PIL import Image
from utils.dataset_utils import new_label_encoder, label_change, process_box, label_encoder
from utils.eval_util import *
from utils.vision_util import Visualize

'''
                                    减少内存使用，将一些数据处理放到get中
'''
class YCBDataSet:
    def __init__(self, data_path=None, file_name=None, data_file=None, mode='train', train=0.95):
        self.mode = mode
        self.data_path = data_path
        self.file_name = file_name
        self.data_file = data_file
        self.train = train
        self.data_list = []
        self.mapping_dict = {'009_gelatin_box': 1, '025_mug': 2, '061_foam_brick': 3}  # 建立名字-id映射关系
        self.load_data()
        self.train_list = self.data_list[:int(len(self.data_list) * self.train)]
        self.test_list = self.data_list[int(len(self.data_list) * self.train):]

        self.objs_id = np.array([1, 2, 3])
        self.cam_intrinsic = np.array([[1066.778, 0, 312.9869],
                                       [0, 1067.487, 241.3109],
                                       [0, 0, 1]])
        self.resolution = [640, 480]
        self.visualizer = None
        self.model = self.load_model()



    def load_data(self):
        type1 = 'box'
        type2 = 'color'
        type3 = 'depth'
        type4 = 'label'
        type5 = 'meta'
        file_list = os.listdir(os.path.join(self.data_path, self.file_name))
        file_list = sorted(file_list, key=lambda x: int(x.split('-')[0]))
        i = 0
        data = {'rgb': None, 'depth': None, 'label': None, 'bbx': None, 'RT': None, 'center': None}
        for file in file_list:
            if type1 in file:
                bbox = process_box(os.path.join(os.path.join(self.data_path, self.file_name), file),
                                   mapping_dict=self.mapping_dict, is_mapping=True)
                data['bbx'] = bbox
                # data['bbx'] = file
                i = i + 1
            if type2 in file:
                img = Image.open(os.path.join(os.path.join(self.data_path, self.file_name), file))
                img = np.asarray(img)
                data['rgb'] = img
                # data['rgb'] = file
                i = i + 1
            if type3 in file:
                depth_img = Image.open(os.path.join(os.path.join(self.data_path, self.file_name), file))
                depth_img = np.asarray(depth_img)
                data['depth'] = depth_img
                # data['depth'] = file
                i = i + 1
            if type4 in file:
                label_img = Image.open(os.path.join(os.path.join(self.data_path, self.file_name), file))
                label_img = np.asarray(label_img)
                data['label'] = label_img
                # data['label'] = file
                i = i + 1
            if type5 in file:
                meta = scipy.io.loadmat(os.path.join(os.path.join(self.data_path, self.file_name), file))
                data['RT'] = meta['poses'].transpose(2, 0, 1)
                data['center'] = meta['center']
                # data['RT'] = file
                # data['center'] = file
                i = i + 1
            if i == 5:
                self.data_list.append(data)
                data = {'rgb': None, 'depth': None, 'label': None, 'bbx': None, 'RT': None, 'center': None}
                i = 0


    def load_model(self):
        model_path = os.path.join(self.data_path, "models")
        objpathdict = {
            1: ["009_gelatin_box", os.path.join(model_path, "009_gelatin_box", "textured.obj")],
            2: ["025_mug", os.path.join(model_path, "025_mug", "textured.obj")],
            3: ["061_foam_brick", os.path.join(model_path, "061_foam_brick", "textured.obj")],
        }
        self.visualizer = Visualize(objpathdict, self.cam_intrinsic, self.resolution)
        models_pcd_dict = {index: np.array(self.visualizer.objnode[index]['mesh'].vertices) for index in self.visualizer.objnode}
        models_pcd = np.zeros((len(models_pcd_dict), 1024, 3))
        for m in models_pcd_dict:
            model = models_pcd_dict[m]
            models_pcd[m - 1] = model[np.random.randint(0, model.shape[0], 1024)]
        return models_pcd


    def __getitem__(self, item):
        if self.mode == 'train':
            data = self.train_list[item]
        else:
            data = self.test_list[item]

        rgb_img = data['rgb']
        rgb_img = rgb_img.astype(float) / 255
        rgb = rgb_img.transpose((2, 0, 1))
        label = label_change(data['label'])
        label = new_label_encoder(4, label)
        depth = data['depth']
        bbx = data['bbx']
        center = data['center']
        RT = data['RT']
        RTs = np.zeros((3, 3, 4))
        centermaps = np.zeros((3, 3, 480, 640))
        for i in range(3):
            tmp_RT = RT[i]
            new_RT = np.zeros((4, 4))
            new_RT[:3, :3] = tmp_RT[:3, :3]
            new_RT[:3, [3]] = tmp_RT[:3, [3]]
            new_RT[3, 3] = 1
            RTs[i] = new_RT[:3]
            the_center = center[i]
            x = np.linspace(0, 639, 640)
            y = np.linspace(0, 479, 480)
            xv, yv = np.meshgrid(x, y)
            dx, dy = the_center[0] - xv, the_center[1] - yv
            distance = np.sqrt(dx ** 2 + dy ** 2)
            nx, ny = dx / distance, dy / distance
            tz = np.ones((480, 640)) * new_RT[2, 3]
            centermaps[i] = np.array([nx, ny, tz])

        centermaps = centermaps.reshape(-1, 480, 640)


        data = {'rgb': rgb, 'depth': depth, 'objs_id': self.objs_id,
                'label': label, 'bbx': bbx, 'RTs': RTs,
                'center': center, 'centermaps': centermaps}

        return data



    def __len__(self):
        if self.mode == 'train':
            return len(self.train_list)
        else:
            return len(self.test_list)


if __name__ == '__main__':
    dataset = YCBDataSet(data_path='E:/Archive files/dataset/YCB_Video_Dataset/', file_name='data/0000')
    print(dataset[0])
    print('-------------------')
    print(dataset.test_list)



'''
    ====================================================================================================================
    ====================================================================================================================
                                          以下dataset用于验证label判断分支
    ====================================================================================================================
    ====================================================================================================================
'''


class YCBDataSet1:

    def __init__(self, data_path=None, file_name=None, data_file=None, mode='train', train=0.8):
        self.mode = mode
        self.file_path = data_path
        self.file_name = file_name
        self.data_file = data_file
        self.train = train
        self.images = []
        self.labels = []
        self.old_labels = []
        self.obj_id = [1, 2, 3]
        self.id_name = {1: '009_gelatin_box ', 2: '025_mug', 3: '061_foam_brick'}
        self.create_data()
        self.train_img = self.images[:int(len(self.images) * self.train)]
        self.test_img = self.images[int(len(self.images) * self.train):]
        self.train_labels = self.labels[:int(len(self.labels) * self.train)]
        self.test_labels = self.labels[int(len(self.labels) * self.train):]
        self.test_old_labels = self.old_labels[int(len(self.labels) * self.train):]

    def create_data(self):
        type1 = 'color'
        type2 = 'label'
        file_list = os.listdir(os.path.join(self.file_path, self.file_name))
        for file in file_list:
            if type1 in file:
                img = Image.open(os.path.join(os.path.join(self.file_path, self.file_name), file))
                img = np.asarray(img)
                img = torch.Tensor(img)
                self.images.append(img)
            if type2 in file:
                img = Image.open(os.path.join(os.path.join(self.file_path, self.file_name), file))
                img = np.asarray(img)
                # 将标签重新映射 8->1, 14->2, 21->3
                img = label_change(img)
                img = torch.Tensor(img)
                self.old_labels.append(img)
                # 处理通道问题，例如 0->[1,0,0,0], 1->[0,1,0,0]
                label = label_encoder(4, img)
                self.labels.append(label)

    def __getitem__(self, item):
        if self.mode == 'train':
            return self.train_img[item] / 255, self.train_labels[item]
        else:
            return self.test_img[item] / 255, self.test_labels[item], self.test_old_labels[item]

    # 数据长度
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_img)
        else:
            return len(self.test_img)



'''

if __name__ == '__main__':
    file_path = 'E:/Archive files/dataset/YCB_Video_Dataset/data'
    y = YCBDataSet(data_path=file_path, file_name='0000', train=1)
    (a, b) = y[0]
    print(1)
'''
