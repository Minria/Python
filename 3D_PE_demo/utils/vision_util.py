
from torchvision.utils import make_grid
import os
import pyrender
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import cv2
from PIL import Image
from utils.dataset_utils import *
from utils.train_utils import *


def show_bbx(image, bbx, standardization=True):
    if standardization:
        image = image*255
    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    for i in range(bbx.shape[0]):
        x1 = bbx[i, 1].item()
        y1 = bbx[i, 2].item()
        x2 = bbx[i, 3].item()
        y2 = bbx[i, 4].item()
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
    image.show()

# test for show_bbx

# file_path = 'E:/Archive files/dataset/YCB_Video_Dataset/data/0000'
# color_name = '000001-color.png'
# label_name = '000001-label.png'
# image = Image.open(os.path.join(file_path, color_name))
# image = np.asarray(image)/255
# label = Image.open(os.path.join(file_path, label_name))
# label = np.asarray(label)
# label = label_change(label)
# label = torch.from_numpy(label)
# label = label.unsqueeze(0)
# bbx = get_pred_bbx(4, label)
# show_bbx(image, bbx)


def visualize_dataset(pose_dataset, num_samples=1, alpha=0.5):

    plt.text(300, -40, 'RGB', ha="center")
    plt.text(950, -40, 'Pose', ha="center")
    # plt.text(1600, -40, 'Depth', ha="center")
    # plt.text(2250, -40, 'Segmentation', ha="center")
    # plt.text(2900, -40, 'Centermaps[0]', ha="center")

    samples = []
    for sample_i in range(num_samples):
        sample_idx = random.randint(0, len(pose_dataset) - 1)
        sample = pose_dataset[sample_idx]
        rgb = (sample['rgb'].transpose(1, 2, 0) * 255).astype(np.uint8)
        # depth = ((np.tile(sample['depth'], (3, 1, 1)) / sample['depth'].max()) * 255).astype(np.uint8)
        # segmentation = (sample['label'] * np.arange(11).reshape((11, 1, 1))).sum(0, keepdims=True).astype(np.float64)
        # segmentation /= segmentation.max()
        # segmentation = (np.tile(segmentation, (3, 1, 1)) * 255).astype(np.uint8)
        # ctrs = sample['centermaps'].reshape(10, 3, 480, 640)[0]
        # ctrs -= ctrs.min()
        # ctrs /= ctrs.max()
        # ctrs = (ctrs * 255).astype(np.uint8)
        pose_dict = format_gt_RTs(sample['RTs'])
        render = pose_dataset.visualizer.vis_oneview(
            ipt_im=rgb,
            obj_pose_dict=pose_dict,
            alpha=alpha
        )
        samples.append(torch.tensor(rgb.transpose(2, 0, 1)))
        samples.append(torch.tensor(render.transpose(2, 0, 1)))
        # samples.append(torch.tensor(depth))
        # samples.append(torch.tensor(segmentation))
        # samples.append(torch.tensor(ctrs))
    img = make_grid(samples, nrow=5).permute(1, 2, 0)
    return img

def visualize_output(pose_dataset, num_samples=1, alpha=0.5, sample=None):

    # plt.text(300, -40, 'RGB', ha="center")
    # plt.text(950, -40, 'Target_Pose', ha="center")
    # plt.text(1600, -40, 'Pred_Pose', ha="center")
    plt.text(300, -40, 'Target_Pose', ha="center")
    plt.text(950, -40, 'Pred_Pose', ha="center")
    samples = []
    for sample_i in range(num_samples):
        rgb = (sample['rgb'][0].transpose(1, 2, 0) * 255).astype(np.uint8)
        pose_dict = format_gt_RTs(sample['RTs'][0])
        pose_dict2 = format_gt_RTs(sample['RTs2'][0])
        render = pose_dataset.visualizer.vis_oneview(
            ipt_im=rgb,
            obj_pose_dict=pose_dict,
            alpha=alpha
        )
        render2 = pose_dataset.visualizer.vis_oneview(
            ipt_im=rgb,
            obj_pose_dict=pose_dict2,
            alpha=alpha
        )
        # samples.append(torch.tensor(rgb.transpose(2, 0, 1)))
        samples.append(torch.tensor(render.transpose(2, 0, 1)))
        samples.append(torch.tensor(render2.transpose(2, 0, 1)))
    img = make_grid(samples, nrow=5).permute(1, 2, 0)
    return img


def format_gt_RTs(RTs):
    return {idx + 1: np.concatenate((RTs[idx], [[0, 0, 0, 1]])) for idx in range(len(RTs))}






class Visualize:
    def __init__(self, object_dict, cam_intrinsic, resolution):
        self.objnode = {}
        self.render = pyrender.OffscreenRenderer(resolution[0], resolution[1])
        self.scene = pyrender.Scene()
        cam = pyrender.camera.IntrinsicsCamera(cam_intrinsic[0, 0],
                                               cam_intrinsic[1, 1],
                                               cam_intrinsic[0, 2],
                                               cam_intrinsic[1, 2],
                                               znear=0.05, zfar=100.0, name=None)
        self.intrinsic = cam_intrinsic
        Axis_align = np.array([[1, 0, 0, 0],
                               [0, -1, 0, 0],
                               [0, 0, -1, 0],
                               [0, 0, 0, 1]])
        self.nc = pyrender.Node(camera=cam, matrix=Axis_align)
        self.scene.add_node(self.nc)

        for obj_label in object_dict:
            objname = object_dict[obj_label][0]
            objpath = object_dict[obj_label][1]
            tm = trimesh.load(objpath)
            mesh = pyrender.Mesh.from_trimesh(tm, smooth=False)
            node = pyrender.Node(mesh=mesh, matrix=np.eye(4))
            node.mesh.is_visible = False
            self.objnode[obj_label] = {"name": objname, "node": node, "mesh": tm}
            self.scene.add_node(node)
        self.cmp = self.color_map(N=len(object_dict))
        self.object_dict = object_dict

    def vis_oneview(self, ipt_im, obj_pose_dict, alpha=0.5, axis_len=30):
        img = ipt_im.copy()
        for obj_label in obj_pose_dict:
            if obj_label in self.object_dict:
                pose = obj_pose_dict[obj_label]
                node = self.objnode[obj_label]['node']
                node.mesh.is_visible = True
                self.scene.set_pose(node, pose=pose)
        full_depth = self.render.render(self.scene, flags=pyrender.constants.RenderFlags.DEPTH_ONLY)
        for obj_label in obj_pose_dict:
            if obj_label in self.object_dict:
                node = self.objnode[obj_label]['node']
                node.mesh.is_visible = False
        for obj_label in self.object_dict:
            node = self.objnode[obj_label]['node']
            node.mesh.is_visible = False
        for obj_label in obj_pose_dict:
            if obj_label in self.object_dict:
                node = self.objnode[obj_label]['node']
                node.mesh.is_visible = True
                depth = self.render.render(self.scene, flags=pyrender.constants.RenderFlags.DEPTH_ONLY)
                node.mesh.is_visible = False
                mask = np.logical_and(
                    (np.abs(depth - full_depth) < 1e-6), np.abs(full_depth) > 0.2
                )
                if np.sum(mask) > 0:
                    color = self.cmp[obj_label - 1]
                    img[mask, :] = alpha * img[mask, :] + (1 - alpha) * color[:]
                    obj_pose = obj_pose_dict[obj_label]
                    obj_center = self.project2d(self.intrinsic, obj_pose[:3, -1])
                    rgb_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
                    for j in range(3):
                        obj_xyz_offset_2d = self.project2d(self.intrinsic, obj_pose[:3, -1] + obj_pose[:3, j] * 0.001)
                        obj_axis_endpoint = obj_center + (obj_xyz_offset_2d - obj_center) / np.linalg.norm(
                            obj_xyz_offset_2d - obj_center) * axis_len
                        cv2.arrowedLine(img, (int(obj_center[0]), int(obj_center[1])),
                                        (int(obj_axis_endpoint[0]), int(obj_axis_endpoint[1])), rgb_colors[j],
                                        thickness=2, tipLength=0.15)
        return img

    def color_map(self, N=256, normalized=False):
        def bitget(byteval, idx):
            return ((byteval & (1 << idx)) != 0)

        dtype = 'float32' if normalized else 'uint8'
        cmap = np.zeros((N, 3), dtype=dtype)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7 - j)
                g = g | (bitget(c, 1) << 7 - j)
                b = b | (bitget(c, 2) << 7 - j)
                c = c >> 3
            cmap[i] = np.array([r, g, b])
        cmap = cmap / 255 if normalized else cmap
        return cmap

    def project2d(self, intrinsic, point3d):
        return (intrinsic @ (point3d / point3d[2]))[:2]



