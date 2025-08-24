import os
import numpy as np
import torch
from torch.utils.data import Dataset
from util.data_util import data_prepare  # 复用现有预处理工具


class MyDataset(Dataset):
    def __init__(self, split='train', data_root=None, transform=None,
                 voxel_size=0.04, voxel_max=None, shuffle_index=True, loop=1):
        super().__init__()
        self.split = split
        self.data_root = data_root  # 数据集根目录（包含merged文件夹和txt文件）
        self.transform = transform  # 数据增强
        self.voxel_size = voxel_size  # 体素化参数
        self.voxel_max = voxel_max  # 最大点数量限制
        self.shuffle_index = shuffle_index  # 是否打乱点顺序
        self.loop = loop  # 训练时重复数据集的次数

        # 读取划分文件（train/val/test_scenes.txt）
        split_file = os.path.join(data_root, f'{split}_scenes.txt')
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"划分文件 {split_file} 不存在")

        # 加载样本路径列表
        with open(split_file, 'r') as f:
            self.data_list = [line.strip() for line in f.readlines()]
        # 补充完整路径（如果txt中是相对路径）
        self.data_list = [os.path.join(data_root, path) for path in self.data_list]

        print(f"Totally {len(self.data_list)} samples in {split} set.")

    def __getitem__(self, idx):
        # 循环索引（处理loop参数）
        data_idx = idx % len(self.data_list)
        data_path = self.data_list[data_idx]

        # 加载.npy文件（10通道：坐标3 + 颜色3 + 法向量3 + 标签1）
        data = np.load(data_path)  # shape: (N, 10)

        # 提取坐标、特征、标签
        coord = data[:, 0:3]  # 前3列：xyz坐标
        feat = data[:, 3:9]  # 中间6列：颜色(3) + 法向量(3)
        label = data[:, 9]  # 最后1列：标签（0/1/2）

        # 数据预处理（体素化、裁剪、增强等，复用现有工具）
        coord, feat, label = data_prepare(
            coord=coord,
            feat=feat,
            label=label,
            split=self.split,
            voxel_size=self.voxel_size,
            voxel_max=self.voxel_max,
            transform=self.transform,
            shuffle_index=self.shuffle_index
        )

        # 处理特征数据
        if isinstance(feat, np.ndarray):
            feat = feat.astype(np.float32)  # NumPy数组用astype
        elif isinstance(feat, torch.Tensor):
            feat = feat.type(torch.float32)  # Tensor用type
        else:
            raise TypeError(f"不支持的特征数据类型: {type(feat)}")

        # 处理坐标数据
        if isinstance(coord, np.ndarray):
            coord = coord.astype(np.float32)  # NumPy数组用astype
        elif isinstance(coord, torch.Tensor):
            coord = coord.type(torch.float32)  # Tensor用type
        else:
            raise TypeError(f"不支持的坐标数据类型: {type(coord)}")

        if isinstance(coord, np.ndarray):
            coord = torch.from_numpy(coord).float()
        if isinstance(feat, np.ndarray):
            feat = torch.from_numpy(feat).float()

        return coord, feat, label

    def __len__(self):
        return len(self.data_list) * self.loop