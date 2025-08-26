import os
import numpy as np
import torch
from torch.utils.data import Dataset
from util.data_util import data_prepare  # 复用现有预处理工具


class MyDataset(Dataset):
    def __init__(self, split='train', data_root=None, transform=None,
                 voxel_size=0.04, voxel_max=None, shuffle_index=True, loop=1):
        super().__init__()
        # 新增：打印实际访问的split目录路径
        split_dir = os.path.join(data_root, split)
        print(f"正在查找 {split} 数据的目录: {split_dir}")  # 关键日志
        if not os.path.isdir(split_dir):
            raise NotADirectoryError(f"划分目录 {split_dir} 不存在")
        self.split = split
        self.data_root = data_root  # 数据集根目录（包含train/val/test文件夹）
        self.transform = transform  # 数据增强
        self.voxel_size = voxel_size  # 体素化参数
        self.voxel_max = voxel_max  # 最大点数量限制
        self.shuffle_index = shuffle_index  # 是否打乱点顺序
        self.loop = loop  # 训练时重复数据集的次数

        # 构建场景路径列表（每个场景是一个文件夹）
        self.scene_dirs = []
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            raise NotADirectoryError(f"划分目录 {split_dir} 不存在")

        # 获取所有场景文件夹
        for scene in os.listdir(split_dir):
            scene_path = os.path.join(split_dir, scene)
            if os.path.isdir(scene_path):
                # 检查必要文件是否存在
                required_files = ['color.npy', 'coord.npy', 'normal.npy', 'segment20.npy']
                if all(os.path.exists(os.path.join(scene_path, f)) for f in required_files):
                    self.scene_dirs.append(scene_path)
                else:
                    print(f"警告: 场景 {scene} 缺少必要文件，已跳过")

        if not self.scene_dirs:
            raise ValueError(f"在 {split_dir} 中未找到有效的场景数据")

        print(f"Totally {len(self.scene_dirs)} samples in {split} set.")

    def __getitem__(self, idx):
        # 循环索引（处理loop参数）
        data_idx = idx % len(self.scene_dirs)
        scene_dir = self.scene_dirs[data_idx]

        # 加载各个独立的npy文件
        coord = np.load(os.path.join(scene_dir, 'coord.npy'))  # 坐标 (N, 3)
        color = np.load(os.path.join(scene_dir, 'color.npy'))  # 颜色 (N, 3)
        normal = np.load(os.path.join(scene_dir, 'normal.npy'))  # 法向量 (N, 3)
        label = np.load(os.path.join(scene_dir, 'segment20.npy'))  # 标签 (N,)

        # 确保所有数组的点数量一致
        num_points = coord.shape[0]
        assert color.shape[0] == num_points, f"颜色点数量与坐标不匹配 in {scene_dir}"
        assert normal.shape[0] == num_points, f"法向量点数量与坐标不匹配 in {scene_dir}"
        assert label.shape[0] == num_points, f"标签点数量与坐标不匹配 in {scene_dir}"

        # 组合特征（颜色 + 法向量）
        feat = np.concatenate([color, normal], axis=1)  # (N, 6)

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

        # 转换为Tensor（如果还不是的话）
        if isinstance(coord, np.ndarray):
            coord = torch.from_numpy(coord).float()
        if isinstance(feat, np.ndarray):
            feat = torch.from_numpy(feat).float()
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label).long()

        print(f"feat shape before model: {feat.shape}")  # 确保最后一维是固定值（如6或3）
        return coord, feat, label

    def __len__(self):
        return len(self.scene_dirs) * self.loop
