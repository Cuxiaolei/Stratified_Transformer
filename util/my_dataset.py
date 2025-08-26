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

        # 核心修改：根据split控制是否下采样
        # 测试时(split='test')不进行下采样，训练和验证时正常处理
        if self.split == 'test':
            # 测试模式：仅进行必要的预处理，跳过下采样
            processed_coord, processed_feat, processed_label = data_prepare(
                coord=coord,
                feat=feat,
                label=label,
                split=self.split,
                voxel_size=self.voxel_size,
                voxel_max=None,  # 强制禁用下采样
                transform=None,  # 测试时不使用数据增强
                shuffle_index=False  # 测试时保持点顺序
            )
        else:
            # 训练/验证模式：正常进行下采样和数据增强
            processed_coord, processed_feat, processed_label = data_prepare(
                coord=coord,
                feat=feat,
                label=label,
                split=self.split,
                voxel_size=self.voxel_size,
                voxel_max=self.voxel_max,  # 使用指定的下采样阈值
                transform=self.transform,
                shuffle_index=self.shuffle_index
            )

        # 处理特征数据
        if isinstance(processed_feat, np.ndarray):
            processed_feat = processed_feat.astype(np.float32)
        elif isinstance(processed_feat, torch.Tensor):
            processed_feat = processed_feat.type(torch.float32)
        else:
            raise TypeError(f"不支持的特征数据类型: {type(processed_feat)}")

        # 处理坐标数据
        if isinstance(processed_coord, np.ndarray):
            processed_coord = processed_coord.astype(np.float32)
        elif isinstance(processed_coord, torch.Tensor):
            processed_coord = processed_coord.type(torch.float32)
        else:
            raise TypeError(f"不支持的坐标数据类型: {type(processed_coord)}")

        # 转换为Tensor（如果还不是的话）
        if isinstance(processed_coord, np.ndarray):
            processed_coord = torch.from_numpy(processed_coord).float()
        if isinstance(processed_feat, np.ndarray):
            processed_feat = torch.from_numpy(processed_feat).float()
        if isinstance(processed_label, np.ndarray):
            processed_label = torch.from_numpy(processed_label).long()

        print(f"feat shape before model: {processed_feat.shape}")
        return processed_coord, processed_feat, processed_label

    def __len__(self):
        return len(self.scene_dirs) * self.loop
