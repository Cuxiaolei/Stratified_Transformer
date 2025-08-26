import os
import time
import random
import numpy as np
import logging
import pickle
import argparse
import collections

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import csv
from util import config, transform
from util.common_util import AverageMeter, intersectionAndUnion, check_makedirs
from util.voxelize import voxelize
import torch_points_kernels as tp
import torch.nn.functional as F
from util.my_dataset import MyDataset
random.seed(123)
np.random.seed(123)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Classification / Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/s3dis/s3dis_pointweb.yaml', help='config file')
    parser.add_argument('opts', help='see config/s3dis/s3dis_pointweb.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def main():
    global args, logger
    args = get_parser()
    logger = get_logger()
    logger.info(args)
    assert args.classes > 1
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    # get model
    if args.arch == 'stratified_transformer':

        from model.stratified_transformer import Stratified

        args.patch_size = args.grid_size * args.patch_size
        args.window_size = [args.patch_size * args.window_size * (2**i) for i in range(args.num_layers)]
        args.grid_sizes = [args.patch_size * (2**i) for i in range(args.num_layers)]
        args.quant_sizes = [args.quant_size * (2**i) for i in range(args.num_layers)]

        model = Stratified(args.downsample_scale, args.depths, args.channels, args.num_heads, args.window_size, \
            args.up_k, args.grid_sizes, args.quant_sizes, rel_query=args.rel_query, \
            rel_key=args.rel_key, rel_value=args.rel_value, drop_path_rate=args.drop_path_rate, concat_xyz=args.concat_xyz, num_classes=args.classes, \
            ratio=args.ratio, k=args.k, prev_grid_size=args.grid_size, sigma=1.0, num_layers=args.num_layers, stem_transformer=args.stem_transformer, in_channels=args.in_channels)

    elif args.arch == 'swin3d_transformer':

        from model.swin3d_transformer import Swin

        args.patch_size = args.grid_size * args.patch_size
        args.window_sizes = [args.patch_size * args.window_size * (2**i) for i in range(args.num_layers)]
        args.grid_sizes = [args.patch_size * (2**i) for i in range(args.num_layers)]
        args.quant_sizes = [args.quant_size * (2**i) for i in range(args.num_layers)]

        model = Swin(args.depths, args.channels, args.num_heads, \
            args.window_sizes, args.up_k, args.grid_sizes, args.quant_sizes, rel_query=args.rel_query, \
            rel_key=args.rel_key, rel_value=args.rel_value, drop_path_rate=args.drop_path_rate, \
            concat_xyz=args.concat_xyz, num_classes=args.classes, \
            ratio=args.ratio, k=args.k, prev_grid_size=args.grid_size, sigma=1.0, num_layers=args.num_layers, stem_transformer=args.stem_transformer, in_channels=args.in_channels)

    else:
        raise Exception('architecture {} not supported yet'.format(args.arch))

    model = model.cuda()

    #model = torch.nn.DataParallel(model.cuda())
    logger.info(model)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
    names = ['铁塔', '背景', '导线']
    if os.path.isfile(args.model_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        state_dict = checkpoint['state_dict']
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name.replace("item", "stem")] = v
        model.load_state_dict(new_state_dict, strict=True)
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))
        args.epoch = checkpoint['epoch']
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))


    # transform
    test_transform_set = []
    test_transform_set.append(None) # for None aug
    test_transform_set.append(None) # for permutate

    # aug 90
    logger.info("augmentation roate")
    logger.info("rotate_angle: {}".format(90))
    test_transform = transform.RandomRotate(rotate_angle=90, along_z=args.get('rotate_along_z', True))
    test_transform_set.append(test_transform)

    # aug 180
    logger.info("augmentation roate")
    logger.info("rotate_angle: {}".format(180))
    test_transform = transform.RandomRotate(rotate_angle=180, along_z=args.get('rotate_along_z', True))
    test_transform_set.append(test_transform)

    # aug 270
    logger.info("augmentation roate")
    logger.info("rotate_angle: {}".format(270))
    test_transform = transform.RandomRotate(rotate_angle=270, along_z=args.get('rotate_along_z', True))
    test_transform_set.append(test_transform)

    if args.data_name == 's3dis':

        # shift +0.2
        test_transform = transform.RandomShift_test(shift_range=0.2)
        test_transform_set.append(test_transform)

        # shift -0.2
        test_transform = transform.RandomShift_test(shift_range=-0.2)
        test_transform_set.append(test_transform)

    test_transform = None  # 无增强，仅必要预处理
    test(model, criterion, names, test_transform)


def data_prepare():
    if args.data_name == 's3dis':
        data_list = sorted(os.listdir(args.data_root))
        data_list = [item[:-4] for item in data_list if 'Area_{}'.format(args.test_area) in item]
    elif args.data_name == 'scannetv2':
        data_list = sorted(os.listdir(args.data_root_val))
        data_list = [item[:-4] for item in data_list if '.pth' in item]
    # ---------------------- 新增：my_dataset 分支 ----------------------
    elif args.data_name == 'my_dataset':
        # 读取 test_scenes.txt 划分文件（与训练集逻辑一致）
        val_split_file = os.path.join(args.data_root, 'test_scenes.txt')
        if not os.path.exists(val_split_file):
            raise FileNotFoundError(f"my_dataset 测试集划分文件 {val_split_file} 不存在！")
        # 加载样本路径列表（每行是 .npy 文件的相对路径，如 "merged/sample_1.npy"）
        with open(val_split_file, 'r') as f:
            data_list = [line.strip() for line in f.readlines()]
        # 提取样本名称（去掉路径和后缀，用于后续保存预测结果，如 "sample_1"）
        data_list = [os.path.splitext(os.path.basename(path))[0] for path in data_list]
    # -------------------------------------------------------------------
    else:
        raise Exception('dataset {} not supported yet'.format(args.data_name))
    print("Totally {} samples in val set.".format(len(data_list)))
    return data_list


def data_load(data_name, transform):
    # data_name：从 data_prepare 传来的样本名称（如 "sample_1"）
    # 需拼接回完整 .npy 路径（与 val_scenes.txt 中的路径一致）
    if args.data_name == 's3dis':
        data_path = os.path.join(args.data_root, data_name + '.npy')
        data = np.load(data_path)  # xyzrgbl, N*7
        coord, feat, label = data[:, :3], data[:, 3:6], data[:, 6]
    elif args.data_name == 'scannetv2':
        data_path = os.path.join(args.data_root_val, data_name + '.pth')
        data = torch.load(data_path)  # xyzrgbl, N*7
        coord, feat, label = data[0], data[1], data[2]
    # ---------------------- 新增：my_dataset 分支 ----------------------
    elif args.data_name == 'my_dataset':
        # 1. 拼接完整数据路径（需与 val_scenes.txt 中的路径匹配）
        # 示例：val_scenes.txt 中是 "merged/sample_1.npy"，则拼接为 args.data_root/merged/sample_1.npy
        # 先从 val_scenes.txt 重新读取完整路径（避免 data_name 仅含文件名）
        val_split_file = os.path.join(args.data_root, 'test_scenes.txt')
        with open(val_split_file, 'r') as f:
            full_paths = [line.strip() for line in f.readlines()]
        # 根据 data_name（如 "sample_1"）找到对应的完整路径
        data_path = None
        for path in full_paths:
            if data_name == os.path.splitext(os.path.basename(path))[0]:
                data_path = os.path.join(args.data_root, path)
                break
        if data_path is None or not os.path.exists(data_path):
            raise FileNotFoundError(f"未找到 my_dataset 样本 {data_name}，路径 {data_path} 无效！")

        # 2. 加载 10 通道数据并提取维度
        data = np.load(data_path)  # shape: [N, 10]（xyz3 + rgb3 + 法向量3 + label1）
        coord = data[:, 0:3]  # 坐标：[N, 3]
        feat = data[:, 3:9]  # 特征：[N, 6]（rgb3 + 法向量3）
        label = data[:, 9]  # 标签：[N]（整数类型）
    # -------------------------------------------------------------------

    # # 数据增强（验证集增强如旋转，与训练集逻辑一致）
    # if transform:
    #     # 注意：需确保 transform 函数支持 6 维 feat（若增强仅影响 coord，可保持不变）
    #     coord, feat = transform(coord, feat)

    # 体素化（复用原有逻辑，与训练集 voxel_size 保持一致）
    idx_data = []
    if args.voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        idx_sort, count = voxelize(coord, args.voxel_size, mode=1)
        for i in range(count.max()):
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
            idx_part = idx_sort[idx_select]
            idx_data.append(idx_part)
    else:
        idx_data.append(np.arange(label.shape[0]))
    return coord, feat, label, idx_data


# test.py #startLine: 232 #endLine: 247（修改后）
def input_normalize(coord, feat):
    # 坐标归一化（与训练一致：平移到原点）
    coord_min = np.min(coord, 0)
    coord -= coord_min

    # 特征归一化（与训练逻辑对齐）
    if args.data_name == 's3dis':
        feat = feat / 255.  # S3DIS 特征全为颜色，整体归一化
    # 统一 my_dataset 归一化逻辑
    elif args.data_name == 'my_dataset':
        # 仅颜色通道（前3列）归一化，法向量（后3列）不处理
        feat = feat.copy()  # 避免修改原数组
        feat[:, 0:3] = feat[:, 0:3] / 255.0  # 颜色 0-255 → 0-1
        # 法向量保持原始值，不做归一化
    return coord, feat

def create_test_dataset(transform):
    """创建测试数据集（复用MyDataset）"""
    if args.data_name == 'my_dataset':
        return MyDataset(
            split='test',  # 对应test_scenes.txt
            data_root=args.data_root,
            transform=transform,  # 传入当前增强策略
            voxel_size=args.voxel_size,  # 与训练保持一致
            voxel_max=args.voxel_max,  # 与训练保持一致
            shuffle_index=True,
            loop=1  # 测试集仅遍历1次
        )
    else:
        raise Exception(f'dataset {args.data_name} not supported yet')



# test.py (不进行数据增强的版本)
def test(model, criterion, names, test_transform = None):  # 修改参数，仅接收单一transform
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    args.batch_size_test = 1  # 点云数据通常单样本加载
    model.eval()

    check_makedirs(args.save_folder)
    pred_save, label_save = [], []

    # 1. 创建单一数据集（不使用数据增强，仅必要预处理）
    dataset = create_test_dataset(test_transform)  # 单一数据集
    sample_names = data_prepare()  # 复用该函数获取txt中的样本名称列表
    total_samples = len(dataset)
    logger.info(f"Totally {len(sample_names)} samples in test set.")

    # 2. 遍历每个样本
    for sample_idx in range(total_samples):
        item = sample_names[sample_idx]
        end = time.time()
        pred_save_path = os.path.join(args.save_folder, f'{item}_{args.epoch}_pred.npy')
        label_save_path = os.path.join(args.save_folder, f'{item}_{args.epoch}_label.npy')

        if os.path.isfile(pred_save_path) and os.path.isfile(label_save_path):
            logger.info(f'{sample_idx + 1}/{total_samples}: {item}, 已加载现有预测结果')
            pred, label = np.load(pred_save_path), np.load(label_save_path)
        else:
            # 3. 直接加载原始数据（不进行增强）
            coord, feat, label = dataset[sample_idx]  # 调用MyDataset的__getitem__
            label = label.type(torch.int64)  # Tensor类型转换

            # 4. 处理点云分块（如果需要，根据voxel_max拆分）
            idx_data = []
            if args.voxel_max and coord.shape[0] > args.voxel_max:
                # 随机分块（保持与原有逻辑一致）
                coord_p = np.random.rand(coord.shape[0]) * 1e-3
                idx_uni = np.array([])
                while idx_uni.size < coord.shape[0]:
                    init_idx = np.argmin(coord_p)
                    dist = np.sum(np.power(coord - coord[init_idx], 2), 1)
                    idx_crop = np.argsort(dist)[:args.voxel_max]
                    idx_uni = np.unique(np.concatenate((idx_uni, idx_crop))).astype(np.int64)
                    idx_data.append(idx_crop)
            else:
                idx_data.append(np.arange(coord.shape[0]))

            # 5. 模型推理
            pred = np.zeros((label.shape[0], args.classes), dtype=np.float32)
            for idx_part in idx_data:
                coord_part = coord[idx_part]
                feat_part = feat[idx_part]

                # 转换为Tensor并移至GPU
                coord_part = torch.FloatTensor(coord_part).cuda(non_blocking=True)
                feat_part = torch.FloatTensor(feat_part).cuda(non_blocking=True)
                offset_part = torch.IntTensor([len(coord_part)]).cuda(non_blocking=True)
                batch = torch.zeros(len(coord_part), dtype=torch.long).cuda(non_blocking=True)

                # 计算邻域（与原有逻辑一致）
                sigma = 1.0
                radius = 2.5 * args.grid_size * sigma
                neighbor_idx = tp.ball_query(
                    radius, args.max_num_neighbors,
                    coord_part, coord_part,
                    mode="partial_dense",
                    batch_x=batch, batch_y=batch
                )[0].cuda(non_blocking=True)

                # 拼接坐标（如果需要）
                if args.concat_xyz:
                    feat_part = torch.cat([feat_part, coord_part], dim=1)

                # 模型预测
                with torch.no_grad():
                    pred_part = model(feat_part, coord_part, offset_part, batch, neighbor_idx)
                    pred_part = F.softmax(pred_part, dim=-1).cpu().numpy()

                pred[idx_part] += pred_part
                torch.cuda.empty_cache()

            # 无需增强结果融合，直接使用原始预测
            loss = criterion(
                torch.FloatTensor(pred).cuda(),
                torch.LongTensor(label).cuda(non_blocking=True)
            )  # 参考损失
            pred = pred.argmax(1)  # 取概率最大的类别

        # 6. 计算指标
        intersection, union, target = intersectionAndUnion(
            pred, label, args.classes, args.ignore_label
        )
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)

        accuracy = sum(intersection) / (sum(target) + 1e-10)
        batch_time.update(time.time() - end)
        logger.info(
            f'Test: [{sample_idx + 1}/{total_samples}]-{label.size} '
            f'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
            f'Accuracy {accuracy:.4f}.'
        )

        # 7. 保存结果
        pred_save.append(pred)
        label_save.append(label)
        if not os.path.isfile(pred_save_path):
            np.save(pred_save_path, pred)
        if not os.path.isfile(label_save_path):
            np.save(label_save_path, label)

    # 8. 计算整体指标
    if not os.path.exists(os.path.join(args.save_folder, "pred.pickle")):
        with open(os.path.join(args.save_folder, "pred.pickle"), 'wb') as handle:
            pickle.dump({'pred': pred_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if not os.path.exists(os.path.join(args.save_folder, "label.pickle")):
        with open(os.path.join(args.save_folder, "label.pickle"), 'wb') as handle:
            pickle.dump({'label': label_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # 计算指标（与原有逻辑一致）
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU1 = np.mean(iou_class)
    mAcc1 = np.mean(accuracy_class)
    allAcc1 = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    intersection, union, target = intersectionAndUnion(
        np.concatenate(pred_save), np.concatenate(label_save),
        args.classes, args.ignore_label
    )
    iou_class = intersection / (union + 1e-10)
    accuracy_class = intersection / (target + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection) / (sum(target) + 1e-10)

    logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    logger.info('Val1 result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU1, mAcc1, allAcc1))

    for i in range(args.classes):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i],
                                                                                    names[i]))
        # 新增：保存测试指标到CSV
        test_csv_path = os.path.join(args.save_folder, 'test_metrics.csv')
        with open(test_csv_path, 'w', newline='') as f:
            csv_writer = csv.writer(f)
            # 表头：整体指标 + 每类iou + 每类acc + 每类名称
            header = ['mIoU', 'mAcc', 'allAcc']
            for i in range(args.classes):
                header.extend([f'class_{i}_iou', f'class_{i}_acc', f'class_{i}_name'])
            csv_writer.writerow(header)

            # 行数据：整体指标 + 每类iou + 每类acc + 每类名称
            row = [mIoU, mAcc, allAcc]
            for i in range(args.classes):
                row.append(iou_class[i])
                row.append(accuracy_class[i])
                row.append(names[i] if i < len(names) else f'class_{i}')  # 类别名称
            csv_writer.writerow(row)
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


if __name__ == '__main__':
    main()
