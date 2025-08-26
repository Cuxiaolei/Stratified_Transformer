import os
import time
import random
import numpy as np
import logging
import pickle
import argparse
import collections
import csv

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn.functional as F

import torch_points_kernels as tp

from util import config, transform
from util.common_util import AverageMeter, intersectionAndUnion, check_makedirs
from util.voxelize import voxelize
from util.my_dataset import MyDataset  # 导入数据加载器

# 设置随机种子，确保结果可复现
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation Testing')
    parser.add_argument('--config', type=str, default='config/s3dis/s3dis_pointweb.yaml', help='config file')
    parser.add_argument('opts', help='see config/s3dis/s3dis_pointweb.yaml for all options', default=None,
                        nargs=argparse.REMAINDER)
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


def data_prepare():
    """获取测试场景列表，与MyDataset保持一致的场景识别逻辑"""
    if args.data_name == 's3dis':
        data_list = sorted(os.listdir(args.data_root))
        data_list = [item[:-4] for item in data_list if 'Area_{}'.format(args.test_area) in item]
    elif args.data_name == 'scannetv2':
        data_list = sorted(os.listdir(args.data_root_val))
        data_list = [item[:-4] for item in data_list if '.pth' in item]
    # 针对my_dataset的处理逻辑：与MyDataset保持一致
    elif args.data_name == 'my_dataset':
        # 构建测试数据目录路径
        test_dir = os.path.join(args.data_root, 'test')
        if not os.path.isdir(test_dir):
            raise NotADirectoryError(f"my_dataset测试文件夹 {test_dir} 不存在！")

        # 获取所有有效的场景文件夹（与MyDataset逻辑一致）
        data_list = []
        for scene in os.listdir(test_dir):
            scene_path = os.path.join(test_dir, scene)
            if os.path.isdir(scene_path):
                # 检查必要文件是否存在（与MyDataset保持一致）
                required_files = ['color.npy', 'coord.npy', 'normal.npy', 'segment20.npy']
                if all(os.path.exists(os.path.join(scene_path, f)) for f in required_files):
                    data_list.append(scene)
                else:
                    logger.warning(f"场景 {scene} 缺少必要文件，已跳过")

        # 验证是否找到有效场景
        if not data_list:
            raise Exception(f"在 {test_dir} 下未找到任何有效的场景文件夹！")

        logger.info(f"测试集有效场景文件夹总数: {len(data_list)}")
    else:
        raise Exception(f'数据集 {args.data_name} 不支持')

    return data_list


def main():
    global args, logger
    args = get_parser()
    logger = get_logger()
    logger.info(args)
    assert args.classes > 1
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    # 加载模型
    if args.arch == 'stratified_transformer':
        from model.stratified_transformer import Stratified
        args.patch_size = args.grid_size * args.patch_size
        args.window_size = [args.patch_size * args.window_size * (2 ** i) for i in range(args.num_layers)]
        args.grid_sizes = [args.patch_size * (2 ** i) for i in range(args.num_layers)]
        args.quant_sizes = [args.quant_size * (2 ** i) for i in range(args.num_layers)]

        model = Stratified(
            args.downsample_scale, args.depths, args.channels, args.num_heads,
            args.window_size, args.up_k, args.grid_sizes, args.quant_sizes,
            rel_query=args.rel_query, rel_key=args.rel_key, rel_value=args.rel_value,
            drop_path_rate=args.drop_path_rate, concat_xyz=args.concat_xyz,
            num_classes=args.classes, ratio=args.ratio, k=args.k,
            prev_grid_size=args.grid_size, sigma=1.0, num_layers=args.num_layers,
            stem_transformer=args.stem_transformer, in_channels=args.in_channels
        )

    elif args.arch == 'swin3d_transformer':
        from model.swin3d_transformer import Swin
        args.patch_size = args.grid_size * args.patch_size
        args.window_sizes = [args.patch_size * args.window_size * (2 ** i) for i in range(args.num_layers)]
        args.grid_sizes = [args.patch_size * (2 ** i) for i in range(args.num_layers)]
        args.quant_sizes = [args.quant_size * (2 ** i) for i in range(args.num_layers)]

        model = Swin(
            args.depths, args.channels, args.num_heads, args.window_sizes,
            args.up_k, args.grid_sizes, args.quant_sizes, rel_query=args.rel_query,
            rel_key=args.rel_key, rel_value=args.rel_value, drop_path_rate=args.drop_path_rate,
            concat_xyz=args.concat_xyz, num_classes=args.classes, ratio=args.ratio,
            k=args.k, prev_grid_size=args.grid_size, sigma=1.0, num_layers=args.num_layers,
            stem_transformer=args.stem_transformer, in_channels=args.in_channels
        )

    else:
        raise Exception(f'Architecture {args.arch} not supported yet')

    model = model.cuda()
    logger.info(model)

    # 损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()

    # 类别名称
    class_names = ['铁塔', '背景', '导线']
    if len(class_names) != args.classes:
        logger.warning(f"类别名称数量({len(class_names)})与类别数量({args.classes})不匹配")
        class_names = [f'类别_{i}' for i in range(args.classes)]

    # 加载最佳模型
    if os.path.isfile(args.model_path):
        logger.info(f"=> 加载模型 checkpoint '{args.model_path}'")
        checkpoint = torch.load(args.model_path)
        state_dict = checkpoint['state_dict']
        new_state_dict = collections.OrderedDict()

        # 调整参数名以适配模型结构
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k  # 移除可能的module前缀
            new_state_dict[name.replace("item", "stem")] = v

        model.load_state_dict(new_state_dict, strict=True)
        logger.info(f"=> 成功加载 checkpoint '{args.model_path}' (epoch {checkpoint['epoch']})")
        args.epoch = checkpoint['epoch']
    else:
        raise RuntimeError(f"=> 未找到模型 checkpoint '{args.model_path}'")

    # 创建测试数据集（与MyDataset完美适配）
    test_dataset = MyDataset(
        split='test',
        data_root=args.data_root,
        transform=None,  # 测试时不使用数据增强
        voxel_size=args.voxel_size,
        voxel_max=args.voxel_max,
        shuffle_index=False,  # 测试时不打乱索引
        loop=1,
    )

    # 获取样本名称列表（场景文件夹名称）
    sample_names = data_prepare()

    # 验证数据加载器加载的样本数与场景文件夹数是否一致
    if len(test_dataset) != len(sample_names):
        logger.warning(f"数据加载器样本数({len(test_dataset)})与场景文件夹数({len(sample_names)})不匹配")
    else:
        logger.info(f"数据加载器与场景文件夹数量匹配: {len(test_dataset)} 个样本")

    # 测试
    test(model, criterion, class_names, test_dataset, sample_names)


def input_normalize(coord, feat):
    """数据归一化，与训练时保持一致"""
    # 对于MyDataset，数据已经是Tensor格式，直接在GPU上处理以提高效率
    if isinstance(coord, np.ndarray):
        coord = torch.from_numpy(coord).float()
    if isinstance(feat, np.ndarray):
        feat = torch.from_numpy(feat).float()

    # 坐标归一化：平移到原点
    coord_min = torch.min(coord, dim=0)[0]
    coord = coord - coord_min

    # 特征归一化（与MyDataset数据格式匹配）
    if args.data_name == 'my_dataset':
        # 颜色通道(前3列)归一化到0-1
        feat[:, 0:3] = feat[:, 0:3] / 255.0
        # 法向量(后3列)保持不变
    elif args.data_name == 's3dis':
        # S3DIS数据集特征归一化
        feat = feat / 255.0

    return coord, feat


def test(model, criterion, class_names, test_dataset, sample_names):
    """测试函数，优化与MyDataset的交互"""
    logger.info('>>>>>>>>>>>>>>>> 开始评估 >>>>>>>>>>>>>>>>')

    # 初始化指标计算器
    batch_time = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    # 设置模型为评估模式
    model.eval()

    # 创建保存结果的文件夹
    check_makedirs(args.save_folder)

    # 存储所有预测和标签
    all_preds = []
    all_labels = []

    total_samples = len(test_dataset)
    logger.info(f"测试集样本总数: {total_samples}")

    # 遍历每个场景
    for sample_idx in range(total_samples):
        scene_name = sample_names[sample_idx]  # 使用场景文件夹名称
        start_time = time.time()

        # 结果保存路径
        pred_save_path = os.path.join(args.save_folder, f'{scene_name}_pred.npy')
        label_save_path = os.path.join(args.save_folder, f'{scene_name}_label.npy')

        # 如果已经有预测结果，直接加载
        if os.path.exists(pred_save_path) and os.path.exists(label_save_path):
            logger.info(f'[{sample_idx + 1}/{total_samples}]: 场景 {scene_name}, 已加载现有预测结果')
            pred = np.load(pred_save_path)
            label = np.load(label_save_path)
        else:
            # 从MyDataset获取数据，已适配其返回格式(coord, feat, label)
            coord, feat, label = test_dataset[sample_idx]

            # 移至CPU并转为numpy用于后续处理
            label_np = label.cpu().numpy()
            num_points = label_np.shape[0]
            logger.info(f"场景 {scene_name} 包含 {num_points} 个点")

            # 数据归一化（保持与训练一致）
            coord, feat = input_normalize(coord, feat)

            # 处理点云分块（如果点云过大）
            idx_data = []
            if args.voxel_max and num_points > args.voxel_max:
                # 随机分块处理
                coord_np = coord.cpu().numpy()
                coord_p = np.random.rand(num_points) * 1e-3
                idx_uni = np.array([])
                while idx_uni.size < num_points:
                    init_idx = np.argmin(coord_p)
                    dist = np.sum(np.power(coord_np - coord_np[init_idx], 2), 1)
                    idx_crop = np.argsort(dist)[:args.voxel_max]
                    idx_uni = np.unique(np.concatenate((idx_uni, idx_crop))).astype(np.int64)
                    idx_data.append(idx_crop)
            else:
                idx_data.append(np.arange(num_points))

            # 模型推理
            pred = np.zeros((num_points, args.classes), dtype=np.float32)
            for idx_part in idx_data:
                # 获取点云子集
                coord_part = coord[idx_part].cuda(non_blocking=True)
                feat_part = feat[idx_part].cuda(non_blocking=True)
                offset_part = torch.IntTensor([len(coord_part)]).cuda(non_blocking=True)
                batch = torch.zeros(len(coord_part), dtype=torch.long).cuda(non_blocking=True)

                # 计算邻域
                sigma = 1.0
                radius = 2.5 * args.grid_size * sigma
                neighbor_idx = tp.ball_query(
                    radius, args.max_num_neighbors,
                    coord_part, coord_part,
                    mode="partial_dense",
                    batch_x=batch, batch_y=batch
                )[0].cuda(non_blocking=True)

                # 如果需要拼接坐标信息
                if args.concat_xyz:
                    feat_part = torch.cat([feat_part, coord_part], dim=1)

                # 模型预测
                with torch.no_grad():
                    pred_part = model(feat_part, coord_part, offset_part, batch, neighbor_idx)
                    pred_part = F.softmax(pred_part, dim=-1).cpu().numpy()

                # 累加预测结果
                pred[idx_part] += pred_part
                torch.cuda.empty_cache()  # 清理GPU缓存

            # 获取最终预测类别
            pred = pred.argmax(1)

            # 保存预测结果
            np.save(pred_save_path, pred)
            np.save(label_save_path, label_np)
            label = label_np  # 统一变量格式

        # 计算评估指标
        intersection, union, target = intersectionAndUnion(
            pred, label, args.classes, args.ignore_label
        )
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)

        # 计算准确率
        accuracy = sum(intersection) / (sum(target) + 1e-10)
        batch_time.update(time.time() - start_time)

        logger.info(
            f'测试进度: [{sample_idx + 1}/{total_samples}]-场景 {scene_name} ({label.size}个点) '
            f'耗时 {batch_time.val:.3f}s (平均 {batch_time.avg:.3f}s) '
            f'准确率 {accuracy:.4f}'
        )

        # 保存所有预测和标签
        all_preds.append(pred)
        all_labels.append(label)

    # 保存所有预测和标签的汇总
    with open(os.path.join(args.save_folder, "pred.pickle"), 'wb') as handle:
        pickle.dump({'pred': all_preds}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(args.save_folder, "label.pickle"), 'wb') as handle:
        pickle.dump({'label': all_labels}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # 计算整体评估指标
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU1 = np.mean(iou_class)
    mAcc1 = np.mean(accuracy_class)
    allAcc1 = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    # 合并所有预测和标签计算指标
    combined_pred = np.concatenate(all_preds)
    combined_label = np.concatenate(all_labels)
    intersection, union, target = intersectionAndUnion(
        combined_pred, combined_label, args.classes, args.ignore_label
    )

    iou_class = intersection / (union + 1e-10)
    accuracy_class = intersection / (target + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection) / (sum(target) + 1e-10)

    # 输出评估结果
    logger.info(f'验证结果: mIoU/mAcc/allAcc {mIoU:.4f}/{mAcc:.4f}/{allAcc:.4f}')
    logger.info(f'验证结果1: mIoU/mAcc/allAcc {mIoU1:.4f}/{mAcc1:.4f}/{allAcc1:.4f}')

    # 输出每个类别的评估结果
    for i in range(args.classes):
        logger.info(
            f'类别_{i} 结果: iou/accuracy {iou_class[i]:.4f}/{accuracy_class[i]:.4f}, '
            f'名称: {class_names[i]}'
        )

    # 保存评估指标到CSV文件
    metrics_csv_path = os.path.join(args.save_folder, 'test_metrics.csv')
    with open(metrics_csv_path, 'w', newline='', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        # 表头
        header = ['mIoU', 'mAcc', 'allAcc']
        for i in range(args.classes):
            header.extend([f'class_{i}_iou', f'class_{i}_acc', f'class_{i}_name'])
        csv_writer.writerow(header)

        # 数据行
        row = [mIoU, mAcc, allAcc]
        for i in range(args.classes):
            row.append(iou_class[i])
            row.append(accuracy_class[i])
            row.append(class_names[i])
        csv_writer.writerow(row)

    logger.info('<<<<<<<<<<<<<<<<< 评估结束 <<<<<<<<<<<<<<<<<')


if __name__ == '__main__':
    main()
