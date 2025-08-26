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
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch_points_kernels as tp

from util import config, transform
from util.common_util import AverageMeter, intersectionAndUnion, check_makedirs
from util.voxelize import voxelize
from util.my_dataset import MyDataset  # 导入自定义数据集

random.seed(123)
np.random.seed(123)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/scannetv2_stratified_transformer.yaml',
                        help='config file')
    parser.add_argument('opts', help='see config for all options', default=None, nargs=argparse.REMAINDER)
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


def log_memory_usage(stage):
    """打印当前CUDA内存使用情况"""
    allocated = torch.cuda.memory_allocated() / 1024 ** 3  # GB
    cached = torch.cuda.memory_reserved() / 1024 ** 3  # GB
    logger.info(f"[Memory] {stage} - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")


def main():
    global args, logger
    args = get_parser()
    logger = get_logger()
    logger.info(args)
    assert args.classes > 1
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    # 模型初始化
    if args.arch == 'stratified_transformer':
        from model.stratified_transformer import Stratified
        args.patch_size = args.grid_size * args.patch_size
        args.window_size = [args.patch_size * args.window_size * (2 ** i) for i in range(args.num_layers)]
        args.grid_sizes = [args.patch_size * (2 ** i) for i in range(args.num_layers)]
        args.quant_sizes = [args.quant_size * (2 ** i) for i in range(args.num_layers)]
        model = Stratified(
            args.downsample_scale, args.depths, args.channels, args.num_heads, args.window_size,
            args.up_k, args.grid_sizes, args.quant_sizes, rel_query=args.rel_query,
            rel_key=args.rel_key, rel_value=args.rel_value, drop_path_rate=args.drop_path_rate,
            concat_xyz=args.concat_xyz, num_classes=args.classes, ratio=args.ratio, k=args.k,
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
            args.depths, args.channels, args.num_heads, args.window_sizes, args.up_k,
            args.grid_sizes, args.quant_sizes, rel_query=args.rel_query, rel_key=args.rel_key,
            rel_value=args.rel_value, drop_path_rate=args.drop_path_rate, concat_xyz=args.concat_xyz,
            num_classes=args.classes, ratio=args.ratio, k=args.k, prev_grid_size=args.grid_size,
            sigma=1.0, num_layers=args.num_layers, stem_transformer=args.stem_transformer,
            in_channels=args.in_channels
        )
    else:
        raise Exception(f'architecture {args.arch} not supported')

    model = model.cuda()
    logger.info(model)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
    names = ['铁塔', '背景', '导线']

    # 加载模型权重
    if os.path.isfile(args.model_path):
        logger.info(f"=> loading checkpoint '{args.model_path}'")
        checkpoint = torch.load(args.model_path)
        state_dict = checkpoint['state_dict']
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name.replace("item", "stem")] = v
        model.load_state_dict(new_state_dict, strict=True)
        logger.info(f"=> loaded checkpoint '{args.model_path}' (epoch {checkpoint['epoch']})")
        args.epoch = checkpoint['epoch']
    else:
        raise RuntimeError(f"=> no checkpoint found at '{args.model_path}'")

    # 初始化时检查显存
    log_memory_usage("After model initialization")
    test(model, criterion, names, test_transform=None)


def create_test_dataset(transform):
    return MyDataset(
        split='test',
        data_root=args.data_root_val,
        transform=transform,
        voxel_size=args.voxel_size,
        voxel_max=args.voxel_max,
        shuffle_index=False,
        loop=1
    )


def get_sample_names(dataset):
    sample_names = []
    for scene_dir in dataset.scene_dirs:
        sample_name = os.path.basename(scene_dir)
        sample_names.append(sample_name)
    return sample_names


def test(model, criterion, names, test_transform):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    check_makedirs(args.save_folder)
    pred_save, label_save = [], []

    dataset = create_test_dataset(test_transform)
    sample_names = get_sample_names(dataset)
    total_samples = len(dataset)
    logger.info(f"Totally {total_samples} samples in test set.")

    for sample_idx in range(total_samples):
        item = sample_names[sample_idx]
        end = time.time()
        pred_save_path = os.path.join(args.save_folder, f'{item}_epoch{args.epoch}_pred.npy')
        label_save_path = os.path.join(args.save_folder, f'{item}_epoch{args.epoch}_label.npy')

        if os.path.exists(pred_save_path) and os.path.exists(label_save_path):
            logger.info(f'{sample_idx + 1}/{total_samples}: {item} - loading existing results')
            pred = np.load(pred_save_path)
            label = np.load(label_save_path)
        else:
            logger.info(f'{sample_idx + 1}/{total_samples}: {item} - starting processing')

            # 加载数据时添加日志
            logger.info(f"[Step 1/5] Loading data for {item}")
            coord, feat, label = dataset[sample_idx]
            label = label.long()
            logger.info(f"[Data Info] coord shape: {coord.shape}, feat shape: {feat.shape}, label shape: {label.shape}")
            log_memory_usage(f"After loading data for {item}")

            # 点云分块处理
            logger.info(f"[Step 2/5] Starting point cloud chunking for {item}")
            idx_data = []
            if args.voxel_max and coord.shape[0] > args.voxel_max:
                logger.info(f"Point count ({coord.shape[0]}) exceeds voxel_max ({args.voxel_max}), starting chunking")
                coord_np = coord.numpy()
                coord_p = np.random.rand(coord_np.shape[0]) * 1e-3
                idx_uni = np.array([], dtype=np.int64)
                chunk_idx = 0
                while idx_uni.size < coord_np.shape[0]:
                    init_idx = np.argmin(coord_p)
                    dist = np.sum(np.power(coord_np - coord_np[init_idx], 2), 1)
                    idx_crop = np.argsort(dist)[:args.voxel_max]
                    idx_uni = np.unique(np.concatenate((idx_uni, idx_crop))).astype(np.int64)
                    idx_data.append(idx_crop)
                    chunk_idx += 1
                    logger.info(
                        f"Generated chunk {chunk_idx}, current covered points: {idx_uni.size}/{coord_np.shape[0]}")
                logger.info(f"Total chunks generated: {len(idx_data)}")
            else:
                idx_data.append(np.arange(coord.shape[0]))
                logger.info(f"No chunking needed (voxel_max: {args.voxel_max}, point count: {coord.shape[0]})")
            log_memory_usage(f"After chunking for {item}")

            # 模型推理
            logger.info(f"[Step 3/5] Starting model inference for {item}")
            pred = np.zeros((label.shape[0], args.classes), dtype=np.float32)
            for chunk_id, idx_part in enumerate(idx_data):
                logger.info(f"Processing chunk {chunk_id + 1}/{len(idx_data)}, chunk size: {len(idx_part)}")

                # 准备分块数据
                logger.info(f"[Chunk {chunk_id + 1}] Preparing data")
                coord_part = coord[idx_part].cuda(non_blocking=True)
                feat_part = feat[idx_part].cuda(non_blocking=True)
                offset_part = torch.tensor([len(coord_part)], dtype=torch.int32).cuda(non_blocking=True)
                batch = torch.zeros(len(coord_part), dtype=torch.long).cuda(non_blocking=True)
                log_memory_usage(f"[Chunk {chunk_id + 1}] After moving data to GPU")

                # 计算邻域
                logger.info(f"[Chunk {chunk_id + 1}] Starting ball query")
                radius = 2.5 * args.grid_size * 1.0
                start_ball = time.time()
                neighbor_idx = tp.ball_query(
                    radius, args.max_num_neighbors,
                    coord_part, coord_part,
                    mode="partial_dense",
                    batch_x=batch, batch_y=batch
                )[0].cuda(non_blocking=True)
                end_ball = time.time()
                logger.info(
                    f"[Chunk {chunk_id + 1}] Ball query finished in {end_ball - start_ball:.2f}s, neighbor_idx shape: {neighbor_idx.shape}")
                log_memory_usage(f"[Chunk {chunk_id + 1}] After ball query")

                # 拼接坐标特征
                if args.concat_xyz:
                    logger.info(f"[Chunk {chunk_id + 1}] Concatenating xyz to features")
                    feat_part = torch.cat([feat_part, coord_part], dim=1)
                    logger.info(f"[Chunk {chunk_id + 1}] New feature shape: {feat_part.shape}")

                # 模型预测
                logger.info(f"[Chunk {chunk_id + 1}] Starting model forward pass")
                start_model = time.time()
                with torch.no_grad():
                    pred_part = model(feat_part, coord_part, offset_part, batch, neighbor_idx)
                    pred_part = F.softmax(pred_part, dim=-1).cpu().numpy()
                end_model = time.time()
                logger.info(f"[Chunk {chunk_id + 1}] Model forward finished in {end_model - start_model:.2f}s")

                # 累加预测结果
                pred[idx_part] += pred_part
                torch.cuda.empty_cache()
                log_memory_usage(f"[Chunk {chunk_id + 1}] After forward and cache clear")

            # 处理最终预测
            logger.info(f"[Step 4/5] Processing final prediction for {item}")
            pred = pred.argmax(1)
            logger.info(f"[Step 5/5] Saving results for {item}")
            np.save(pred_save_path, pred)
            np.save(label_save_path, label.numpy())
            log_memory_usage(f"After saving results for {item}")

        # 计算评估指标
        logger.info(f"Calculating metrics for {item}")
        intersection, union, target = intersectionAndUnion(
            pred, label.numpy(), args.classes, args.ignore_label
        )
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)

        accuracy = sum(intersection) / (sum(target) + 1e-10)
        batch_time.update(time.time() - end)
        logger.info(
            f'Test: [{sample_idx + 1}/{total_samples}] {item} ({label.size(0)} points) '
            f'Time {batch_time.val:.3f}s (avg: {batch_time.avg:.3f}s) '
            f'Accuracy {accuracy:.4f}'
        )

        pred_save.append(pred)
        label_save.append(label.numpy())

    # 保存整体结果和指标（保持原有逻辑）
    with open(os.path.join(args.save_folder, "pred_all.pickle"), 'wb') as f:
        pickle.dump({'pred': pred_save, 'names': sample_names}, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.save_folder, "label_all.pickle"), 'wb') as f:
        pickle.dump({'label': label_save, 'names': sample_names}, f, protocol=pickle.HIGHEST_PROTOCOL)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    logger.info('=' * 50)
    logger.info(f'Overall Test Result: mIoU={mIoU:.4f}, mAcc={mAcc:.4f}, allAcc={allAcc:.4f}')
    logger.info('=' * 50)

    for i in range(args.classes):
        logger.info(
            f'Class {i} ({names[i]}): '
            f'IoU={iou_class[i]:.4f}, Accuracy={accuracy_class[i]:.4f}'
        )

    csv_path = os.path.join(args.save_folder, 'test_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['mIoU', f'{mIoU:.4f}'])
        writer.writerow(['mAcc', f'{mAcc:.4f}'])
        writer.writerow(['allAcc', f'{allAcc:.4f}'])
        writer.writerow([])
        writer.writerow(['Class', 'Name', 'IoU', 'Accuracy'])
        for i in range(args.classes):
            writer.writerow([i, names[i], f'{iou_class[i]:.4f}', f'{accuracy_class[i]:.4f}'])

    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


if __name__ == '__main__':
    main()
