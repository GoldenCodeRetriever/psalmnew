import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import json
import numpy as np
import cv2
import pickle
import transformers
from dataclasses import dataclass, field
from typing import Optional, List
from tqdm import tqdm
from enum import Enum
from pathlib import Path
from torch.utils.data import DataLoader
import torch.distributed as dist

# PSALM 模块导入
from psalm.model.builder import load_pretrained_model
from psalm.utils import disable_torch_init
from psalm.mm_utils import get_model_name_from_path
from psalm import conversation as conversation_lib
from psalm.train.train_datasets import Cross_interactive_dataset, DataCollatorForCOCODatasetV2

from pycocotools import mask as mask_utils

# ==================== 指标计算辅助类 ====================

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    """
    计算交集和并集 (GPU版本)
    output: 预测掩码 (Flattened)
    target: 真实掩码 (Flattened)
    K: 类别数 (通常为2: 背景, 前景)
    """
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    
    # 过滤 ignore_index
    valid_mask = (target != ignore_index)
    output = output[valid_mask]
    target = target[valid_mask]
    
    # 限制范围
    output = torch.clamp(output, 0, K-1)
    target = torch.clamp(target, 0, K-1)

    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def fuse_masks(masks):
    """如果预测了多个mask，将它们合并（取并集）"""
    fused_mask = None
    for mask_ in masks:
        if fused_mask is None:
            fused_mask = mask_
        else:
            fused_mask = np.logical_or(fused_mask, mask_)
    return fused_mask

def compute_metric(intersection_meter, union_meter, acc_iou_meter, results_list):
    """
    计算指标的核心函数
    保留原有的 metric 逻辑：
    1. 计算每个样本的 IoU
    2. 累加 Intersection 和 Union 用于计算 cIoU
    3. 累加 IoU 用于计算 gIoU
    """
    for results in results_list:
        gt = results['gt'] # Target Image GT
        preds = results['pred'].astype(np.uint8)
        scores = results['scores']
        
        # 策略：筛选高置信度的 mask 进行融合 (阈值可调，原逻辑为0.4)
        pred_mask = []
        for i, score_ in enumerate(scores):
            if score_ > 0.4:
                pred_mask.append(preds[i])
        
        # 如果没有大于阈值的，取全黑，或者取最大置信度的那个
        if len(pred_mask) == 0:
             # 如果完全没有预测，尝试取置信度最高的 top-1
             if len(scores) > 0:
                 max_idx = np.argmax(scores)
                 pred_mask = [preds[max_idx]]
             else:
                 pred_mask = [np.zeros_like(gt, dtype=np.uint8)]
        else:
            pred_mask = [fuse_masks(pred_mask)]

        # 计算 IoU
        max_acc_iou = -1
        max_iou = 0
        max_intersection = 0
        max_union = 0
        
        # 实际上经过 fuse 后 pred_mask 通常只有一个，但保留循环结构以兼容
        for i, pred_ in enumerate(pred_mask):
            intersection, union, _ = intersectionAndUnionGPU(
                torch.tensor(pred_).int().cuda().contiguous(), 
                torch.tensor(gt).int().cuda().contiguous(), 
                2, ignore_index=255
            )
            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            
            # 计算当前样本的 IoU (Class 1: Foreground)
            acc_iou = intersection / (union + 1e-5)
            acc_iou[union == 0] = 1.0  # 防止分母为0
            
            fore_acc_iou = acc_iou[1] # 只看前景
            
            if fore_acc_iou > max_acc_iou:
                max_acc_iou = fore_acc_iou
                max_iou = acc_iou
                max_intersection = intersection
                max_union = union
        
        # 更新累加器
        intersection_meter.update(max_intersection)
        union_meter.update(max_union)
        acc_iou_meter.update(max_iou, n=1) # 记录该样本的 IoU

def parse_outputs(outputs, gt_masks_batch):
    """解析模型输出"""
    res_list = []
    # outputs 通常是一个 list，对应 batch 中的每个样本
    for output, gt_mask in zip(outputs, gt_masks_batch):
        pred_mask = output['instances'].pred_masks.cpu().numpy()
        scores = output['instances'].scores.transpose(1,0).cpu().numpy()
        gt_mask = gt_mask.astype(np.uint8)
        
        # 对每一个 GT 对象计算（通常 Cross Segmentation 是一对一或一对多）
        # 这里假设 batch 中每个 instance 对应一个 query
        for i in range(gt_mask.shape[0]):
            res = {
                'pred': pred_mask,
                'gt': gt_mask[i],
                'scores': scores[i] if i < len(scores) else scores[0], # 安全处理
            }
            res_list.append(res)
    return res_list

# ==================== 参数配置 ====================

@dataclass
class DataArguments:
    # 模型路径
    model_path: Optional[str] = field(default="/nfs-data1/lipeilang/output/checkpoint/PSALM_deformable/checkpoint-15000", metadata={"help": "Path to the trained model."})
    # 数据路径
    json_path: str = field(default="/nfs-data1/public/12.03/interactive_category_6.json", metadata={"help": "Path to the json file."})
    # 图像路径 (Cross Task 通常需要指定 source 和 target 的 folder，如果都在一个目录下则用同一个)
    region_cross_image_folder: Optional[str] = field(default='/nfs-data1/public/SIOR/images')
    
    # 其他配置
    mask_config: Optional[str] = field(default="./psalm/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml")
    is_multimodal: bool = True
    image_aspect_ratio: str = 'square'
    model_map_name: str = 'psalm'
    version: str = 'llava_phi'
    
    # 评测配置
    eval_batch_size: int = 1 # 建议设为 1 以避免 mask 尺寸不一致的 padding 问题
    dataloader_num_workers: int = 4
    seg_task: Optional[str] = field(default="region") # Cross segmentation 通常沿用 region 的配置
    region_mask_type: Optional[str] = field(default="point_visual_prompt_mask") # 或者是 box_visual_prompt_mask
    
    # 占位符
    lazy_preprocess: bool = False
    image_grid_pinpoints: Optional[str] = None

# ==================== 主评测逻辑 ====================

def evaluation():
    # 1. 解析参数
    parser = transformers.HfArgumentParser(DataArguments)
    data_args = parser.parse_args_into_dataclasses()[0]
    
    disable_torch_init()
    
    model_path = os.path.expanduser(data_args.model_path)
    model_name = get_model_name_from_path(model_path)
    
    print(f"Loading model from: {model_path}")
    print(f"Mask Config: {data_args.mask_config}")

    # 2. 加载模型
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, 
        model_args=data_args, 
        mask_config=data_args.mask_config, 
        device='cuda'
    )
    
    data_args.image_processor = image_processor
    conversation_lib.default_conversation = conversation_lib.conv_templates[data_args.version]

    # 3. 加载数据集 (使用 Cross_interactive_dataset)
    print(f"Loading dataset from: {data_args.json_path}")
    eval_dataset = Cross_interactive_dataset(
        json_path=data_args.json_path, 
        tokenizer=tokenizer, 
        data_args=data_args
    )
    
    data_collator = DataCollatorForCOCODatasetV2(tokenizer=tokenizer)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=data_args.eval_batch_size,
        collate_fn=data_collator,
        num_workers=data_args.dataloader_num_workers,
        shuffle=False,
        drop_last=False
    )

    # 4. 加载 GT JSON 用于生成 Ground Truth Masks
    with open(data_args.json_path) as f:
        gt_data_json = json.load(f)
        # 如果 json 是列表
        if isinstance(gt_data_json, list):
            gt_data = gt_data_json
        else:
            gt_data = gt_data_json['annotations'] # 视具体 json 结构而定，通常 psalm 的 json 是 list

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device=device, dtype=torch.float).eval()

    # 5. 初始化指标统计器
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    print("Start Evaluation...")
    
    with torch.no_grad():
        for batch_idx, inputs in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
            
            # 准备 GT Masks (从原始 JSON 中读取，因为 DataLoader 可能做了 resize/crop)
            # 注意：Eval 时最好 batch_size=1，否则这里索引对齐比较麻烦
            start_idx = batch_idx * data_args.eval_batch_size
            end_idx = start_idx + data_args.eval_batch_size
            sample_indices = list(range(start_idx, min(end_idx, len(eval_dataset))))
            
            gt_masks_batch = []
            for sample_idx in sample_indices:
                # 跨图任务的目标是 Target Image，所以读取 annotations1
                # 注意：Cross_interactive_dataset 也是读 annotations1
                raw_sample = gt_data[sample_idx]
                annotations = raw_sample['annotations1'] 
                h = raw_sample['image1_info']['height']
                w = raw_sample['image1_info']['width']
                
                masks = []
                for annotation in annotations:
                    # 生成 mask
                    if isinstance(annotation['segmentation'], list):
                        segm = np.zeros((h, w), dtype=np.uint8)
                        for poly in annotation['segmentation']:
                            poly = np.array(poly, dtype=np.int32).reshape(-1, 2)
                            cv2.fillPoly(segm, [poly], 1)
                        masks.append(segm.astype(np.bool_))
                    else:
                        # RLE 格式
                        if isinstance(annotation['segmentation']['counts'], list):
                            rle = mask_utils.frPyObjects(annotation['segmentation'], *annotation['segmentation']['size'])
                            segm = mask_utils.decode(rle)
                        else:
                            segm = mask_utils.decode(annotation['segmentation'])
                        masks.append(segm.astype(np.bool_))
                
                if len(masks) > 0:
                    gt_mask = np.stack([m.astype(np.uint8) for m in masks], axis=0)
                else:
                    gt_mask = np.zeros((1, h, w), dtype=np.uint8)
                
                gt_masks_batch.append(gt_mask)

            # 模型推理
            # 关键修复：Cross Segmentation 需要同时传入 images (Source) 和 images1 (Target)
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            
            try:
                outputs = model.eval_seg(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    images=inputs['images'].float(),      # Source Image
                    images1=inputs['images1'].float(),    # Target Image (关键!)
                    seg_info=inputs['seg_info'],
                    labels=inputs.get('labels', None)
                )
            except Exception as e:
                print(f'Error in batch {batch_idx}: {e}')
                continue

            # 解析结果并更新指标
            cur_res = parse_outputs(outputs, gt_masks_batch)
            compute_metric(intersection_meter, union_meter, acc_iou_meter, cur_res)

    # 6. 计算最终指标
    # cIoU: 全局 Intersection Sum / 全局 Union Sum
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1] # 取前景类
    
    # gIoU: 所有样本 IoU 的平均值
    giou = acc_iou_meter.avg[1] # 取前景类

    result_msg = f"Evaluation Result:\n  Model: {model_name}\n  gIoU (Mean IoU): {giou:.4f}\n  cIoU (Global IoU): {ciou:.4f}"
    print("-" * 30)
    print(result_msg)
    print("-" * 30)
    
    # 保存结果到 txt
    save_path = os.path.join(data_args.model_path, 'eval_results_cross.txt')
    with open(save_path, 'w') as f:
        f.write(result_msg)
    print(f"Results saved to {save_path}")

if __name__ == '__main__':
    evaluation()