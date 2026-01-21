import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # 您可以根据需要修改GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
import torch
from enum import Enum
import json
from tqdm import tqdm
import numpy as np
from psalm.model.builder import load_pretrained_model
from psalm.utils import disable_torch_init
from psalm.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import cv2
from torch.utils.data import Dataset, DataLoader

from psalm import conversation as conversation_lib
from psalm.train.train_datasets import *

from detectron2.data import MetadataCatalog, DatasetCatalog
from pycocotools import mask
from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
import transformers 
import pickle
from pathlib import Path

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

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)
        return fmtstr.format(** self.__dict__)


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    valid_mask = (target != ignore_index)
    output = output[valid_mask]
    target = target[valid_mask]
    output = torch.clamp(output, 0, K-1)
    target = torch.clamp(target, 0, K-1)
    output = output.view(-1)
    target = target.view(-1)
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    area_union = torch.clamp(area_union, min=0.0)
    return area_intersection, area_union, area_target


@dataclass
class EvalArguments:
    model_path: str = field(metadata={"help": "/nfs-data1/lipeilang/output/checkpoint/PSALM_multi_query_deformable_1.12/checkpoint-1295"})
    eval_data_path: str = field(metadata={"help": "/nfs-data1/public/12.03/interactive_category_6.json"})
    image_folder: str = field(metadata={"help": "/nfs-data1/public/SIOR/images"})
    output_dir: str = field(default='/nfs-data1/lipeilang/output/interactive_segmentation', metadata={"help": "Directory to save evaluation results."})
    model_base: Optional[str] = field(default=None)
    conv_mode: Optional[str] = field(default='phi')
    eval_batch_size: int = field(default=1)
    # 从 region_cross_eval_batch.py 继承的参数
    mm_vision_tower: Optional[str] = field(default=None)
    with_norm: bool = field(default=False)
    with_layernorm: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)


def compute_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / (union + 1e-6)
    return iou

def evaluation():
    parser = transformers.HfArgumentParser(EvalArguments)
    eval_args = parser.parse_args_into_dataclasses()[0]

    disable_torch_init()
    model_name = get_model_name_from_path(eval_args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        eval_args.model_path, eval_args.model_base, model_name, model_args=eval_args, device_map='cuda'
    )
    
    # 初始化vision modules
    model.model.initialize_vision_modules(model_args=eval_args)

    # 加载评测数据集
    with open(eval_args.eval_data_path, 'r') as f:
        eval_data = json.load(f)

    # 初始化指标
    giou_meter = AverageMeter("gIoU", fmt=":.4f")
    ciou_meter = AverageMeter("cIoU", fmt=":.4f")

    # 创建输出目录
    os.makedirs(eval_args.output_dir, exist_ok=True)

    for item in tqdm(eval_data):
        image_file1 = os.path.join(eval_args.image_folder, item['image1'])
        image_file2 = os.path.join(eval_args.image_folder, item['image2'])
        
        # 加载图像
        image1 = Image.open(image_file1).convert('RGB')
        image2 = Image.open(image_file2).convert('RGB')

        # 准备输入
        conv = conversation_lib.conv_templates[eval_args.conv_mode].copy()
        inp = item['instruction'] + " <image> <image>"
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # 对图像进行预处理
        # 注意：这里的 image_processor 需要根据您的模型进行适配
        # 您可能需要使用与训练时相同的预处理方式
        image_tensor1 = image_processor.preprocess(image1, return_tensors='pt')['pixel_values'][0].to(model.device, dtype=torch.float16)
        image_tensor2 = image_processor.preprocess(image2, return_tensors='pt')['pixel_values'][0].to(model.device, dtype=torch.float16)

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != conversation_lib.SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=[image_tensor1, image_tensor2], # 传递两个图像
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        
        # 解析模型输出的mask
        # 这里的解析逻辑需要根据您模型的实际输出来确定
        # 以下是一个示例，您需要替换成您自己的逻辑
        try:
            # 假设模型输出的是一个mask的字符串表示，例如RLE格式
            pred_mask_rle = outputs # ... 解析outputs得到RLE
            pred_mask = mask.decode(pred_mask_rle)
        except Exception as e:
            print(f"Error parsing mask: {e}")
            continue

        # 加载GT mask
        gt_mask_rle = item['segmentation']
        gt_mask = mask.decode(gt_mask_rle)

        # 计算指标
        # gIoU
        intersection, union, _ = intersectionAndUnionGPU(
            torch.from_numpy(pred_mask).cuda().int(),
            torch.from_numpy(gt_mask).cuda().int(),
            2
        )
        iou = intersection[1] / (union[1] + 1e-6)
        giou_meter.update(iou.item())

        # cIoU
        ciou = compute_iou(pred_mask, gt_mask)
        ciou_meter.update(ciou)

    print(f"gIoU: {giou_meter.avg:.4f}")
    print(f"cIoU: {ciou_meter.avg:.4f}")

    # 保存结果
    with open(os.path.join(eval_args.output_dir, 'results.txt'), 'w') as f:
        f.write(f"gIoU: {giou_meter.avg:.4f}\n")
        f.write(f"cIoU: {ciou_meter.avg:.4f}\n")

if __name__ == "__main__":
    evaluation()