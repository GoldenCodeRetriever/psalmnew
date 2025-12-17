import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
    """Computes and stores the average and current value（保留原有，不删all_reduce，单卡不调用不影响）"""
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

    def all_reduce(self):  
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist() + [self.count],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

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
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    region_cross_image_folder: Optional[str] = field(default='/nfs-data1/public/test/images')
    model_path: Optional[str] = field(default="/nfs-data1/lipeilang/output/checkpoint/PSALM_multi_query_deformable_gaosi/checkpoint-15440")
    mask_config: Optional[str] = field(default="./psalm/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml")
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    json_path: str = '/nfs-data1/public/test/newinteractive_category_1.json'
    model_map_name: str = 'psalm'
    version: str = 'llava_phi'  
    output_dir: str = './output/interactive_segmentation_newtest0.6'
    segmentation: bool = True
    eval_batch_size: int = 4  
    dataloader_num_workers: int = 8
    seg_task: Optional[str] = field(default="region")
    region_mask_type: Optional[str] = field(default="box_visual_prompt_mask") #'point_visual_prompt_mask||box_visual_prompt_mask||scribble_visual_prompt_mask||'


def parse_outputs(outputs, gt_masks_batch):
    res_list = []
    for output, gt_mask in zip(outputs, gt_masks_batch):
        pred_mask = output['instances'].pred_masks.cpu().numpy()
        scores = output['instances'].scores.transpose(1,0).cpu().numpy()
        gt_mask = gt_mask.astype(np.uint8)
        try:
            pred_cls = output['instances'].pred_classes.cpu().numpy()
        except:
            pred_cls = None
        for i in range(gt_mask.shape[0]):
            res = {
                'pred': pred_mask,
                'gt': gt_mask[i],
                'scores': scores[i],
                'pred_cls': pred_cls
            }
            res_list.append(res)
    return res_list


def fuse_masks(masks):
    fused_mask = None
    for mask_ in masks:
        if fused_mask is None:
            fused_mask = mask_
        else:
            fused_mask = np.logical_or(fused_mask, mask_)
    return fused_mask


def compute_metric(intersection_meter, union_meter, acc_iou_meter, results_list):
    pred_list = []
    gt_list = []
    results_list = list(results_list)
    for results in results_list:
        gt = results['gt']
        preds = results['pred'].astype(np.uint8)
        scores = results['scores']
        pred_mask = []
        for i, score_ in enumerate(scores):
            if score_ > 0.6:
                pred_mask.append(preds[i])
        if len(pred_mask) == 0:
            pred_mask = [np.zeros_like(gt, dtype=np.uint8)]
        else:
            pred_mask = [fuse_masks(pred_mask)] 
        max_acc_iou = -1
        max_iou = 0
        max_intersection = 0
        max_union = 0
        max_i = 0
        for i,pred_ in enumerate(pred_mask):
            intersection, union, _ = intersectionAndUnionGPU(
                torch.tensor(pred_).int().cuda().contiguous().clone(), 
                torch.tensor(gt).int().cuda().contiguous(), 
                2, ignore_index=255
            )
            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            acc_iou = intersection / (union + 1e-5)
            acc_iou[union == 0] = 1.0
            fore_acc_iou = acc_iou[1]
            if fore_acc_iou > max_acc_iou:
                max_acc_iou = fore_acc_iou
                max_iou = acc_iou
                max_intersection = intersection
                max_union = union
                max_i = i
        intersection_meter.update(max_intersection)
        union_meter.update(max_union)
        acc_iou_meter.update(max_iou, n=1)
        pred_list.append(pred_mask[max_i])
        gt_list.append(gt)
    return pred_list, gt_list


def evaluation():
    parser = transformers.HfArgumentParser(DataArguments)
    data_args = parser.parse_args_into_dataclasses()[0]
    disable_torch_init()
    model_path = os.path.expanduser(data_args.model_path)
    model_name = get_model_name_from_path(model_path)
    save_suffix = os.path.basename(data_args.json_path).split('.')[0]
    if data_args.region_mask_type is not None:
        save_suffix += '_' + data_args.region_mask_type.split('_')[0]
    print(f'save suffix is {save_suffix}')
    print(f'current model is {model_path}')
    print(f'eval_batch_size = {data_args.eval_batch_size}')

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, model_args=data_args, 
        mask_config=data_args.mask_config, device='cuda'
    )
    data_args.image_processor = image_processor
    data_args.is_multimodal = True
    conversation_lib.default_conversation = conversation_lib.conv_templates[data_args.version]

    eval_dataset = Cross_interactive_dataset(
        json_path=data_args.json_path, tokenizer=tokenizer, data_args=data_args
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

    gt_json_path = data_args.json_path
    with open(gt_json_path) as f:
        gt_data = json.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device=device, dtype=torch.float).eval()

    save_list = []
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)


    vis_dir = os.path.join(data_args.output_dir, f'vis_{save_suffix}')
    os.makedirs(vis_dir, exist_ok=True)
    print(f'Visualization results will be saved to: {vis_dir}')


    with torch.no_grad():
        for batch_idx, inputs in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
    
            start_idx = batch_idx * data_args.eval_batch_size
            end_idx = start_idx + data_args.eval_batch_size
            sample_indices = list(range(start_idx, min(end_idx, len(gt_data))))  

            gt_masks_batch = []
            for sample_idx in sample_indices:
                annotations = gt_data[sample_idx]['annotations1']
                h, w = gt_data[sample_idx]['image1_info']['height'], gt_data[sample_idx]['image1_info']['width']
                masks = []
                for annotation in annotations:
                    if isinstance(annotation['segmentation'], list):
                        segm = np.zeros((h, w), dtype=np.uint8)
                        for poly in annotation['segmentation']:
                            poly = np.array(poly, dtype=np.int32).reshape(-1, 2)
                            cv2.fillPoly(segm, [poly], 1)
                        masks.append(segm.astype(np.bool_))
                    else:
                        if isinstance(annotation['segmentation']['counts'], list):
                            rle = mask.frPyObjects(annotation['segmentation'], *annotation['segmentation']['size'])
                            segm = mask.decode(rle)
                        else:
                            segm = mask.decode(annotation['segmentation'])
                        masks.append(segm.astype(np.bool_))
                gt_mask = [mask_.astype(np.uint8) for mask_ in masks]
                gt_mask = np.stack(gt_mask, axis=0)
                gt_masks_batch.append(gt_mask)  

            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            try:
                outputs = model.eval_seg(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    images=inputs['images'].float(),
                    images1=inputs['images1'].float(),
                    seg_info=inputs['seg_info'],
                    labels=inputs['labels']
                )
            except Exception as e:
                print(f'Batch {batch_idx} skipped. Error: {e}')
                continue

            print(f"Batch outputs length: {len(outputs)}")

            cur_res = parse_outputs(outputs, gt_masks_batch)
            pred, gt_mask = compute_metric(intersection_meter, union_meter, acc_iou_meter, cur_res)


            for i in range(len(sample_indices)):
                sample_idx = sample_indices[i]
                cur_pred_mask = pred[i] # 预测的Mask (H, W) 0/1
                
                # 获取原图路径 (Image1是目标图)
                img_info = gt_data[sample_idx]['image1_info']
                img_name = img_info['image']
                
                # 拼接完整路径
                if data_args.region_cross_image_folder:
                    img_path = os.path.join(data_args.region_cross_image_folder, img_name)
                else:
                    img_path = img_name
                
                # 读取图像
                img = cv2.imread(img_path)
                if img is None:
                    # 尝试不加前缀读取，防止路径拼接错误
                    img = cv2.imread(img_name)
                
                if img is not None:
                    # 确保 Mask 尺寸与图像一致 (模型内部已做postprocess，理论上一致)
                    if img.shape[:2] != cur_pred_mask.shape:
                        cur_pred_mask = cv2.resize(cur_pred_mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                    
                    # 绘制红色Mask叠加
                    overlay = img.copy()
                    # BGR格式: 红色 (0, 0, 255)
                    overlay[cur_pred_mask > 0] = np.array([0, 0, 255], dtype=np.uint8)
                    
                    # 混合图片 (0.5透明度)
                    alpha = 0.5
                    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
                    
                    # 绘制Mask轮廓 (更清晰)
                    contours, _ = cv2.findContours(cur_pred_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
                    
                    # 保存图片
                    save_name = f"{os.path.splitext(os.path.basename(img_name))[0]}_{sample_idx}_pred.jpg"
                    cv2.imwrite(os.path.join(vis_dir, save_name), img)
                else:
                    print(f"[Warning] Could not read image: {img_path}")


            for i in range(len(sample_indices)):
                save_info = {
                    'pred': [mask.encode(np.asfortranarray(p)) for p in pred[i:i+1]],
                    'gt': [mask.encode(np.asfortranarray(g)) for g in gt_mask[i:i+1]],
                    'name': inputs['seg_info'][i]['file_name']  
                }
                save_list.append(save_info)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]
    miou = (iou_class[0] + iou_class[1]) / 2.0  
    msg = f"benchmark: {save_suffix}: giou: {giou:.4f}, ciou: {ciou:.4f}, miou: {miou:.4f}"
    print(f"intersection_meter.sum:{intersection_meter.sum}, union_meter.sum:{union_meter.sum}")
    print(msg)

    save_path = os.path.join(data_args.model_path, 'pred_pkl')
    Path(save_path).mkdir(parents=True, exist_ok=True)
    # with open(os.path.join(save_path, f'pred_{save_suffix}.pkl'), 'wb') as f:
    #     pickle.dump(save_list, f)
    with open(os.path.join(save_path, f'pred_{save_suffix}.txt'), 'w') as f:
        f.write(msg)


if __name__ == '__main__': 
    evaluation()