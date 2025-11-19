import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
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
from pycocotools import mask as mask_utils

from psalm import conversation as conversation_lib
from psalm.train.train_datasets import *

from detectron2.data import MetadataCatalog, DatasetCatalog
from pycocotools import mask
from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
import torch.distributed as dist
import transformers
import pickle
from pathlib import Path

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    region_image_folder: Optional[str] = field(default='/nfs-data1/zhangkunquan/SIOR/images')
    model_path: Optional[str] = field(default="/home/zhangkunquan/code/PSALM/output/checkpoint/RRSIS-D_SIOR_SIORCross_10/checkpoint-5240")
    mask_config: Optional[str] = field(default="./psalm/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml")
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    json_path: str = '/nfs-data1/zhangkunquan/SIOR/interactive_test.json'
    model_map_name: str = 'psalm'
    version: str = 'llava_phi'
    output_dir: str = './output/interactive_segmentation'
    segmentation: bool = True
    eval_batch_size: int = 1
    dataloader_num_workers: int = 4
    seg_task: Optional[str] = field(default="region")
    region_mask_type: Optional[str] = field(default="scribble_visual_prompt_mask") #'box_visual_prompt_mask||scribble_visual_prompt_mask||point_visual_prompt_mask'
    thr: float = 0.6

def parse_outputs(outputs,gt_mask):
    res_list = []
    for output in outputs:

        pred_mask = output['instances'].pred_masks
        pred_mask = pred_mask.cpu().numpy()
        scores = output['instances'].scores.transpose(1,0).cpu().numpy()
        gt_mask = output['gt'].cpu().numpy().astype(np.uint8)
        try:
            pred_cls = output['instances'].pred_classes.cpu().numpy()
        except:
            pred_cls = None
        assert scores.shape[0] == gt_mask.shape[0]
        for i in range(gt_mask.shape[0]):
            res = {
                'pred':pred_mask,
                'gt': gt_mask[i],
                'scores':scores[i],
                'pred_cls':pred_cls
            }
            res_list.append(res)
    return res_list

def fuse_masks(masks):
    fused_mask = None
    for mask_ in masks:
        if fused_mask is None:
            fused_mask = mask_
        else:                                                                                                                                                                                                                                          
            fused_mask = np.logical_or(fused_mask,mask_)

    return fused_mask

def evaluation(data_args,thr=0.6):
    disable_torch_init()
    model_path = os.path.expanduser(data_args.model_path)
    model_name = get_model_name_from_path(model_path)
    save_suffix = os.path.basename(data_args.json_path).split('.')[0]
    if data_args.region_mask_type is not None:
        save_suffix += '_' + data_args.region_mask_type.split('_')[0]
    print(f'save suffix is {save_suffix}')
    print(f'current model is {model_path}')
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, model_args=data_args, mask_config=data_args.mask_config, device='cuda')
    # ckpt = torch.load(os.path.join(model_path,'pytorch_model.bin'))
    # model.load_state_dict(ckpt,strict=True)

    data_args.image_processor = image_processor
    data_args.is_multimodal = True
    conversation_lib.default_conversation = conversation_lib.conv_templates[data_args.version]

    eval_dataset = interactive_dataset(json_path=data_args.json_path, tokenizer=tokenizer, data_args=data_args)
    data_collator = DataCollatorForCOCODatasetV2(tokenizer=tokenizer)

    dataloader_params = {
        "batch_size": data_args.eval_batch_size,
        "num_workers": data_args.dataloader_num_workers,
    }
    eval_dataloader = DataLoader(eval_dataset, batch_size=dataloader_params['batch_size'], collate_fn=data_collator,
                                 num_workers=dataloader_params['num_workers'])

    def load_ref_dataset():
        return interactive_dataset(json_path=data_args.json_path, tokenizer=tokenizer, data_args=data_args)

    DatasetCatalog.register('refcoco_dataset', load_ref_dataset)
    MetadataCatalog.get('refcoco_dataset').set(stuff_classes=['object'],)
    gt_json_path = data_args.json_path
    with open(gt_json_path) as f:
        gt_data = json.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device=device,dtype=torch.float).eval()
    save_list = []

    seg_image_save_dir = os.path.join(data_args.output_dir, save_suffix)
    Path(seg_image_save_dir).mkdir(parents=True, exist_ok=True) 
    print(f"segmentation image save to : {seg_image_save_dir}")
    

    with torch.no_grad():
        for idx, inputs in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
            gt = gt_data[idx]['annotations']
            h, w = gt_data[idx]['image_info']['height'], gt_data[idx]['image_info']['width']

            # generate gt mask
            masks = []
            prompt_masks = []
            for annotation in gt:
                prompt_mask = annotation[data_args.region_mask_type]
                if isinstance(prompt_mask['counts'], list):
                    rle = mask_utils.frPyObjects(prompt_mask, *prompt_mask['size'])
                    prompt_segm = mask_utils.decode(rle)
                else:
                    prompt_segm = mask_utils.decode(prompt_mask)

                if isinstance(annotation['segmentation'], list):
                    segm = np.zeros((h, w), dtype=np.uint8)
                    for poly in annotation['segmentation']:
                        poly = np.array(poly, dtype=np.int32).reshape(-1, 2)
                        cv2.fillPoly(segm, [poly], 1)
                    masks.append(segm.astype(np.bool_))
                else:
                    if isinstance(annotation['segmentation']['counts'], list):
                        rle = mask_utils.frPyObjects(annotation['segmentation'], *annotation['segmentation']['size'])
                        segm = mask_utils.decode(rle)
                    else:
                        segm = mask_utils.decode(annotation['segmentation'])

                prompt_masks.append(prompt_segm.astype(np.bool_))
                masks.append(segm.astype(np.bool_))

            prompt_mask = [prompt_mask_.astype(np.uint8) for prompt_mask_ in prompt_masks]
            prompt_mask = np.stack(prompt_mask, axis=0)
            original_prompt = prompt_mask[0]

            gt_mask = [mask_.astype(np.uint8) for mask_ in masks]
            gt_mask = np.stack(gt_mask, axis=0)
            original_gt = gt_mask[0]

            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

            # print(f"\n3. 单个样本seg_info的类型：{type(sample_seg_info)}")
            # print(f"4. 单个样本seg_info包含的字段（keys）：{list(inputs['seg_info'][0].keys())}")  

            try:
                outputs = model.eval_seg(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    images=inputs['images'].float(),
                    seg_info=inputs['seg_info'],
                    labels=inputs['labels']
                )
            except:
                print('can not find region masks, skip')
                continue

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            cur_res = parse_outputs(outputs,gt_mask)

            img_file_name = inputs['seg_info'][0]['file_name']
            raw_img_path = os.path.join(data_args.region_image_folder, img_file_name)

            for results in cur_res: 
                save_img_name = os.path.basename(img_file_name).replace('.jpg',f'_{idx}_instance_pred_gt.jpg')
                save_img_path = os.path.join(seg_image_save_dir, save_img_name)
                preds = results['pred'].astype(np.uint8)
                scores = results['scores']
                gt = results['gt']

                # 仅保留分数最高的1个预测掩码，不融合，每个GT对应一个最佳预测
                # scores_tensor = torch.tensor(scores)
                # topk_scores, topk_idx = torch.topk(scores_tensor, k=1)  # 取分数最高的1个
                # topk_idx = topk_idx.cpu().numpy()[0]  # 获取最高分数对应的索引
                # pred_mask = [preds[topk_idx].astype(np.uint8)]  # 直接使用该最佳预测掩码，不融合
                
                # # 保留原阈值逻辑：若最高分数低于阈值，设为全0掩码
                # if topk_scores.cpu().numpy()[0] < thr:
                #     pred_mask = [np.zeros_like(gt, dtype=np.uint8)]

                raw_img = cv2.imread(raw_img_path)
                alpha = 0.5

                gt_overlay = raw_img.copy()
                gt_overlay[gt == 1] = gt_overlay[gt == 1] * (1 - alpha) + np.array([0, 0, 255]) * alpha
                gt_overlay = gt_overlay.astype(np.uint8)

                prompt_overlay = raw_img.copy()
                prompt_overlay[original_prompt == 1] = gt_overlay[original_prompt == 1] * (1 - alpha) + np.array([0, 0, 255]) * alpha
                prompt_overlay = prompt_overlay.astype(np.uint8)

                pred_overlay = raw_img.copy()
                pred_overlay[preds[0] == 1] = pred_overlay[preds[0] == 1] * (1 - alpha) + np.array([0, 255, 0]) * alpha
                pred_overlay = pred_overlay.astype(np.uint8)

                cv2.putText(gt_overlay, 'GT Mask (Red)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(prompt_overlay, 'Prompt Mask (Red)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(pred_overlay, 'Pred Mask (Green)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

                combined_img = np.hstack((gt_overlay, prompt_overlay, pred_overlay))
                cv2.imwrite(save_img_path, combined_img)
                print(f"Visualization saved to: {save_img_path}")


if __name__ == '__main__':
    parser = transformers.HfArgumentParser(DataArguments)
    data_args = parser.parse_args_into_dataclasses()[0]
    thr = data_args.thr
    evaluation(data_args,thr)