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
    region_cross_image_folder: Optional[str] = field(default='/nfs-data1/zhangkunquan/SIOR/images')
    model_path: Optional[str] = field(default="/home/zhangkunquan/code/PSALM/output/weight/RRSIS-D_SIOR_10")
    mask_config: Optional[str] = field(default="./psalm/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml")
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    json_path: str = '/nfs-data1/zhangkunquan/SIOR/test2.json'
    model_map_name: str = 'psalm'
    version: str = 'llava_phi'
    output_dir: str = './output/interactive_segmentation'
    segmentation: bool = True
    eval_batch_size: int = 1
    dataloader_num_workers: int = 4
    seg_task: Optional[str] = field(default="region")
    region_mask_type: Optional[str] = field(default="scribble_visual_prompt_mask") #'box_visual_prompt_mask||scribble_visual_prompt_mask||point_visual_prompt_mask'
    thr: float = 0.6

def parse_outputs(outputs,gt_mask,gt_mask1):
    # print(f"-----------------------------")
    # print(f"Number of outputs: {len(outputs)}")
    # print(f"Number of GT masks: {len(gt_mask)}")
    # print(f"Number of GT masks1: {len(gt_mask1)}")
    res_list = []
    for output in outputs:
        pred_mask = output['instances'].pred_masks
        pred_mask = pred_mask.cpu().numpy()
        # scores = output['instances'].scores.transpose(1,0).cpu().numpy()
        scores = output['instances'].scores.cpu().numpy().squeeze()
        ogt_mask = output['gt']
        ogt_mask = ogt_mask.cpu().numpy().astype(np.uint8)
        # gt_mask = gt_mask.astype(np.uint8)
        gt_mask1 = gt_mask1.astype(np.uint8)
        # print(f"Output keys: {output.keys()}")  
        print(f"Number of output['gt']: {len(output['gt'])}")
        # print(f"Number of output['instances'].scores: {len(output['instances'].scores)}")
        # print(f"Number of output['instances'].pred_masks: {len(output['instances'].pred_masks)}")
        # print(f"Scores shape: {output['instances'].scores.shape}")
        # print(f"pred_masks shape: {output['instances'].pred_masks.shape}")
        try:
            pred_cls = output['instances'].pred_classes.cpu().numpy()
        except:
            pred_cls = None
        
        # assert scores.shape[0] == gt_mask.shape[0] , f"scores shape {scores.shape[0]} and gt_mask shape {gt_mask.shape[0]} do not match!"
        for i in range(ogt_mask.shape[0]):
            res = {
                'pred':pred_mask,
                'gt': ogt_mask[i],
                'gt1': gt_mask1[i],
                'scores':scores,
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

    eval_dataset = Cross_interactive_dataset(json_path=data_args.json_path, tokenizer=tokenizer, data_args=data_args)
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
            annotations = gt_data[idx]['annotations']
            annotations1 = gt_data[idx]['annotations1']
            h, w = gt_data[idx]['image_info']['height'], gt_data[idx]['image_info']['width']
            h1, w1 = gt_data[idx]['image1_info']['height'], gt_data[idx]['image1_info']['width']

            # generate gt mask
            masks = []
            prompt_masks = []
            for annotation in annotations:
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
                else:
                    if isinstance(annotation['segmentation']['counts'], list):
                        rle = mask_utils.frPyObjects(annotation['segmentation'], *annotation['segmentation']['size'])
                        segm = mask_utils.decode(rle)
                    else:
                        segm = mask_utils.decode(annotation['segmentation'])

                masks.append(segm.astype(np.bool_))
                prompt_masks.append(prompt_segm.astype(np.bool_))


            masks1 = []
            for annotation in annotations1:
                # prompt_mask = annotation[data_args.region_mask_type]
                # if isinstance(prompt_mask['counts'], list):
                #     rle = mask_utils.frPyObjects(prompt_mask, *prompt_mask['size'])
                #     prompt_segm = mask_utils.decode(rle)
                # else:
                #     prompt_segm = mask_utils.decode(prompt_mask)

                if isinstance(annotation['segmentation'], list):
                    segm1 = np.zeros((h1, w1), dtype=np.uint8)
                    for poly in annotation['segmentation']:
                        poly = np.array(poly, dtype=np.int32).reshape(-1, 2)
                        cv2.fillPoly(segm1, [poly], 1)
                else:
                    if isinstance(annotation['segmentation']['counts'], list):
                        rle = mask_utils.frPyObjects(annotation['segmentation'], *annotation['segmentation']['size'])
                        segm1 = mask_utils.decode(rle)
                    else:
                        segm1 = mask_utils.decode(annotation['segmentation'])
                masks1.append(segm1.astype(np.bool_))

            prompt_mask = [prompt_mask_.astype(np.uint8) for prompt_mask_ in prompt_masks]
            # print(f"len prompt_mask: {len(prompt_mask)}")
            prompt_mask = np.stack(prompt_mask, axis=0)
            original_prompt = prompt_mask[0]
            
            # print(f"masks len :{len(masks)}")
            gt_mask = [mask_.astype(np.uint8) for mask_ in masks]
            gt_mask = np.stack(gt_mask, axis=0)
            original_gt = gt_mask[0]

            gt_mask1 = [mask_.astype(np.uint8) for mask_ in masks1]
            gt_mask1 = np.stack(gt_mask1, axis=0)
            

            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

            outputs = model.eval_seg(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                images=inputs['images'].float(),
                images1=inputs['images1'].float(),
                seg_info=inputs['seg_info'],
                labels=inputs['labels']
            )


            if torch.cuda.is_available():
                torch.cuda.synchronize()
            cur_res = parse_outputs(outputs,gt_mask,gt_mask1)

            img_file_name = inputs['seg_info'][0]['file_name']
            img_file_name1 = inputs['seg_info'][0]['file_name1']
            raw_img_path = os.path.join(data_args.region_cross_image_folder, img_file_name)
            raw_img1_path = os.path.join(data_args.region_cross_image_folder, img_file_name1)
            save_img_name = os.path.basename(img_file_name).replace('.jpg',f'_{idx}_pred_gt.jpg')
            save_img_path = os.path.join(seg_image_save_dir, save_img_name)
            
            print(f"len cur_res: {len(cur_res)}")
            print(f"cur_res[0].key: {cur_res[0].keys()}")
            results = cur_res[0]  
            preds = results['pred'].astype(np.uint8)
            scores = results['scores']
            gt = results['gt']
            gt1 = results['gt1']

            pred_mask = []
            for i, score_ in enumerate(scores):
                if score_ > 0.4:
                    pred_mask.append(preds[i])
            print(f"len(pred_mask):{len(pred_mask)}")
            if len(pred_mask) == 0:
                pred_mask = [np.zeros_like(gt, dtype=np.uint8)]
            else:
                pred_mask = [fuse_masks(pred_mask)]  

            raw_img = cv2.imread(raw_img_path)
            raw_img1 = cv2.imread(raw_img1_path)
            alpha = 0.5

            gt_overlay = raw_img.copy()
            gt_overlay[gt == 1] = gt_overlay[gt == 1] * (1 - alpha) + np.array([0, 0, 255]) * alpha
            gt_overlay = gt_overlay.astype(np.uint8)

            prompt_overlay = raw_img.copy()
            prompt_overlay[original_prompt == 1] = prompt_overlay[original_prompt == 1] * (1 - alpha) + np.array([0, 0, 255]) * alpha
            prompt_overlay = prompt_overlay.astype(np.uint8)

            gt1_overlay = raw_img1.copy()
            gt1_overlay[gt1 == 1] = gt1_overlay[gt1 == 1] * (1 - alpha) + np.array([0, 0, 255]) * alpha
            gt1_overlay = gt1_overlay.astype(np.uint8)

            pred_overlay = raw_img1.copy()
            pred_overlay[pred_mask[0] == 1] = pred_overlay[pred_mask[0] == 1] * (1 - alpha) + np.array([0, 255, 0]) * alpha
            pred_overlay = pred_overlay.astype(np.uint8)

            cv2.putText(gt_overlay, ' Reference GT Mask (Red)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(prompt_overlay, ' Reference Prompt Mask (Red)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(gt1_overlay, ' Target GT Mask (Red)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(pred_overlay, ' Target Pred Mask (Green)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
 
            combined_img = np.hstack((gt_overlay, prompt_overlay, gt1_overlay, pred_overlay))
            cv2.imwrite(save_img_path, combined_img)
            print(f"Visualization saved to: {save_img_path}")


if __name__ == '__main__':
    parser = transformers.HfArgumentParser(DataArguments)
    data_args = parser.parse_args_into_dataclasses()[0]
    thr = data_args.thr
    evaluation(data_args,thr)
