import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import argparse
import torch
import os
from enum import Enum
import json
from tqdm import tqdm
import shortuuid
import numpy as np
from psalm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN, DEFAULT_SEG_TOKEN, SEG_TOKEN_INDEX, CLS_TOKEN_INDEX, REFER_TOKEN_INDEX
from psalm.conversation import conv_templates, SeparatorStyle
from psalm.model.builder import load_pretrained_model
from psalm.utils import disable_torch_init
from psalm.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from psalm.eval.segmentation_evaluation.instance_evaluation import InstanceSegEvaluator, my_coco_evaluator
from psalm.eval.segmentation_evaluation.referring_evaluation import my_refcoco_evaluator
from transformers import StoppingCriteria, StoppingCriteriaList
import cv2
from torch.utils.data import Dataset, DataLoader

from psalm import conversation as conversation_lib
from psalm.train.train_datasets import DataCollatorForCOCODatasetV2, RefCOCO_dataset

from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from pycocotools import mask
from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
import torch.distributed as dist
import transformers
import pickle
from pathlib import Path

def parse_outputs(outputs,gt_mask):
    res_list = []
    for output in outputs:
        # gt = output['gt'].cpu().numpy().astype(np.uint8)

        pred_mask = output['instances'].pred_masks
        pred_mask = pred_mask.cpu().numpy()
        scores = output['instances'].scores.cpu().numpy()
        try:
            pred_cls = output['instances'].pred_classes.cpu().numpy()
        except:
            pred_cls = None
        res = {
            'pred':pred_mask,
            'gt': gt_mask,
            'scores':scores,
            'pred_cls':pred_cls
        }
        res_list.append(res)
    return res_list




@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    referring_image_folder: Optional[str] = field(default='/nfs-data1/public/psalm_data/RRSIS-D_split/referring/train/JPEGImages')
    model_path: Optional[str] = field(default="/home/zhangkunquan/code/PSALM/output/checkpoint/RRSIS-D_5")
    mask_config: Optional[str] = field(default="./psalm/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml")
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    json_path: str = '/nfs-data1/public/psalm_data/RRSIS-D_split/referring/train/referring.json'
    model_map_name: str = 'psalm'
    version: str = 'llava_phi'
    output_dir: str = './output/referring_segmentation'
    segmentation: bool = True
    eval_batch_size: int = 1
    dataloader_num_workers: int = 4
    seg_task: Optional[str] = field(default="referring")
    thr: float = 0.6



class gRefcoco_Dataset(RefCOCO_dataset):
    def __getitem__(self, idx):
        data = self.data[idx]
        image_file = data['image_info']['image']
        image_folder = self.data_args.referring_image_folder

        data_dict = {}
        data_dict['file_name'] = os.path.join(image_folder, image_file)
        data_dict['height'] = data['image_info']['height']
        data_dict['width'] = data['image_info']['width']
        data_dict['image_id'] = data['image_info']['image_id']
        data_dict['annotations'] = data['annotations']
        for annotation in data_dict['annotations']:
            annotation['bbox_mode'] = BoxMode.XYXY_ABS
            # annotation['category_id'] = self.coco_id_to_cont_id[annotation['category_id']]
            if annotation['category_id'] in self.coco_id_to_cont_id:
                annotation['category_id'] = self.coco_id_to_cont_id[annotation['category_id']]
            elif annotation['category_id'] in self.coco_id_to_cont_id.values():
                annotation['category_id'] = annotation['category_id']
            else:
                raise ValueError
            annotation['image_id'] = data['image_info']['image_id']

        if isinstance(self.data_args.image_processor,dict):
            processor = self.data_args.image_processor['instance']
        else:
            processor = self.data_args.image_processor
        data_dict = processor.preprocess(data_dict, mask_format=self.mask_format)
        # instruction = data['instruction']
        sentences = data['instruction']
        # prefix_inst = 'Referring Segmentation according to the following instruction:'
        prefix_inst = 'This is an image <image>, Please doing Referring Segmentation according to the following instruction:'
        instruction = ''
        for sent in sentences:
            instruction += ' {}.'.format(sent['sent'])
        sources = [[{'from': 'human', 'value': prefix_inst + '\n<refer>'},
                    {'from': 'gpt', 'value': '\nSure, the segmentation result is <seg>'}]]

        text_dict = self.preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        labels = text_dict['labels'][0]

        token_refer_id = self.preprocess_referring_instruction(instruction)
        refer_embedding_indices = torch.zeros_like(input_ids)
        refer_embedding_indices[input_ids == REFER_TOKEN_INDEX] = 1

        data_dict['input_ids'] = text_dict['input_ids'][0]
        data_dict['labels'] = text_dict['labels'][0]
        data_dict['dataset_type'] = 'referring_coco'

        data_dict['token_refer_id'] = token_refer_id
        data_dict['refer_embedding_indices'] = refer_embedding_indices
        return data_dict

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
    print(f'save suffix is {save_suffix}')
    print(f'current model is {model_path}')
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, model_args=data_args, mask_config=data_args.mask_config, device='cuda')

    data_args.image_processor = image_processor
    data_args.is_multimodal = True
    conversation_lib.default_conversation = conversation_lib.conv_templates[data_args.version]
    
    eval_dataset = gRefcoco_Dataset(json_path=data_args.json_path, tokenizer=tokenizer, data_args=data_args)
    data_collator = DataCollatorForCOCODatasetV2(tokenizer=tokenizer)
    dataloader_params = {
        "batch_size": data_args.eval_batch_size,
        "num_workers": data_args.dataloader_num_workers,
    }
    eval_dataloader = DataLoader(eval_dataset, batch_size=dataloader_params['batch_size'], collate_fn=data_collator,
                                 num_workers=dataloader_params['num_workers'])

    def load_ref_dataset():
        return RefCOCO_dataset(json_path=data_args.json_path, tokenizer=tokenizer, data_args=data_args)

    try:
        DatasetCatalog.register('refcoco_dataset', load_ref_dataset)
    except:
        print('dataset have been registed')
    MetadataCatalog.get('refcoco_dataset').set(stuff_classes=['object'],)
    gt_json_path = data_args.json_path
    with open(gt_json_path) as f:
        gt_data = json.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device=device,dtype=torch.float).eval()
    save_list = []

    seg_image_save_dir = os.path.join(data_args.output_dir)
    Path(seg_image_save_dir).mkdir(parents=True, exist_ok=True) 
    print(f"segmentation image save to : {seg_image_save_dir}")

    with torch.no_grad():
        for idx, inputs in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):

            # print(f"batch idx={idx} inputs 的键：", inputs.keys())
            # print(f"batch idx={idx} attention_mask 形状: {inputs['attention_mask'].shape}")

            current_sample = gt_data[idx]
            sentences = current_sample['instruction'] 
            seg_instruction = ''
            for sent in sentences:
                seg_instruction += ' {}.'.format(sent['sent'])
            seg_instruction = seg_instruction.strip()

            gt = gt_data[idx]['annotations']
            h, w = gt_data[idx]['image_info']['height'], gt_data[idx]['image_info']['width']

            gt_segmentations = [] 
            # generate gt mask
            masks = []
            for annotation in gt:
                gt_segmentations.append(annotation['segmentation'])
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
            if len(masks) == 0:
                gt_mask = np.zeros((h,w), dtype=np.uint8)
            else:
                gt_mask = fuse_masks(masks)

            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            inputs['token_refer_id'] = [ids.to(device) for ids in inputs['token_refer_id']]


            outputs = model.eval_seg(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                images=inputs['images'].float(),
                seg_info=inputs['seg_info'],
                token_refer_id = inputs['token_refer_id'],
                refer_embedding_indices=inputs['refer_embedding_indices'],
                labels=inputs['labels']
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            cur_res = parse_outputs(outputs,gt_mask)

            img_file_name = inputs['seg_info'][0]['file_name']
            raw_img_path = os.path.join(data_args.referring_image_folder, img_file_name)
            save_img_name = os.path.basename(img_file_name).replace('.jpg',f'_{idx}_pred_gt.jpg')
            save_img_path = os.path.join(seg_image_save_dir, save_img_name)

            results = cur_res[0]  
            preds = results['pred'].astype(np.uint8)
            scores = results['scores']
            gt = results['gt']

            pred_mask = []
            for i, score_ in enumerate(scores):
                if score_ > thr:
                    pred_mask.append(preds[i])
                    
            if len(pred_mask) == 0:
                pred_mask = [np.zeros_like(gt, dtype=np.uint8)]
            else:
                pred_mask = [fuse_masks(pred_mask)]  

            raw_img = cv2.imread(raw_img_path)
            alpha = 0.5

            gt_overlay = raw_img.copy()
            gt_overlay[gt == 1] = gt_overlay[gt == 1] * (1 - alpha) + np.array([0, 0, 255]) * alpha
            gt_overlay = gt_overlay.astype(np.uint8)

            seg_str = str(gt_segmentations)  
            max_display_len = 300  
            if len(seg_str) > max_display_len:
                seg_str = seg_str[:max_display_len] + "... (truncated, total len: {})".format(len(seg_str))
            lines = [seg_str[i:i+60] for i in range(0, len(seg_str), 60)]
            for i, line in enumerate(lines):
                y_pos = 30 + i * 20
                if y_pos >= gt_overlay.shape[0] - 10:  
                    break
                cv2.putText(gt_overlay, line, (10, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) 
                

            pred_overlay = raw_img.copy()
            pred_overlay[pred_mask[0] == 1] = pred_overlay[pred_mask[0] == 1] * (1 - alpha) + np.array([0, 255, 0]) * alpha
            pred_overlay = pred_overlay.astype(np.uint8)

            cv2.putText(gt_overlay, 'GT Mask (Red)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(pred_overlay, 'Pred Mask (Green)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
 
            combined_img = np.hstack((gt_overlay, pred_overlay))
            cv2.putText(combined_img, seg_instruction,(10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imwrite(save_img_path, combined_img)
            print(f"Visualization saved to: {save_img_path}")


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(DataArguments)
    data_args = parser.parse_args_into_dataclasses()[0]
    thr = data_args.thr
    evaluation(data_args,thr)