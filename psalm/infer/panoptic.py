import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from pycocotools import mask
import numpy as np
import cv2
from psalm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN, DEFAULT_SEG_TOKEN, SEG_TOKEN_INDEX, CLS_TOKEN_INDEX
from psalm.conversation import conv_templates, SeparatorStyle
from psalm.model.builder import load_pretrained_model
from psalm.utils import disable_torch_init
from psalm.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from psalm.eval.segmentation_evaluation.instance_evaluation import InstanceSegEvaluator, my_coco_evaluator
from psalm.eval.segmentation_evaluation.panoptic_evaluation import my_coco_panoptic_evaluator, my_SemSegEvaluator
from transformers import StoppingCriteria, StoppingCriteriaList

from torch.utils.data import Dataset, DataLoader

from psalm import conversation as conversation_lib
from psalm.model.datasets_mapper.coco_instance_mapper import COCOInstanceNewBaselineDatasetMapperForEval

from panopticapi.utils import id2rgb
import io
from PIL import Image
import math
import copy
from detectron2.structures import BoxMode
from detectron2.evaluation import inference_on_dataset, COCOEvaluator
from detectron2.data import MetadataCatalog, DatasetCatalog

from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
from psalm.train.train_datasets import DataCollatorForCOCODatasetV2, COCO_panoptic_dataset

import transformers




@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default='/path/to/val2017')
    model_path: Optional[str] = field(default="/path/to/model")
    mask_config: Optional[str] = field(default="./psalm/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml")
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    json_path: str = '/path/to/coco'
    model_map_name: str = 'psalm'
    version: str = 'llava_phi'
    output_dir: str = './output/panoptic_segmentation'
    segmentation: bool = True
    eval_batch_size: int = 1
    dataloader_num_workers: int = 4
    seg_task: Optional[str] = field(default="panoptic")




class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_token = input_ids[0][-1]
        for stop in self.stops:
            if stop == last_token:
                return True
        return False


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def manual_color_map(id_val, num_classes=26):
    colors = [
        (255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255),
        (128,0,0), (0,128,0), (0,0,128), (128,128,0), (128,0,128), (0,128,128),
        (255,128,0), (255,0,128), (0,255,128), (128,255,0), (128,0,255), (0,128,255),
        (255,128,128), (128,255,128), (128,128,255), (255,64,64), (64,255,64), (64,64,255), (255,192,192)
    ]
    if id_val == 0:
        return (0, 0, 0)  
    elif 1 <= id_val <= num_classes-1:
        return colors[id_val-1]  
    else:
        return (128,128,128)  


def visualize_panoptic_pred(raw_img_path, panoptic_img, save_path):
    """
    Generate the visualization of panoptic segmentation predictions and save it concatenated with the original image.
    Args:
        raw_img_path: Path to the original image (e.g., /path/val2017/000000123456.jpg)
        panoptic_img: ID matrix of panoptic segmentation predictions ([H, W], where each pixel value = category ID * 1000 + instance ID)
        save_path: Path to save the visualization image
    """

    raw_img = cv2.imread(raw_img_path)
    if raw_img is None:
        print("No raw_img!")
    raw_img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

    panoptic_img = panoptic_img.copy()
    panoptic_img[panoptic_img == -1] = 0  

    # panoptic_img_coco = panoptic_img * 1000 + 0
    # panoptic_rgb = id2rgb(panoptic_img_coco)  # [H, W, 3]，RGB

    height, width = panoptic_img.shape
    panoptic_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for h in range(height):
        for w in range(width):
            id_val = panoptic_img[h, w]
            panoptic_rgb[h, w] = manual_color_map(id_val)  

    panoptic_bgr = cv2.cvtColor(panoptic_rgb, cv2.COLOR_RGB2BGR)
    mask = (panoptic_img != 0) 
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    alpha = 0.5
    panoptic_overlay = np.zeros_like(raw_img, dtype=np.uint8)
    panoptic_overlay[mask_3d] = panoptic_bgr[mask_3d]

    pred_overlay = cv2.addWeighted(
        raw_img, (1 - alpha),  
        panoptic_overlay, alpha,  
        0 
    )


    cv2.putText(
        raw_img_rgb, "Raw Image", (10, 30),  
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2  
    )
    cv2.putText(
        pred_overlay, "Panoptic Prediction", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )

    combined_img = np.hstack((raw_img_rgb, pred_overlay))

    combined_img_bgr = cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, combined_img_bgr)
    print(f"save to:{save_path}")


def evaluation():
    parser = transformers.HfArgumentParser(DataArguments)
    data_args = parser.parse_args_into_dataclasses()[0]
    disable_torch_init()
    model_path = os.path.expanduser(data_args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name,mask_config=data_args.mask_config,model_args=data_args)
    data_args.image_processor = image_processor
    data_args.is_multimodal = True

    conversation_lib.default_conversation = conversation_lib.conv_templates[data_args.version]
    eval_dataset = COCO_panoptic_dataset(json_path=data_args.json_path, tokenizer=tokenizer,
                                                                    data_args=data_args, is_train=False)

    data_collator = DataCollatorForCOCODatasetV2(tokenizer=tokenizer)
    dataloader_params = {
        "batch_size": data_args.eval_batch_size,
        "num_workers": data_args.dataloader_num_workers,
    }
    eval_dataloader = DataLoader(eval_dataset, batch_size=dataloader_params['batch_size'], collate_fn=data_collator,
                                 num_workers=dataloader_params['num_workers'])

    def load_instruction_dataset():
        return eval_dataset

    DatasetCatalog.register('instruction_dataset', load_instruction_dataset)
    MetadataCatalog.get('instruction_dataset').set(panoptic_json=eval_dataset.panoptic_json_path,
                                                   panoptic_root=eval_dataset.panoptic_gt_path)

    evaluator = my_coco_panoptic_evaluator('instruction_dataset',
                                  output_dir=data_args.output_dir, dataset_id_to_cont_id=eval_dataset.coco_id_to_cont_id, is_thing_list=eval_dataset.coco_is_thing)
    sem_evaluator = my_SemSegEvaluator('instruction_dataset',
                                  output_dir=data_args.output_dir, dataset_id_to_cont_id=eval_dataset.coco_id_to_cont_id, class_name=eval_dataset.coco_class_name[:-1],ignore_label=255)
    evaluator.reset()
    sem_evaluator.reset()

    vis_save_dir = os.path.join(data_args.output_dir, "segmentation_visualizations")
    os.makedirs(vis_save_dir, exist_ok=True)  
    print(f"segmentation_visualizations save to:{vis_save_dir}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(dtype=torch.float32, device=device).eval()
    with torch.no_grad():
        for idx, inputs in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            outputs = model.eval_seg(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                images=inputs['images'].float(),
                seg_info=inputs['seg_info'],
                class_name_embedding_indices=inputs['class_name_embedding_indices'],
                class_name_ids=inputs['class_name_ids'],
                cls_indices=inputs['cls_indices'],
                labels=inputs['labels'],
                is_thing_list=eval_dataset.coco_is_thing
            )
            save_dir = os.path.join(data_args.output_dir, "segmentation_visualizations")
            os.makedirs(save_dir, exist_ok=True)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            seg_info = inputs['seg_info'][0]  
            raw_img_filename_with_prefix = seg_info.get('file_name')
            print(f"raw_img_filename_with_prefix:{raw_img_filename_with_prefix}")
            raw_img_filename = os.path.basename(raw_img_filename_with_prefix)
            print(f"raw_img_filename:{raw_img_filename}")
            raw_img_path = os.path.join(data_args.image_folder, raw_img_filename)
            print(f"raw_img_path:{raw_img_path}")

            output = outputs[0]
            panoptic_img, _ = output['panoptic_seg'] # panoptic_seg = (panoptic_img, segments_info)
            panoptic_img_np = panoptic_img.cpu().numpy() 

            save_filename = f"{os.path.splitext(raw_img_filename)[0]}_idx{idx}_panoptic_pred.jpg"
            save_path = os.path.join(vis_save_dir, save_filename)

            visualize_panoptic_pred(raw_img_path, panoptic_img_np, save_path)


if __name__ == "__main__":
    evaluation()

