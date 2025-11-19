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

from psalm import conversation as conversation_lib
from psalm.train.train_datasets import COCO_interactive_dataset, DataCollatorForCOCODatasetV2

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
    seg_task: Optional[str] = field(default="region")
    region_mask_type: Optional[str] = field(default=None)

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

def fuse_masks(masks, scores, threshold=0.5):
    """
    将多个预测mask融合成一个mask
    Args:
        masks: 多个预测mask，形状为 [N, H, W]
        scores: 对应的置信度分数，形状为 [N]
        threshold: 分数阈值，高于此阈值的mask才会被融合
    Returns:
        fused_mask: 融合后的二值mask，形状为 [H, W]
    """
    if len(masks) == 0:
        return np.zeros((1, 1), dtype=np.uint8)  # 返回空mask
    
    # 创建一个空的融合mask
    fused_mask = np.zeros(masks[0].shape, dtype=np.uint8)
    
    # 遍历所有mask，根据分数阈值进行融合
    for i, mask in enumerate(masks):
        if scores[i] > threshold:
            # 使用逻辑或操作融合mask
            fused_mask = np.logical_or(fused_mask, mask).astype(np.uint8)
    
    return fused_mask

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
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, model_args=data_args, mask_config=data_args.mask_config, device='cuda')
    # ckpt = torch.load(os.path.join(model_path,'pytorch_model.bin'))
    # model.load_state_dict(ckpt,strict=True)

    data_args.image_processor = image_processor
    data_args.is_multimodal = True
    conversation_lib.default_conversation = conversation_lib.conv_templates[data_args.version]

    eval_dataset = COCO_interactive_dataset(json_path=data_args.json_path, tokenizer=tokenizer, data_args=data_args)
    data_collator = DataCollatorForCOCODatasetV2(tokenizer=tokenizer)

    dataloader_params = {
        "batch_size": data_args.eval_batch_size,
        "num_workers": data_args.dataloader_num_workers,
    }
    eval_dataloader = DataLoader(eval_dataset, batch_size=dataloader_params['batch_size'], collate_fn=data_collator,
                                 num_workers=dataloader_params['num_workers'])

    def load_ref_dataset():
        return COCO_interactive_dataset(json_path=data_args.json_path, tokenizer=tokenizer, data_args=data_args)

    DatasetCatalog.register('refcoco_dataset', load_ref_dataset)
    MetadataCatalog.get('refcoco_dataset').set(stuff_classes=['object'],)
    gt_json_path = data_args.json_path
    with open(gt_json_path) as f:
        gt_data = json.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device=device,dtype=torch.float).eval()
    save_list = []

    with torch.no_grad():
        for idx, inputs in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
            gt = gt_data[idx]['anns']
            h, w = gt_data[idx]['image_info']['height'], gt_data[idx]['image_info']['width']
            # generate gt mask
            masks = []
            for annotation in gt:
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
            gt_mask = np.stack(gt_mask,axis=0)

            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
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

            cur_res = parse_outputs(outputs,gt_mask)
            
            # 融合所有预测mask为一个mask
            if len(cur_res) > 0:
                # 获取所有预测mask和对应的分数
                all_pred_masks = []
                all_scores = []
                
                for res in cur_res:
                    # 每个res包含多个预测mask，我们取分数最高的一个
                    if len(res['scores']) > 0:
                        max_score_idx = np.argmax(res['scores'])
                        all_pred_masks.append(res['pred'][max_score_idx])
                        all_scores.append(res['scores'][max_score_idx])
                
                # 融合所有mask
                if len(all_pred_masks) > 0:
                    fused_mask = fuse_masks(all_pred_masks, all_scores, threshold=0.5)
                else:
                    fused_mask = np.zeros((h, w), dtype=np.uint8)
            else:
                fused_mask = np.zeros((h, w), dtype=np.uint8)
            
            # 只保存融合后的预测结果
            save_info = {
                'fused_pred': mask.encode(np.asfortranarray(fused_mask)),
                'name': inputs['seg_info'][0]['file_name']
            }
            save_list.append(save_info)
    
    # 只保存预测结果，不输出指标
    save_path = os.path.join(data_args.model_path,'pred_pkl')
    Path(save_path).mkdir(parents=True,exist_ok=True)
    with open(os.path.join(save_path,f'pred_{save_suffix}.pkl'),'wb') as f:
        pickle.dump(save_list, f)
    print(f"Predictions saved to: {os.path.join(save_path, f'pred_{save_suffix}.pkl')}")

if __name__ == '__main__':
    evaluation()