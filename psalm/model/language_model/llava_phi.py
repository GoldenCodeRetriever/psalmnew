import os
from typing import List, Optional, Tuple, Union
from addict import Dict
from dataclasses import dataclass
import torch.nn.functional as F
import numpy as np
import pickle
import torch
import math
import cv2
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from psalm.model.visual_prompt_module.context_cluster import region_pooling
from transformers import AutoConfig, AutoModelForCausalLM

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

from psalm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, IMAGE1_TOKEN_INDEX, IMAGE_DEFORM_TOKEN_INDEX, IMAGE1_DEFORM_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN, SEG_TOKEN_INDEX, CLS_TOKEN_INDEX, REGION_TOKEN_INDEX, REFER_TOKEN_INDEX
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.utils.memory import retry_if_cuda_oom
from ..mask_decoder.Mask2Former_Simplify.modeling.transformer_decoder.mask2former_transformer_decoder import \
    MultiScaleMaskedTransformerDecoderForOPTPreTrain
from ..mask_decoder.Mask2Former_Simplify.modeling.pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from ..multimodal_projector.builder import build_vision_projector
from ..multimodal_encoder.swin_trans import build_swin_b, build_swin_l


from ..datasets_mapper.coco_instance_mapper import COCOInstanceNewBaselineDatasetMapper
from ..datasets_mapper.coco_panoptic_mapper import COCOPanopticNewBaselineDatasetMapper
from ..datasets_mapper.coco_semantic_mapper import COCOSemanticNewBaselineDatasetMapper
from psalm.model.mask_decoder.mask_criterion.pretrain_criterion import PSALM_criterion, hungarian_matcher_PSALM
from transformers import PhiModel, PhiForCausalLM, PhiConfig
class LlavaConfig(PhiConfig):
    model_type = "llava_phi"

@dataclass
class CausalOutputWithMask(CausalLMOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    loss_mask: Optional[torch.FloatTensor] = None
    loss_dice: Optional[torch.FloatTensor] = None
    loss_SEG_class: Optional[torch.FloatTensor] = None
    loss_class_name_class: Optional[torch.FloatTensor] = None
    loss_region_class: Optional[torch.FloatTensor] = None
    loss_llm: Optional[torch.FloatTensor] = None
    loss_proposal: Optional[torch.FloatTensor] = None


class PSALMModel(LlavaMetaModel, PhiModel):
    config_class = LlavaConfig

    def __init__(self, config: PhiConfig, mask_decoder_cfg=None):
        super(PSALMModel, self).__init__(config)
        self.cfg = mask_decoder_cfg
        self.projector_outdim = config.hidden_size 
        if hasattr(config, "mm_vision_tower"):
            swin_type = getattr(config,'swin_type','base')
            if swin_type == 'base':
                self.vision_tower = build_swin_b(None)
            else:
                self.vision_tower = build_swin_l(None)
            self.mm_projector = build_vision_projector(config)

            if getattr(config, 'mm_projector_type', 'conv') == 'deformable':
                import copy
                # 确保引用 build_vision_projector
                
                print(f"Initializing baseline_projector in __init__ (swin_type={swin_type})...")
                baseline_config = copy.deepcopy(config)
                # 强制使用 swin_conv 以匹配 Res5 的真实通道数
                baseline_config.mm_projector_type = 'swin_conv'
                # 根据 Swin 类型强制指定输入维度
                baseline_config.mm_input_embeds = 1024 if swin_type == 'base' else 1536
                
                self.baseline_projector = build_vision_projector(baseline_config)

            self.vision_tower.image_processor = {}
            self.vision_tower.image_processor['panoptic'] = COCOPanopticNewBaselineDatasetMapper(self.cfg)
            self.vision_tower.image_processor['instance'] = COCOInstanceNewBaselineDatasetMapper(self.cfg)
            self.vision_tower.image_processor['semantic'] = COCOSemanticNewBaselineDatasetMapper(self.cfg)


    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    # def initialize_vision_modules(self, model_args, fsdp=None):
    #     vision_tower = model_args.vision_tower if hasattr(model_args, 'vision_tower') else model_args.mm_vision_tower
    #     with_norm = model_args.with_norm
    #     with_layernorm = model_args.with_layernorm
    #     pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter if hasattr(model_args,
    #                                                                             'pretrain_mm_mlp_adapter') else None
    #     projector_outdim = self.projector_outdim

    #     self.config.mm_vision_tower = vision_tower
    #     swin_type = getattr(model_args,'swin_type','base')
    #     self.config.swin_type = swin_type
    #     if swin_type == 'base':
    #         vision_tower = build_swin_b(vision_tower)
    #     else:
    #         print('current visual encoder is swin large')
    #         vision_tower = build_swin_l(vision_tower)
    #         self.config.mm_input_embeds = 1536

    #     if fsdp is not None and len(fsdp) > 0:
    #         self.vision_tower = [vision_tower]
    #     else:
    #         self.vision_tower = vision_tower
    #     # if model_args.float32:
    #     #     print('convert sam parameters from fp16 to fp32')
    #     #     for name, module in self.vision_tower.named_modules():
    #     #         module = module.to(torch.float32)z

    #     self.config.use_mm_proj = True
    #     vision_tower.hidden_size = 256
    #     vision_tower.image_processor = {}
    #     vision_tower.image_processor['panoptic'] = COCOPanopticNewBaselineDatasetMapper(self.cfg)
    #     vision_tower.image_processor['instance'] = COCOInstanceNewBaselineDatasetMapper(self.cfg)
    #     vision_tower.image_processor['semantic'] = COCOSemanticNewBaselineDatasetMapper(self.cfg)
    #     # if model_args.seg_task == 'instance':
    #     #     vision_tower.image_processor = COCOInstanceNewBaselineDatasetMapper(self.cfg)
    #     # else:
    #     #     vision_tower.image_processor = COCOPanopticNewBaselineDatasetMapper(self.cfg)
    #     self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'conv')
    #     print(f'current mm_project_type is {self.config.mm_projector_type}, the output dim is {projector_outdim}')
    #     self.config.mm_hidden_size = vision_tower.hidden_size
    #     self.config.with_norm = with_norm
    #     self.config.with_layernorm = with_layernorm
    #     self.config.projector_outdim = projector_outdim
    #     # Deformable模块相关参数
    #     self.config.mm_hidden_dim = getattr(model_args, 'mm_hidden_dim', 256)
    #     self.config.mm_n_heads = getattr(model_args, 'mm_n_heads', 8)
    #     self.config.mm_n_points = getattr(model_args, 'mm_n_points', 4)

    #     if not hasattr(self, "mm_projector"):
    #         self.mm_projector = build_vision_projector(self.config)
    #     else:
    #         print('exist mm_projector, skip init')

    #     if pretrain_mm_mlp_adapter is not None:
    #         mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

    #         def get_w(weights, keyword):
    #             return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

    #         # import ipdb;ipdb.set_trace()
    #         self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=False)
    #         print('load mm_projector pth successfully')

    #     # 初始化 baseline_projector：用于在 deformable 模式下计算 baseline 特征。
    #     # baseline_projector 的输入通道应基于 swin_type（base -> 1024, large -> 1536）。
    #     try:
    #         from psalm.model.multimodal_projector.builder import build_vision_projector
    #         import copy
    #         baseline_config = copy.deepcopy(self.config)
    #         # 使用 swin_conv 变体以接受 res5 的通道数
    #         baseline_config.mm_projector_type = 'swin_conv'
    #         baseline_config.mm_input_embeds = 1024 if getattr(self.config, 'swin_type', 'base') == 'base' else 1536
    #         # 创建 baseline_projector 并绑定到 model
    #         if not hasattr(self, 'baseline_projector'):
    #             self.baseline_projector = build_vision_projector(baseline_config)
    #         else:
    #             print('exist baseline_projector, skip init')

    #         # 若存在预训练权重（mm_projector_weights），将其权重映射并加载到 baseline_projector
    #         if pretrain_mm_mlp_adapter is not None:
    #             try:
    #                 baseline_w = get_w(mm_projector_weights, 'mm_projector')
    #                 # 加载到 baseline_projector（非严格模式以允许少量键不匹配）
    #                 self.baseline_projector.load_state_dict(baseline_w, strict=False)
    #                 print('load baseline_projector pth successfully')
    #             except Exception:
    #                 print('warning: failed to load baseline_projector weights from pretrain_mm_mlp_adapter')
    #     except Exception:
    #         # 不要中断初始化流程，仅打印警告；后续调用会抛出错误以强制用户初始化
    #         print('warning: failed to initialize baseline_projector in initialize_vision_modules')



    def initialize_vision_modules(self, model_args, fsdp=None):
        # ==============================================================================
        # [修复核心] 将 import 移到函数最开头，防止 UnboundLocalError
        # ==============================================================================
        from psalm.model.multimodal_projector.builder import build_vision_projector
        import copy
        # ==============================================================================

        vision_tower = model_args.vision_tower if hasattr(model_args, 'vision_tower') else model_args.mm_vision_tower
        with_norm = model_args.with_norm
        with_layernorm = model_args.with_layernorm
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter if hasattr(model_args,
                                                                                'pretrain_mm_mlp_adapter') else None
        projector_outdim = self.projector_outdim

        # [修复点 1] 提前加载权重文件
        mm_projector_weights = None
        if pretrain_mm_mlp_adapter is not None:
            print(f'Loading mm_projector_weights from {pretrain_mm_mlp_adapter} ...')
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

        def get_w(weights, keyword):
            return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

        self.config.mm_vision_tower = vision_tower
        swin_type = getattr(model_args, 'swin_type', 'base')
        self.config.swin_type = swin_type

        if swin_type == 'base':
            vision_tower = build_swin_b(vision_tower)
        else:
            print('current visual encoder is swin large')
            vision_tower = build_swin_l(vision_tower)
            self.config.mm_input_embeds = 1536

        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower = vision_tower

        self.config.use_mm_proj = True
        vision_tower.hidden_size = 256
        vision_tower.image_processor = {}
        vision_tower.image_processor['panoptic'] = COCOPanopticNewBaselineDatasetMapper(self.cfg)
        vision_tower.image_processor['instance'] = COCOInstanceNewBaselineDatasetMapper(self.cfg)
        vision_tower.image_processor['semantic'] = COCOSemanticNewBaselineDatasetMapper(self.cfg)

        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'conv')
        print(f'current mm_project_type is {self.config.mm_projector_type}, the output dim is {projector_outdim}')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.with_norm = with_norm
        self.config.with_layernorm = with_layernorm
        self.config.projector_outdim = projector_outdim

        # Deformable模块相关参数
        self.config.mm_hidden_dim = getattr(model_args, 'mm_hidden_dim', 256)
        self.config.mm_n_heads = getattr(model_args, 'mm_n_heads', 8)
        self.config.mm_n_points = getattr(model_args, 'mm_n_points', 4)

        # ==============================================================================
        # [修复点 2] 初始化 Baseline Projector (仅在 Deformable 模式下)
        # ==============================================================================
        if self.config.mm_projector_type == 'deformable':
            print("--> Initializing Baseline Projector for Deformable Mode...")
            
            baseline_config = copy.deepcopy(self.config)
            # 强制使用 swin_conv 以匹配 Res5 的真实通道数
            baseline_config.mm_projector_type = 'swin_conv'
            # 根据 Swin 类型强制指定输入维度 (解决 256 vs 1024 报错)
            baseline_config.mm_input_embeds = 1024 if swin_type == 'base' else 1536

            if not hasattr(self, 'baseline_projector'):
                self.baseline_projector = build_vision_projector(baseline_config)
                print(f"    Created baseline_projector with input dim: {baseline_config.mm_input_embeds}")
            else:
                print('    exist baseline_projector, skip init')

            # 加载权重到 baseline_projector
            if mm_projector_weights is not None:
                try:
                    print("    Loading weights into baseline_projector...")
                    baseline_w = get_w(mm_projector_weights, 'mm_projector')
                    # 加载权重
                    incompatible_keys = self.baseline_projector.load_state_dict(baseline_w, strict=False)
                    print(f"    Baseline weights loaded. Incompatible keys: {incompatible_keys}")
                except Exception as e:
                    print(f'    [WARNING] Failed to load baseline_projector weights: {e}')
            else:
                print("    [WARNING] No weights found for baseline_projector (Random Init)!")
        # ==============================================================================

        # 初始化原本的 mm_projector (Deformable 模式下这里初始化的是 Deformable 模块)
        if not hasattr(self, "mm_projector"):
            # 这里调用 build_vision_projector 不会再报错了，因为 import 已经在最上面执行了
            self.mm_projector = build_vision_projector(self.config)
        else:
            print('exist mm_projector, skip init')

        # 加载权重 (非 Deformable 模式下给 mm_projector 加载)
        if self.config.mm_projector_type != 'deformable' and mm_projector_weights is not None:
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=False)
            print('load mm_projector pth successfully')





class PSALM(PhiForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config, mask_decoder_cfg=None, add_cross_attn=True, cross_attn_index=None):
        super(PSALM, self).__init__(config)

        self.model = PSALMModel(config, mask_decoder_cfg)
        self.init_config = config
        self.mask_decoder_cfg = mask_decoder_cfg
        self.cross_attn_index = cross_attn_index
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        is_train_mask_decode = getattr(config, 'mask_decode_train', False)
        self.is_train_mask_decode = is_train_mask_decode
        self.refer_pooling = nn.AdaptiveAvgPool1d(output_size=1)
        self.class_name_pooling = nn.AdaptiveAvgPool1d(output_size=1)
        self.region_sampler = region_pooling(num_sample_point=256)
        self.region_projector = nn.Linear(config.hidden_size, mask_decoder_cfg.MODEL.MASK_FORMER.HIDDEN_DIM)

        if is_train_mask_decode:
            print('Mask Decoder has been trained, init directly')
            self.initial_mask_module()
        self.post_init()

    def initial_mask_module(self, pretrained_path=None, model_args=None):
        if not self.is_train_mask_decode:
            print('Initialize mask modules...')
            self.config.mask_decode_train = True
        self.seg_query = nn.Parameter(
            torch.zeros([self.mask_decoder_cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES, self.config.hidden_size]))
        self.num_queries = self.mask_decoder_cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        self.num_classes = self.mask_decoder_cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        self.test_topk_per_image = self.mask_decoder_cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        input_shape = self.output_shape()
        self.pixel_decoder = self.pixel_decoder_init(cfg=self.mask_decoder_cfg, input_shape=input_shape)
        self.predictor = self.predictor_init(cfg=self.mask_decoder_cfg)

        self.seg_query_projector = nn.Linear(self.config.hidden_size, self.mask_decoder_cfg.MODEL.MASK_FORMER.HIDDEN_DIM)
        self.SEG_token_projector = nn.Linear(self.config.hidden_size, self.mask_decoder_cfg.MODEL.MASK_FORMER.HIDDEN_DIM)
        self.class_name_projector = nn.Linear(self.config.hidden_size, self.mask_decoder_cfg.MODEL.MASK_FORMER.HIDDEN_DIM)

        self.mask_decoder_training_init(self.mask_decoder_cfg)
        if pretrained_path is not None:
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            def change_w(weights, old_name, new_name):
                weights[new_name] = weights[old_name]
                weights.pop(old_name)

            if pretrained_path.endswith('.pkl'):
                with open(pretrained_path, 'rb') as f:
                    ckpt = pickle.load(f)
            else:
                ckpt = torch.load(pretrained_path)
            pixel_decoder_weights = get_w(ckpt['model'],'sem_seg_head.pixel_decoder')
            predictor_weights = get_w(ckpt['model'],'sem_seg_head.predictor')
            pixel_decoder_weights = {k: torch.tensor(v) for k, v in pixel_decoder_weights.items()}
            predictor_weights = {k: torch.tensor(v) for k, v in predictor_weights.items()}

            #deal some diff keys
            change_w(pixel_decoder_weights,'adapter_1.weight','adapter_1.0.weight')
            change_w(pixel_decoder_weights,'adapter_1.norm.weight','adapter_1.1.weight')
            change_w(pixel_decoder_weights,'adapter_1.norm.bias','adapter_1.1.bias')
            change_w(pixel_decoder_weights,'layer_1.weight','layer_1.0.weight')
            change_w(pixel_decoder_weights,'layer_1.norm.weight','layer_1.1.weight')
            change_w(pixel_decoder_weights,'layer_1.norm.bias','layer_1.1.bias')
            if 'static_query.weight' in predictor_weights:
                change_w(predictor_weights,'static_query.weight','query_feat.weight')
            if predictor_weights['query_embed.weight'].shape[0] == 200:
                predictor_weights['query_embed.weight'] = predictor_weights['query_embed.weight'][:100,:]
            diff_pixel_msg = self.pixel_decoder.load_state_dict(pixel_decoder_weights,strict=False)
            diff_predictor_msg = self.predictor.load_state_dict(predictor_weights,strict=False)
            print(diff_predictor_msg)
            print(diff_pixel_msg)


    def get_vision_tower_feature(self, images):
        features = self.get_model().get_vision_tower()(images)
        features_dict = {
            'res2': features[0],
            'res3': features[1],
            'res4': features[2],
            'res5': features[3],
        }
        return features_dict
    def mask_decoder_training_init(self, cfg):
        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        # boundary_weight = cfg.MODEL.MASK_FORMER.BOUNDARY_WEIGHT

        matcher = hungarian_matcher_PSALM(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_SEG_class": class_weight, "loss_class_name_class": class_weight, "loss_mask": mask_weight,
                       "loss_dice": dice_weight, "loss_region_class": class_weight}
        self.weight_dict = weight_dict
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        losses = ["SEG_labels", "class_name_labels", "masks", "region_labels"]
        self.criterion = PSALM_criterion(
            matcher=matcher,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            device=self.device
        )
        self.size_divisibility = 32
        if cfg.MODEL.MASK_FORMER.SEG_TASK == 'semantic':
            self.semantic_on = True
            self.instance_on = False
            self.panoptic_on = False
            self.referring_on = False
            self.region_on = False

        elif cfg.MODEL.MASK_FORMER.SEG_TASK == 'instance':
            self.semantic_on = False
            self.instance_on = True
            self.panoptic_on = False
            self.referring_on = False
            self.region_on = False
        elif cfg.MODEL.MASK_FORMER.SEG_TASK == 'panoptic':
            self.semantic_on = True
            self.instance_on = True
            self.panoptic_on = True
            self.referring_on = False
            self.region_on = False
        elif cfg.MODEL.MASK_FORMER.SEG_TASK == 'referring':
            self.semantic_on = False
            self.instance_on = False
            self.panoptic_on = False
            self.referring_on = True
            self.region_on = False
        elif cfg.MODEL.MASK_FORMER.SEG_TASK == 'region':
            self.semantic_on = False
            self.instance_on = False
            self.panoptic_on = False
            self.referring_on = False
            self.region_on = True
        else:
            raise NotImplementedError
        self.sem_seg_postprocess_before_inference = self.instance_on or self.panoptic_on or self.referring_on or self.region_on
    def get_region_embedding(self, hidden_states, region_embedding_masks):
        region_embedding_list = []
        for sample_hidden_satates, sample_region_embedding_masks in zip(hidden_states, region_embedding_masks):
            sample_region_embedding = sample_hidden_satates[sample_region_embedding_masks.bool()]
            region_embedding_list.append(sample_region_embedding)
        return region_embedding_list
    def SEG_instance_inference(self, SEG_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        scores = F.sigmoid(SEG_cls)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)

        mask_pred = mask_pred[topk_indices]

        result = Instances(image_size)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))

        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (
                result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        return result
    def class_name_panoptic_inference(self, SEG_cls, class_name_cls, mask_pred):

        scores, labels = F.softmax(class_name_cls, dim=-1).max(-1)
        num_classes = class_name_cls.shape[-1] - 1
        mask_pred = mask_pred.sigmoid()

        object_mask_threshold = 0.8
        overlap_threshold = 0.8

        keep = labels.ne(num_classes) & (scores > object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = class_name_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = self.is_thing_list[pred_class]
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info
    def region_inference(self, region_cls, mask_pred):
        image_size = mask_pred.shape[-2:]

        scores = F.sigmoid(region_cls)





        result = Instances(image_size)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))

        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (
                result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = (scores * mask_scores_per_image[None,...].repeat(scores.shape[0],1)).transpose(1,0)
        return result

    def class_name_semantic_inference(self, SEG_cls, class_name_cls, mask_pred):
        mask_cls = F.softmax(class_name_cls, dim=-1)[:, :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg
    def class_name_instance_inference(self, SEG_cls, class_name_cls, mask_pred):
        image_size = mask_pred.shape[-2:]

        cls_scores = F.softmax(class_name_cls, dim=-1)[:, :-1]
        scores = cls_scores

        num_classes = scores.shape[-1]

        labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)

        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(5000, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // num_classes
        mask_pred = mask_pred[topk_indices]


        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = self.is_thing_list[lab]

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (
                result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result
    def encode_images(self, images, queries_z_q=None):
        """
        编码图像特征。支持两种模式：
        1. 原始模式（baseline）：仅使用res5特征，通过线性/卷积投影
        2. Deformable模式：使用多尺度特征，通过可变注意力采样
        
        Args:
            images: 输入图像 (B, C, H, W)
            queries_z_q: Deformable模式下的查询特征 (B, N_q, query_dim)，原始模式下为None
        
        Returns:
            投影后的特征 (B, seq_len, hidden_dim)
        """
        from psalm.model.multimodal_projector.deformable_alignment import MultiScaleDeformableCrossAttentionAlignment
        
        # 获取多尺度视觉特征
        multi_scale_features_list = self.get_model().get_vision_tower()(images)  # [res2, res3, res4, res5]
        
        projector = self.get_model().mm_projector
        
        # 检查 projector 是否是Deformable模块
        if isinstance(projector, MultiScaleDeformableCrossAttentionAlignment):
            # --- Deformable多尺度模式 ---
            # 将list转换为dict格式
            multi_scale_features = {
                'res2': multi_scale_features_list[0],
                'res3': multi_scale_features_list[1],
                'res4': multi_scale_features_list[2],
                'res5': multi_scale_features_list[3],
            }
            
            # 调用Deformable模块
            if queries_z_q is None:
                raise ValueError(
                    "Deformable projector requires queries_z_q, but got None. "
                    "Please ensure create_deformable_queries() is called before encode_images()."
                )
            
            projected = projector(
                queries=queries_z_q,
                multi_scale_features=multi_scale_features
            )  # (B, N_q, projector_outdim)
        else:
            # --- 原始模式（baseline） ---
            # 仅使用res5特征
            res5_features = multi_scale_features_list[-1]  # (B, C, H, W)
            # 调用原有projector
            projected = projector(res5_features)
        
        return projected
    

    def predictor_init(self, cfg):
        in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        hidden_dim = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        num_queries = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        nheads = cfg.MODEL.MASK_FORMER.NHEADS
        dim_feedforward = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        pre_norm = cfg.MODEL.MASK_FORMER.PRE_NORM
        mask_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        enforce_input_project = False
        seg_norm = cfg.MODEL.MASK_FORMER.SEG_NORM
        seg_proj = cfg.MODEL.MASK_FORMER.SEG_PROJ
        seg_fuse_score = cfg.MODEL.MASK_FORMER.FUSE_SCORE
        seg_concat = False
        print(f'current seg concat mode: {seg_concat}, seg_norm: {seg_norm}, seg_proj: {seg_proj}, seg_fuse_score: {seg_fuse_score}')
        predictor = MultiScaleMaskedTransformerDecoderForOPTPreTrain(in_channels,
                                                                     hidden_dim,
                                                                     num_queries,
                                                                     nheads,
                                                                     dim_feedforward,
                                                                     dec_layers,
                                                                     pre_norm,
                                                                     mask_dim,
                                                                     enforce_input_project,
                                                                     seg_norm,
                                                                     seg_concat,
                                                                     seg_proj,
                                                                     seg_fuse_score
                                                                     )
        return predictor


    def get_model(self):
        return self.model
    def output_shape(self):
        out_features = self.mask_decoder_cfg.MODEL.SWIN.OUT_FEATURES
        out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        num_features = [int(self.mask_decoder_cfg.MODEL.SWIN.EMBED_DIM * 2 ** i) for i in
                        range(len(self.mask_decoder_cfg.MODEL.SWIN.DEPTHS))]
        out_feature_channels = {
            "res2": num_features[0],
            "res3": num_features[1],
            "res4": num_features[2],
            "res5": num_features[3],
        }
        backbone_feature_shape = dict()
        for name in out_features:
            backbone_feature_shape[name] = Dict(
                {'channel': out_feature_channels[name], 'stride': out_feature_strides[name]})
        return backbone_feature_shape

    def get_encoder_image(self, images):
        encode_image_features = self.get_model().get_vision_tower()(images)
        return encode_image_features

    def pixel_decoder_init(self, cfg, input_shape):
        common_stride = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        transformer_dropout = cfg.MODEL.MASK_FORMER.DROPOUT
        transformer_nheads = cfg.MODEL.MASK_FORMER.NHEADS
        transformer_dim_feedforward = 1024
        transformer_enc_layers = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS
        conv_dim = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        mask_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        transformer_in_features = cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES  # ["res3", "res4", "res5"]

        pixel_decoder = MSDeformAttnPixelDecoder(input_shape,
                                                 transformer_dropout,
                                                 transformer_nheads,
                                                 transformer_dim_feedforward,
                                                 transformer_enc_layers,
                                                 conv_dim,
                                                 mask_dim,
                                                 transformer_in_features,
                                                 common_stride)
        return pixel_decoder
    # def prepare_targets(self, targets, images):
    #     h_pad, w_pad = images.shape[-2:]
    #     new_targets = []
        
    #     for targets_per_image in targets:
    #         # pad gt
    #         gt_masks = targets_per_image.gt_masks
    #         padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
    #         padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
    #         new_targets.append(
    #             {
    #                 "labels": targets_per_image.gt_classes,
    #                 "masks": padded_masks,
    #             }
    #         ) 
    #     return new_targets

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.shape[-2:]
        new_targets = []
        
        # 创建保存目录（如果不存在）
        save_dir = "./output/test"
        os.makedirs(save_dir, exist_ok=True)
        
        for idx, targets_per_image in enumerate(targets):
            # 处理GT mask（填充到图像尺寸）
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), 
                                    dtype=gt_masks.dtype, 
                                    device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append({
                "labels": targets_per_image.gt_classes,
                "masks": padded_masks,
            })
            
            # 保存红色mask（黑色背景），只处理第一个mask
            if gt_masks.numel() == 0:
                continue  # 没有mask则跳过
            
            # 取第一个mask并转换为numpy
            mask_np = padded_masks[0].cpu().detach().numpy()
            mask_np = (mask_np > 0).astype(np.uint8)  # 二值化（0或1）
            
            # 创建黑色背景 (H, W, 3)
            black_bg = np.zeros((h_pad, w_pad, 3), dtype=np.uint8)
            # 将mask区域设为红色 (BGR格式，红色为(0,0,255))
            black_bg[mask_np == 1] = [0, 0, 255]
            
            # 保存图像
            save_path = os.path.join(save_dir, f"red_mask_{idx}.png")
            cv2.imwrite(save_path, black_bg)
        
        return new_targets


    def get_special_token(self, SEG, EOS):
        self.SEG_id = SEG
        self.EOS_id = EOS

    def get_class_name_embedding(self, hidden_states, cls_token_indices):
        class_name_embedding_list = []
        for current_hidden_state, current_token_indice in zip(hidden_states, cls_token_indices):
            class_id = torch.unique(current_token_indice)
            class_id = class_id[class_id != 0]
            current_class_name_embedding_list = []
            for id in class_id:
                current_class_mask = (current_token_indice == id)
                current_class_state = current_hidden_state[current_class_mask]
                current_class_name_embedding_list.append(current_class_state)
            current_pool_class_name_embedding = [self.class_name_pooling(class_name.transpose(-2, -1)).transpose(-2, -1)
                                                 for class_name in current_class_name_embedding_list]
            class_name_embedding_list.append(torch.cat(current_pool_class_name_embedding, dim=0))
        return torch.stack(class_name_embedding_list, dim=0)
    def embed_class_ids(self, class_name_ids, cls_indices):
        if class_name_ids is None:
            return None
        num_class = cls_indices.unique_consecutive()
        num_class = num_class[num_class >= 0]
        class_name_ids = [class_name_ids[cls_indices == idx] for idx in num_class]
        embedded_class_name = [self.get_model().embed_tokens(id) for id in class_name_ids]

        return embedded_class_name

    def embed_refer_ids(self, refer_ids):
        if refer_ids is None:
            return None
        embedded_refer = self.get_model().embed_tokens(refer_ids)
        return embedded_refer
    
    def create_deformable_queries(self, task_type, batch_size, device, 
                                  class_name_embedding=None, 
                                  region_feature_list=None,
                                  refer_embeddings=None):
        """
        为Deformable Cross-Attention生成查询特征。
        
        根据任务类型生成相应的查询向量：
        - 'referring': 从文本指代词的嵌入中提取
        - 'panoptic'/'semantic': 从类别名称的嵌入中提取
        - 'interactive'/'region': 从用户交互提示的池化特征中提取
        - 'cross_image': 从跨图提示特征中提取
        - 其他: 使用学习的分割查询向量
        
        Args:
            task_type: str, 任务类型标识
            batch_size: int, 批量大小
            device: torch.device, 设备
            class_name_embedding: (B, num_classes, hidden_dim), 可选，类别名称嵌入
            region_feature_list: list, 可选，区域特征列表
            refer_embeddings: (B, 1, hidden_dim), 可选，指代词的嵌入特征
        
        Returns:
            queries_z_q: (B, N_q, query_dim), 查询特征
        """
        from psalm.model.multimodal_projector.deformable_alignment import MultiScaleDeformableCrossAttentionAlignment
        
        projector = self.get_model().mm_projector
        if not isinstance(projector, MultiScaleDeformableCrossAttentionAlignment):
            return None, None

        query_dim = projector.query_dim
        
        # 方案1：Panoptic/Semantic分割 - 使用类别名称嵌入作为查询
        if 'panoptic' in task_type or 'semantic' in task_type:
            if class_name_embedding is not None:
                # class_name_embedding: (B, num_classes, hidden_dim)
                queries_z_q = class_name_embedding
                source = 'class_name'
            else:
                # 后备方案：使用学习的分割查询
                queries_z_q = self.seg_query.unsqueeze(0).expand(batch_size, -1, -1)
                source = 'seg_query'
                # 投影到query_dim
                if queries_z_q.shape[-1] != query_dim:
                    # 使用线性投影
                    proj_layer = nn.Linear(queries_z_q.shape[-1], query_dim).to(device)
                    queries_z_q = proj_layer(queries_z_q)
        
        # 方案2：Referring分割 - 使用指代词嵌入作为查询
        elif 'referring' in task_type:
            if refer_embeddings is not None:
                queries_z_q = refer_embeddings
                source = 'refer'
                # 确保维度匹配
                if queries_z_q.dim() == 2: # (B, C) -> (B, 1, C)
                    queries_z_q = queries_z_q.unsqueeze(1)
                
                if queries_z_q.shape[-1] != query_dim:
                    proj_layer = nn.Linear(queries_z_q.shape[-1], query_dim).to(device)
                    queries_z_q = proj_layer(queries_z_q)
            
            else:   
                # 后备方案
                queries_z_q = self.seg_query.unsqueeze(0).expand(batch_size, -1, -1)
                source = 'seg_query'
                if queries_z_q.shape[-1] != query_dim:
                    proj_layer = nn.Linear(queries_z_q.shape[-1], query_dim).to(device)
                    queries_z_q = proj_layer(queries_z_q)
        
        # 方案3：Interactive/Region分割 - 使用区域提示特征作为查询
        elif 'interactive' in task_type or 'region' in task_type:
            if region_feature_list is not None:
                # region_feature_list: list of (num_regions, hidden_dim) tensors
                # 对每个样本进行处理
                queries_list = []
                for region_feat in region_feature_list:
                    if region_feat is not None and region_feat.numel() > 0:
                        # region_feat: (num_regions, hidden_dim)
                        queries_list.append(region_feat)
                    else:
                        # 无区域信息，使用零向量
                        queries_list.append(torch.zeros(1, query_dim, device=device))
                
                # 需要对齐所有查询的长度
                max_num_regions = max(q.shape[0] for q in queries_list) if queries_list else 1
                queries_aligned = []
                for q in queries_list:
                    if q.shape[0] < max_num_regions:
                        padding = torch.zeros(max_num_regions - q.shape[0], q.shape[1], device=device)
                        q = torch.cat([q, padding], dim=0)
                    queries_aligned.append(q)
                queries_z_q = torch.stack(queries_aligned, dim=0)  # (B, max_num_regions, hidden_dim)
                source = 'region'
                # 投影到query_dim
                if queries_z_q.shape[-1] != query_dim:
                    proj_layer = nn.Linear(queries_z_q.shape[-1], query_dim).to(device)
                    queries_z_q = proj_layer(queries_z_q)
            else:
                # 后备方案
                queries_z_q = self.seg_query.unsqueeze(0).expand(batch_size, -1, -1)
                source = 'seg_query'
                if queries_z_q.shape[-1] != query_dim:
                    proj_layer = nn.Linear(queries_z_q.shape[-1], query_dim).to(device)
                    queries_z_q = proj_layer(queries_z_q)
        
        # 方案4：Cross-image分割 - 使用跨图提示特征作为查询
        elif 'cross_image' in task_type or 'cross' in task_type:
            # 跨图任务可以使用区域特征或默认查询
            if region_feature_list is not None:
                queries_list = []
                for region_feat in region_feature_list:
                    if region_feat is not None and region_feat.numel() > 0:
                        queries_list.append(region_feat)
                    else:
                        queries_list.append(torch.zeros(1, query_dim, device=device))
                
                max_num_regions = max(q.shape[0] for q in queries_list) if queries_list else 1
                queries_aligned = []
                for q in queries_list:
                    if q.shape[0] < max_num_regions:
                        padding = torch.zeros(max_num_regions - q.shape[0], q.shape[1], device=device)
                        q = torch.cat([q, padding], dim=0)
                    queries_aligned.append(q)
                queries_z_q = torch.stack(queries_aligned, dim=0)
                source = 'region'
                if queries_z_q.shape[-1] != query_dim:
                    proj_layer = nn.Linear(queries_z_q.shape[-1], query_dim).to(device)
                    queries_z_q = proj_layer(queries_z_q)
            else:
                queries_z_q = self.seg_query.unsqueeze(0).expand(batch_size, -1, -1)
                source = 'seg_query'
                if queries_z_q.shape[-1] != query_dim:
                    proj_layer = nn.Linear(queries_z_q.shape[-1], query_dim).to(device)
                    queries_z_q = proj_layer(queries_z_q)
        
        # 其他任务 - 使用学习的分割查询向量
        else:
            queries_z_q = self.seg_query.unsqueeze(0).expand(batch_size, -1, -1)
            source = 'seg_query'
            if queries_z_q.shape[-1] != query_dim:
                proj_layer = nn.Linear(queries_z_q.shape[-1], query_dim).to(device)
                queries_z_q = proj_layer(queries_z_q)
        
        # 确保查询维度匹配
        # if queries_z_q.shape[-1] != query_dim:
        #     raise ValueError(
        #         f"Query dimension mismatch: expected {query_dim}, got {queries_z_q.shape[-1]}. "
        #         f"Please check the projection layer."
        #     )
        
        if queries_z_q.dim() == 4:
            # 如果形状是 [B, N, 1, C]，压缩第2维
            if queries_z_q.shape[2] == 1:
                queries_z_q = queries_z_q.squeeze(2)
            # 如果形状是 [B, N, C, 1]，压缩第3维
            elif queries_z_q.shape[3] == 1:
                queries_z_q = queries_z_q.squeeze(3)
            else:
                # 如果是其他情况，尝试 flatten，假设最后两维是空间维度或者多余维度
                # 但通常 deformable query 应该是 (B, N, C)
                # 这里做一个安全处理：合并 B, N 之后的维度
                B, N = queries_z_q.shape[0], queries_z_q.shape[1]
                queries_z_q = queries_z_q.view(B, N, -1)
                
        # 再次确认维度
        if queries_z_q.shape[-1] != query_dim:
            # 不匹配时仍返回（上层会处理）
            pass

        return queries_z_q, source

        
    def concat_image_seg_cls_embeds(self, input_id, img_feature, img1_feature, label, seg_query, seg_query_mask, class_embed,
                                    class_name_embedding_indices,region_embedding_mask=None, region_feature_list=None, refer_embedding_indices=None,
                refer_embedding=None, img_deform_feature=None, img1_deform_feature=None):
       

        image_token_indices = torch.where(input_id == IMAGE_TOKEN_INDEX)[0]
        image1_token_indices = torch.where(input_id == IMAGE1_TOKEN_INDEX)[0]
        image_deform_token_indices = torch.where(input_id == IMAGE_DEFORM_TOKEN_INDEX)[0]
        image1_deform_token_indices = torch.where(input_id == IMAGE1_DEFORM_TOKEN_INDEX)[0]
        seg_query_indices = torch.where(input_id == SEG_TOKEN_INDEX)[0]
        cls_token_indices = torch.where(input_id == CLS_TOKEN_INDEX)[0]
        region_token_indices = torch.where(input_id == REGION_TOKEN_INDEX)[0]
        assert len(image_token_indices) <= 1, 'not supporting multi image index'
        assert len(seg_query_indices) == 1, 'not supporting multi seg index'
        if class_name_embedding_indices is not None:
            assert len(cls_token_indices) == len(class_embed), 'the number of <cls> tokens and class_embed needs to be same'
        if region_feature_list is not None:
            assert len(region_feature_list) == len(
                region_token_indices), 'the munber of <region> tokens and regions needs to be same'
        cur_new_input_embeds = []
        cur_new_seg_query_mask = []
        if label is not None:
            cur_new_label = []
            assert label.shape == input_id.shape
        else:
            cur_new_label = None
        cur_class_name_embedding_indices = [] if class_name_embedding_indices is not None else None
        cur_refer_embedding_indices = [] if refer_embedding_indices is not None else None

        if region_embedding_mask is not None:
            enable_region_mask = True
            cur_new_region_embedding_mask = []
        else:
            enable_region_mask = False
            cur_new_region_embedding_mask = None
        chunks = []
        current_chunk = []

        for id in input_id:
            if id >= 0:
                current_chunk.append(id.item())
            else:
                if current_chunk:
                    chunks.append(torch.tensor(current_chunk, device=input_id.device))
                    current_chunk = []
                chunks.append([id])
        if current_chunk:
            chunks.append(torch.tensor(current_chunk, device=input_id.device))

        cls_idx = 0
        region_idx = 0
        for chunk in chunks:
            chunk_len = len(chunk)
            if chunk_len == 1 and chunk[0] == IMAGE_TOKEN_INDEX:
                cur_new_input_embeds.append(img_feature)
                cur_new_seg_query_mask.append(torch.zeros(img_feature.shape[0]))
                if class_name_embedding_indices is not None:
                    cur_class_name_embedding_indices.append(
                        torch.full((img_feature.shape[0],), 0, device=input_id.device,
                                   dtype=input_id.dtype))
                if refer_embedding_indices is not None:
                    cur_refer_embedding_indices.append(
                        torch.full((img_feature.shape[0],), 0, device=input_id.device,
                                   dtype=input_id.dtype))
                if label is not None:
                    cur_new_label.append(
                        torch.full((img_feature.shape[0],), IGNORE_INDEX, device=label.device,
                                   dtype=label.dtype)
                    )
                if enable_region_mask:
                    cur_new_region_embedding_mask.append(torch.zeros(img_feature.shape[0]))

            elif chunk_len == 1 and chunk[0] == IMAGE1_TOKEN_INDEX:
                cur_new_input_embeds.append(img1_feature)
                cur_new_seg_query_mask.append(torch.zeros(img1_feature.shape[0]))
                if class_name_embedding_indices is not None:
                    cur_class_name_embedding_indices.append(
                        torch.full((img1_feature.shape[0],), 0, device=input_id.device, dtype=input_id.dtype)
                    )
                if refer_embedding_indices is not None:
                    cur_refer_embedding_indices.append(
                        torch.full((img1_feature.shape[0],), 0, device=input_id.device, dtype=input_id.dtype)
                    )
                if label is not None:
                    cur_new_label.append(
                        torch.full((img1_feature.shape[0],), IGNORE_INDEX, device=label.device, dtype=label.dtype)
                    )
                if enable_region_mask:
                    cur_new_region_embedding_mask.append(torch.zeros(img1_feature.shape[0]))  

            elif chunk_len == 1 and chunk[0] == IMAGE_DEFORM_TOKEN_INDEX:
                # Deformable模块的特征token
                if img_deform_feature is not None:
                    cur_new_input_embeds.append(img_deform_feature)
                    cur_new_seg_query_mask.append(torch.zeros(img_deform_feature.shape[0]))
                    if class_name_embedding_indices is not None:
                        cur_class_name_embedding_indices.append(
                            torch.full((img_deform_feature.shape[0],), 0, device=input_id.device, dtype=input_id.dtype)
                        )
                    if refer_embedding_indices is not None:
                        cur_refer_embedding_indices.append(
                            torch.full((img_deform_feature.shape[0],), 0, device=input_id.device, dtype=input_id.dtype)
                        )
                    if label is not None:
                        cur_new_label.append(
                            torch.full((img_deform_feature.shape[0],), IGNORE_INDEX, device=label.device, dtype=label.dtype)
                        )
                    if enable_region_mask:
                        cur_new_region_embedding_mask.append(torch.zeros(img_deform_feature.shape[0]))
                else:
                    # 如果没有deformable特征，跳过（不应该发生）
                    pass

            elif chunk_len == 1 and chunk[0] == IMAGE1_DEFORM_TOKEN_INDEX:
                # Deformable模块的第二个图像特征token
                if img1_deform_feature is not None:
                    cur_new_input_embeds.append(img1_deform_feature)
                    #print("img1_deform_feature appended, shape:", img1_deform_feature.shape)
                    cur_new_seg_query_mask.append(torch.zeros(img1_deform_feature.shape[0]))
                    if class_name_embedding_indices is not None:
                        cur_class_name_embedding_indices.append(
                            torch.full((img1_deform_feature.shape[0],), 0, device=input_id.device, dtype=input_id.dtype)
                        )
                    if refer_embedding_indices is not None:
                        cur_refer_embedding_indices.append(
                            torch.full((img1_deform_feature.shape[0],), 0, device=input_id.device, dtype=input_id.dtype)
                        )
                    if label is not None:
                        cur_new_label.append(
                            torch.full((img1_deform_feature.shape[0],), IGNORE_INDEX, device=label.device, dtype=label.dtype)
                        )
                    if enable_region_mask:
                        cur_new_region_embedding_mask.append(torch.zeros(img1_deform_feature.shape[0]))
                else:
                    # 如果没有deformable特征，跳过（不应该发生）
                    pass

            elif chunk_len == 1 and chunk[0] == SEG_TOKEN_INDEX:
                cur_new_input_embeds.append(seg_query)
                cur_new_seg_query_mask.append(torch.ones(seg_query.shape[0]))
                if class_name_embedding_indices is not None:
                    cur_class_name_embedding_indices.append(torch.full((seg_query.shape[0],), 0, device=label.device,
                                                                       dtype=label.dtype))
                if refer_embedding_indices is not None:
                    cur_refer_embedding_indices.append(torch.full((seg_query.shape[0],), 0, device=label.device,
                                                                       dtype=label.dtype))
                if label is not None:
                    cur_new_label.append(
                        torch.full((seg_query.shape[0],), IGNORE_INDEX, device=label.device,
                                   dtype=label.dtype))
                if enable_region_mask:
                    cur_new_region_embedding_mask.append(torch.zeros(seg_query.shape[0]))
            elif chunk_len == 1 and chunk[0] == CLS_TOKEN_INDEX:
                cls_embed = class_embed[cls_idx]
                if len(cls_embed.shape) == 1:
                    cls_embed = cls_embed.unsqueeze(0)
                cls_idx += 1
                cur_new_input_embeds.append(cls_embed)
                cur_new_seg_query_mask.append(torch.zeros(cls_embed.shape[0]))
                if enable_region_mask:
                    cur_new_region_embedding_mask.append(torch.zeros(cls_embed.shape[0]))
                if class_name_embedding_indices is not None:
                    cur_class_name_embedding_indices.append(
                        torch.full((cls_embed.shape[0],), cls_idx, device=input_id.device,
                                   dtype=input_id.dtype))
                if refer_embedding_indices is not None:
                    cur_refer_embedding_indices.append(
                        torch.full((cls_embed.shape[0],), 0, device=input_id.device,
                                   dtype=input_id.dtype))
                if label is not None:
                    cur_new_label.append(
                        torch.full((cls_embed.shape[0],), IGNORE_INDEX, device=label.device,
                                   dtype=label.dtype)
                    )
            elif chunk_len == 1 and chunk[0] == REGION_TOKEN_INDEX:
                region_feature = region_feature_list[region_idx]
                region_idx += 1
                cur_new_input_embeds.append(region_feature)
                cur_new_seg_query_mask.append(torch.zeros(region_feature.shape[0]))
                if class_name_embedding_indices is not None:
                    cur_class_name_embedding_indices.append(
                        torch.full((region_feature.shape[0],), 0, device=input_id.device,
                                   dtype=input_id.dtype))
                if refer_embedding_indices is not None:
                    cur_refer_embedding_indices.append(
                        torch.full((region_feature.shape[0],), 0, device=input_id.device,
                                   dtype=input_id.dtype))
                if label is not None:
                    cur_new_label.append(
                        torch.full((region_feature.shape[0],), IGNORE_INDEX, device=label.device,
                                   dtype=label.dtype)
                    )
                if enable_region_mask:
                    cur_new_region_embedding_mask.append(torch.ones(region_feature.shape[0]))
            elif chunk_len == 1 and chunk[0] == REFER_TOKEN_INDEX:
                refer_embed = refer_embedding
                if len(refer_embed.shape) == 1:
                    refer_embed = refer_embed.unsqueeze(0)
                cur_new_input_embeds.append(refer_embed)
                cur_new_seg_query_mask.append(torch.zeros(refer_embed.shape[0]))
                if enable_region_mask:
                    cur_new_region_embedding_mask.append(torch.zeros(refer_embed.shape[0]))
                if class_name_embedding_indices is not None:
                    cur_class_name_embedding_indices.append(
                        torch.full((refer_embed.shape[0],), 0, device=input_id.device,
                                   dtype=input_id.dtype))
                if refer_embedding_indices is not None:
                    cur_refer_embedding_indices.append(
                        torch.full((refer_embed.shape[0],), 1, device=input_id.device,
                                   dtype=input_id.dtype))
                if label is not None:
                    cur_new_label.append(
                        torch.full((refer_embed.shape[0],), IGNORE_INDEX, device=label.device,
                                   dtype=label.dtype)
                    )
            else:
                cur_new_input_embeds.append(self.get_model().embed_tokens(input_id[:chunk_len]))
                cur_new_seg_query_mask.append(seg_query_mask[:chunk_len])
                if class_name_embedding_indices is not None:
                    cur_class_name_embedding_indices.append(class_name_embedding_indices[:chunk_len])
                if refer_embedding_indices is not None:
                    cur_refer_embedding_indices.append(refer_embedding_indices[:chunk_len])
                if label is not None:
                    cur_new_label.append(label[:chunk_len])
                if enable_region_mask:
                    cur_new_region_embedding_mask.append(region_embedding_mask[:chunk_len])

            # --- 下面是改动的部分 ---
            input_id = input_id[chunk_len:]
            seg_query_mask = seg_query_mask[chunk_len:]
            if label is not None:
                label = label[chunk_len:]

            # 检查是否为新插入的 token
            is_inserted_token = (chunk_len == 1) and (chunk[0] == IMAGE_DEFORM_TOKEN_INDEX or chunk[0] == IMAGE1_DEFORM_TOKEN_INDEX)

            if not is_inserted_token:
                if class_name_embedding_indices is not None:
                    class_name_embedding_indices = class_name_embedding_indices[chunk_len:]
                if refer_embedding_indices is not None:
                    refer_embedding_indices = refer_embedding_indices[chunk_len:]
                if enable_region_mask:
                    region_embedding_mask = region_embedding_mask[chunk_len:]

        cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
        cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
        if label is not None:
            cur_new_label = [x.to(device=self.device) for x in cur_new_label]
            cur_new_label = torch.cat(cur_new_label, dim=0)
        cur_new_seg_query_mask = [x.to(device=self.device) for x in cur_new_seg_query_mask]
        cur_new_seg_query_mask = torch.cat(cur_new_seg_query_mask, dim=0)
        if class_name_embedding_indices is not None:
            cur_class_name_embedding_indices = [x.to(device=self.device) for x in cur_class_name_embedding_indices]
            cur_class_name_embedding_indices = torch.cat(cur_class_name_embedding_indices, dim=0)
        if refer_embedding_indices is not None:
            cur_refer_embedding_indices = [x.to(device=self.device) for x in cur_refer_embedding_indices]
            cur_refer_embedding_indices = torch.cat(cur_refer_embedding_indices, dim=0)

        if enable_region_mask:
            cur_new_region_embedding_mask = [x.to(device=self.device) for x in cur_new_region_embedding_mask]
            cur_new_region_embedding_mask = torch.cat(cur_new_region_embedding_mask, dim=0)

        return cur_new_input_embeds, cur_new_label, cur_new_seg_query_mask, cur_class_name_embedding_indices, cur_new_region_embedding_mask, cur_refer_embedding_indices
    
    def prepare_inputs_labels_for_multimodal(
            self, input_ids, attention_mask, past_key_values, labels, images, image1=None, class_name_embedding_indices=None,
            class_name_ids=None, cls_indices=None, instances=None, token_refer_id=None, refer_embedding_indices=None,
            dataset_type=None
    ):
        from psalm.model.multimodal_projector.deformable_alignment import MultiScaleDeformableCrossAttentionAlignment
        
        vision_tower = self.get_vision_tower()
        seg_query_mask = torch.zeros_like(input_ids)
        aux_similarity_maps = []

        # 非多模态场景
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[
                1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                                            dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels, seg_query_mask

        # 检查是否是Deformable模式
        projector = self.get_model().mm_projector
        is_deformable_mode = isinstance(projector, MultiScaleDeformableCrossAttentionAlignment)
        # 初始化 queries 源标识（后续由 create_deformable_queries 填充）
        queries_z_q_source = 'N/A'
        queries_z_q_image1_source = 'N/A'
        
        # 从dataset_type推断任务类型
        task_type = ''
        is_cross_image_task = False
        if dataset_type is not None:
            if isinstance(dataset_type, list) and len(dataset_type) > 0:
                # 取第一个作为batch的任务类型（假设batch内任务类型一致）
                batch_dataset_type = dataset_type[0] if isinstance(dataset_type[0], str) else ''
                if 'panoptic' in batch_dataset_type:
                    task_type = 'panoptic'
                elif 'referring' in batch_dataset_type:
                    task_type = 'referring'
                elif 'region_cross_seg' in batch_dataset_type or 'cross' in batch_dataset_type:
                    task_type = 'cross_image'
                    is_cross_image_task = True
                elif 'region' in batch_dataset_type:
                    task_type = 'region'
                else:
                    task_type = batch_dataset_type
        
        # 如果是跨图任务，需要特殊处理：先提取提示图的区域特征
        # 对于跨图任务，提示图只使用baseline特征，不需要deformable特征
        # 目标图使用提示图的区域特征作为zq，做deformable attention
        if is_cross_image_task and image1 is not None and is_deformable_mode:
            # 跨图任务：先处理提示图，提取区域特征
            # 暂时不创建queries_z_q，等提取区域特征后再创建queries_z_q_image1
            queries_z_q = None
            queries_z_q_image1 = None
        elif is_deformable_mode:
            # 非跨图任务或单图任务：正常创建查询
            batch_size = input_ids.shape[0]
            device = input_ids.device
            queries_z_q_source = 'N/A'
            queries_z_q_image1_source = 'N/A'

            refer_embeddings_batch = None
            if 'referring' in task_type and token_refer_id is not None:
                # token_refer_id 是一个列表，包含每个样本的指代词 Token IDs
                ref_embeds_list = []
                for t_ids in token_refer_id:
                    if t_ids is not None:
                        # 1. 获取 Embedding: [Len, Hidden_Dim]
                        # 注意：需要移到 device
                        cur_embeds = self.get_model().embed_tokens(t_ids.to(device))
                        # 2. 平均池化得到 Sentence Embedding: [1, Hidden_Dim]
                        pooled_embed = cur_embeds.mean(dim=0, keepdim=True)
                        ref_embeds_list.append(pooled_embed)
                    else:
                        # 空数据处理
                        ref_embeds_list.append(torch.zeros(1, self.config.hidden_size, device=device))
                
                if len(ref_embeds_list) > 0:
                    refer_embeddings_batch = torch.stack(ref_embeds_list, dim=0) # [B, 1, C]

            # 为images创建查询
            queries_z_q, queries_z_q_source = self.create_deformable_queries(
                task_type=task_type,
                batch_size=batch_size,
                device=device,
                class_name_embedding=None,  # 将在后续处理中填充
                region_feature_list=None,  # 此时region_feature_list也不可用
                refer_embeddings=refer_embeddings_batch
            )
            
            # 为image1创建查询（非跨图任务的双图场景）
            if image1 is not None:
                queries_z_q_image1, queries_z_q_image1_source = self.create_deformable_queries(
                    task_type=task_type,
                    batch_size=batch_size,
                    device=device,
                    class_name_embedding=None,
                    region_feature_list=None
                )
            else:
                queries_z_q_image1 = None
        else:
            queries_z_q = None
            queries_z_q_image1 = None

        # baseline_projector 应在 initialize_vision_modules 中创建并加载权重
        baseline_projector = getattr(self.get_model(), 'baseline_projector', None) if is_deformable_mode else None
        if is_deformable_mode and baseline_projector is None:
            raise RuntimeError('baseline_projector is not initialized. Call initialize_vision_modules with a proper pretrain_mm_mlp_adapter so baseline_projector is constructed and weights are loaded.')
        
        # 对于跨图任务，需要先处理提示图，提取区域特征
        prompt_region_features = None
        if is_cross_image_task and image1 is not None and is_deformable_mode:
            # 1. 先处理提示图（images），提取baseline特征
            if type(images) is list or images.ndim == 5:
                concat_images = torch.cat([image for image in images], dim=0)
                multi_scale_features_list = self.get_model().get_vision_tower()(concat_images)
                res5_features = multi_scale_features_list[-1]
                baseline_features_prompt = baseline_projector(res5_features)
                
                split_sizes = [image.shape[0] for image in images]
                baseline_features_prompt = torch.split(baseline_features_prompt, split_sizes, dim=0)
                baseline_features_prompt = [x.flatten(0, 1) for x in baseline_features_prompt]
            else:
                multi_scale_features_list = self.get_model().get_vision_tower()(images)
                res5_features = multi_scale_features_list[-1]
                baseline_features_prompt = baseline_projector(res5_features)
                # 转换为list格式，方便后续处理
                if len(baseline_features_prompt.shape) == 3:  # (B, H*W, hidden_dim)
                    baseline_features_prompt = [baseline_features_prompt[i] for i in range(baseline_features_prompt.shape[0])]
                else:  # (H*W, hidden_dim)
                    baseline_features_prompt = [baseline_features_prompt]
            
            # 2. 从提示图中提取区域特征（如果有region masks）
            if (input_ids == REGION_TOKEN_INDEX).sum() != 0 and instances is not None:
                region_masks_list = [instance.region_masks.tensor for instance in instances]
                # 使用提示图的baseline特征提取区域特征
                prompt_region_features = self.region_sampler(
                    baseline_features_prompt, 
                    region_masks_list,
                    original_dtype=baseline_features_prompt[0].dtype if len(baseline_features_prompt) > 0 else torch.float32,
                    return_dtype=baseline_features_prompt[0].dtype if len(baseline_features_prompt) > 0 else torch.float32
                )
                # prompt_region_features 已生成（静默，不打印）
                pass
            
            # 3. 使用提示图的区域特征作为zq，创建目标图的查询
            if prompt_region_features is not None:
                batch_size = input_ids.shape[0]
                device = input_ids.device
                queries_z_q_image1, queries_z_q_image1_source = self.create_deformable_queries(
                    task_type='cross_image',
                    batch_size=batch_size,
                    device=device,
                    class_name_embedding=None,
                    region_feature_list=prompt_region_features  # 使用提示图的区域特征
                )
            else:
                # 如果没有区域特征，使用seg_query作为后备
                batch_size = input_ids.shape[0]
                device = input_ids.device
                queries_z_q_image1, queries_z_q_image1_source = self.create_deformable_queries(
                    task_type='cross_image',
                    batch_size=batch_size,
                    device=device,
                    class_name_embedding=None,
                    region_feature_list=None
                )
            
            # 4. 提示图只使用baseline特征，不需要deformable特征
            if type(images) is list or images.ndim == 5:
                image_features = baseline_features_prompt
                image_deform_features = None
            else:
                # 如果是单个tensor，需要保持batch维度
                if isinstance(baseline_features_prompt, list):
                    # 如果已经是list，直接使用
                    image_features = baseline_features_prompt
                else:
                    # 如果是tensor，转换为list
                    if len(baseline_features_prompt.shape) == 3:
                        image_features = [baseline_features_prompt[i] for i in range(baseline_features_prompt.shape[0])]
                    else:
                        image_features = [baseline_features_prompt]
                image_deform_features = None
        else:
            # 非跨图任务：正常处理
            #单图像/多图像
            if type(images) is list or images.ndim == 5:
                concat_images = torch.cat([image for image in images], dim=0)
                if is_deformable_mode:
                    # Deformable模式：同时计算baseline和deformable特征
                    multi_scale_features_list = self.get_model().get_vision_tower()(concat_images)
                    res5_features = multi_scale_features_list[-1]  # (B, C, H, W)
                    baseline_features = baseline_projector(res5_features)  # (B, H*W, hidden_dim)
                    
                    multi_scale_features = {
                        'res2': multi_scale_features_list[0],
                        'res3': multi_scale_features_list[1],
                        'res4': multi_scale_features_list[2],
                        'res5': multi_scale_features_list[3],
                    }
                    deformable_features, similarity_map = projector(queries=queries_z_q, multi_scale_features=multi_scale_features, task_type=task_type)
                    if similarity_map is not None:
                        aux_similarity_maps.append(similarity_map)
                    print("deformable_features, shape:", deformable_features.shape)
                        
                    split_sizes = [image.shape[0] for image in images]
                    baseline_features = torch.split(baseline_features, split_sizes, dim=0)
                    deformable_features = torch.split(deformable_features, split_sizes, dim=0)
                    baseline_features = [x.flatten(0, 1) for x in baseline_features]
                    deformable_features = [x.flatten(0, 1) for x in deformable_features]
                    image_features = baseline_features
                    image_deform_features = deformable_features
                else:
                    image_features = self.encode_images(concat_images, queries_z_q=queries_z_q)
                    split_sizes = [image.shape[0] for image in images]
                    image_features = torch.split(image_features, split_sizes, dim=0)
                    image_features = [x.flatten(0, 1) for x in image_features]
                    image_deform_features = None
            else:
                if is_deformable_mode:
                    # Deformable模式：同时计算baseline和deformable特征
                    multi_scale_features_list = self.get_model().get_vision_tower()(images)
                    res5_features = multi_scale_features_list[-1]  # (B, C, H, W)
                    baseline_features = baseline_projector(res5_features)  # (B, H*W, hidden_dim)
                    
                    multi_scale_features = {
                        'res2': multi_scale_features_list[0],
                        'res3': multi_scale_features_list[1],
                        'res4': multi_scale_features_list[2],
                        'res5': multi_scale_features_list[3],
                    }
                    deformable_features, similarity_map = projector(queries=queries_z_q, multi_scale_features=multi_scale_features, task_type=task_type)
                    if similarity_map is not None:
                        aux_similarity_maps.append(similarity_map)
                    print("deformable_features, shape:", deformable_features.shape)
                    # 保持batch维度，不flatten，在batch循环中处理
                    image_features = baseline_features  # (B, H*W, hidden_dim)
                    image_deform_features = deformable_features  # (B, N_q, hidden_dim)
                else:
                    image_features = self.encode_images(images, queries_z_q=queries_z_q)
                    image_deform_features = None

        image1_features = None
        image1_deform_features = None
        if image1 is not None:
            if type(image1) is list or image1.ndim == 5:
                concat_image1 = torch.cat([img for img in image1], dim=0)
                if is_deformable_mode:
                    # Deformable模式：同时计算baseline和deformable特征
                    multi_scale_features_list1 = self.get_model().get_vision_tower()(concat_image1)
                    res5_features1 = multi_scale_features_list1[-1]
                    baseline_features1 = baseline_projector(res5_features1)
                    
                    multi_scale_features1 = {
                        'res2': multi_scale_features_list1[0],
                        'res3': multi_scale_features_list1[1],
                        'res4': multi_scale_features_list1[2],
                        'res5': multi_scale_features_list1[3],
                    }
                    deformable_features1, similarity_map1 = projector(queries=queries_z_q_image1, multi_scale_features=multi_scale_features1, task_type=task_type)
                    if similarity_map1 is not None:
                        aux_similarity_maps.append(similarity_map1)
                    print("deformable_features1, shape:", deformable_features1.shape)
                    
                    split_sizes1 = [img.shape[0] for img in image1]
                    baseline_features1 = torch.split(baseline_features1, split_sizes1, dim=0)
                    deformable_features1 = torch.split(deformable_features1, split_sizes1, dim=0)
                    baseline_features1 = [x.flatten(0, 1) for x in baseline_features1]
                    deformable_features1 = [x.flatten(0, 1) for x in deformable_features1]
                    image1_features = baseline_features1
                    image1_deform_features = deformable_features1
                else:
                    image1_features = self.encode_images(concat_image1, queries_z_q=queries_z_q_image1) 
                    split_sizes1 = [img.shape[0] for img in image1]
                    image1_features = torch.split(image1_features, split_sizes1, dim=0)
                    image1_features = [x.flatten(0, 1) for x in image1_features]
                    image1_deform_features = None
            else:
                if is_deformable_mode:
                    # Deformable模式：同时计算baseline和deformable特征
                    multi_scale_features_list1 = self.get_model().get_vision_tower()(image1)
                    res5_features1 = multi_scale_features_list1[-1]
                    baseline_features1 = baseline_projector(res5_features1)
                    
                    multi_scale_features1 = {
                        'res2': multi_scale_features_list1[0],
                        'res3': multi_scale_features_list1[1],
                        'res4': multi_scale_features_list1[2],
                        'res5': multi_scale_features_list1[3],
                    }
                    deformable_features1, similarity_map1 = projector(queries=queries_z_q_image1, multi_scale_features=multi_scale_features1, task_type=task_type)
                    if similarity_map1 is not None:
                        aux_similarity_maps.append(similarity_map1)
                    print("deformable_features1, shape:", deformable_features1.shape)
                    
                    # 保持batch维度，不flatten
                    image1_features = baseline_features1  # (B, H*W, hidden_dim)
                    image1_deform_features = deformable_features1  # (B, N_q, hidden_dim)
                else:
                    image1_features = self.encode_images(image1, queries_z_q=queries_z_q_image1)
                    image1_deform_features = None 


        # Single concise English-only debug print (after queries creation)
        try:
            zq_src = queries_z_q_source if 'queries_z_q_source' in locals() else 'N/A'
            zq1_src = queries_z_q_image1_source if 'queries_z_q_image1_source' in locals() else 'N/A'
            mode_str = 'Deformable' if is_deformable_mode else 'Baseline'
            print(f"[PSALM] task={task_type} mode={mode_str} zq={zq_src} zq1={zq1_src}")
        except Exception:
            pass

        # 拓展分割查询到batch层次
        expanded_seg_query = self.seg_query.unsqueeze(0).expand(input_ids.shape[0], -1, -1)


        #区域特征提取
        # 对于跨图任务，区域特征已经在前面从提示图中提取过了
        if is_cross_image_task and image1 is not None:
            # 跨图任务：使用前面提取的prompt_region_features
            region_features = prompt_region_features
            region_embedding_masks = torch.zeros_like(input_ids)
        elif (input_ids == REGION_TOKEN_INDEX).sum() != 0 and instances is not None:
            region_masks_list = [instance.region_masks.tensor for instance in instances]

            # [region_features_per_batch: [num_region, 1, dims]], len(region_features) = batch_size
            # 使用baseline特征进行区域采样（deformable模式下使用baseline特征）
            # region_sampler期望list格式的输入
            if isinstance(image_features, list):
                region_features = self.region_sampler(image_features, region_masks_list,
                                                      original_dtype=image_features[0].dtype if len(image_features) > 0 else torch.float32,
                                                      return_dtype=image_features[0].dtype if len(image_features) > 0 else torch.float32)
            else:
                # 非list情况：转换为list格式
                # image_features可能是(B, seq_len, hidden_dim)或(seq_len, hidden_dim)
                if len(image_features.shape) == 3:  # (B, seq_len, hidden_dim)
                    image_features_list = [image_features[i] for i in range(image_features.shape[0])]
                else:  # (seq_len, hidden_dim) - 单batch
                    image_features_list = [image_features]
                region_features = self.region_sampler(image_features_list, region_masks_list,
                                                      original_dtype=image_features.dtype,
                                                      return_dtype=image_features.dtype)
             # 初始化区域嵌入掩码，标记输入中区域特征的位置
            region_embedding_masks = torch.zeros_like(input_ids)
        else:
            region_features = None
            region_embedding_masks = None


        # 找到这个位置（大约在 1730 行附近）
        if is_deformable_mode:
            new_input_ids_list = []
            new_labels_list = [] if labels is not None else None
            # [修复 1] 初始化 mask 列表
            new_attention_mask_list = [] if attention_mask is not None else None
            
            for batch_idx, cur_input_ids in enumerate(input_ids):
                cur_new_input_ids = []
                cur_new_labels = [] if labels is not None else None
                
                # [修复 2] 获取当前样本的 attention_mask
                cur_attention_mask = attention_mask[batch_idx] if attention_mask is not None else None
                cur_new_attention_mask = [] if cur_attention_mask is not None else None
                
                i = 0
                while i < len(cur_input_ids):
                    cur_new_input_ids.append(cur_input_ids[i])
                    if labels is not None:
                        cur_new_labels.append(labels[batch_idx][i])
                    
                    # [修复 3] 同步复制原 mask 的值
                    if cur_new_attention_mask is not None:
                        cur_new_attention_mask.append(cur_attention_mask[i])
                    
                    # 检查是否插入 Token
                    insert_token = False
                    insert_val = None
                    
                    if is_cross_image_task and image1 is not None:
                        if cur_input_ids[i] == IMAGE1_TOKEN_INDEX:
                            insert_token = True
                            insert_val = IMAGE1_DEFORM_TOKEN_INDEX
                    else:
                        if cur_input_ids[i] == IMAGE_TOKEN_INDEX:
                            insert_token = True
                            insert_val = IMAGE_DEFORM_TOKEN_INDEX
                        elif cur_input_ids[i] == IMAGE1_TOKEN_INDEX:
                            insert_token = True
                            insert_val = IMAGE1_DEFORM_TOKEN_INDEX
                    
                    if insert_token:
                        cur_new_input_ids.append(insert_val)
                        if labels is not None:
                            cur_new_labels.append(IGNORE_INDEX)
                        # [修复 4: 关键!] 插入 Token 时，Mask 也要补 1 (True)
                        if cur_new_attention_mask is not None:
                            cur_new_attention_mask.append(True)
                            
                    i += 1
                
                new_input_ids_list.append(torch.tensor(cur_new_input_ids, device=cur_input_ids.device, dtype=cur_input_ids.dtype))
                if labels is not None:
                    new_labels_list.append(torch.tensor(cur_new_labels, device=labels.device, dtype=labels.dtype))
                # [修复 5] 保存新的 mask
                if new_attention_mask_list is not None:
                    new_attention_mask_list.append(torch.tensor(cur_new_attention_mask, device=attention_mask.device, dtype=attention_mask.dtype))
            
            # 对齐 Batch
            if len(new_input_ids_list) > 0:
                max_len = max(len(ids) for ids in new_input_ids_list)
                aligned_input_ids = []
                aligned_labels = [] if labels is not None else None
                # [修复 6] 初始化对齐列表
                aligned_attention_masks = [] if attention_mask is not None else None
                
                for idx in range(len(new_input_ids_list)):
                    ids = new_input_ids_list[idx]
                    pad_len = max_len - len(ids)
                    if pad_len > 0:
                        pad_ids = torch.full((pad_len,), IGNORE_INDEX, device=ids.device, dtype=ids.dtype)
                        ids = torch.cat([ids, pad_ids], dim=0)
                    aligned_input_ids.append(ids)
                    
                    if labels is not None:
                        lbls = new_labels_list[idx]
                        if pad_len > 0:
                            pad_lbls = torch.full((pad_len,), IGNORE_INDEX, device=lbls.device, dtype=lbls.dtype)
                            lbls = torch.cat([lbls, pad_lbls], dim=0)
                        aligned_labels.append(lbls)
                    
                    # [修复 7] 对 Mask 进行 Padding (补 False)
                    if attention_mask is not None:
                        msk = new_attention_mask_list[idx]
                        if pad_len > 0:
                            pad_msk = torch.full((pad_len,), False, device=msk.device, dtype=msk.dtype)
                            msk = torch.cat([msk, pad_msk], dim=0)
                        aligned_attention_masks.append(msk)

                input_ids = torch.stack(aligned_input_ids, dim=0)
                if labels is not None:
                    labels = torch.stack(aligned_labels, dim=0)
                
                # [修复 8: 最终赋值] 更新 attention_mask
                if attention_mask is not None:
                    attention_mask = torch.stack(aligned_attention_masks, dim=0)
                
                seg_query_mask = torch.zeros_like(input_ids)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        new_seg_query_masks = []
        new_class_name_embedding_indices = [] if class_name_embedding_indices is not None else None
        new_refer_embedding_indices = [] if refer_embedding_indices is not None else None
        new_region_embedding_masks = [] if region_features is not None else None


        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_seg_query_mask = seg_query_mask[batch_idx]
            cur_seg_query = expanded_seg_query[batch_idx]

            # 处理baseline特征
            if isinstance(image_features, list):
                cur_image_feature = image_features[batch_idx]
            else:
                # 非list情况：直接按batch索引
                if len(image_features.shape) == 3:  # (B, seq_len, hidden_dim)
                    cur_image_feature = image_features[batch_idx]  # (seq_len, hidden_dim)
                else:  # (seq_len, hidden_dim) - 单batch情况
                    cur_image_feature = image_features
            
            if image1_features is not None:
                if isinstance(image1_features, list):
                    cur_image1_feature = image1_features[batch_idx]
                else:
                    if len(image1_features.shape) == 3:  # (B, seq_len, hidden_dim)
                        cur_image1_feature = image1_features[batch_idx]
                    else:
                        cur_image1_feature = image1_features
            else:
                cur_image1_feature = None
            
            # 处理deformable特征（防护：可能为 None）
            if is_deformable_mode:
                if image_deform_features is None:
                    cur_image_deform_feature = None
                else:
                    if isinstance(image_deform_features, list):
                        cur_image_deform_feature = image_deform_features[batch_idx]
                    else:
                        # 非list情况：直接按batch索引
                        if len(image_deform_features.shape) == 3:  # (B, N_q, hidden_dim)
                            cur_image_deform_feature = image_deform_features[batch_idx]  # (N_q, hidden_dim)
                        else:  # (N_q, hidden_dim) - 单batch情况
                            cur_image_deform_feature = image_deform_features

                if image1_deform_features is None:
                    cur_image1_deform_feature = None
                else:
                    if isinstance(image1_deform_features, list):
                        cur_image1_deform_feature = image1_deform_features[batch_idx]
                    else:
                        if len(image1_deform_features.shape) == 3:  # (B, N_q, hidden_dim)
                            cur_image1_deform_feature = image1_deform_features[batch_idx]
                        else:
                            cur_image1_deform_feature = image1_deform_features
            else:
                cur_image_deform_feature = None
                cur_image1_deform_feature = None

            cur_class_name_embedding_indices = class_name_embedding_indices[batch_idx] if class_name_embedding_indices is not None else None
            cur_refer_embedding_indices = refer_embedding_indices[batch_idx] if refer_embedding_indices is not None else None
            cur_region_feature_list = region_features[batch_idx] if region_features is not None else None
            cur_region_embedding_mask = region_embedding_masks[batch_idx] if region_features is not None else None
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0 and (cur_input_ids == IMAGE1_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                # ensure gradients back propagation, not changing cur_input_embeds
                cur_input_embeds = cur_input_embeds + (
                        0. * self.get_model().mm_projector(vision_tower.dummy_feature)).sum()
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                new_seg_query_masks.append(cur_seg_query_mask)
                # cur_image_idx += 1
                continue

            if labels is not None:
                cur_label = labels[batch_idx]
            else:
                cur_label = None

            if class_name_ids is not None:
                cur_class_name_ids = class_name_ids[batch_idx]
                cur_cls_indices = cls_indices[batch_idx]
            else:
                cur_class_name_ids = None
                cur_cls_indices = None
            if token_refer_id is not None:
                cur_token_refer_id = token_refer_id[batch_idx]
            else:
                cur_token_refer_id = None


            cur_class_name_embedding = self.embed_class_ids(cur_class_name_ids, cur_cls_indices)
            cur_refer_embedding = self.embed_refer_ids(cur_token_refer_id)

            cur_input_embeds, cur_label, cur_seg_query_mask, cur_class_name_embedding_indices, cur_region_embedding_mask, cur_refer_embedding_indices = self.concat_image_seg_cls_embeds(
                input_id=cur_input_ids,
                img_feature=cur_image_feature,
                img1_feature=cur_image1_feature,
                label=cur_label,
                seg_query=cur_seg_query,
                seg_query_mask=cur_seg_query_mask,
                class_embed=cur_class_name_embedding,
                class_name_embedding_indices=cur_class_name_embedding_indices,
                region_embedding_mask=cur_region_embedding_mask,
                region_feature_list=cur_region_feature_list,
                refer_embedding_indices=cur_refer_embedding_indices,
                refer_embedding=cur_refer_embedding,
                img_deform_feature=cur_image_deform_feature,
                img1_deform_feature=cur_image1_deform_feature
            )
            assert cur_input_embeds.shape[0] == cur_seg_query_mask.shape[0]

            new_input_embeds.append(cur_input_embeds)
            if labels is not None:
                new_labels.append(cur_label)
            new_seg_query_masks.append(cur_seg_query_mask)
            if class_name_embedding_indices is not None:
                new_class_name_embedding_indices.append(cur_class_name_embedding_indices)
            if refer_embedding_indices is not None:
                new_refer_embedding_indices.append(cur_refer_embedding_indices)
            if new_region_embedding_masks is not None:
                new_region_embedding_masks.append(cur_region_embedding_mask)
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed,
                                           torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                                                       dtype=cur_new_embed.dtype, device=cur_new_embed.device)),
                                          dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label,
                                               torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX,
                                                          dtype=cur_new_label.dtype, device=cur_new_label.device)),
                                              dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            new_seg_query_masks_align = []
            for new_seg_query_mask in new_seg_query_masks:
                new_seg_query_mask = torch.cat(
                    (new_seg_query_mask, torch.zeros((max_len - new_seg_query_mask.shape[0]),dtype=new_seg_query_mask.dtype, device=new_seg_query_mask.device)),
                    dim=0)
                new_seg_query_masks_align.append(new_seg_query_mask)
            new_seg_query_masks = torch.stack(new_seg_query_masks_align, dim=0)

            new_class_name_embedding_indices_align = []

            if class_name_embedding_indices is not None:
                for new_class_name_embedding_indice in new_class_name_embedding_indices:
                    new_class_name_embedding_indice = torch.cat(
                        (new_class_name_embedding_indice,
                         torch.zeros((max_len - new_class_name_embedding_indice.shape[0]),dtype=new_class_name_embedding_indice.dtype, device=new_class_name_embedding_indice.device)),
                        dim=0)
                    new_class_name_embedding_indices_align.append(new_class_name_embedding_indice)
                new_class_name_embedding_indices = torch.stack(new_class_name_embedding_indices_align, dim=0)

            if refer_embedding_indices is not None:
                new_refer_embedding_indices_align = []
                for new_refer_embedding_indice in new_refer_embedding_indices:
                    new_refer_embedding_indice = torch.cat(
                        (new_refer_embedding_indice,
                         torch.zeros((max_len - new_refer_embedding_indice.shape[0]),dtype=new_refer_embedding_indice.dtype, device=new_refer_embedding_indice.device)),
                        dim=0)
                    new_refer_embedding_indices_align.append(new_refer_embedding_indice)
                new_refer_embedding_indices = torch.stack(new_refer_embedding_indices_align, dim=0)

            if new_region_embedding_masks is not None:
                new_region_embedding_masks_align = []
                for new_region_embedding_mask in new_region_embedding_masks:
                    new_region_embedding_mask = torch.cat(
                        (new_region_embedding_mask, torch.zeros((max_len - new_region_embedding_mask.shape[0]),dtype=new_region_embedding_mask.dtype, device=new_region_embedding_mask.device)),
                        dim=0)
                    new_region_embedding_masks_align.append(new_region_embedding_mask)
                new_region_embedding_masks = torch.stack(new_region_embedding_masks_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels,
                                                                                    new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True,
                                                        dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                                                         False, dtype=attention_mask.dtype,
                                                         device=attention_mask.device)
                    cur_new_attention_mask = torch.cat(
                        (new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape

        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            new_seg_query_masks = torch.stack(new_seg_query_masks, dim=0)
            if class_name_embedding_indices is not None:
                new_class_name_embedding_indices = torch.stack(new_class_name_embedding_indices, dim=0)
            if refer_embedding_indices is not None:
                new_refer_embedding_indices = torch.stack(new_refer_embedding_indices, dim=0)

            if new_region_embedding_masks is not None:
                new_region_embedding_masks = torch.stack(new_region_embedding_masks, dim=0)

            if attention_mask is not None:
                # pad length should be based on current attention_mask width to avoid mismatch
                pad_len = new_input_embeds.shape[1] - attention_mask.shape[1]
                if pad_len < 0:
                    # unexpected: new_input_embeds shorter than attention_mask, raise for debug
                    raise AssertionError(f"new_input_embeds length {new_input_embeds.shape[1]} < attention_mask length {attention_mask.shape[1]}")
                new_attn_mask_pad_left = torch.full(
                    (attention_mask.shape[0], pad_len), True,
                    dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                # if new_region_embedding_masks is not None:
                #     # pad region masks on the left to align with new_input_embeds
                #     left_pad_region = torch.zeros((new_region_embedding_masks.shape[0], pad_len),
                #                                   dtype=new_region_embedding_masks.dtype,
                #                                   device=new_region_embedding_masks.device)
                #     new_region_embedding_masks = torch.cat((left_pad_region, new_region_embedding_masks), dim=1)

                if attention_mask.shape != new_input_embeds.shape[:2]:
                    try:
                        print('[PSALM-DEBUG] attention_mask.shape:', attention_mask.shape, 'new_input_embeds.shape:', new_input_embeds.shape)
                        print('[PSALM-DEBUG] input_ids.shape:', input_ids.shape)
                        # print first sample tokens and special token counts
                        sample_input_ids = input_ids[0].tolist() if hasattr(input_ids, 'tolist') else None
                        print('[PSALM-DEBUG] sample input_ids (first row, truncated):', sample_input_ids[:200] if sample_input_ids is not None else sample_input_ids)
                        # count special tokens per sample
                        try:
                            # 使用模块顶部已导入的常量，避免在函数中再次导入导致名称被视为局部变量
                            for i in range(min(4, input_ids.shape[0])):
                                row = input_ids[i]
                                counts = {
                                    'IMAGE': int((row == IMAGE_TOKEN_INDEX).sum().item()),
                                    'IMAGE1': int((row == IMAGE1_TOKEN_INDEX).sum().item()),
                                    'IMAGE_DEFORM': int((row == IMAGE_DEFORM_TOKEN_INDEX).sum().item()),
                                    'IMAGE1_DEFORM': int((row == IMAGE1_DEFORM_TOKEN_INDEX).sum().item()),
                                    'SEG': int((row == SEG_TOKEN_INDEX).sum().item()),
                                    'CLS': int((row == CLS_TOKEN_INDEX).sum().item()),
                                    'REGION': int((row == REGION_TOKEN_INDEX).sum().item()),
                                }
                                print(f'[PSALM-DEBUG] row {i} special token counts:', counts)
                        except Exception as e:
                            print('[PSALM-DEBUG] failed to compute special token counts:', e)
                    except Exception:
                        pass
                    raise AssertionError(f"attention_mask.shape {attention_mask.shape} != new_input_embeds.shape[:2] {new_input_embeds.shape[:2]}")

        if len(aux_similarity_maps) > 0:
            # 假设每次 append 的是 [B_chunk, 1, H, W]，需要拼回去
            # 如果 logic 比较复杂，可以先简化：只取第一个非空的作为代表，或者在这个函数里先不拼，直接返回 list
            # 简单起见，如果是在 batch loop 外面算的，它应该就是 [B, 1, H, W]
            # 如果是在 loop 里算的，需要 cat
            if isinstance(aux_similarity_maps[0], torch.Tensor):
                # 这种情况下通常已经是完整的 batch 或者 list of chunks
                if len(aux_similarity_maps) == 1:
                    aux_similarity_maps = aux_similarity_maps[0]
                else:
                    aux_similarity_maps = torch.cat(aux_similarity_maps, dim=0)
        else:
            aux_similarity_maps = None            

        return None, attention_mask, past_key_values, new_input_embeds, new_labels, new_seg_query_masks, new_class_name_embedding_indices, new_region_embedding_masks, new_refer_embedding_indices, aux_similarity_maps


    def get_SEG_embedding(self,hidden_states, refer_embedding_indices):
        refer_embedding_list = []
        for current_hidden_state, current_token_indice in zip(hidden_states, refer_embedding_indices):
            current_refer_state = current_hidden_state[current_token_indice.bool()]
            current_pool_refer_state = self.refer_pooling(current_refer_state.transpose(-2, -1)).transpose(-2, -1)
            refer_embedding_list.append(current_pool_refer_state)
        return torch.stack(refer_embedding_list, dim=0)
    
    def extract_prompt_features(self, prompt_images, prompt_info):
        """
        从提示图像中提取提示区域的特征
        prompt_images: 提示图像张量 [batch_size, 3, H, W]
        prompt_info: 每个样本的提示信息，含掩码或框
        返回：提示区域特征 [batch_size, num_prompts, hidden_dim]
        """
        if prompt_images is None or prompt_info is None:
            return None
        
        # 1. 提取提示图像的多尺度视觉特征（同目标图像处理）
        prompt_vision_features = self.get_vision_tower_feature(prompt_images)  # 字典：res2/res3/res4/res5
        # 取最高级特征（res5）用于区域提取
        prompt_high_feature = prompt_vision_features["res5"]  # [batch_size, C, H', W']
        
        prompt_region_features = []
        for i in range(prompt_images.shape[0]):
            # 2. 获取当前样本的提示掩码（假设prompt_info[i]['mask']是二值掩码 [H, W]）
            prompt_mask = prompt_info[i]['mask'].to(prompt_high_feature.device)  # [H, W]
            # 下采样掩码到特征图尺寸（与res5特征对齐）
            feat_h, feat_w = prompt_high_feature.shape[2], prompt_high_feature.shape[3]
            prompt_mask_down = F.interpolate(
                prompt_mask.unsqueeze(0).unsqueeze(0).float(),
                size=(feat_h, feat_w),
                mode="bilinear"
            ).squeeze()  # [feat_h, feat_w]
            
            # 3. 根据掩码提取提示区域的特征（取掩码内特征的平均值）
            mask_flat = prompt_mask_down.flatten(0)  # [feat_h*feat_w]
            feat_flat = prompt_high_feature[i].flatten(1)  # [C, feat_h*feat_w]
            # 掩码内特征的平均值（排除背景）
            if mask_flat.sum() > 0:
                region_feat = (feat_flat * mask_flat.unsqueeze(0)).sum(dim=1) / mask_flat.sum()  # [C]
            else:
                # 无提示时用零向量
                region_feat = torch.zeros_like(feat_flat[:, 0])
            
            prompt_region_features.append(region_feat)
        
        # 4. 投影到掩码解码器的隐藏维度
        prompt_region_features = torch.stack(prompt_region_features, dim=0)  # [batch_size, C]
        prompt_region_features = self.region_projector(prompt_region_features)  # [batch_size, hidden_dim]
        return prompt_region_features.unsqueeze(1)  # [batch_size, 1, hidden_dim]（增加num_prompts维度）

    def save_saved_data(self, save_dir: str):
        if hasattr(self, 'saved_masks') and hasattr(self, 'saved_images'):
            saved_masks = torch.cat(self.saved_masks, dim=0)
            saved_images = torch.cat(self.saved_images, dim=0)
            
            torch.save(saved_masks, os.path.join(save_dir, "predicted_masks.pt"))
            torch.save(saved_images, os.path.join(save_dir, "original_images.pt"))
            
            del self.saved_masks
            del self.saved_images

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            images1: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            seg_info=None,
            class_name_ids=None,
            class_name_embedding_indices=None,
            cls_indices=None,
            random_idx=None,
            token_refer_id=None,
            refer_embedding_indices=None,
            dataset_type=None
    ) -> Union[Tuple, CausalLMOutputWithPast]: 
        
        
        if dataset_type is not None:
            assert all(item == dataset_type[0] for item in dataset_type), f'this batch contain different dataset_type: {dataset_type}'
            batch_dataset_type = dataset_type[0]
            # batch_dataset_type obtained (silent)
        else:
            batch_dataset_type = []


        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 理含分割任务的多模态输入（有 <seg> 令牌）
        if (input_ids == SEG_TOKEN_INDEX).sum() != 0:
            # 若有 <region> 令牌，从 seg_info 提取实例信息，区域分割任务需区域掩码
            if (input_ids == REGION_TOKEN_INDEX).sum() != 0:
                instances = [i['instances'] for i in seg_info]
            else:
                instances = None
            
            input_ids, attention_mask, past_key_values, inputs_embeds, labels, seg_query_mask, class_name_embedding_indices, region_embedding_masks, refer_embedding_indices, aux_similarity_maps = self.prepare_inputs_labels_for_multimodal(
                input_ids, attention_mask, past_key_values, labels, images, images1, class_name_embedding_indices,
                class_name_ids, cls_indices, instances, token_refer_id, refer_embedding_indices, dataset_type=dataset_type)
        # 处理普通多模态输入（无 <seg> 令牌）
        else:
            seg_query_mask = None
            class_name_embedding_indices = None
            region_embedding_masks = None
            SEG_token_indices = None
            input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.mm_conv_prepare_inputs_labels_for_multimodal(
                input_ids, attention_mask, past_key_values, labels, images)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        # 提取模型输出的最后一层隐藏状态，last_hidden_state 融合了图文特征的语义表示
        hidden_states = outputs.last_hidden_state
        # 预测下一个token
        logits = self.lm_head(hidden_states)


        # 处理类别嵌入，用于分割任务的类别预测，用于掩码解码器的类别预测
        if class_name_embedding_indices is not None:
            # 从 hidden_states 中提取类别相关的隐藏状态，并通过平均池化得到类别嵌入
            class_name_embedding = self.get_class_name_embedding(hidden_states, class_name_embedding_indices)
            # 将类别嵌入投影到掩码解码器的隐藏维度，匹配 Mask2Former 输入
            class_name_embedding = self.class_name_projector(class_name_embedding)
        else:
            class_name_embedding = None

        # 用 random_idx 随机采样类别嵌入
        if class_name_embedding is not None:
            class_name_embedding = torch.gather(class_name_embedding,dim=1,index=random_idx.unsqueeze(-1).repeat(1, 1, class_name_embedding.shape[-1]))

        # 处理区域嵌入，用于区域分割任务，区域分割任务需区域嵌入。region_embedding_masks 标记区域令牌位置，提取对应语义特征
        if region_embedding_masks is not None:
            # Ensure region_embedding_masks aligns with hidden_states length to avoid indexing errors
            try:
                hs_len = hidden_states.shape[1]
                # if masks is a tensor of shape (B, L) or list
                if isinstance(region_embedding_masks, torch.Tensor):
                    aligned_masks = []
                    for m in region_embedding_masks:
                        if m.shape[0] > hs_len:
                            # trim left (keep last hs_len positions)
                            m = m[-hs_len:]
                        elif m.shape[0] < hs_len:
                            pad = torch.zeros((hs_len - m.shape[0],), dtype=m.dtype, device=m.device)
                            m = torch.cat((pad, m), dim=0)
                        aligned_masks.append(m)
                    region_embedding_masks = torch.stack(aligned_masks, dim=0)
                else:
                    # assume it's a list of masks
                    aligned_masks = []
                    for m in region_embedding_masks:
                        if m.shape[0] > hs_len:
                            m = m[-hs_len:]
                        elif m.shape[0] < hs_len:
                            pad = torch.zeros((hs_len - m.shape[0],), dtype=m.dtype, device=m.device)
                            m = torch.cat((pad, m), dim=0)
                        aligned_masks.append(m)
                    region_embedding_masks = torch.stack(aligned_masks, dim=0)
            except Exception:
                # fall back to original mask if alignment fails for unexpected reason
                pass
            region_embedding_list = self.get_region_embedding(hidden_states, region_embedding_masks)
             # 将每个区域的嵌入投影到掩码解码器的隐藏维度
            region_embedding_list = [self.region_projector(region_embedding) for region_embedding in
                                     region_embedding_list]
        else:
            region_embedding_list = None
        # 指代/区域任务不需要类别预测，置空类别嵌入
        if 'referring' in batch_dataset_type or 'region' in batch_dataset_type:
            class_name_embedding = None


        loss = None
        #仅语言模型任务，无分割
        if labels is not None and seg_query_mask is None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            llm_loss = loss_fct(shift_logits, shift_labels)

        # 分割任务，含掩码解码器
        if seg_query_mask is not None:
            # 从 hidden_states 中，按 seg_query_mask 掩码，提取分割查询向量
            seg_query = self.get_seg_query(hidden_states, seg_query_mask)
            # 投影到掩码解码器的隐藏维度，以匹配 Mask2Former 输入
            seg_query = self.seg_query_projector(seg_query)
            
            if images1 is not None:
                image_features = self.get_vision_tower_feature(images1)
                mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
                image_features)
            else:
                image_features = self.get_vision_tower_feature(images)
                mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
                image_features)

            
            if refer_embedding_indices is not None:
                SEG_embedding = self.get_SEG_embedding(hidden_states, refer_embedding_indices)
                SEG_embedding = self.SEG_token_projector(SEG_embedding)
            else:
                SEG_embedding = None
            

            if 'panoptic' in batch_dataset_type or 'region' in batch_dataset_type:
                SEG_embedding = None


            mask_outputs = self.predictor(multi_scale_features, 
                                          mask_features, 
                                          None, 
                                          seg_query, 
                                          SEG_embedding,
                                          class_name_embedding, 
                                          region_embedding_list
                                          )
            
            # pred_masks = mask_outputs["pred_masks"] 
            # original_images = images
            # if not hasattr(self, 'saved_masks'):
            #     self.saved_masks = []
            #     self.saved_images = []
            # self.saved_masks.append(pred_masks.detach().cpu())
            # self.saved_images.append(original_images.detach().cpu())


            if seg_info is not None:
                # if "instances" in seg_info[0]:
                #     gt_instances = [x["instances"].to(self.device) for x in seg_info]
                #     # images = ImageList.from_tensors(images, self.size_divisibility)
                #     targets = self.prepare_targets(gt_instances, target_images)
                # else:
                #     targets = None

                # for i, seg in enumerate(seg_info):
                #     print(f"\n--- 样本 {i} 的键: {list(seg.keys())} ---")

                if images1 is not None and "instances1" in seg_info[0]:
                    # print("____________________________________________")
                    # h, w = images1.shape[-2:]
                    # print(f"Image1 size: {w} * {h}") 
                    # for i, x in enumerate(seg_info):
                    #     print(f"\n--- sample {i} keys: {list(x.keys())} ---")
                    #     print(f"  Type of x['instances1']: {type(x['instances1'])}")
                    #     print(f"  Content of x['instances1']:", x['instances1'])

                    gt_instances = [x["instances1"].to(self.device) for x in seg_info]
                    target_images = images1
                else:
                    # print("____________________________________________")
                    # h, w = images.shape[-2:]
                    # print(f"Image size: {w} * {h}") 
                    # for i, x in enumerate(seg_info):
                    #     print(f"\n--- sample {i} keys: {list(x.keys())} ---")
                    #     print(f"  Type of x['instances']: {type(x['instances'])}")
                    #     print(f"  Content of x['instances']:", x['instances'])

                    gt_instances = [x["instances"].to(self.device) for x in seg_info]
                    target_images = images

                targets = self.prepare_targets(gt_instances, target_images)
                

    
                mask_losses = self.criterion(mask_outputs, targets)
                weight_dict = self.weight_dict

                # ================= [新增/修改：计算 Proposal Loss] =================
                loss_proposal = torch.tensor(0.0, device=self.device)
                
                # 只有在 (跨图任务) 且 (有 aux_similarity_maps) 时计算
                # 注意：aux_similarity_maps 是我们之前修改 prepare_inputs... 返回的
                loss_condition = ('cross' in batch_dataset_type or 'referring' in batch_dataset_type)
                
                if loss_condition and aux_similarity_maps is not None:
                    
                    # 1. 获取特征图尺寸 (从 similarity map 获取)
                    # aux_similarity_maps: [B, 1, H_feat, W_feat]
                    feat_h, feat_w = aux_similarity_maps.shape[-2:]
                    
                    # 2. 获取原图尺寸 (直接使用 target_images，它肯定是对的那张图)
                    if isinstance(target_images, list):
                        img_h, img_w = target_images[0].shape[-2:]
                    else:
                        img_h, img_w = target_images.shape[-2:]
                        
                    # 3. 生成 GT Heatmap (直接复用 gt_instances)
                    # 这些 instances 已经被移动到 device 上了，且对应 target_images
                    gt_heatmap = self.generate_gaussian_heatmap(
                        (img_h, img_w), 
                        (feat_h, feat_w), 
                        gt_instances
                    )
                    
                    # 4. 计算 MSE Loss
                    # 这里的 300.0 是权重，可以根据 loss 大小调整
                    loss_proposal = F.mse_loss(aux_similarity_maps.float(), gt_heatmap.float()) * 500.0
                
        
                loss_mask = 0.0
                loss_dice = 0.0
                loss_SEG_class = 0.0
                loss_class_name_class = 0.0
                loss_region_class = 0.0
                for k in list(mask_losses.keys()):
                    if k in weight_dict:
                        if mask_losses[k] is not None:
                            mask_losses[k] *= weight_dict[k]
                        if '_SEG' in k and mask_losses[k] is not None:
                            loss_SEG_class += mask_losses[k]
                        elif '_name' in k and mask_losses[k] is not None:
                            loss_class_name_class += mask_losses[k]
                        elif '_mask' in k:
                            loss_mask += mask_losses[k]
                        elif '_dice' in k:
                            loss_dice += mask_losses[k]
                        elif '_region' in k and mask_losses[k] is not None:
                            loss_region_class += mask_losses[k]
                    else:
                        mask_losses.pop(k)

                loss_region_class= loss_region_class*0.4 
                loss_mask= loss_mask*1.5       
                mask_loss = loss_mask + loss_dice + loss_SEG_class + loss_class_name_class + loss_region_class + loss_proposal
                
                # 确保损失为张量类型
                if isinstance(loss_class_name_class, float):
                    loss_class_name_class = torch.tensor(loss_class_name_class, device=mask_loss.device)
                if isinstance(loss_SEG_class, float):
                    loss_SEG_class = torch.tensor(loss_SEG_class, device=mask_loss.device)
                if isinstance(loss_region_class, float):
                    loss_region_class = torch.tensor(loss_region_class, device=mask_loss.device)
            llm = torch.tensor(0.0, device=mask_loss.device)
            if labels is not None:
                # loss = llm_loss + mask_loss
                loss = mask_loss

            # 返回分割任务的输出，含所有损失项和模型输出
            return CausalOutputWithMask(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                loss_mask=loss_mask.detach(),
                loss_dice=loss_dice.detach(),
                loss_SEG_class=loss_SEG_class.detach(),
                loss_class_name_class=loss_class_name_class.detach(),
                loss_region_class=loss_region_class.detach(),
                loss_proposal=loss_proposal.detach(),
                loss_llm=llm.detach(),
            )

        # 若有标签且无分割任务，总损失 = 语言模型损失
        if labels is not None and seg_query_mask is None:
            loss_mask = torch.tensor(0.0, device=llm_loss.device)
            loss_dice = torch.tensor(0.0, device=llm_loss.device)
            loss_SEG_class = torch.tensor(0.0, device=llm_loss.device)
            loss_class_name_class = torch.tensor(0.0, device=llm_loss.device)
            loss_region_class = torch.tensor(0.0, device=llm_loss.device)
            loss = llm_loss
        else:
            # 无标签时，仅返回模型输出，不返回损失
            return CausalOutputWithMask(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        # 返回普通语言任务的输出（含语言损失和其他信息）
        return CausalOutputWithMask(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            loss_mask=loss_mask.detach(),
            loss_dice=loss_dice.detach(),
            loss_SEG_class=loss_SEG_class.detach(),
            loss_class_name_class=loss_class_name_class.detach(),
            loss_region_class=loss_region_class.detach(),
            loss_llm=llm_loss.detach(),
        )

    def mm_conv_prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                # ensure gradients back propagation, not changing cur_input_embeds
                cur_input_embeds = cur_input_embeds + (0. * self.get_model().mm_projector(vision_tower.dummy_feature)).sum()
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            # concat text and image embedding. prepare labels, IGNORE_INDEX for image tokens
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start-1]).detach())
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start-1:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start+1:image_token_start+2]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[image_token_start:image_token_start+1])
                        cur_labels = cur_labels[image_token_start+2:]
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[image_token_start+1:]
                cur_image_idx += 1
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_input_ids = cur_input_ids[image_token_start+2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_start+1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        # Align embedddings, labels, attn_mask from different sample into a batch
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def get_seg_query(self, hidden_states, seg_query_masks):
        seg_query_list = []
        for sample_hidden_state, sample_query_mask in zip(hidden_states, seg_query_masks):
            if torch.sum(sample_query_mask) == 0:
                continue

            unique_query_value = torch.unique(sample_query_mask)
            unique_query_value = unique_query_value[unique_query_value != 0]

            for value in unique_query_value:
                current_query_mask = (sample_query_mask == value)
                current_query = sample_hidden_state[current_query_mask]

                seg_query_list.append(current_query)

        seg_query = torch.stack(seg_query_list, dim=0)

        return seg_query
    def eval_seg(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            images1: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            seg_info=None,
            class_name_ids=None,
            class_name_embedding_indices=None,
            cls_indices=None,
            token_refer_id=None,
            refer_embedding_indices=None,
            is_thing_list=None
    ):
        if self.panoptic_on:
            assert is_thing_list is not None, 'is_thing_list need to be given'
            self.is_thing_list = is_thing_list
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids == REGION_TOKEN_INDEX).sum() != 0:
            if images1 is not None:
                instances1 = [i['instances1'] for i in seg_info]
            instances = [i['instances'] for i in seg_info]
        else:
            instances = None
            
        # 从 seg_info 中提取 dataset_type
        dataset_type = None
        if seg_info is not None and len(seg_info) > 0 and 'dataset_type' in seg_info[0]:
            dataset_type = [item['dataset_type'] for item in seg_info]
        
        # 将 dataset_type 传递给 prepare_inputs_labels_for_multimodal
        input_ids, attention_mask, past_key_values, inputs_embeds, labels, seg_query_mask, class_name_embedding_indices, region_embedding_masks, refer_embedding_indices, _ = self.prepare_inputs_labels_for_multimodal(
                input_ids, attention_mask, past_key_values, labels, images, images1, class_name_embedding_indices,
                class_name_ids, cls_indices, instances, token_refer_id, refer_embedding_indices, 
                dataset_type=dataset_type)  

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs.last_hidden_state
        seg_query = self.get_seg_query(hidden_states, seg_query_mask)
        seg_query = self.seg_query_projector(seg_query)

        if images1 is not None:
            image_features = self.get_vision_tower_feature(images1)
            mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
            image_features)
        else:
            image_features = self.get_vision_tower_feature(images)
            mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
            image_features)

        if refer_embedding_indices is not None:
            SEG_embedding = self.get_SEG_embedding(hidden_states, refer_embedding_indices)
            SEG_embedding = self.SEG_token_projector(SEG_embedding)
        else:
            SEG_embedding = None

        if class_name_embedding_indices is not None:
            class_name_embedding = self.get_class_name_embedding(hidden_states, class_name_embedding_indices)
            class_name_embedding = self.class_name_projector(class_name_embedding)
        else:
            class_name_embedding = None

        if region_embedding_masks is not None:
            region_embedding_list = self.get_region_embedding(hidden_states, region_embedding_masks)
            region_embedding_list = [self.region_projector(region_embedding) for region_embedding in
                                     region_embedding_list]
        else:
            region_embedding_list = None

        mask_outputs = self.predictor(multi_scale_features, mask_features, None, seg_query, SEG_embedding,
                                      class_name_embedding, region_embedding_list)

        SEG_cls_results = mask_outputs['pred_SEG_logits']
        class_name_cls_results = mask_outputs['pred_class_name_logits']
        mask_pred_results = mask_outputs["pred_masks"]
        region_cls_results = mask_outputs['pred_region_logits']
        images = [x for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        del mask_outputs
        processed_results = []
        batch_size = len(seg_info)
        if SEG_cls_results is None:
            SEG_cls_results = [None] * batch_size
        if class_name_cls_results is None:
            class_name_cls_results = [None] * batch_size
        if region_cls_results is None:
            region_cls_results = [None] * batch_size
            
        for _seg_info, SEG_cls_result, class_name_cls_result, mask_pred_result, region_cls_result, input_per_image, image_size in zip(
                seg_info, SEG_cls_results, class_name_cls_results, mask_pred_results, region_cls_results, seg_info, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            padding_mask = input_per_image.get("padding_mask")
            non_padding_indices = np.where(~ np.array(padding_mask))
            min_y, max_y = np.min(non_padding_indices[0]), np.max(non_padding_indices[0])
            min_x, max_x = np.min(non_padding_indices[1]), np.max(non_padding_indices[1])
            original_height = max_y - min_y + 1
            original_width = max_x - min_x + 1
            processed_results.append({})
            # gt = _seg_info['instances'].gt_masks
            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, [original_height, original_width], height, width
                )
                if SEG_cls_result is not None:
                    SEG_cls_result = SEG_cls_result.to(mask_pred_result)

            if self.semantic_on:
                semantic_r = retry_if_cuda_oom(self.class_name_semantic_inference)(None,
                                                                                   class_name_cls_result.float(),
                                                                                   mask_pred_result.float())
                if not self.sem_seg_postprocess_before_inference:
                    semantic_r = retry_if_cuda_oom(sem_seg_postprocess)(
                    semantic_r, [original_height, original_width], height, width
                )
                processed_results[-1]["sem_seg"] = semantic_r

            if self.instance_on:
                instance_r = retry_if_cuda_oom(self.class_name_instance_inference)(None,
                                                                                   class_name_cls_result.float(),
                                                                                   mask_pred_result.float())
                processed_results[-1]["instances"] = instance_r
            if self.panoptic_on:
                panoptic_r = retry_if_cuda_oom(self.class_name_panoptic_inference)(None,
                                                                                   class_name_cls_result.float(),
                                                                                   mask_pred_result.float())
                processed_results[-1]["panoptic_seg"] = panoptic_r
            if self.referring_on:
                instance_r = retry_if_cuda_oom(self.SEG_instance_inference)(SEG_cls_result.float(),
                                                                            mask_pred_result.float())
                processed_results[-1]["instances"] = instance_r
            if self.region_on:
                gt = _seg_info['instances'].gt_masks
                gt_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    gt, [original_height, original_width], height, width
                )
                current_region_cls = region_cls_result.to(mask_pred_result)
                instance_r = retry_if_cuda_oom(self.region_inference)(current_region_cls.float(),
                                                                            mask_pred_result.float())
                processed_results[-1]["instances"] = instance_r
                processed_results[-1]["gt"] = gt_result

        return processed_results

    def generate_gaussian_heatmap(self, image_size, feature_size, instances, sigma=2.0):
        """
        生成高斯热图 GT
        Args:
            image_size: (H_img, W_img) 原始图片大小
            feature_size: (H_feat, W_feat) 特征图大小 (32, 32)
            instances: Detectron2 Instances 对象列表，包含 gt_masks
        Returns:
            heatmap: [B, 1, H_feat, W_feat]
        """
        B = len(instances)
        H, W = feature_size
        device = instances[0].gt_classes.device
        gt_heatmap = torch.zeros(B, 1, H, W, device=device)
        
        for b_idx, inst in enumerate(instances):
            if len(inst) == 0: continue
            
            # 获取当前图片所有 GT 的中心点
            # 假设 inst.gt_masks 是 BitMasks 或 Tensor
            if hasattr(inst, 'gt_masks'):
                if isinstance(inst.gt_masks, torch.Tensor):
                    masks = inst.gt_masks
                else: # BitMasks
                    masks = inst.gt_masks.tensor
                
                # 计算重心
                # masks: [N, H_img, W_img]
                for mask in masks:
                    # 简单粗暴的方法：nonzero 取平均
                    # 注意：这里 mask 是原图大小，需要缩放到 feature map 大小
                    y_indices, x_indices = torch.where(mask)
                    if len(y_indices) == 0: continue
                    
                    center_y = y_indices.float().mean()
                    center_x = x_indices.float().mean()
                    
                    # 映射到 Feature Map 坐标
                    feat_y = int(center_y / image_size[0] * H)
                    feat_x = int(center_x / image_size[1] * W)
                    
                    # 限制范围
                    feat_y = min(max(feat_y, 0), H - 1)
                    feat_x = min(max(feat_x, 0), W - 1)
                    
                    # 在 (feat_x, feat_y) 处生成高斯点
                    # 创建网格
                    y_grid, x_grid = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device))
                    gaussian = torch.exp(-((x_grid - feat_x)**2 + (y_grid - feat_y)**2) / (2 * sigma**2))
                    
                    # 叠加到 Heatmap (取最大值，处理重叠)
                    gt_heatmap[b_idx, 0] = torch.maximum(gt_heatmap[b_idx, 0], gaussian)
                    
        return gt_heatmap



class PSALMForDAVISEval(PSALM):
    def eval_seg(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            seg_info=None,
            class_name_ids=None,
            class_name_embedding_indices=None,
            cls_indices=None,
            token_refer_id=None,
            refer_embedding_indices=None,
            is_thing_list=None,
            vp_images=None
    ):
        if self.panoptic_on:
            assert is_thing_list is not None, 'is_thing_list need to be given'
            self.is_thing_list = is_thing_list
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids == REGION_TOKEN_INDEX).sum() != 0:
            instances = [i['instances'] for i in seg_info]
        else:
            instances = None
        input_ids, attention_mask, past_key_values, inputs_embeds, labels, seg_query_mask, class_name_embedding_indices, region_embedding_masks, refer_embedding_indices = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images,vp_images, class_name_embedding_indices,
            class_name_ids, cls_indices, instances, token_refer_id, refer_embedding_indices)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs.last_hidden_state
        seg_query = self.get_seg_query(hidden_states, seg_query_mask)
        seg_query = self.seg_query_projector(seg_query)

        image_features = self.get_vision_tower_feature(images)
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
            image_features)

        if refer_embedding_indices is not None:
            SEG_embedding = self.get_SEG_embedding(hidden_states, refer_embedding_indices)
            SEG_embedding = self.SEG_token_projector(SEG_embedding)
        else:
            SEG_embedding = None

        if class_name_embedding_indices is not None:
            class_name_embedding = self.get_class_name_embedding(hidden_states, class_name_embedding_indices)
            class_name_embedding = self.class_name_projector(class_name_embedding)
        else:
            class_name_embedding = None

        if region_embedding_masks is not None:
            region_embedding_list = self.get_region_embedding(hidden_states, region_embedding_masks)
            region_embedding_list = [self.region_projector(region_embedding) for region_embedding in
                                     region_embedding_list]
        else:
            region_embedding_list = None

        mask_outputs = self.predictor(multi_scale_features, mask_features, None, seg_query, SEG_embedding,
                                      class_name_embedding, region_embedding_list)

        SEG_cls_results = mask_outputs['pred_SEG_logits']
        class_name_cls_results = mask_outputs['pred_class_name_logits']
        mask_pred_results = mask_outputs["pred_masks"]
        region_cls_results = mask_outputs['pred_region_logits']
        images = [x for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        del mask_outputs
        processed_results = []
        batch_size = len(seg_info)
        if SEG_cls_results is None:
            SEG_cls_results = [None] * batch_size
        if class_name_cls_results is None:
            class_name_cls_results = [None] * batch_size
        for _seg_info, SEG_cls_result, class_name_cls_result, mask_pred_result, input_per_image, image_size in zip(
                seg_info, SEG_cls_results, class_name_cls_results, mask_pred_results, seg_info, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            padding_mask = input_per_image.get("padding_mask")
            non_padding_indices = np.where(~ np.array(padding_mask))
            min_y, max_y = np.min(non_padding_indices[0]), np.max(non_padding_indices[0])
            min_x, max_x = np.min(non_padding_indices[1]), np.max(non_padding_indices[1])
            original_height = max_y - min_y + 1
            original_width = max_x - min_x + 1
            processed_results.append({})
            # gt = _seg_info['instances'].gt_masks
            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, [original_height, original_width], height, width
                )
                # gt_result = retry_if_cuda_oom(sem_seg_postprocess)(
                #     gt, [original_height, original_width], height, width
                # )
                if SEG_cls_result is not None:
                    SEG_cls_result = SEG_cls_result.to(mask_pred_result)

            if self.semantic_on:
                semantic_r = retry_if_cuda_oom(self.class_name_semantic_inference)(None,
                                                                                   class_name_cls_result.float(),
                                                                                   mask_pred_result.float())
                if not self.sem_seg_postprocess_before_inference:
                    semantic_r = retry_if_cuda_oom(sem_seg_postprocess)(
                    semantic_r, [original_height, original_width], height, width
                )
                processed_results[-1]["sem_seg"] = semantic_r

            if self.instance_on:
                instance_r = retry_if_cuda_oom(self.class_name_instance_inference)(None,
                                                                                   class_name_cls_result.float(),
                                                                                   mask_pred_result.float())
                processed_results[-1]["instances"] = instance_r
            if self.panoptic_on:
                panoptic_r = retry_if_cuda_oom(self.class_name_panoptic_inference)(None,
                                                                                   class_name_cls_result.float(),
                                                                                   mask_pred_result.float())
                processed_results[-1]["panoptic_seg"] = panoptic_r
            if self.referring_on:
                instance_r = retry_if_cuda_oom(self.SEG_instance_inference)(SEG_cls_result.float(),
                                                                            mask_pred_result.float())
                processed_results[-1]["instances"] = instance_r
            if self.region_on:
                gt = _seg_info['instances'].gt_masks
                gt_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    gt, [original_height, original_width], height, width
                )
                region_cls_results = region_cls_results[0].to(mask_pred_result)
                instance_r = retry_if_cuda_oom(self.region_inference)(region_cls_results.float(),
                                                                            mask_pred_result.float())
                processed_results[-1]["instances"] = instance_r
                processed_results[-1]["gt"] = gt_result

        return processed_results

    def prepare_inputs_labels_for_multimodal(
            self, input_ids, attention_mask, past_key_values, labels, images, vp_images=None, class_name_embedding_indices=None,
            class_name_ids=None, cls_indices=None, instances=None, token_refer_id=None, refer_embedding_indices=None
    ):
        vision_tower = self.get_vision_tower()
        seg_query_mask = torch.zeros_like(input_ids)
        
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[
                1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                                            dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels, seg_query_mask

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)

        expanded_seg_query = self.seg_query.unsqueeze(0).expand(input_ids.shape[0], -1, -1)

        if (input_ids == REGION_TOKEN_INDEX).sum() != 0 and instances is not None:
            region_masks_list = [instance.vp_region_masks.tensor for instance in instances]
            vp_image_features = self.encode_images(vp_images)

            # [region_features_per_batch: [num_region, 1, dims]], len(region_features) = batch_size
            region_features = self.region_sampler(vp_image_features, region_masks_list,
                                                  original_dtype=vp_image_features.dtype,
                                                  return_dtype=vp_image_features.dtype)
            region_embedding_masks = torch.zeros_like(input_ids)
        else:
            region_features = None
            region_embedding_masks = None
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        new_seg_query_masks = []
        new_class_name_embedding_indices = [] if class_name_embedding_indices is not None else None
        new_refer_embedding_indices = [] if refer_embedding_indices is not None else None
        new_region_embedding_masks = [] if region_features is not None else None
        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_seg_query_mask = seg_query_mask[batch_idx]
            cur_seg_query = expanded_seg_query[batch_idx]
            cur_image_feature = image_features[batch_idx]
            # 为PSALMForDAVISEval设置默认值（该函数不支持image1和deformable）
            cur_image1_feature = None
            cur_image_deform_feature = None
            cur_image1_deform_feature = None
            cur_class_name_embedding_indices = class_name_embedding_indices[batch_idx] if class_name_embedding_indices is not None else None
            cur_refer_embedding_indices = refer_embedding_indices[batch_idx] if refer_embedding_indices is not None else None
            cur_region_feature_list = region_features[batch_idx] if region_features is not None else None
            cur_region_embedding_mask = region_embedding_masks[batch_idx] if region_features is not None else None
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                # ensure gradients back propagation, not changing cur_input_embeds
                cur_input_embeds = cur_input_embeds + (
                        0. * self.get_model().mm_projector(vision_tower.dummy_feature)).sum()
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                new_seg_query_masks.append(cur_seg_query_mask)
                # cur_image_idx += 1
                continue

            if labels is not None:
                cur_label = labels[batch_idx]
            else:
                cur_label = None

            if class_name_ids is not None:
                cur_class_name_ids = class_name_ids[batch_idx]
                cur_cls_indices = cls_indices[batch_idx]
            else:
                cur_class_name_ids = None
                cur_cls_indices = None
            if token_refer_id is not None:
                cur_token_refer_id = token_refer_id[batch_idx]
            else:
                cur_token_refer_id = None


            cur_class_name_embedding = self.embed_class_ids(cur_class_name_ids, cur_cls_indices)
            cur_refer_embedding = self.embed_refer_ids(cur_token_refer_id)

            cur_input_embeds, cur_label, cur_seg_query_mask, cur_class_name_embedding_indices, cur_region_embedding_mask, cur_refer_embedding_indices = self.concat_image_seg_cls_embeds(
                input_id=cur_input_ids,
                img_feature=cur_image_feature,
                img1_feature=cur_image1_feature,
                label=cur_label,
                seg_query=cur_seg_query,
                seg_query_mask=cur_seg_query_mask,
                class_embed=cur_class_name_embedding,
                class_name_embedding_indices=cur_class_name_embedding_indices,
                region_embedding_mask=cur_region_embedding_mask,
                region_feature_list=cur_region_feature_list,
                refer_embedding_indices=cur_refer_embedding_indices,
                refer_embedding=cur_refer_embedding,
                img_deform_feature=cur_image_deform_feature,
                img1_deform_feature=cur_image1_deform_feature
            )
            assert cur_input_embeds.shape[0] == cur_seg_query_mask.shape[0]

            new_input_embeds.append(cur_input_embeds)
            if labels is not None:
                new_labels.append(cur_label)
            new_seg_query_masks.append(cur_seg_query_mask)
            if class_name_embedding_indices is not None:
                new_class_name_embedding_indices.append(cur_class_name_embedding_indices)
            if refer_embedding_indices is not None:
                new_refer_embedding_indices.append(cur_refer_embedding_indices)
            if new_region_embedding_masks is not None:
                new_region_embedding_masks.append(cur_region_embedding_mask)
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed,
                                           torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                                                       dtype=cur_new_embed.dtype, device=cur_new_embed.device)),
                                          dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label,
                                               torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX,
                                                          dtype=cur_new_label.dtype, device=cur_new_label.device)),
                                              dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            new_seg_query_masks_align = []
            for new_seg_query_mask in new_seg_query_masks:
                new_seg_query_mask = torch.cat(
                    (new_seg_query_mask, torch.zeros((max_len - new_seg_query_mask.shape[0]),dtype=new_seg_query_mask.dtype, device=new_seg_query_mask.device)),
                    dim=0)
                new_seg_query_masks_align.append(new_seg_query_mask)
            new_seg_query_masks = torch.stack(new_seg_query_masks_align, dim=0)

            new_class_name_embedding_indices_align = []

            if class_name_embedding_indices is not None:
                for new_class_name_embedding_indice in new_class_name_embedding_indices:
                    new_class_name_embedding_indice = torch.cat(
                        (new_class_name_embedding_indice,
                         torch.zeros((max_len - new_class_name_embedding_indice.shape[0]),dtype=new_class_name_embedding_indice.dtype, device=new_class_name_embedding_indice.device)),
                        dim=0)
                    new_class_name_embedding_indices_align.append(new_class_name_embedding_indice)
                new_class_name_embedding_indices = torch.stack(new_class_name_embedding_indices_align, dim=0)

            if refer_embedding_indices is not None:
                new_refer_embedding_indices_align = []
                for new_refer_embedding_indice in new_refer_embedding_indices:
                    new_refer_embedding_indice = torch.cat(
                        (new_refer_embedding_indice,
                         torch.zeros((max_len - new_refer_embedding_indice.shape[0]),dtype=new_refer_embedding_indice.dtype, device=new_refer_embedding_indice.device)),
                        dim=0)
                    new_refer_embedding_indices_align.append(new_refer_embedding_indice)
                new_refer_embedding_indices = torch.stack(new_refer_embedding_indices_align, dim=0)

            if new_region_embedding_masks is not None:
                new_region_embedding_masks_align = []
                for new_region_embedding_mask in new_region_embedding_masks:
                    new_region_embedding_mask = torch.cat(
                        (new_region_embedding_mask, torch.zeros((max_len - new_region_embedding_mask.shape[0]),dtype=new_region_embedding_mask.dtype, device=new_region_embedding_mask.device)),
                        dim=0)
                    new_region_embedding_masks_align.append(new_region_embedding_mask)
                new_region_embedding_masks = torch.stack(new_region_embedding_masks_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels,
                                                                                    new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True,
                                                        dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                                                         False, dtype=attention_mask.dtype,
                                                         device=attention_mask.device)
                    cur_new_attention_mask = torch.cat(
                        (new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape

        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            new_seg_query_masks = torch.stack(new_seg_query_masks, dim=0)
            if class_name_embedding_indices is not None:
                new_class_name_embedding_indices = torch.stack(new_class_name_embedding_indices, dim=0)
            if refer_embedding_indices is not None:
                new_refer_embedding_indices = torch.stack(new_refer_embedding_indices, dim=0)

            if new_region_embedding_masks is not None:
                new_region_embedding_masks = torch.stack(new_region_embedding_masks, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True,
                    dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels, new_seg_query_masks, new_class_name_embedding_indices, new_region_embedding_masks, new_refer_embedding_indices
    def eval_video(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            vp_images: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            seg_info=None,
            class_name_ids=None,
            class_name_embedding_indices=None,
            cls_indices=None,
            token_refer_id=None,
            refer_embedding_indices=None,
            is_thing_list=None
    ):
        if self.panoptic_on:
            assert is_thing_list is not None, 'is_thing_list need to be given'
            self.is_thing_list = is_thing_list
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids == REGION_TOKEN_INDEX).sum() != 0:
            instances = [i['instances'] for i in seg_info]
        else:
            instances = None
        input_ids, attention_mask, past_key_values, inputs_embeds, labels, seg_query_mask, class_name_embedding_indices, region_embedding_masks, refer_embedding_indices = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images,vp_images, class_name_embedding_indices,
            class_name_ids, cls_indices, instances, token_refer_id, refer_embedding_indices)


        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs.last_hidden_state
        seg_query = self.get_seg_query(hidden_states, seg_query_mask)
        seg_query = self.seg_query_projector(seg_query)

        image_features = self.get_vision_tower_feature(images)
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
            image_features)

        if refer_embedding_indices is not None:
            SEG_embedding = self.get_SEG_embedding(hidden_states, refer_embedding_indices)
            SEG_embedding = self.SEG_token_projector(SEG_embedding)
        else:
            SEG_embedding = None

        if class_name_embedding_indices is not None:
            class_name_embedding = self.get_class_name_embedding(hidden_states, class_name_embedding_indices)
            class_name_embedding = self.class_name_projector(class_name_embedding)
        else:
            class_name_embedding = None

        if region_embedding_masks is not None:
            region_embedding_list = self.get_region_embedding(hidden_states, region_embedding_masks)
            region_embedding_list = [self.region_projector(region_embedding) for region_embedding in
                                     region_embedding_list]
        else:
            region_embedding_list = None

        mask_outputs = self.predictor(multi_scale_features, mask_features, None, seg_query, SEG_embedding,
                                      class_name_embedding, region_embedding_list)

        SEG_cls_results = mask_outputs['pred_SEG_logits']
        class_name_cls_results = mask_outputs['pred_class_name_logits']
        mask_pred_results = mask_outputs["pred_masks"]
        region_cls_results = mask_outputs['pred_region_logits']
        images = [x for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        del mask_outputs
        processed_results = []
        batch_size = len(seg_info)
        if SEG_cls_results is None:
            SEG_cls_results = [None] * batch_size
        if class_name_cls_results is None:
            class_name_cls_results = [None] * batch_size
        for _seg_info, SEG_cls_result, class_name_cls_result, mask_pred_result, input_per_image, image_size in zip(
                seg_info, SEG_cls_results, class_name_cls_results, mask_pred_results, seg_info, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            padding_mask = input_per_image.get("padding_mask")
            non_padding_indices = np.where(~ np.array(padding_mask))
            min_y, max_y = np.min(non_padding_indices[0]), np.max(non_padding_indices[0])
            min_x, max_x = np.min(non_padding_indices[1]), np.max(non_padding_indices[1])
            original_height = max_y - min_y + 1
            original_width = max_x - min_x + 1
            processed_results.append({})
            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, [original_height, original_width], height, width
                )
                if SEG_cls_result is not None:
                    SEG_cls_result = SEG_cls_result.to(mask_pred_result)

            if self.semantic_on:
                semantic_r = retry_if_cuda_oom(self.class_name_semantic_inference)(None,
                                                                                   class_name_cls_result.float(),
                                                                                   mask_pred_result.float())
                if not self.sem_seg_postprocess_before_inference:
                    semantic_r = retry_if_cuda_oom(sem_seg_postprocess)(
                        semantic_r, [original_height, original_width], height, width
                    )
                processed_results[-1]["sem_seg"] = semantic_r

            if self.instance_on:
                instance_r = retry_if_cuda_oom(self.class_name_instance_inference)(None,
                                                                                   class_name_cls_result.float(),
                                                                                   mask_pred_result.float())
                processed_results[-1]["instances"] = instance_r
            if self.panoptic_on:
                panoptic_r = retry_if_cuda_oom(self.class_name_panoptic_inference)(None,
                                                                                   class_name_cls_result.float(),
                                                                                   mask_pred_result.float())
                processed_results[-1]["panoptic_seg"] = panoptic_r
            if self.referring_on:
                instance_r = retry_if_cuda_oom(self.SEG_instance_inference)(SEG_cls_result.float(),
                                                                            mask_pred_result.float())
                processed_results[-1]["instances"] = instance_r
            if self.region_on:
                gt = _seg_info['instances'].gt_masks
                gt_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    gt, [original_height, original_width], height, width
                )
                region_cls_results = region_cls_results[0].to(mask_pred_result)
                instance_r = retry_if_cuda_oom(self.region_inference)(region_cls_results.float(),
                                                                      mask_pred_result.float())
                processed_results[-1]["instances"] = instance_r
                processed_results[-1]["gt"] = gt_result

        return processed_results


AutoConfig.register("llava_phi", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, PSALMModel)
