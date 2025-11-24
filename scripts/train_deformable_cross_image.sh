#!/bin/bash
# 使用Deformable模块进行跨图提示分割任务的训练脚本
# 该脚本专门用于训练跨图提示分割任务，充分利用Deformable模块的跨图特征对齐能力

export DISABLE_ADDMM_CUDA_LT=1
# export CUDA_VISIBLE_DEVICES=0,1,2,3  # 根据需要设置可见的GPU

# 使用deepspeed进行分布式训练
deepspeed --include localhost:0,1,2,3 --master_port 29501 psalm/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path "/path/to/phi-1_5_dev" \
    --version "llava_phi" \
    --region_cross_json_path "/path/to/interactive_train.json" \
    --region_cross_image_folder "/path/to/images" \
    --region_json_path "/path/to/coco_interactive_train_psalm.json" \
    --region_image_folder "/path/to/coco/train2017" \
    --panoptic_json_path "/path/to/coco" \
    --referring_json_path "/path/to/referring/train/referring.json" \
    --referring_image_folder "/path/to/referring/train/JPEGImages" \
    --ref_coco_path "/path/to/refcoco/refcoco_train.json" \
    --ref_coco_plus_path "/path/to/refcoco+/refcoco+_train.json" \
    --ref_coco_g_path "/path/to/refcocog/refcocog_train.json" \
    --refcoco_image_folder "/path/to/coco/train2014" \
    --image_folder "/path/to/coco/train2017" \
    --mmconv_path "/path/to/llava" \
    --vision_tower "/path/to/Swin-B_Mask2former/model_final_54b88a.pkl" \
    --pretrain_mm_mlp_adapter "/path/to/mm_projector.bin" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_projector_type 'deformable' \
    --mm_hidden_dim 256 \
    --mm_n_heads 8 \
    --mm_n_points 4 \
    --projector_outdim 2048 \
    --swin_type 'base' \
    --fp16 True \
    --output_dir ./output/checkpoint/PSALM_deformable_cross_image \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 5 \
    --learning_rate 1e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to none \
    --seg_task 'panoptic' \
    --cross_image_seg_task True \
    
    # 区域分割任务的提示类型（可选）
    # --region_mask_type 'box_visual_prompt_mask||scribble_visual_prompt_mask||point_visual_prompt_mask' \

# ========== 跨图任务说明 ==========
# 跨图提示分割任务的特点：
# 1. 使用两个图像：提示图（prompt image）和目标图（target image）
# 2. Deformable模块会使用提示图的区域特征作为查询，在目标图的多尺度特征中进行稀疏采样
# 3. 这样可以实现跨图的特征对齐和匹配
#
# Deformable模块在跨图任务中的工作流程：
# 1. 提示图（images）: 提取baseline特征，然后使用region_sampler提取区域特征
# 2. 目标图（image1）: 使用提示图的区域特征作为zq，通过deformable attention采样多尺度特征
# 3. 最终输出: 提示图的baseline特征 + 目标图的baseline特征 + 目标图的deformable特征
#
# ========== 训练建议 ==========
# 1. 跨图任务通常需要更小的batch_size（建议1-2），因为需要处理两个图像
# 2. 可以适当增加gradient_accumulation_steps来保持有效batch_size
# 3. 学习率建议从较小的值开始（1e-6），因为跨图任务相对复杂
# 4. 建议与单图任务（panoptic、region）混合训练，提高模型泛化能力
# 5. 训练过程中注意观察loss，确保跨图特征对齐正常

