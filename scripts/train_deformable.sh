#!/bin/bash
# 使用Deformable模块的训练脚本
# 该脚本启用MultiScaleDeformableCrossAttentionAlignment模块进行训练

# export CUDA_VISIBLE_DEVICES=0,1,2,3  # 根据需要设置可见的GPU
export PYTHONPATH=/home/lipeilang/projects/PSALM_new:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024

# Optional: path to pretrained weights for deformable projector
# Can be set via environment variable: PRETRAIN_MM_DEFORMABLE_ADAPTER=/path/to/file
# PRETRAIN_MM_DEFORMABLE_ADAPTER=${PRETRAIN_MM_DEFORMABLE_ADAPTER:-""}

# 使用deepspeed进行分布式训练
# --include: 指定使用的GPU节点，格式为 localhost:gpu_id1,gpu_id2,...
# --master_port: 主节点端口号，避免冲突
deepspeed --include localhost:4,5,6,7 --master_port 29501 psalm/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path "/home/lipeilang/projects/PSALM_new/PSALM/phi-1_5_dev" \
    --version "llava_phi" \
    --region_json_path "/nfs-data1/public/psalm_data/RRSIS-D_split/region_full/train/region_cleaned.json" \
    --region_image_folder "/nfs-data1/public/psalm_data/RRSIS-D_split/region_full/train/JPEGImages" \
    --region_cross_json_path "/nfs-data1/public/test/updatedcategoryid2.0_1_6_14_17_train2.json" \
    --region_cross_image_folder "/nfs-data1/public/test/images" \
    --panoptic_json_path "/path/to/coco" \
    --referring_json_path "/nfs-data1/public/referring/train/updated_referring.json" \
    --referring_image_folder "/nfs-data1/public/referring/train/JPEGImages" \
    --ref_coco_path "/path/to/refcoco/refcoco_train.json" \
    --ref_coco_plus_path "/path/to/refcoco+/refcoco+_train.json" \
    --ref_coco_g_path "/path/to/refcocog/refcocog_train.json" \
    --refcoco_image_folder "/path/to/coco/train2014" \
    --image_folder "/path/to/coco/train2017" \
    --mmconv_path "/home/lipeilang/projects/PSALM_new/PSALM/llava" \
    --vision_tower "/home/lipeilang/projects/PSALM_new/PSALM/Siwn-B Mask2former/model_final_54b88a.pkl" \
    --pretrain_mm_mlp_adapter "/home/lipeilang/projects/PSALM_new/PSALM/PSALM_stage1/mm_projector.bin" \
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
    --output_dir /nfs-data1/lipeilang/output/checkpoint/PSALM_multi_query_deformable_1.21 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 20 \
    --learning_rate 5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --seg_task 'region' \
    --cross_image_seg_task True \
    --region_mask_type "point_visual_prompt_mask||box_visual_prompt_mask||scribble_visual_prompt_mask||mask_visual_prompt_mask"
    # --train_backbone True \
    # --freeze_vision_tower False \
    
    # 区域分割任务的提示类型（可选）
    # --region_mask_type 'box_visual_prompt_mask||scribble_visual_prompt_mask||point_visual_prompt_mask' \

# ========== 使用说明 ==========False
# 1. 修改所有路径为实际数据路径
# 2. 根据GPU数量调整 --include 参数
# 3. 根据显存大小调整 batch_size 和 gradient_accumulation_steps
# 4. 如果使用Swin-L，设置 --swin_type 'large' 和 --projector_outdim 2560
# 5. 如果从baseline checkpoint继续训练，确保 --pretrain_mm_mlp_adapter 指向正确的权重文件
#
# ========== Deformable模块说明 ==========
# - mm_projector_type='deformable': 启用Deformable多尺度注意力模块
# - mm_hidden_dim: Deformable attention的隐藏维度（默认256）
# - mm_n_heads: 注意力头数（默认8）
# - mm_n_points: 每个头每个尺度的采样点数（默认4）
# - projector_outdim: 输出维度，应与LLM的hidden_size一致（Phi-1.5为2048）
# - swin_type: 'base'对应vision_dims=[128,256,512,1024], 'large'对应[192,384,768,1536]
#
# ========== 注意事项 ==========
# 1. Deformable模块会增加模型参数量和计算量，建议适当减小batch_size
# 2. 首次训练建议使用较小的learning_rate（如1e-6）进行warmup
# 3. 如果显存不足，可以增加gradient_accumulation_steps来保持有效batch_size
# 4. 训练过程中会打印projector类型信息，确认是否成功启用deformable模块

