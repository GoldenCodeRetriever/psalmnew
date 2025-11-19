export DISABLE_ADDMM_CUDA_LT=1
# export CUDA_VISIBLE_DEVICES=5  

deepspeed --include localhost:4,5,6,7  --master_port 29505 psalm/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path "/home/zhangkunquan/code/PSALM/PSALM/phi-1_5_dev" \
    --version "llava_phi" \
    --region_json_path "/nfs-data1/zhangkunquan/COCO_data/coco_interactive_train_psalm.json" \
    --panoptic_json_path "/nfs-data1/zhangkunquan/COCO_data/coco" \
    --ref_coco_path "/nfs-data1/zhangkunquan/COCO_data/refseg/refcoco/refcoco_train.json" \
    --ref_coco_plus_path "/nfs-data1/zhangkunquan/COCO_data/refseg/refcoco+/refcoco+_train.json" \
    --ref_coco_g_path "/nfs-data1/zhangkunquan/COCO_data/refseg/refcocog/refcocog_train.json" \
    --image_folder "/nfs-data1/zhangkunquan/COCO_data/coco/train2017" \
    --refcoco_image_folder "/nfs-data1/zhangkunquan/COCO_data/refseg/images/mscoco/train2014" \
    --mmconv_path "/home/zhangkunquan/code/PSALM/PSALM/llava" \
    --vision_tower "/home/zhangkunquan/code/PSALM/PSALM/Siwn-B Mask2former/model_final_54b88a.pkl" \
    --pretrain_mm_mlp_adapter "/home/zhangkunquan/code/PSALM/PSALM/PSALM_stage1/mm_projector.bin" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --fp16 True \
    --output_dir ./output/checkpoint/COCO_data_10 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 15000 \
    --save_total_limit 2 \
    --learning_rate 6e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to none \
    --seg_task 'region'\
    --region_mask_type 'box_visual_prompt_mask||scribble_visual_prompt_mask||point_visual_prompt_mask'
#region\panoptic\referring
# --seg_task 'referring'\
# --region_mask_type 'box_visual_prompt_mask||scribble_visual_prompt_mask||point_visual_prompt_mask'

