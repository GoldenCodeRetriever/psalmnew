export DISABLE_ADDMM_CUDA_LT=1
# export CUDA_VISIBLE_DEVICES=5  

deepspeed --include localhost:7 --master_port 29501  psalm/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path "/home/zhangkunquan/code/PSALM/PSALM/phi-1_5_dev" \
    --version "llava_phi" \
    --region_cross_json_path "/nfs-data1/zhangkunquan/SIOR/test1.json" \
    --region_cross_image_folder "/nfs-data1/zhangkunquan/SIOR/images" \
    --region_json_path "/nfs-data1/zhangkunquan/SIOR/interactive_train.json" \
    --region_image_folder "/nfs-data1/zhangkunquan/SIOR/images" \
    --panoptic_json_path "/nfs-data1/zhangkunquan/psalm-data/coco" \
    --referring_json_path "/nfs-data1/public/psalm_data/RRSIS-D_split/referring/train/referring.json" \
    --referring_image_folder "/nfs-data1/public/psalm_data/RRSIS-D_split/referring/train/JPEGImages" \
    --ref_coco_path "/share/zhangyudong6-local/PSALM/datasets/refseg/refcoco/refcoco_train.json" \
    --ref_coco_plus_path "/share/zhangyudong6-local/PSALM/datasets/refseg/refcoco+/refcoco+_train.json" \
    --ref_coco_g_path "/share/zhangyudong6-local/PSALM/datasets/refseg/refcocog/refcocog_train.json" \
    --image_folder "/nfs-data1/zhangkunquan/psalm-data/coco/train2017" \
    --refcoco_image_folder "/share/zhangyudong6-local/PSALM/datasets/refseg/images/mscoco/train2014" \
    --mmconv_path "/home/zhangkunquan/code/PSALM/PSALM/llava" \
    --vision_tower "/home/zhangkunquan/code/PSALM/PSALM/Siwn-B Mask2former/model_final_54b88a.pkl" \
    --pretrain_mm_mlp_adapter "/home/zhangkunquan/code/PSALM/PSALM/PSALM_stage1/mm_projector.bin" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --fp16 True \
    --output_dir ./output/checkpoint/test \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --learning_rate 1e-6 \
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
    --seg_task 'panoptic'\
    --cross_image_seg_task False \
    
# --region_mask_type 'box_visual_prompt_mask' 
# --max_grad_norm 0.2
#region\panoptic\referring
# --seg_task 'referring'
# --region_mask_type 'box_visual_prompt_mask||scribble_visual_prompt_mask||point_visual_prompt_mask'

