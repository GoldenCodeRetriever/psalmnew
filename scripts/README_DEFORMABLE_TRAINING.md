# Deformable模块训练指南

本文档说明如何使用Deformable模块进行训练。

## 📋 文件说明

### 训练脚本

1. **`train_deformable.sh`** - 通用Deformable训练脚本
   - 适用于所有任务类型（Panoptic、Semantic、Instance、Referring、Region）
   - 支持单图和双图任务
   - 推荐用于首次使用Deformable模块的训练

2. **`train_deformable_cross_image.sh`** - 跨图任务专用训练脚本
   - 专门针对跨图提示分割任务优化
   - 启用 `--cross_image_seg_task True`
   - 充分利用Deformable模块的跨图特征对齐能力

## 🚀 快速开始

### 1. 准备工作

#### 1.1 检查依赖
确保已安装所有依赖：
```bash
pip install -r requirements.txt
```

#### 1.2 准备数据
确保所有数据路径正确：
- COCO数据集（Panoptic、Instance、Semantic）
- RefCOCO系列数据集（Referring）
- 区域分割数据集（Region）
- 跨图提示分割数据集（Cross-image，可选）

#### 1.3 准备预训练权重
- LLM模型：`phi-1_5_dev`
- Vision Tower：Swin-B/L Mask2Former模型
- Baseline mm_projector（可选，用于权重初始化）

### 2. 配置训练脚本

#### 2.1 修改路径
编辑训练脚本，修改以下路径：
```bash
# 模型路径
--model_name_or_path "/path/to/phi-1_5_dev"
--vision_tower "/path/to/Swin-B_Mask2former/model_final_54b88a.pkl"
--pretrain_mm_mlp_adapter "/path/to/mm_projector.bin"

# 数据路径
--region_json_path "/path/to/coco_interactive_train_psalm.json"
--panoptic_json_path "/path/to/coco"
# ... 其他数据路径
```

#### 2.2 配置GPU
根据可用GPU数量修改：
```bash
# 单GPU
deepspeed --include localhost:0 ...

# 多GPU（例如4个GPU）
deepspeed --include localhost:0,1,2,3 ...

# 指定特定GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

#### 2.3 调整超参数
根据显存和需求调整：
```bash
--per_device_train_batch_size 2      # 根据显存调整
--gradient_accumulation_steps 2       # 保持有效batch_size
--learning_rate 6e-5                  # 学习率
--num_train_epochs 10                  # 训练轮数
```

### 3. 运行训练

#### 3.1 通用训练（所有任务）
```bash
bash scripts/train_deformable.sh
```

#### 3.2 跨图任务训练
```bash
bash scripts/train_deformable_cross_image.sh
```

## ⚙️ Deformable模块参数说明

### 核心参数

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `--mm_projector_type` | 投影器类型 | `'swin_conv'` | `'deformable'` |
| `--mm_hidden_dim` | Deformable attention隐藏维度 | 256 | 256 |
| `--mm_n_heads` | 注意力头数 | 8 | 8 |
| `--mm_n_points` | 每个头每个尺度的采样点数 | 4 | 4 |
| `--projector_outdim` | 输出维度（应与LLM hidden_size一致） | 2048 | 2048（Phi-1.5）或2560（Phi-3） |
| `--swin_type` | Swin Transformer类型 | `'base'` | `'base'` 或 `'large'` |

### 参数选择建议

1. **`mm_hidden_dim`**: 
   - 较小值（128-256）：减少参数量，适合显存受限
   - 较大值（512-1024）：增加模型容量，可能提升性能

2. **`mm_n_heads`**: 
   - 通常设置为8，与LLM的注意力头数对齐
   - 可以尝试4或16，但需要确保能被hidden_dim整除

3. **`mm_n_points`**: 
   - 默认4，每个尺度采样4个点
   - 增加点数（如8）可能提升精度，但会增加计算量

4. **`projector_outdim`**: 
   - 必须与LLM的`hidden_size`一致
   - Phi-1.5: 2048
   - Phi-3: 2560

5. **`swin_type`**: 
   - `'base'`: vision_dims=[128,256,512,1024]，参数量较少
   - `'large'`: vision_dims=[192,384,768,1536]，参数量更多，性能可能更好

## 📊 训练监控

### 检查Deformable模块是否启用

训练开始时会打印projector信息：
```
✓ mm_projector 类型: MultiScaleDeformableCrossAttentionAlignment
  - Deformable参数: n_levels=4, n_heads=8, n_points=4
```

如果看到上述信息，说明Deformable模块已成功启用。

### 训练日志

关注以下指标：
- **Loss**: 总体损失应逐渐下降
- **loss_mask**: 掩码损失
- **loss_dice**: Dice损失
- **loss_SEG_class**: 分割类别损失
- **loss_class_name_class**: 类别名称损失
- **loss_region_class**: 区域类别损失（如果使用）

## 🔧 常见问题

### 1. 显存不足（OOM）

**解决方案**：
- 减小 `--per_device_train_batch_size`
- 增加 `--gradient_accumulation_steps`
- 启用 `--gradient_checkpointing True`
- 减小 `--mm_hidden_dim` 或 `--mm_n_heads`

### 2. 训练速度慢

**解决方案**：
- 增加 `--dataloader_num_workers`
- 使用 `--fp16 True` 或 `--bf16 True`
- 检查数据加载是否成为瓶颈
- 考虑使用更少的GPU但更大的batch_size

### 3. Loss不下降

**解决方案**：
- 检查学习率是否合适（尝试1e-6到1e-4）
- 确保数据路径正确
- 检查预训练权重是否正确加载
- 尝试从baseline checkpoint继续训练

### 4. Deformable模块未启用

**检查**：
- 确认 `--mm_projector_type 'deformable'` 已设置
- 查看训练日志中的projector类型信息
- 检查 `builder.py` 是否正确导入deformable模块

## 📝 训练建议

### 首次训练

1. **从小规模开始**：
   - 使用较小的batch_size（1-2）
   - 使用较小的learning_rate（1e-6）
   - 训练少量epochs验证流程

2. **逐步扩展**：
   - 确认训练正常后，增加batch_size
   - 调整learning_rate到推荐值（6e-5）
   - 增加训练轮数

### 从Baseline继续训练

如果已有baseline checkpoint：
1. 使用 `tools/adapt_pretrain_to_deformable.py` 转换权重
2. 设置 `--pretrain_mm_mlp_adapter` 指向转换后的权重
3. 使用较小的learning_rate（1e-6）进行微调

### 混合任务训练

建议同时训练多个任务以提高泛化能力：
```bash
--seg_task 'panoptic' \
--cross_image_seg_task True \
# 同时提供panoptic、region、referring等数据
```

## 🎯 任务特定配置

### Panoptic分割
```bash
--seg_task 'panoptic'
# 需要提供panoptic_json_path和image_folder
```

### Semantic分割
```bash
--seg_task 'semantic'
# 需要提供panoptic_json_path和image_folder
```

### Instance分割
```bash
--seg_task 'instance'
# 需要提供panoptic_json_path和image_folder
```

### Referring分割
```bash
--seg_task 'referring'
# 需要提供ref_coco_path、ref_coco_plus_path、ref_coco_g_path
```

### Region分割
```bash
--seg_task 'region'
--region_mask_type 'box_visual_prompt_mask||scribble_visual_prompt_mask||point_visual_prompt_mask'
# 需要提供region_json_path和region_image_folder
```

### Cross-image分割
```bash
--seg_task 'panoptic'  # 或其他任务
--cross_image_seg_task True
# 需要提供region_cross_json_path和region_cross_image_folder
```

## 📚 相关文档

- Deformable模块集成说明：`DEFORMABLE_INTEGRATION_SUMMARY.md`
- 模型架构：`psalm/model/language_model/llava_phi.py`
- Deformable实现：`psalm/model/multimodal_projector/deformable_alignment.py`

## 💡 提示

1. **保存checkpoint**：定期保存checkpoint，避免训练中断导致损失
2. **监控显存**：使用 `nvidia-smi` 监控GPU显存使用情况
3. **日志分析**：保存训练日志，便于后续分析
4. **实验记录**：记录每次实验的超参数和结果，便于对比

---

**最后更新**：2025年

