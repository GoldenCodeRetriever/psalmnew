# Deformable模块集成说明文档

本文档详细说明了PSALM_new项目中deformable attention模块的集成方式，包括各任务的特征传递方式和区域特征提取方法。

---

## 一、各任务的特征传递方式总结（Deformable模式）

### 1. Panoptic Segmentation（全景分割）

**传入LLM的Token序列：**
```
[文本] <image> <image_deform> [文本] <cls> <cls> ... <cls> [文本] <seg> [文本]
```

**传入的特征：**
- `<image>`: baseline特征（从res5通过baseline_projector得到）
- `<image_deform>`: deformable特征（使用类别名称嵌入作为zq，对多尺度特征做deformable attention）
- `<cls>`: 类别名称嵌入（每个类别一个token）
- `<seg>`: 分割查询向量（可学习的seg_query）

**zq来源：** 类别名称嵌入（`class_name_embedding`）

**提示性话语：**
```
"This is an image <image>, Please do Panoptic Segmentation.
This is all the candidate categories: <cls>, <cls>, ..., <cls>.
Note: After the image token (<image>), there will be a deformable feature token 
(<image_deform>) that contains early-fused features combining category information 
with multi-scale visual features through deformable attention. You can use this 
deformable feature as auxiliary information to better understand the relationship 
between categories and visual content for more accurate panoptic segmentation."
```

---

### 2. Semantic Segmentation（语义分割）

**传入LLM的Token序列：**
```
[文本] <image> <image_deform> [文本] <cls> <cls> ... <cls> [文本] <seg> [文本]
```

**传入的特征：**
- `<image>`: baseline特征
- `<image_deform>`: deformable特征（使用类别名称嵌入作为zq）
- `<cls>`: 类别名称嵌入
- `<seg>`: 分割查询向量

**zq来源：** 类别名称嵌入（`class_name_embedding`）

**提示性话语：**
```
"This is an image <image>, Please do Semantic Segmentation.
This is all the candidate categories: <cls>, <cls>, ..., <cls>.
Note: After the image token (<image>), there will be a deformable feature token 
(<image_deform>) that contains early-fused features combining category information 
with multi-scale visual features through deformable attention. You can use this 
deformable feature as auxiliary information to better understand the relationship 
between categories and visual content for more accurate semantic segmentation."
```

---

### 3. Region Segmentation（区域分割）

**传入LLM的Token序列：**
```
[文本] <image> <image_deform> [文本] <region> <region> ... <region> [文本] <seg> [文本]
```

**传入的特征：**
- `<image>`: baseline特征
- `<image_deform>`: deformable特征（使用区域提示特征作为zq，对多尺度特征做deformable attention）
- `<region>`: 区域特征（从baseline特征中通过region_sampler提取）
- `<seg>`: 分割查询向量

**zq来源：** 区域提示特征（`region_feature_list`，从图像baseline特征中提取）

**提示性话语：**
```
"This is an image <image>, Please segment by given regions.
This is all regions: <region>, <region>, ..., <region>.
Note: After the image token (<image>), there will be a deformable feature token 
(<image_deform>) that contains early-fused features combining the given region 
prompts with multi-scale visual features through deformable attention. You can use 
this deformable feature as auxiliary information to better understand the 
relationship between the region prompts and visual content for more accurate 
region-based segmentation."
```

---

### 4. Referring Segmentation（指代分割）

**传入LLM的Token序列：**
```
[文本] <image> <image_deform> [文本] <refer> [文本] <seg> [文本]
```

**传入的特征：**
- `<image>`: baseline特征
- `<image_deform>`: deformable特征（使用指代词嵌入作为zq，对多尺度特征做deformable attention）
- `<refer>`: 指代词嵌入（从文本指令中提取）
- `<seg>`: 分割查询向量

**zq来源：** 指代词嵌入（从`hidden_states`中提取`<refer>` token对应的隐藏状态，然后池化）

**提示性话语：**
```
"This is an image <image>, Please doing Referring Segmentation according to the 
following instruction:
<refer>
Note: After the image token (<image>), there will be a deformable feature token 
(<image_deform>) that contains early-fused features combining the referring 
expression with multi-scale visual features through deformable attention. You can 
use this deformable feature as auxiliary information to better understand the 
relationship between the referring expression and visual content for more accurate 
referring segmentation."
```

---

### 5. Instance Segmentation（实例分割）

**传入LLM的Token序列：**
```
[文本] <image> <image_deform> [文本] <cls> <cls> ... <cls> [文本] <seg> [文本]
```

**传入的特征：**
- `<image>`: baseline特征
- `<image_deform>`: deformable特征（使用类别名称嵌入作为zq）
- `<cls>`: 类别名称嵌入
- `<seg>`: 分割查询向量

**zq来源：** 类别名称嵌入（`class_name_embedding`）

**提示性话语：**
```
"This is an image <image>, Please do Panoptic Segmentation.
This is all the candidate categories: <cls>, <cls>, ..., <cls>.
Note: After the image token (<image>), there will be a deformable feature token 
(<image_deform>) that contains early-fused features combining category information 
with multi-scale visual features through deformable attention. You can use this 
deformable feature as auxiliary information to better understand the relationship 
between categories and visual content for more accurate panoptic segmentation."
```

---

### 6. Cross-image Segmentation（跨图提示分割）

**传入LLM的Token序列：**
```
[文本] <image> [文本] <region> <region> ... <region> [文本] <image1> <image1_deform> 
[文本] <seg> [文本]
```

**传入的特征：**
- `<image>`: baseline特征（提示图，只有baseline，无deformable）
- `<region>`: 区域特征（从提示图的baseline特征中提取）
- `<image1>`: baseline特征（目标图）
- `<image1_deform>`: deformable特征（使用提示图的区域特征作为zq，对目标图的多尺度特征做deformable attention）
- `<seg>`: 分割查询向量

**zq来源：** 提示图的区域特征（`prompt_region_features`，从提示图的baseline特征中提取）

**提示性话语：**
```
"Based on the regions indicated in Image <image>, Please doing Semantic Segmentation 
on this Image <image1>.
This is all regions: <region>, <region>, ..., <region>.
Note: After the target image token (<image1>), there will be a deformable feature 
token (<image1_deform>) that contains early-fused features. These features are 
computed by using the region features extracted from Image <image> (the prompt image) 
as queries to sample from multi-scale visual features of Image <image1> (the target 
image) through deformable attention. This allows the model to find regions in Image 
<image1> that match the prompts from Image <image>. You can use this deformable 
feature as auxiliary information to better understand the cross-image relationship 
for more accurate segmentation."
```

---

## 二、任务对比总结表

| 任务类型 | zq来源 | 传入的特征 | 特殊Token |
|---------|--------|-----------|----------|
| **Panoptic** | 类别名称嵌入 | `<image>`(baseline) + `<image_deform>`(deformable) + `<cls>` + `<seg>` | `<image_deform>` |
| **Semantic** | 类别名称嵌入 | `<image>`(baseline) + `<image_deform>`(deformable) + `<cls>` + `<seg>` | `<image_deform>` |
| **Region** | 区域提示特征 | `<image>`(baseline) + `<image_deform>`(deformable) + `<region>` + `<seg>` | `<image_deform>` |
| **Referring** | 指代词嵌入 | `<image>`(baseline) + `<image_deform>`(deformable) + `<refer>` + `<seg>` | `<image_deform>` |
| **Instance** | 类别名称嵌入 | `<image>`(baseline) + `<image_deform>`(deformable) + `<cls>` + `<seg>` | `<image_deform>` |
| **Cross-image** | 提示图区域特征 | `<image>`(baseline) + `<region>` + `<image1>`(baseline) + `<image1_deform>`(deformable) + `<seg>` | `<image1_deform>` |

**关键点：**
1. 所有任务都保留baseline特征，确保性能不低于baseline
2. Deformable特征作为辅助信息，通过特殊token传入
3. zq来源与任务相关：类别名称、区域特征、指代词或跨图区域特征
4. 跨图任务特殊：提示图只有baseline，目标图有baseline和deformable

---

## 三、跨图任务 vs 区域分割任务：区域特征提取对比

### 1. 提取方法是否相同？

**✅ 完全相同。** 两种任务都使用相同的 `region_sampler`：

**跨图任务**（`llava_phi.py` 第1212行）：
```python
prompt_region_features = self.region_sampler(
    baseline_features_prompt,  # 提示图的baseline特征
    region_masks_list,          # 区域掩码列表
    ...
)
```

**区域分割任务**（`llava_phi.py` 第1388行）：
```python
region_features = self.region_sampler(
    image_features,             # 当前图像的baseline特征
    region_masks_list,          # 区域掩码列表
    ...
)
```

两者都使用 `region_pooling(num_sample_point=256)` 的实例。

### 2. region_sampler 的工作原理

从 `context_cluster.py` 的实现可以看到：

```python
class region_pooling(nn.Module):
    def __init__(self, num_sample_point):
        self.num_sample_point = num_sample_point  # 256
        self.pooler = nn.AdaptiveAvgPool1d(output_size=1)
    
    def extract_region_feature(self, region_feature_map, region_masks, ...):
        # 对每个mask：
        # 1. 使用 m.nonzero() 获取mask内的所有非零位置
        # 2. 从这些位置随机采样 num_sample_point=256 个点
        # 3. 使用 point_sample 从特征图中提取这些点的特征
        # 4. 通过 AdaptiveAvgPool1d 池化得到单个区域特征向量
```

**工作流程：**
1. 输入：图像特征（list格式）和区域掩码列表
2. 对每个区域掩码，提取掩码内的点位置
3. 从这些位置随机采样 256 个点
4. 使用 `point_sample` 从特征图中提取这些点的特征
5. 通过 `AdaptiveAvgPool1d` 池化得到区域特征向量

### 3. 是否都支持点、框、涂鸦三种 prompt？

**✅ 都支持。** 从 `coco_instance_mapper.py` 可以看到：

```python
# 第260-278行
if 'point_visual_prompt_mask' in annos[0]:
    if region_mask_type is None:
        region_mask_type = [
            'point_visual_prompt_mask',      # 点提示
            'mask_visual_prompt_mask',       # 完整mask（涂鸦的一种形式）
            'box_visual_prompt_mask',        # 框提示
            'scribble_visual_prompt_mask'   # 涂鸦提示
        ]
    
    for anno in annos:
        # 随机选择一个可用的mask类型
        used_mask_type = random.choice(non_empty_masks)
        region_mask = decode(anno[used_mask_type])
        
        # 对于点和涂鸦，使用圆形增强
        if used_mask_type in ['point_visual_prompt_mask', 'scribble_visual_prompt_mask']:
            radius = 10 if used_mask_type == 'point_visual_prompt_mask' else 5
            region_mask = enhance_with_circles(region_mask, radius)
        
        # 应用变换并添加到region_masks
        scale_region_mask = transforms.apply_segmentation(region_mask)
        region_masks.append(scale_region_mask)
```

两种任务的数据预处理都调用：
```python
# 区域分割任务（train_datasets.py 第342行）
data_dict = processor.preprocess(data_dict, region_mask_type=region_mask_type)

# 跨图任务（train_datasets_cross.py 第713行）
data_dict = processor.preprocess(data_dict, region_mask_type=region_mask_type)
```

### 4. 三种 prompt 类型的处理方式

| Prompt类型 | 处理方式 | 特殊处理 |
|-----------|---------|---------|
| **点 (point)** | 解码为点位置mask | 使用半径10的圆形增强 |
| **框 (box)** | 解码为矩形mask | 无特殊处理 |
| **涂鸦 (scribble)** | 解码为涂鸦路径mask | 使用半径5的圆形增强 |
| **完整mask** | 直接使用mask | 无特殊处理 |

### 5. 对比总结表

| 项目 | 跨图任务 | 区域分割任务 | 是否相同 |
|------|---------|------------|---------|
| **提取方法** | `region_sampler` | `region_sampler` | ✅ 相同 |
| **输入特征** | `baseline_features_prompt` | `image_features` | ✅ 都是baseline特征 |
| **输入mask** | `instance.region_masks.tensor` | `instance.region_masks.tensor` | ✅ 相同格式 |
| **支持点提示** | ✅ | ✅ | ✅ 都支持 |
| **支持框提示** | ✅ | ✅ | ✅ 都支持 |
| **支持涂鸦提示** | ✅ | ✅ | ✅ 都支持 |
| **采样点数** | 256 | 256 | ✅ 相同 |

### 6. 结论

1. **两种任务使用完全相同的 `region_sampler` 方法**
2. **都支持点、框、涂鸦三种 prompt 类型**
3. **处理流程完全一致：mask → 采样点 → 提取特征 → 池化**
4. **唯一区别：跨图任务从提示图的baseline特征中提取，区域分割任务从当前图像的baseline特征中提取**

---

## 四、技术实现细节

### 1. Deformable模块集成策略

采用"**增加式集成**"策略，而非替换：
- **保留baseline特征**：所有任务都保留原始的baseline特征（通过baseline_projector处理res5特征）
- **增加deformable特征**：通过新的特殊token（`<image_deform>`、`<image1_deform>`）传入deformable特征
- **保证性能下限**：即使deformable模块效果不佳，整体性能也不会低于baseline

### 2. 跨图任务的特殊处理

跨图任务采用"**单次deformable attention**"策略：
- **提示图（image）**：只使用baseline特征，不进行deformable attention
- **目标图（image1）**：使用提示图的区域特征作为zq，对目标图的多尺度特征做deformable attention
- **优势**：
  - 语义清晰：用提示图的区域特征指导目标图的分割
  - 计算高效：只做一次deformable attention
  - 对齐明确：通过deformable attention实现跨图对齐

### 3. 查询向量（zq）生成策略

不同任务使用不同的zq来源：
- **Panoptic/Semantic/Instance**：类别名称嵌入（`class_name_embedding`）
- **Region**：区域提示特征（从baseline特征中通过region_sampler提取）
- **Referring**：指代词嵌入（从hidden_states中提取并池化）
- **Cross-image**：提示图的区域特征（从提示图的baseline特征中提取）

### 4. Token插入策略

- **跨图任务**：只在`<image1>`后插入`<image1_deform>`，`<image>`后不插入
- **其他任务**：在`<image>`后插入`<image_deform>`
- **动态插入**：在`prepare_inputs_labels_for_multimodal`函数中动态插入，而非在数据预处理阶段

---

## 五、文件修改清单

### 核心文件修改

1. **`psalm/model/language_model/llava_phi.py`**
   - 添加`create_deformable_queries`函数
   - 修改`encode_images`函数支持deformable模式
   - 修改`prepare_inputs_labels_for_multimodal`函数实现跨图任务特殊处理
   - 修改`concat_image_seg_cls_embeds`函数支持deformable特征token

2. **`psalm/model/language_model/llava_phi-cross.py`**
   - 同步`llava_phi.py`的所有修改

3. **`psalm/model/multimodal_projector/builder.py`**
   - 添加'deformable'类型的projector构建支持

4. **`psalm/constants.py`**
   - 添加`IMAGE_DEFORM_TOKEN_INDEX`和`IMAGE1_DEFORM_TOKEN_INDEX`常量

5. **`psalm/train/train_datasets.py`**
   - 为Panoptic、Semantic、Region、Referring、Instance任务添加deformable说明文本

6. **`psalm/train/train_datasets_cross.py`**
   - 为Cross-image任务添加deformable说明文本

### 新增文件

1. **`psalm/model/multimodal_projector/deformable_alignment.py`**
   - 从PSALM项目复制，实现MultiScaleDeformableCrossAttentionAlignment模块

---

## 六、使用说明

### 启用Deformable模式

在训练配置中设置：
```python
model_args.mm_projector_type = 'deformable'
```

### 配置参数

```python
# Deformable模块相关配置
config.mm_hidden_dim = 256          # 隐藏维度
config.mm_n_heads = 8               # 注意力头数
config.mm_n_points = 4               # 每个查询的采样点数
config.mm_vision_dims = [128, 256, 512, 1024]  # 多尺度特征维度
config.mm_n_levels = 4              # 特征层级数
config.projector_outdim = query_dim # 输出维度（通常等于hidden_size）
```

### 区域提示类型配置

在数据配置中设置：
```python
data_args.region_mask_type = 'point_visual_prompt_mask||box_visual_prompt_mask||scribble_visual_prompt_mask'
```

支持的类型：
- `point_visual_prompt_mask`：点提示
- `box_visual_prompt_mask`：框提示
- `scribble_visual_prompt_mask`：涂鸦提示
- `mask_visual_prompt_mask`：完整mask

---

## 七、注意事项

1. **Baseline Projector**：在deformable模式下，会自动创建一个baseline_projector（类型为'conv'）来处理res5特征，确保baseline特征正常传递。

2. **跨图任务的区域特征提取时机**：跨图任务的区域特征在`prepare_inputs_labels_for_multimodal`函数中提前提取，而不是在后续的区域特征提取阶段。

3. **Token序列长度**：启用deformable模式后，输入序列长度会增加（每个图像token后增加一个deformable token），需要注意序列长度限制。

4. **区域特征提取的一致性**：跨图任务和区域分割任务使用完全相同的region_sampler，确保特征提取的一致性。

---

## 八、参考资料

- Deformable Attention模块实现：`psalm/model/multimodal_projector/deformable_alignment.py`
- 区域特征提取实现：`psalm/model/visual_prompt_module/context_cluster.py`
- 数据预处理实现：`psalm/model/datasets_mapper/coco_instance_mapper.py`

---

**文档版本：** v1.0  
**最后更新：** 2025年

