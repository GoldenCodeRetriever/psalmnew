# ------------------------------------------------------------------------------------------------
# Multi-Scale Deformable Cross-Attention Alignment Module for PSALM
# 基于 Deformable DETR (arXiv:2010.04159) 实现的视觉-语言对齐模块
# 使用文本/提示特征作为查询，从多尺度视觉特征中稀疏采样
# ------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import math

# 导入项目中已有的 CUDA 优化的 MSDeformAttn
from psalm.model.mask_decoder.Mask2Former_Simplify.modeling.pixel_decoder.ops.modules.ms_deform_attn import MSDeformAttn


class MultiScaleDeformableCrossAttentionAlignment(nn.Module):
    """
    Multi-Scale Deformable Cross-Attention 对齐模块
    
    该模块取代了原始 PSALM 中将完整视觉特征图展平并注入 LLM 的方式。
    
    工作流程：
    1. 接收查询 (queries, z_q)：
       - Referring 任务：文本描述的嵌入
       - Panoptic/Semantic 任务：类别名称的嵌入
       - Interactive 任务：通过 mask pooling 得到的视觉特征
       - Cross-image 任务：跨图提示分割的查询特征
    
    2. 接收键/值 (keys/values)：来自 Swin Transformer 的多尺度视觉特征图
    
    3. 查询通过 Deformable Attention 从多尺度特征图中稀疏采样：
       - 查询预测采样偏移量和注意力权重
       - 根据偏移量从多尺度特征中采样
       - 加权聚合采样到的特征
    
    4. 输出：与查询长度相同的视觉增强特征序列 V'
    """
    
    def __init__(
        self, 
        query_dim=2560,           # LLM 的隐藏维度（例如 Phi 的 2560）
        vision_dims=[128, 256, 512, 1024],  # Swin Transformer 各层级的通道数
        hidden_dim=256,           # Deformable Attention 的隐藏维度
        n_levels=4,               # 多尺度特征的级数（res2, res3, res4, res5）
        n_heads=8,                # 注意力头数
        n_points=4,               # 每个头每个尺度的采样点数
        projector_outdim=2560     # 输出维度（与 LLM hidden size 一致）
    ):
        super().__init__()
        
        self.query_dim = query_dim
        self.vision_dims = vision_dims if isinstance(vision_dims, list) else [vision_dims] * n_levels
        self.hidden_dim = hidden_dim
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.projector_outdim = projector_outdim
        
        if hidden_dim % n_heads != 0:
            raise ValueError(
                f'd_model must be divisible by n_heads, but got {hidden_dim} and {n_heads}'
            )
        
        # 1. 查询投影
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        
        # 2. 值投影：为每个尺度单独创建投影层（因为 Swin 各层级通道数不同）
        self.value_projs = nn.ModuleList([
            nn.Linear(vision_dim, hidden_dim) for vision_dim in self.vision_dims
        ])
        
        # 3. 使用项目中已有的 CUDA 优化的 MSDeformAttn 模块
        self.deform_attn = MSDeformAttn(
            d_model=hidden_dim,
            n_levels=n_levels,
            n_heads=n_heads,
            n_points=n_points
        )
        
        # 4. 最终投影层
        self.output_proj = nn.Linear(hidden_dim, projector_outdim)
        
        # 5. 层级位置编码
        self.level_embed = nn.Parameter(torch.Tensor(n_levels, hidden_dim))
        
        self._reset_parameters()

    def _reset_parameters(self):
        """初始化参数"""
        # 查询投影
        xavier_uniform_(self.query_proj.weight.data)
        constant_(self.query_proj.bias.data, 0.)
        
        # 值投影
        for proj in self.value_projs:
            xavier_uniform_(proj.weight.data)
            constant_(proj.bias.data, 0.)
        
        # 输出投影
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)
        
        # 层级嵌入
        nn.init.normal_(self.level_embed)

    def get_reference_points(self, spatial_shapes, device):
        """
        生成参考点
        
        Args:
            spatial_shapes: list of (H, W)，每个级别的空间形状
            device: 设备
        
        Returns:
            reference_points: (1, n_levels, 2)，范围在 [0, 1]
        """
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # 在特征图中心位置生成参考点 (x=0.5, y=0.5)
            # 注意：reference_points 格式是 (x, y)，范围 [0, 1]
            ref_x = 0.5  # 归一化的 x 坐标
            ref_y = 0.5  # 归一化的 y 坐标
            ref = torch.tensor([ref_x, ref_y], dtype=torch.float32, device=device)  # (2,)
            reference_points_list.append(ref)
        
        # 堆叠所有级别：list of (2,) -> (n_levels, 2) -> (1, n_levels, 2)
        reference_points = torch.stack(reference_points_list, dim=0).unsqueeze(0)
        return reference_points


    def get_proposal_based_reference_points(self, query_embed, feature_flatten, spatial_shape):
        """
        [新增] 基于相关性图生成动态参考点 (Proposal-based Reference Points)
        适用于跨图分割等空间不对应的任务。
        
        Args:
            query_embed: (B, N_q, C) 投影后的查询特征
            feature_flatten: (B, L, C) 投影后的某一尺度特征图 (通常取最高层级 res5)
            spatial_shape: (H, W) 该层级特征图的空间形状
            
        Returns:
            reference_points: (B, N_q, 1, 2) 归一化坐标 [0, 1]
        """
        B, N_q, C = query_embed.shape
        H, W = spatial_shape
        
        # 1. 计算相似度矩阵 (Cosine Similarity)
        # 对 Query 和 Feature 进行归一化，使计算结果为余弦相似度，训练更稳定
        q_norm = F.normalize(query_embed, p=2, dim=-1)      # (B, N_q, C)
        f_norm = F.normalize(feature_flatten, p=2, dim=-1)  # (B, HW, C)
        
        # 矩阵乘法: (B, N_q, C) @ (B, C, HW) -> (B, N_q, HW)
        similarity = torch.bmm(q_norm, f_norm.transpose(1, 2))
        
        # 2. 找到响应最大的位置 (Argmax)
        # 这一步不需要梯度，我们只需要坐标
        max_indices = torch.argmax(similarity, dim=-1) # (B, N_q)
        
        # 3. 将一维索引转换为归一化坐标 (x, y)
        y_indices = max_indices // W
        x_indices = max_indices % W
        
        # 加 0.5 取像素中心，并归一化到 [0, 1]
        ref_x = (x_indices.float() + 0.5) / W
        ref_y = (y_indices.float() + 0.5) / H
        
        # 堆叠坐标: (B, N_q, 2)
        ref_points = torch.stack([ref_x, ref_y], dim=-1)
        
        # 扩展维度以匹配后续 repeat 操作: (B, N_q, 1, 2)
        return ref_points.unsqueeze(2)


    def forward(
        self, 
        queries,                    # (B, N_q, query_dim) 查询特征
        multi_scale_features,       # Dict[str, Tensor] 多尺度特征 {'res2': ..., 'res3': ..., 'res4': ..., 'res5': ...}
        task_type=None              # 任务类型（可选，用于调试）
    ):
        """
        前向传播
        
        Args:
            queries: (B, N_q, query_dim) 查询特征
            multi_scale_features: Dict[str, Tensor] 多尺度特征图
                - 'res2': (B, C2, H2, W2)
                - 'res3': (B, C3, H3, W3)
                - 'res4': (B, C4, H4, W4)
                - 'res5': (B, C5, H5, W5)
            task_type: str, 可选的任务类型标识（'panoptic', 'referring', 'region', 'cross_image'）
        
        Returns:
            output: (B, N_q, projector_outdim) 对齐后的特征
        """
        B, N_q, _ = queries.shape
        device = queries.device
        
        # 1. 投影查询到隐藏空间
        query = self.query_proj(queries)  # (B, N_q, hidden_dim)
        
        # 2. 准备多尺度视觉特征
        value_list = []
        spatial_shapes = []
        expected_batch_size = B
        
        for lvl, level_name in enumerate(['res2', 'res3', 'res4', 'res5']):
            if level_name not in multi_scale_features:
                continue
                
            feat = multi_scale_features[level_name]  # (B, C, H, W)
            B_f, C_f, H_f, W_f = feat.shape
            
            # 检查并修复 batch 维度不一致的问题
            if B_f != expected_batch_size:
                import warnings
                warnings.warn(
                    f"Batch size mismatch in {level_name}: expected {expected_batch_size}, got {B_f}. "
                    f"Adjusting to match expected batch size.",
                    UserWarning
                )
                if B_f > expected_batch_size:
                    feat = feat[:expected_batch_size]
                elif B_f < expected_batch_size:
                    pad_size = expected_batch_size - B_f
                    feat = torch.cat([feat, feat[-1:].repeat(pad_size, 1, 1, 1)], dim=0)
                B_f = expected_batch_size
            
            # 展平空间维度: (B, C, H, W) -> (B, H*W, C)
            feat_flat = feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
            
            # 使用对应层级的投影层投影到 hidden_dim: (B, H*W, C) -> (B, H*W, hidden_dim)
            feat_projected = self.value_projs[lvl](feat_flat)
            
            value_list.append(feat_projected)
            spatial_shapes.append((H_f, W_f))
        
        if len(value_list) == 0:
            raise ValueError("No valid multi-scale features found!")
        
        # 3. 拼接所有尺度的特征
        value = torch.cat(value_list, dim=1)  # (B, sum(H*W), hidden_dim)
        
        # 4. 添加层级位置编码
        level_start_index = [0]
        for H_f, W_f in spatial_shapes[:-1]:
            level_start_index.append(level_start_index[-1] + H_f * W_f)
        
        # 为每个位置添加其对应层级的嵌入
        value_with_pos = value.clone()
        for lvl, (start_idx, (H_f, W_f)) in enumerate(
            zip(level_start_index, spatial_shapes)
        ):
            end_idx = start_idx + H_f * W_f
            value_with_pos[:, start_idx:end_idx, :] += self.level_embed[lvl]
        
        # 5. 准备 MSDeformAttn 需要的输入格式
        spatial_shapes_tensor = torch.tensor(
            spatial_shapes, dtype=torch.long, device=device
        )  # (n_levels, 2)
        
        level_start_index_tensor = torch.tensor(
            level_start_index, dtype=torch.long, device=device
        )  # (n_levels,)
        
        if task_type == 'cross_image':
            # --- [DEBUG] 打印标志，确认进入了跨图分支 ---
            # 为了防止刷屏，只在 rank 0 或者偶尔打印，这里简单起见每次都打
            # 建议在确认生效后注释掉
            if torch.rand(1).item() < 0.05: # 5% 的概率打印，防止日志爆炸
                print(f"\n[DEBUG] >>> Entering Cross-Image Alignment Branch! Task: {task_type}")

            last_lvl_feat = value_list[-1]
            last_lvl_shape = spatial_shapes[-1]
            
            # 计算最佳匹配位置
            ref_points_prop = self.get_proposal_based_reference_points(
                query, last_lvl_feat, last_lvl_shape
            )
            
            # --- [DEBUG] 检查生成的坐标是否合理 ---
            if torch.rand(1).item() < 0.05:
                # 打印第一个 Batch 的第一个 Query 的坐标
                sample_pt = ref_points_prop[0, 0, 0].detach().cpu().numpy()
                print(f"[DEBUG] Spatial Shape (res5): {last_lvl_shape}")
                print(f"[DEBUG] Sample Ref Point (Normalized): x={sample_pt[0]:.4f}, y={sample_pt[1]:.4f}")
                # 检查是否都在 [0, 1] 之间
                print(f"[DEBUG] Ref Point Range: min={ref_points_prop.min().item():.4f}, max={ref_points_prop.max().item():.4f}")
                
                # 如果全是 0.5，说明可能是初始状态或者哪里算错了
                if (ref_points_prop == 0.5).all():
                    print("[WARNING] !!! All Reference Points are 0.5 (Center). Check Similarity Calculation !!!")
                else:
                    print("[DEBUG] >>> Dynamic Reference Points generated successfully.")

            # (B, N_q, 1, 2) -> (B, N_q, n_levels, 2)
            reference_points = ref_points_prop.repeat(1, 1, self.n_levels, 1)
            
        else:
            # 策略 A: 默认中心点 (Default Center)
            # 适用于 Referring/Interactive 等任务，或者作为兜底
            print("WRONG!!!")
            # 生成参考点：(1, n_levels, 2) -> (1, 1, n_levels, 2)
            reference_points = self.get_reference_points(spatial_shapes, device).unsqueeze(1)
            # 扩展到 batch 和 query 维度：(B, N_q, n_levels, 2)
            reference_points = reference_points.expand(B, N_q, -1, -1)
        
        # 6. 执行 CUDA 优化的 Deformable Attention
        # MSDeformAttn.forward 参数:
        #   - query: (B, N_q, hidden_dim)
        #   - reference_points: (B, N_q, n_levels, 2)
        #   - input_flatten: (B, sum(H*W), hidden_dim)
        #   - input_spatial_shapes: (n_levels, 2)
        #   - input_level_start_index: (n_levels,)
        #   - input_padding_mask: (B, sum(H*W)), 可选
        output = self.deform_attn(
            query=query,
            reference_points=reference_points,
            input_flatten=value_with_pos,
            input_spatial_shapes=spatial_shapes_tensor,
            input_level_start_index=level_start_index_tensor,
            input_padding_mask=None  # 我们不使用 padding mask
        )  # (B, N_q, hidden_dim)
        
        # 7. 最终投影
        output = self.output_proj(output)  # (B, N_q, projector_outdim)
        
        return output


# ====================================================================================================
# 测试函数
# ====================================================================================================

def test_deformable_alignment():
    """测试 MultiScaleDeformableCrossAttentionAlignment 模块"""
    print("=" * 80)
    print("测试 MultiScaleDeformableCrossAttentionAlignment (CUDA 版本)")
    print("=" * 80)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 参数设置
    batch_size = 2
    query_dim = 2560  # Phi 的隐藏维度
    vision_dims = [128, 256, 512, 1024]  # Swin Transformer 各层级的通道数
    hidden_dim = 256
    n_levels = 4
    n_heads = 8
    n_points = 4
    
    # 创建模块
    print("\n创建 Deformable Alignment 模块...")
    alignment = MultiScaleDeformableCrossAttentionAlignment(
        query_dim=query_dim,
        vision_dims=vision_dims,
        hidden_dim=hidden_dim,
        n_levels=n_levels,
        n_heads=n_heads,
        n_points=n_points,
        projector_outdim=query_dim
    ).to(device)
    
    print(f"模块参数量: {sum(p.numel() for p in alignment.parameters()) / 1e6:.2f}M")
    
    # 测试场景 1: Referring Segmentation
    print("\n" + "-" * 80)
    print("场景 1: Referring Segmentation")
    print("-" * 80)
    
    n_queries_ref = 10  # 文本 token 数量
    queries_ref = torch.randn(batch_size, n_queries_ref, query_dim).to(device)
    
    # 模拟多尺度特征（来自 Swin Transformer）
    multi_scale_features = {
        'res2': torch.randn(batch_size, 128, 64, 64).to(device),   # 1/4 分辨率
        'res3': torch.randn(batch_size, 256, 32, 32).to(device),   # 1/8 分辨率
        'res4': torch.randn(batch_size, 512, 16, 16).to(device),   # 1/16 分辨率
        'res5': torch.randn(batch_size, 1024, 8, 8).to(device),    # 1/32 分辨率
    }
    
    print(f"输入查询形状: {queries_ref.shape}")
    for level_name, feat in multi_scale_features.items():
        print(f"  {level_name}: {feat.shape}")
    
    # 前向传播
    try:
        output_ref = alignment(queries_ref, multi_scale_features, task_type='referring')
        print(f"✓ 输出形状: {output_ref.shape}")
        print(f"✓ 输出统计: mean={output_ref.mean().item():.4f}, std={output_ref.std().item():.4f}")
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试场景 2: Panoptic Segmentation
    print("\n" + "-" * 80)
    print("场景 2: Panoptic Segmentation")
    print("-" * 80)
    
    n_categories = 80  # COCO 类别数
    queries_pan = torch.randn(batch_size, n_categories, query_dim).to(device)
    
    print(f"输入查询形状: {queries_pan.shape}")
    
    try:
        output_pan = alignment(queries_pan, multi_scale_features, task_type='panoptic')
        print(f"✓ 输出形状: {output_pan.shape}")
        print(f"✓ 输出统计: mean={output_pan.mean().item():.4f}, std={output_pan.std().item():.4f}")
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试场景 3: Interactive Segmentation
    print("\n" + "-" * 80)
    print("场景 3: Interactive Segmentation")
    print("-" * 80)
    
    n_prompts = 5  # 用户交互提示数量
    queries_int = torch.randn(batch_size, n_prompts, query_dim).to(device)
    
    print(f"输入查询形状: {queries_int.shape}")
    
    try:
        output_int = alignment(queries_int, multi_scale_features, task_type='interactive')
        print(f"✓ 输出形状: {output_int.shape}")
        print(f"✓ 输出统计: mean={output_int.mean().item():.4f}, std={output_int.std().item():.4f}")
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试场景 4: Cross-image Segmentation
    print("\n" + "-" * 80)
    print("场景 4: Cross-image Segmentation")
    print("-" * 80)
    
    n_cross_queries = 5  # 跨图提示数量
    queries_cross = torch.randn(batch_size, n_cross_queries, query_dim).to(device)
    
    print(f"输入查询形状: {queries_cross.shape}")
    
    try:
        output_cross = alignment(queries_cross, multi_scale_features, task_type='cross_image')
        print(f"✓ 输出形状: {output_cross.shape}")
        print(f"✓ 输出统计: mean={output_cross.mean().item():.4f}, std={output_cross.std().item():.4f}")
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    test_deformable_alignment()

