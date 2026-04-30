#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
推理与融合模块

使用训练好的 Inpainting 模型生成融合后的 COPD 数字孪生 CT
"""

from pathlib import Path
from typing import Union, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    from scipy import ndimage
    from scipy.sparse import diags, csr_matrix
    from scipy.sparse.linalg import spsolve
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from .network import InpaintingUNet, PartialConvUNet, PatchDiscriminator, AttentionUNet, DiffusionUNet, create_model
from ..utils.io import load_nifti, save_nifti
from ..utils.math_ops import normalize_ct, denormalize_ct
from ..utils.logger import get_logger

logger = get_logger(__name__)


def load_model(
    checkpoint_path: Union[str, Path],
    device: str = "cuda",
    model_type: str = "unet"
) -> 'torch.nn.Module':
    """
    加载训练好的模型

    Args:
        checkpoint_path: 检查点路径
        device: 设备
        model_type: 模型类型 ("unet", "partial_conv", "patchgan", "attgan", "mae_patchgan", "ddpm")

    Returns:
        model: 加载的模型
    """
    if torch is None:
        raise ImportError("PyTorch 未安装")

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # 根据模型类型创建对应的网络
    if model_type in ("unet", "patchgan", "mae_patchgan"):
        # patchgan / mae_patchgan 的生成器也是 UNet
        model = InpaintingUNet()
    elif model_type == "partial_conv":
        model = PartialConvUNet()
    elif model_type == "attgan":
        model = AttentionUNet()
    elif model_type == "ddpm":
        model = DiffusionUNet()
    else:
        logger.warning(f"未知模型类型 '{model_type}'，使用默认 UNet")
        model = InpaintingUNet()

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # DDPM 模型优先加载 EMA 权重（采样质量更优）
    if model_type == "ddpm" and 'ema_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['ema_state_dict'])
        logger.info(f"  DDPM: 使用 EMA 权重 (采样质量更优)")
    else:
        model.load_state_dict(checkpoint['generator_state_dict'])
    model.to(device)
    model.eval()

    logger.info(f"模型加载完成: {checkpoint_path}")
    logger.info(f"  模型类型: {model_type}")
    return model


# ============================================================================
# DDPM 专用推理函数
# ============================================================================

def _build_ddpm_schedule(device, num_timesteps=1000, ddim_steps=10,
                          beta_start=1e-4, beta_end=0.02):
    """
    预构建 DDPM 噪声调度参数（只调用一次，避免每个 patch 重复计算）

    Returns:
        alphas_cumprod: (T,) 累积 α 乘积
        timesteps: DDIM 子序列（从大到小）
    """
    betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    step_size = num_timesteps // ddim_steps
    timesteps = list(range(0, num_timesteps, step_size))[::-1]
    return alphas_cumprod, timesteps


def ddpm_inpaint_patch(
    model: 'torch.nn.Module',
    template_patch: 'torch.Tensor',
    mask_patch: 'torch.Tensor',
    device: 'torch.device',
    alphas_cumprod: 'torch.Tensor' = None,
    timesteps: list = None,
    num_timesteps: int = 1000,
    ddim_steps: int = 10,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> 'torch.Tensor':
    """
    DDPM RePaint Inpainting 推理（单个 patch）

    使用 DDIM 加速采样 + RePaint 策略：
    - 从纯高斯噪声 x_T 出发，逐步去噪
    - 每步去噪后，将已知区域（非 mask）替换为对应时间步的加噪真实数据
    - 最终得到 mask 区域的 inpainting 结果

    Args:
        model: DiffusionUNet 模型（已加载权重，eval 模式）
        template_patch: 归一化后的模板 patch (1, 1, D, H, W)
        mask_patch: 病灶 mask patch (1, 1, D, H, W)，mask>0 为待修复区域
        device: 计算设备
        alphas_cumprod: 预计算的累积 α（可选，传入避免重复计算）
        timesteps: 预计算的 DDIM 时间步子序列（可选）
        num_timesteps: 训练时使用的扩散步数 T
        ddim_steps: DDIM 加速采样步数（默认 10，从 50 降低以提速 5 倍）
        beta_start: β 起始值（须与训练一致）
        beta_end: β 终止值（须与训练一致）

    Returns:
        output_patch: 修复后的 patch (1, 1, D, H, W)
    """
    # 如果未传入预计算参数，则构建（向后兼容）
    if alphas_cumprod is None or timesteps is None:
        alphas_cumprod, timesteps = _build_ddpm_schedule(
            device, num_timesteps, ddim_steps, beta_start, beta_end
        )

    # 将模板和 mask 移到设备
    x_known = template_patch.to(device)
    mask_bool = (mask_patch > 0).to(device)  # True = 待修复区域

    # 从纯高斯噪声开始
    x_t = torch.randn_like(x_known)

    model.eval()
    with torch.no_grad():
        for i, t_val in enumerate(timesteps):
            t_tensor = torch.full((x_t.shape[0],), t_val, device=device, dtype=torch.long)

            # 预测噪声
            noise_pred = model(x_t, t_tensor)

            # DDIM 去噪一步
            alpha_t = alphas_cumprod[t_val]
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)

            # 预测 x_0
            x_0_pred = (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
            x_0_pred = torch.clamp(x_0_pred, 0.0, 1.0)  # 限制到有效范围

            # 确定下一个时间步
            if i + 1 < len(timesteps):
                t_next = timesteps[i + 1]
                alpha_next = alphas_cumprod[t_next]
            else:
                t_next = 0
                alpha_next = torch.tensor(1.0, device=device)

            # DDIM 确定性采样（eta=0，无额外噪声）
            sqrt_alpha_next = torch.sqrt(alpha_next)
            sqrt_one_minus_alpha_next = torch.sqrt(1.0 - alpha_next)
            x_t = sqrt_alpha_next * x_0_pred + sqrt_one_minus_alpha_next * noise_pred

            # RePaint 策略：将已知区域替换为对应时间步的加噪真实数据
            if t_next > 0:
                noise_known = torch.randn_like(x_known)
                sqrt_alpha_known = torch.sqrt(alphas_cumprod[t_next])
                sqrt_one_minus_known = torch.sqrt(1.0 - alphas_cumprod[t_next])
                x_known_noisy = sqrt_alpha_known * x_known + sqrt_one_minus_known * noise_known

                # 已知区域用加噪真实数据替换，待修复区域保持去噪结果
                x_t = x_t * mask_bool.float() + x_known_noisy * (1.0 - mask_bool.float())
            else:
                # 最后一步：已知区域直接用真实数据
                x_t = x_t * mask_bool.float() + x_known * (1.0 - mask_bool.float())

    return x_t


def smooth_boundary(
    output: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    boundary_width: int = 3
) -> np.ndarray:
    """
    平滑边界过渡（简化版泊松融合）

    使用高斯加权在边界区域进行平滑过渡

    Args:
        output: 生成的输出
        original: 原始图像
        mask: 病灶 mask
        boundary_width: 边界宽度

    Returns:
        smoothed: 平滑后的输出
    """
    if not HAS_SCIPY:
        logger.warning("scipy 未安装，跳过边界平滑")
        return output

    from scipy import ndimage

    # 创建边界区域 mask
    dilated = ndimage.binary_dilation(mask > 0, iterations=boundary_width)
    eroded = ndimage.binary_erosion(mask > 0, iterations=boundary_width)
    boundary = dilated & ~eroded

    # 计算距离权重
    distance = ndimage.distance_transform_edt(~(mask > 0))
    distance = np.clip(distance, 0, boundary_width) / boundary_width

    # 在边界区域进行加权混合
    result = output.copy()
    result[boundary] = (
        output[boundary] * (1 - distance[boundary]) +
        original[boundary] * distance[boundary]
    )

    return result


def fuse_lesion(
    template_path: Union[str, Path],
    lesion_mask_path: Union[str, Path],
    model: 'InpaintingUNet',
    output_path: Union[str, Path],
    patch_size: Tuple[int, int, int] = (64, 64, 64),
    overlap: int = 16,
    hu_min: float = -1000,
    hu_max: float = 400,
    device: str = "cuda",
    smooth_boundary_width: int = 3,
    model_type: str = "unet",
    patient_condition: Optional[dict] = None,
    model_condition: Optional['torch.Tensor'] = None,
    mask_dilation: int = 0,
    atlas_lung_mask_path: Optional[Union[str, Path]] = None,
    use_adaptive_hu_calibration: bool = False,
    adaptive_mask_dilation: bool = False,
) -> Path:
    """
    将病灶融合到模板中

    流程:
    1. 加载模板和病灶 mask
    2. 在 mask 区域挖空
    3. 使用 Inpainting 模型填充
    4. 边界平滑（可选）
    5. 【阶段①】自适应 HU 校准（基于患者真实病灶统计）
    6. 保存融合结果

    Args:
        template_path: 模板 CT 路径
        lesion_mask_path: 病灶 mask 路径 (已配准到模板空间)
        model: Inpainting 模型
        output_path: 输出路径
        patch_size: 处理的 patch 大小
        overlap: patch 重叠
        hu_min: 归一化最小 HU
        hu_max: 归一化最大 HU
        device: 计算设备
        smooth_boundary_width: 边界平滑宽度（0 表示不平滑）
        model_type: 模型类型，"ddpm" 时使用 RePaint 推理管线
        patient_condition: 患者条件信息（阶段①自适应 HU 校准用）
                          格式: {'lesion_HU_mean': float, 'lesion_HU_std': float}
                          若为 None，则使用固定先验值 (-965.0, 45.0)
        model_condition: 归一化后的 5 维条件向量 tensor（阶段② FiLM 推理用）
                        形状: (1, 5)，已归一化到 [0, 1]
                        当 model 是 ConditionedGenerator 时，每个 patch 推理都会传入此 condition
                        若为 None，则模型以无条件模式运行（Exp-0/Exp-1）

    CICI-FiLM 实验模式说明:
        - Exp-0 (Baseline): patient_condition=None, model_condition=None
        - Exp-1 (自适应校准): patient_condition={c₃,c₄}, model_condition=None
        - Exp-2 (FiLM 微调): patient_condition=None, model_condition=(1,5) tensor

    Returns:
        output_path: 融合后的 CT 路径
    """
    if torch is None:
        raise ImportError("PyTorch 未安装")
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    logger.info(f"开始融合: {Path(lesion_mask_path).name}")
    
    # 加载数据
    template, affine = load_nifti(template_path, return_affine=True)
    lesion_mask = load_nifti(lesion_mask_path)
    
    # 归一化
    template_norm = normalize_ct(template, hu_min, hu_max)
    
    # 创建输入 (mask 区域置为 0)
    input_volume = template_norm.copy()
    input_volume[lesion_mask > 0] = 0
    
    # 创建输出
    output_volume = template_norm.copy()
    weight_volume = np.zeros_like(template_norm)
    
    # 滑动窗口处理
    d, h, w = template.shape
    pd, ph, pw = patch_size
    step = pd - overlap
    
    is_ddpm = model_type == "ddpm"
    ddpm_steps = 10  # DDIM 采样步数（从 50 降到 10 以 5 倍提速，对低 epoch 模型影响极小）

    # DDPM：预构建调度参数（避免每个 patch 重复计算）
    ddpm_alphas_cumprod = None
    ddpm_timesteps = None
    if is_ddpm:
        ddpm_alphas_cumprod, ddpm_timesteps = _build_ddpm_schedule(
            device, num_timesteps=1000, ddim_steps=ddpm_steps
        )
        logger.info(f"  使用 DDPM RePaint 推理管线（DDIM {ddpm_steps} 步加速）")

    # ---- 预扫描：统计含 mask 的 patch 数量 (用于进度日志) ----
    active_patches = 0
    for z in range(0, d - pd + 1, step):
        for y in range(0, h - ph + 1, step):
            for x in range(0, w - pw + 1, step):
                mp = lesion_mask[z:z+pd, y:y+ph, x:x+pw]
                if np.sum(mp) > 0:
                    active_patches += 1
    logger.info(f"  滑动窗口: 体积={d}×{h}×{w}, patch={pd}³, step={step}, 含mask patch={active_patches}")

    import time as _time
    patch_count = 0
    t_start = _time.time()

    with torch.no_grad():
        for z in range(0, d - pd + 1, step):
            for y in range(0, h - ph + 1, step):
                for x in range(0, w - pw + 1, step):
                    # 检查该 patch 是否包含 mask
                    mask_patch = lesion_mask[z:z+pd, y:y+ph, x:x+pw]
                    if np.sum(mask_patch) == 0:
                        continue

                    patch_count += 1

                    # 提取 patch
                    input_patch = input_volume[z:z+pd, y:y+ph, x:x+pw]

                    # 转换为 tensor
                    input_tensor = torch.from_numpy(
                        input_patch[np.newaxis, np.newaxis]
                    ).float().to(device)

                    if is_ddpm:
                        # DDPM 推理：使用 RePaint 策略的迭代去噪
                        template_patch = torch.from_numpy(
                            template_norm[z:z+pd, y:y+ph, x:x+pw][np.newaxis, np.newaxis]
                        ).float().to(device)
                        mask_tensor = torch.from_numpy(
                            mask_patch[np.newaxis, np.newaxis]
                        ).float().to(device)
                        output_result = ddpm_inpaint_patch(
                            model, template_patch, mask_tensor, device,
                            alphas_cumprod=ddpm_alphas_cumprod,
                            timesteps=ddpm_timesteps,
                        )
                        output_patch = output_result.cpu().numpy()[0, 0]
                    else:
                        # 标准模型推理：单次前向传播
                        # 【CICI-FiLM Exp-2】: 如果提供了 model_condition，传给模型
                        # ConditionedGenerator.forward(x, condition) 会激活 FiLM 调制
                        # 无条件模式（Exp-0/Exp-1）: model_condition=None，等价于 backbone(x)
                        if model_condition is not None:
                            # model_condition 形状 (1, 5)，已在调用方构造好
                            cond_tensor = model_condition.to(device)
                            output_patch = model(input_tensor, cond_tensor)
                        else:
                            output_patch = model(input_tensor)
                        output_patch = output_patch.cpu().numpy()[0, 0]

                    # 只更新 mask 区域
                    mask_region = mask_patch > 0
                    output_volume[z:z+pd, y:y+ph, x:x+pw][mask_region] += \
                        output_patch[mask_region]
                    weight_volume[z:z+pd, y:y+ph, x:x+pw][mask_region] += 1

                    # 进度日志（每 10 个 patch 或最后一个 patch）
                    if patch_count % 10 == 0 or patch_count == active_patches:
                        elapsed = _time.time() - t_start
                        avg_t = elapsed / patch_count
                        eta = avg_t * (active_patches - patch_count)
                        logger.info(
                            f"  [{patch_count}/{active_patches}] "
                            f"z={z} y={y} x={x} | "
                            f"{elapsed:.1f}s elapsed, {avg_t:.2f}s/patch, ETA {eta:.0f}s"
                        )
    
    # 处理重叠区域 (平均)
    weight_volume[weight_volume == 0] = 1
    output_volume = output_volume / weight_volume

    # 非 mask 区域保持原样
    output_volume[lesion_mask == 0] = template_norm[lesion_mask == 0]

    # 边界平滑
    if smooth_boundary_width > 0:
        output_volume = smooth_boundary(
            output_volume, template_norm, lesion_mask, smooth_boundary_width
        )

    # 反归一化
    output_hu = denormalize_ct(output_volume, hu_min, hu_max)

    # ============================================================
    # 【核心修改】: 直方图统计匹配校准 (Histogram Statistics Calibration)
    # 目的: 将 AI 生成的深灰色分布强制拉伸到与真实 COPD 一致的水平
    #       基于百分位数的拉伸算法，自动把 AI 生成图像的"最黑的部分"
    #       强制拉伸到与真实 COPD 一致的水平
    # ============================================================

    def match_histogram_stats(source, reference_stats):
        """
        直方图统计匹配 (Calibration)
        将 source(AI) 的统计分布校准到 reference(Real) 的水平

        Exp-0: reference_stats = {'mean': -965.0, 'std': 45.0} (固定先验)
        Exp-1: reference_stats = {'mean': 患者真实值, 'std': 患者真实值} (全量自适应)
        """
        # 1. 计算 AI 当前的统计值
        src_mean = np.mean(source)
        src_std = np.std(source)

        # 2. 获取目标统计值
        ref_mean = reference_stats.get('mean', -970.0)
        ref_std = reference_stats.get('std', 40.0)

        # 3. 线性变换: Z-score 匹配
        scale = ref_std / (src_std + 1e-6)
        scale = np.clip(scale, 0.5, 2.0)

        matched = (source - src_mean) * scale + ref_mean

        # 4. Gamma 压暗
        mask_gray = (matched > -960) & (matched < -800)
        if np.any(mask_gray):
            matched[mask_gray] -= 20.0

        return np.clip(matched, -1024, 400)

    # 1. 定义病灶区域（仅用于 HU 校准，不改变 inpainting 区域）
    calibration_mask = lesion_mask.copy().astype(np.float32)
    atlas_lung = None
    if atlas_lung_mask_path is not None:
        atlas_lung = (load_nifti(atlas_lung_mask_path) > 0).astype(np.float32)

    if mask_dilation > 0:
        try:
            from scipy.ndimage import binary_dilation

            base_mask = lesion_mask > 0
            if adaptive_mask_dilation and patient_condition is not None and atlas_lung is not None:
                lung_voxels = int(atlas_lung.sum())
                target_ei = float(patient_condition.get('global_EI', 0.0)) / 100.0
                target_voxels = int(round(target_ei * lung_voxels))
                best_mask = base_mask.astype(np.float32)
                best_iter = 0
                best_gap = abs(int(base_mask.sum()) - target_voxels)
                for i in range(1, mask_dilation + 1):
                    cand = binary_dilation(base_mask, iterations=i)
                    cand = cand & (atlas_lung > 0)
                    cand_voxels = int(cand.sum())
                    cand_gap = abs(cand_voxels - target_voxels)
                    if cand_gap < best_gap:
                        best_gap = cand_gap
                        best_iter = i
                        best_mask = cand.astype(np.float32)
                calibration_mask = best_mask
                logger.info(
                    f"[方案D-自适应] 目标EI={target_ei * 100:.2f}%, 目标体素={target_voxels}, "
                    f"选择膨胀 {best_iter} 次, 体素={int((calibration_mask > 0).sum())}"
                )
            else:
                calibration_mask = binary_dilation(base_mask, iterations=mask_dilation).astype(np.float32)
                if atlas_lung is not None:
                    calibration_mask = ((calibration_mask > 0) & (atlas_lung > 0)).astype(np.float32)
                orig_v = int(base_mask.sum())
                dil_v = int((calibration_mask > 0).sum())
                logger.info(
                    f"[方案D] Lesion mask 膨胀 {mask_dilation} 次: "
                    f"{orig_v} → {dil_v} 体素 (×{dil_v / max(orig_v, 1):.1f})"
                )
        except ImportError:
            logger.warning("[方案D] scipy 未安装，跳过 mask 膨胀")

    lesion_indices = calibration_mask > 0

    # 2. 仅对病灶区域进行直方图校准
    if np.sum(lesion_indices) > 0:
        lesion_pixels = output_hu[lesion_indices]
        if use_adaptive_hu_calibration and patient_condition is not None:
            target_stats = {
                'mean': patient_condition.get('lesion_HU_mean', -965.0),
                'std': patient_condition.get('lesion_HU_std', 45.0),
            }
            logger.info(
                f"[CICI-FiLM Exp-1] 自适应 HU 校准 (全量): "
                f"mean={target_stats['mean']:.1f} HU, std={target_stats['std']:.1f} HU"
            )
        else:
            target_stats = {'mean': -965.0, 'std': 45.0}
            logger.info(
                f"[Exp-0 Baseline] 固定 HU 校准: "
                f"mean={target_stats['mean']:.1f} HU, std={target_stats['std']:.1f} HU"
            )
        calibrated_pixels = match_histogram_stats(lesion_pixels, target_stats)
        output_hu[lesion_indices] = calibrated_pixels

    # ============================================================

    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_nifti(output_hu, output_path, affine=affine)

    logger.info(f"融合完成: {output_path}")

    return output_path


def batch_fuse(
    template_path: Union[str, Path],
    mask_dir: Union[str, Path],
    checkpoint_path: Union[str, Path],
    output_dir: Union[str, Path],
    pattern: str = "*_warped_lesion.nii.gz"
) -> int:
    """
    批量融合
    
    Args:
        template_path: 模板路径
        mask_dir: mask 目录
        checkpoint_path: 模型检查点路径
        output_dir: 输出目录
        pattern: mask 文件匹配模式
        
    Returns:
        count: 成功处理的数量
    """
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    model = load_model(checkpoint_path)
    
    # 查找所有 mask
    mask_files = list(mask_dir.rglob(pattern))
    logger.info(f"找到 {len(mask_files)} 个 mask 文件")
    
    count = 0
    for mask_path in mask_files:
        try:
            patient_id = mask_path.parent.name
            output_path = output_dir / f"{patient_id}_fused.nii.gz"
            
            fuse_lesion(
                template_path=template_path,
                lesion_mask_path=mask_path,
                model=model,
                output_path=output_path
            )
            count += 1
            
        except Exception as e:
            logger.error(f"融合失败 {mask_path.name}: {e}")
    
    logger.info(f"批量融合完成: {count}/{len(mask_files)}")
    return count


def main(config: dict) -> None:
    """主函数"""
    paths = config.get('paths', {})
    
    template_path = Path(paths.get('atlas', 'data/02_atlas')) / 'standard_template.nii.gz'
    mask_dir = Path(paths.get('mapped', 'data/03_mapped'))
    checkpoint_path = Path(paths.get('checkpoints', 'checkpoints')) / 'best.pth'
    output_dir = Path(paths.get('final_viz', 'data/04_final_viz'))
    
    if not checkpoint_path.exists():
        logger.error(f"模型检查点不存在: {checkpoint_path}")
        return
    
    batch_fuse(template_path, mask_dir, checkpoint_path, output_dir)


if __name__ == "__main__":
    import yaml
    
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)

