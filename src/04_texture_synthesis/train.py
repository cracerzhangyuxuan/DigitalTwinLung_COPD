#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练模块

Inpainting 网络训练逻辑，支持三种模型架构：
- unet: 基线方案
- partial_conv: 进阶方案
- patchgan: 高级方案
"""

import json
import math
from pathlib import Path
from typing import Dict, Optional, Union
from datetime import datetime

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.optim import Adam
    from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, LambdaLR
except ImportError:
    torch = None

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

from .network import InpaintingUNet, PatchDiscriminator, PartialConvUNet, AttentionUNet, DiffusionUNet, create_model
from .losses import InpaintingLoss
from .diffusion_trainer import DiffusionTrainer
from .mae_pretrain import MAEPretrainer
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """
    Inpainting 网络训练器

    支持三种模型架构和多种学习率调度策略
    """

    def __init__(
        self,
        generator: 'nn.Module',
        discriminator: Optional['nn.Module'] = None,
        config: Optional[dict] = None,
        device: str = "cuda",
        use_condition: bool = False
    ):
        if torch is None:
            raise ImportError("PyTorch 未安装")

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")

        # 模型
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device) if discriminator else None
        self.use_condition = use_condition

        # 默认配置
        self.config = config or {}
        train_config = self.config.get('training', {})

        # ---- 方案 C + A 参数 ----
        self.lambda_ei = float(train_config.get('lambda_ei', 0.0))
        self.lambda_contrast = float(train_config.get('lambda_contrast', 0.0))
        self.ei_temperature = float(train_config.get('ei_temperature', 10.0))
        self.hu_min = -1000.0
        self.hu_max = 400.0
        if self.lambda_ei > 0:
            logger.info(f"[方案C] EI Loss: λ={self.lambda_ei}, τ={self.ei_temperature}")
        if self.lambda_contrast > 0:
            logger.info(f"[方案A] 条件响应 Loss: λ={self.lambda_contrast}")
        self.epochs = train_config.get('epochs', 100)

        # 优化器（CICI-FiLM 阶段②：只训练 FiLM 分支）
        lr = train_config.get('learning_rate', 0.0002)
        betas = (train_config.get('beta1', 0.5), train_config.get('beta2', 0.999))

        if use_condition:
            # 只优化条件分支参数（cond_encoder + film）
            trainable_params = [p for p in self.generator.parameters() if p.requires_grad]
            self.g_optimizer = Adam(trainable_params, lr=lr, betas=betas)
            logger.info(f"[CICI-FiLM] 仅训练条件分支: {sum(p.numel() for p in trainable_params):,} 参数")
        else:
            # 训练所有参数（Exp-0 baseline）
            self.g_optimizer = Adam(self.generator.parameters(), lr=lr, betas=betas)

        if self.discriminator:
            self.d_optimizer = Adam(self.discriminator.parameters(), lr=lr, betas=betas)

        # 学习率调度器
        self.g_scheduler = self._create_scheduler(
            self.g_optimizer,
            train_config.get('lr_scheduler', 'step'),
            train_config.get('warmup_epochs', 5)
        )

        # 损失函数
        loss_weights = train_config.get('loss_weights', {})
        self.criterion = InpaintingLoss(
            reconstruction_weight=loss_weights.get('reconstruction', 1.0),
            perceptual_weight=loss_weights.get('perceptual', 0.1),
            adversarial_weight=loss_weights.get('adversarial', 0.01),
            hu_constraint_weight=loss_weights.get('hu_constraint', 0.5),  # 新增：HU 约束权重
            enable_hu_constraint=train_config.get('enable_hu_constraint', True),  # 新增：是否启用
        )

        # 日志输出 HU 约束状态
        if train_config.get('enable_hu_constraint', True):
            logger.info(f"HU 约束损失: 启用 (权重={loss_weights.get('hu_constraint', 0.5)})")
        else:
            logger.info("HU 约束损失: 禁用")

        # TensorBoard
        self.writer = None
        if train_config.get('tensorboard', False) and HAS_TENSORBOARD:
            log_dir = Path(train_config.get('log_dir', 'logs/tensorboard'))
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
            logger.info(f"TensorBoard 日志: {log_dir}")

        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.history = {
            'train_loss': [], 'val_loss': [], 'ssim': [], 'psnr': [],
            'train_recon_loss': [], 'train_ei_loss': [], 'train_contrast_loss': [],
            'val_recon_loss': [], 'val_ei_loss': [], 'val_contrast_loss': [],
        }

    def _create_scheduler(self, optimizer, scheduler_type: str, warmup_epochs: int):
        """创建学习率调度器"""
        if scheduler_type == 'step':
            return StepLR(optimizer, step_size=50, gamma=0.5)
        elif scheduler_type == 'cosine':
            return CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-6)
        elif scheduler_type == 'warmup_cosine':
            def warmup_cosine_lambda(epoch):
                if epoch < warmup_epochs:
                    return epoch / warmup_epochs
                progress = (epoch - warmup_epochs) / (self.epochs - warmup_epochs)
                return 0.5 * (1 + math.cos(math.pi * progress))
            return LambdaLR(optimizer, warmup_cosine_lambda)
        else:
            return StepLR(optimizer, step_size=50, gamma=0.5)

    # ================================================================
    # 方案 C：Soft EI Loss
    # ================================================================
    def _compute_soft_ei(self, pred: 'torch.Tensor') -> 'torch.Tensor':
        """计算 batch 中每个样本的 soft EI（可导近似）

        Args:
            pred: 模型输出，归一化空间 [0,1]，shape (B, 1, D, H, W)

        Returns:
            soft_ei: (B,) 每个样本的 soft EI 值 ∈ (0,1)
        """
        # 反归一化到 HU 空间
        hu = pred * (self.hu_max - self.hu_min) + self.hu_min  # [0,1] → [-1000, 400]
        # soft EI: sigmoid((-950 - hu) / tau)，HU 越低于 -950 值越接近 1
        soft_ei_map = torch.sigmoid((-950.0 - hu) / self.ei_temperature)
        # 对每个样本取空间均值
        B = pred.shape[0]
        return soft_ei_map.view(B, -1).mean(dim=1)

    # ================================================================
    # 方案 A：条件响应一致性损失
    # ================================================================
    def _compute_contrast_loss(
        self, pred: 'torch.Tensor', input_data: 'torch.Tensor',
        condition: 'torch.Tensor'
    ) -> 'torch.Tensor':
        """条件响应一致性：不同 EI 条件应产生方向一致的输出差异

        对 batch 内的每个样本，构造一个扰动条件（EI 翻转到 0.5 的对侧），
        重新 forward 一次，要求：
            EI_orig > EI_pert → 输出应更暗（均值更低）
            EI_orig < EI_pert → 输出应更亮（均值更高）

        Loss = ReLU( -sign(ei_diff) * output_diff )
        当方向一致时 loss=0，方向不一致时 loss>0。

        Args:
            pred: 原始条件下的输出 (B, 1, D, H, W)
            input_data: 输入 (B, 1, D, H, W)
            condition: 原始条件向量 (B, 5)

        Returns:
            contrast_loss: 标量
        """
        B = condition.shape[0]
        if B < 1:
            return torch.tensor(0.0, device=pred.device)

        # 构造扰动条件：将 c₁(EI) 翻转到 0.5 的对侧
        # 如果原始 EI=0.02，扰动 EI=0.98；原始 EI=0.18，扰动 EI=0.82
        cond_pert = condition.clone()
        cond_pert[:, 0] = 1.0 - condition[:, 0]

        # 用扰动条件做一次 forward
        with torch.set_grad_enabled(self.generator.training):
            pred_pert = self.generator(input_data, cond_pert)

        # 计算每个样本的输出均值差异
        pred_mean = pred.view(B, -1).mean(dim=1)
        pred_pert_mean = pred_pert.view(B, -1).mean(dim=1)
        output_diff = pred_mean - pred_pert_mean  # (B,)

        # EI 差异方向
        ei_diff = condition[:, 0] - cond_pert[:, 0]  # (B,)

        # 方向一致性：EI 更高 → 输出应更暗（归一化值更低）→ output_diff < 0
        # 即 sign(ei_diff) 和 sign(output_diff) 应该相反
        # Loss = ReLU( sign(ei_diff) * output_diff )
        loss = torch.relu(ei_diff.sign() * output_diff)
        return loss.mean()
    
    def train_epoch(self, train_loader: 'DataLoader') -> Dict[str, float]:
        """训练一个 epoch（CICI-FiLM 条件化版本 + 方案 C/A）"""
        self.generator.train()
        if self.discriminator:
            self.discriminator.train()

        epoch_losses = {
            'reconstruction': 0, 'perceptual': 0, 'total': 0,
            'ei': 0, 'contrast': 0,
        }

        for batch_idx, batch in enumerate(train_loader):
            input_data = batch['input'].to(self.device)
            target = batch['target'].to(self.device)
            mask = batch['mask'].to(self.device)

            # 获取条件向量（CICI-FiLM 阶段②）
            condition = batch.get('condition', None)
            if condition is not None:
                condition = condition.to(self.device)

            # 生成器前向传播（传递条件向量）
            if self.use_condition and condition is not None:
                pred = self.generator(input_data, condition)
            else:
                pred = self.generator(input_data)

            # GAN 训练：先更新判别器，再更新生成器
            if self.discriminator:
                # ========== 步骤 1: 更新判别器 ==========
                self.d_optimizer.zero_grad()
                real_pred = self.discriminator(target)
                fake_pred = self.discriminator(pred.detach())
                d_losses = self.criterion.discriminator_loss(real_pred, fake_pred)
                d_losses['total'].backward()
                self.d_optimizer.step()

                # ========== 步骤 2: 更新生成器 ==========
                self.g_optimizer.zero_grad()
                disc_pred_for_g = self.discriminator(pred)
                g_losses = self.criterion.generator_loss(pred, target, mask, disc_pred_for_g)
            else:
                self.g_optimizer.zero_grad()
                g_losses = self.criterion.generator_loss(pred, target, mask)

            # ---- 方案 C：EI Loss ----
            ei_loss = torch.tensor(0.0, device=self.device)
            if self.lambda_ei > 0 and condition is not None:
                soft_ei = self._compute_soft_ei(pred)  # (B,)
                ei_target = condition[:, 0]  # c₁ = global_EI / 100
                import torch.nn.functional as _F
                ei_loss = _F.mse_loss(soft_ei, ei_target)
                g_losses['ei'] = ei_loss
                g_losses['total'] = g_losses['total'] + self.lambda_ei * ei_loss

            # ---- 方案 A：条件响应损失 ----
            contrast_loss = torch.tensor(0.0, device=self.device)
            if self.lambda_contrast > 0 and self.use_condition and condition is not None:
                contrast_loss = self._compute_contrast_loss(pred, input_data, condition)
                g_losses['contrast'] = contrast_loss
                g_losses['total'] = g_losses['total'] + self.lambda_contrast * contrast_loss

            g_losses['total'].backward()
            self.g_optimizer.step()

            # 累计损失
            for key in epoch_losses:
                if key in g_losses:
                    epoch_losses[key] += g_losses[key].item()

        # 平均损失
        num_batches = len(train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses
    
    def validate(self, val_loader: 'DataLoader') -> Dict[str, float]:
        """验证（CICI-FiLM 条件化版本 + 方案 C/A）"""
        self.generator.eval()

        val_losses = {'reconstruction': 0, 'total': 0, 'ei': 0, 'contrast': 0}

        with torch.no_grad():
            for batch in val_loader:
                input_data = batch['input'].to(self.device)
                target = batch['target'].to(self.device)
                mask = batch['mask'].to(self.device)

                condition = batch.get('condition', None)
                if condition is not None:
                    condition = condition.to(self.device)

                if self.use_condition and condition is not None:
                    pred = self.generator(input_data, condition)
                else:
                    pred = self.generator(input_data)

                losses = self.criterion.generator_loss(pred, target, mask)

                # ---- 方案 C：EI Loss ----
                if self.lambda_ei > 0 and condition is not None:
                    soft_ei = self._compute_soft_ei(pred)
                    ei_target = condition[:, 0]
                    import torch.nn.functional as _F
                    ei_loss = _F.mse_loss(soft_ei, ei_target)
                    losses['ei'] = ei_loss
                    losses['total'] = losses['total'] + self.lambda_ei * ei_loss

                # ---- 方案 A：条件响应损失（验证时仅记录，不反向传播）----
                if self.lambda_contrast > 0 and self.use_condition and condition is not None:
                    cond_pert = condition.clone()
                    cond_pert[:, 0] = 1.0 - condition[:, 0]
                    pred_pert = self.generator(input_data, cond_pert)
                    B = condition.shape[0]
                    pred_mean = pred.view(B, -1).mean(dim=1)
                    pred_pert_mean = pred_pert.view(B, -1).mean(dim=1)
                    output_diff = pred_mean - pred_pert_mean
                    ei_diff = condition[:, 0] - cond_pert[:, 0]
                    contrast_loss = torch.relu(ei_diff.sign() * output_diff).mean()
                    losses['contrast'] = contrast_loss
                    losses['total'] = losses['total'] + self.lambda_contrast * contrast_loss

                for key in val_losses:
                    if key in losses:
                        val_losses[key] += losses[key].item()

        num_batches = len(val_loader)
        for key in val_losses:
            val_losses[key] /= num_batches

        return val_losses
    
    @staticmethod
    def _compute_psnr_ssim(pred, target, mask):
        """计算 patch 级 PSNR 和 SSIM（仅 mask 区域）"""
        import numpy as np
        p = pred.detach().cpu().numpy().flatten()
        t = target.detach().cpu().numpy().flatten()
        m = mask.detach().cpu().numpy().flatten() > 0
        if m.sum() == 0:
            return 0.0, 0.0
        p_m, t_m = p[m], t[m]
        mse = float(np.mean((p_m - t_m) ** 2))
        psnr = 10.0 * np.log10(1.0 / (mse + 1e-10))
        # 简化 SSIM（全局统计）
        mu_p, mu_t = p_m.mean(), t_m.mean()
        sig_p, sig_t = p_m.std(), t_m.std()
        sig_pt = np.mean((p_m - mu_p) * (t_m - mu_t))
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim = float(((2 * mu_p * mu_t + c1) * (2 * sig_pt + c2)) /
                      ((mu_p ** 2 + mu_t ** 2 + c1) * (sig_p ** 2 + sig_t ** 2 + c2)))
        return psnr, ssim

    def train(
        self,
        train_loader: 'DataLoader',
        val_loader: 'DataLoader',
        epochs: int = 100,
        checkpoint_dir: Union[str, Path] = "checkpoints",
        save_frequency: int = 10,
        patience: int = 0
    ) -> Dict:
        """完整训练流程

        Args:
            patience: Early stopping 耐心值。0 表示不启用。
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"开始训练: {epochs} epochs" +
                    (f", early stopping patience={patience}" if patience > 0 else ""))

        no_improve_count = 0

        for epoch in range(epochs):
            self.current_epoch = epoch + 1

            # 训练
            train_losses = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_losses['total'])
            self.history['train_recon_loss'].append(train_losses.get('reconstruction', 0))
            self.history['train_ei_loss'].append(train_losses.get('ei', 0))
            self.history['train_contrast_loss'].append(train_losses.get('contrast', 0))

            # 验证（含 PSNR/SSIM）
            val_losses = self.validate(val_loader)
            self.history['val_loss'].append(val_losses['total'])
            self.history['val_recon_loss'].append(val_losses.get('reconstruction', 0))
            self.history['val_ei_loss'].append(val_losses.get('ei', 0))
            self.history['val_contrast_loss'].append(val_losses.get('contrast', 0))

            # 计算验证集 PSNR/SSIM
            epoch_psnr, epoch_ssim, n_val = 0.0, 0.0, 0
            self.generator.eval()
            with torch.no_grad():
                for batch in val_loader:
                    inp = batch['input'].to(self.device)
                    tgt = batch['target'].to(self.device)
                    msk = batch['mask'].to(self.device)
                    cond = batch.get('condition', None)
                    if cond is not None:
                        cond = cond.to(self.device)
                    if self.use_condition and cond is not None:
                        pred = self.generator(inp, cond)
                    else:
                        pred = self.generator(inp)
                    for i in range(pred.shape[0]):
                        p, s = self._compute_psnr_ssim(pred[i], tgt[i], msk[i])
                        epoch_psnr += p
                        epoch_ssim += s
                        n_val += 1
            if n_val > 0:
                epoch_psnr /= n_val
                epoch_ssim /= n_val
            self.history['psnr'].append(epoch_psnr)
            self.history['ssim'].append(epoch_ssim)

            # 更新学习率
            current_lr = self.g_optimizer.param_groups[0]['lr']
            self.g_scheduler.step()

            # TensorBoard 日志
            if self.writer:
                self.writer.add_scalar('Loss/train', train_losses['total'], self.current_epoch)
                self.writer.add_scalar('Loss/val', val_losses['total'], self.current_epoch)
                self.writer.add_scalar('Loss/reconstruction', train_losses.get('reconstruction', 0), self.current_epoch)
                self.writer.add_scalar('Loss/perceptual', train_losses.get('perceptual', 0), self.current_epoch)
                self.writer.add_scalar('Loss/ei', train_losses.get('ei', 0), self.current_epoch)
                self.writer.add_scalar('Loss/contrast', train_losses.get('contrast', 0), self.current_epoch)
                self.writer.add_scalar('LearningRate', current_lr, self.current_epoch)
                self.writer.add_scalar('Metrics/PSNR', epoch_psnr, self.current_epoch)
                self.writer.add_scalar('Metrics/SSIM', epoch_ssim, self.current_epoch)

            # 日志
            ei_str = f" | EI: {train_losses.get('ei', 0):.6f}/{val_losses.get('ei', 0):.6f}" if self.lambda_ei > 0 else ""
            ctr_str = f" | Ctr: {train_losses.get('contrast', 0):.6f}/{val_losses.get('contrast', 0):.6f}" if self.lambda_contrast > 0 else ""
            logger.info(
                f"Epoch {self.current_epoch}/{epochs} | "
                f"Train: {train_losses['total']:.4f} | "
                f"Val: {val_losses['total']:.4f} | "
                f"PSNR: {epoch_psnr:.2f} | SSIM: {epoch_ssim:.4f} | "
                f"LR: {current_lr:.6f}{ei_str}{ctr_str}"
            )

            # 保存最佳模型
            if val_losses['total'] < self.best_loss:
                self.best_loss = val_losses['total']
                self.save_checkpoint(checkpoint_dir / "best.pth")
                logger.info(f"  ✓ 保存最佳模型: loss = {self.best_loss:.4f}")
                no_improve_count = 0
            else:
                no_improve_count += 1

            # 定期保存
            if self.current_epoch % save_frequency == 0:
                self.save_checkpoint(checkpoint_dir / "latest.pth")

            # Early stopping
            if patience > 0 and no_improve_count >= patience:
                logger.info(f"Early stopping: {patience} epochs 无改善，停止训练")
                break

        # 关闭 TensorBoard
        if self.writer:
            self.writer.close()

        # 保存训练历史
        with open(checkpoint_dir / "training_log.json", 'w') as f:
            json.dump(self.history, f, indent=2)

        logger.info("训练完成!")
        return self.history
    
    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """保存检查点（原子性保存，防止损坏）"""
        checkpoint = {
            'epoch': self.current_epoch,
            'generator_state_dict': self.generator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'best_loss': self.best_loss,
            'history': self.history,
        }

        if self.discriminator:
            checkpoint['discriminator_state_dict'] = self.discriminator.state_dict()
            checkpoint['d_optimizer_state_dict'] = self.d_optimizer.state_dict()

        # 原子性保存：先保存到临时文件，成功后再重命名
        path = Path(path)
        temp_path = path.parent / f"{path.stem}_tmp{path.suffix}"

        try:
            torch.save(checkpoint, temp_path)
            # 验证文件可读
            torch.load(temp_path, map_location='cpu')
            # 重命名为目标文件
            if path.exists():
                path.unlink()
            temp_path.rename(path)
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise RuntimeError(f"保存 checkpoint 失败: {e}")
    
    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.history = checkpoint.get('history', {'train_loss': [], 'val_loss': []})
        
        if self.discriminator and 'discriminator_state_dict' in checkpoint:
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        
        logger.info(f"加载检查点: epoch {self.current_epoch}, best_loss {self.best_loss:.4f}")


def main(config: dict) -> None:
    """主函数"""
    if torch is None:
        logger.error("PyTorch 未安装")
        return

    from .dataset import create_dataloader

    # 获取路径配置
    paths = config.get('paths', {})
    mapped_dir = Path(paths.get('mapped', 'data/03_mapped'))

    # 收集已配准的 CT 和病灶 mask
    ct_files = []
    mask_files = []

    for patient_dir in sorted(mapped_dir.iterdir()):
        if not patient_dir.is_dir() or patient_dir.name == 'visualizations':
            continue

        warped_ct = patient_dir / f"{patient_dir.name}_warped.nii.gz"
        warped_mask = patient_dir / f"{patient_dir.name}_warped_lesion.nii.gz"

        if warped_ct.exists() and warped_mask.exists():
            ct_files.append(warped_ct)
            mask_files.append(warped_mask)

    if len(ct_files) == 0:
        logger.error("未找到已配准的数据，请先运行 Phase 3A")
        return

    logger.info(f"找到 {len(ct_files)} 例已配准数据")

    # 划分训练/验证集 — 硬编码分界线，杜绝数据泄漏
    # 训练集: copd_001 ~ copd_023 (23 例)
    # 验证集: copd_024 ~ copd_029 (6 例)
    TRAIN_CUTOFF = "copd_024"  # 该 ID 及之后的患者划入验证集
    train_ct, train_mask = [], []
    val_ct, val_mask = [], []
    for ct_f, mk_f in zip(ct_files, mask_files):
        patient_id = ct_f.parent.name  # e.g. "copd_001"
        if patient_id < TRAIN_CUTOFF:
            train_ct.append(ct_f)
            train_mask.append(mk_f)
        else:
            val_ct.append(ct_f)
            val_mask.append(mk_f)

    # 安全回退：如果验证集为空（数据不足），使用训练集最后一个
    if not val_ct:
        val_ct = train_ct[-1:]
        val_mask = train_mask[-1:]

    logger.info(f"数据集划分 (硬编码边界={TRAIN_CUTOFF}):")
    logger.info(f"  训练集: {len(train_ct)} 例 ({train_ct[0].parent.name}~{train_ct[-1].parent.name})")
    logger.info(f"  验证集: {len(val_ct)} 例 ({val_ct[0].parent.name}~{val_ct[-1].parent.name})")

    from .dataset import LungPatchDataset
    from torch.utils.data import DataLoader

    train_config = config.get('training', {})
    patch_size = tuple(train_config.get('patch_size', [64, 64, 64]))
    batch_size = train_config.get('batch_size', 4)
    num_workers = train_config.get('num_workers', 0)  # Windows 下建议设为 0

    train_dataset = LungPatchDataset(
        ct_paths=train_ct,
        mask_paths=train_mask,
        patch_size=patch_size,
        augment=True
    )

    val_dataset = LungPatchDataset(
        ct_paths=val_ct,
        mask_paths=val_mask,
        patch_size=patch_size,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )

    logger.info(f"训练集: {len(train_dataset)} patches, 验证集: {len(val_dataset)} patches")

    # 创建模型（根据 model_type 选择）
    model_type = train_config.get('model_type', 'unet')
    logger.info(f"创建模型: {model_type}")

    # GAN 系列模型返回 (generator, discriminator) 元组
    if model_type in ('patchgan', 'attgan', 'mae_patchgan'):
        generator, discriminator = create_model(model_type)
    elif model_type == 'partial_conv':
        generator = create_model('partial_conv')
        discriminator = None
    elif model_type == 'ddpm':
        # DDPM 使用专用的 DiffusionTrainer
        logger.info("DDPM 使用专用 DiffusionTrainer 训练循环")
        model = create_model('ddpm')
        diff_trainer = DiffusionTrainer(model, config)
        # 注意：pipeline 已将 paths['checkpoints'] 修改为 checkpoints/ddpm，
        #       此处直接使用，不再追加子目录，避免 checkpoints/ddpm/ddpm 双重嵌套
        checkpoint_dir = Path(paths.get('checkpoints', 'checkpoints'))
        diff_trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=train_config.get('epochs', 500),
            checkpoint_dir=checkpoint_dir,
            save_frequency=train_config.get('save_frequency', 50),
        )
        return  # DDPM 训练流程已完成，不走标准 Trainer
    else:  # 默认 unet
        generator = create_model('unet')
        discriminator = None

    # MAE-PatchGAN 预训练权重加载（如果有）
    if model_type == 'mae_patchgan':
        # 使用项目根目录下的固定路径（不受 pipeline 修改 paths['checkpoints'] 影响）
        mae_weights_path = Path('checkpoints') / 'mae_pretrain' / 'encoder_weights.pth'
        if mae_weights_path.exists():
            logger.info(f"加载 MAE 预训练 encoder 权重: {mae_weights_path}")
            encoder_weights = torch.load(mae_weights_path, map_location='cpu')
            missing, unexpected = generator.load_state_dict(encoder_weights, strict=False)
            logger.info(f"  加载完成: {len(encoder_weights)} 个权重, {len(missing)} 个缺失, {len(unexpected)} 个多余")
        else:
            logger.warning(f"MAE 预训练权重未找到: {mae_weights_path}，使用随机初始化")

    # 创建训练器
    trainer = Trainer(generator, discriminator, config)

    # 训练
    checkpoint_dir = Path(paths.get('checkpoints', 'checkpoints'))
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=train_config.get('epochs', 100),
        checkpoint_dir=checkpoint_dir,
        save_frequency=train_config.get('save_frequency', 10),
    )


if __name__ == "__main__":
    import yaml
    
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)

