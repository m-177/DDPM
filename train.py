# -----------------------------
# train_classic_ddpm.py - 经典DDPM加性噪声训练（混合损失函数）
# -----------------------------
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import psutil
import gc
from Dataset import Dataset_UWB


# -----------------------------
# 内存监控工具
# -----------------------------
class MemoryMonitor:
    """监控系统内存和GPU内存使用情况"""

    def __init__(self, device='cuda', memory_warning_threshold=85):
        self.device = device
        self.memory_warning_threshold = memory_warning_threshold
        self.peak_system_memory = 0
        self.peak_gpu_memory = 0

    def get_system_memory_usage(self):
        """获取系统内存使用率 (%)"""
        return psutil.virtual_memory().percent

    def get_system_memory_used(self):
        """获取已使用的系统内存 (GB)"""
        return psutil.virtual_memory().used / (1024 ** 3)

    def get_system_memory_total(self):
        """获取总系统内存 (GB)"""
        return psutil.virtual_memory().total / (1024 ** 3)

    def get_gpu_memory_usage(self):
        """获取GPU内存使用情况"""
        if torch.cuda.is_available() and self.device == 'cuda':
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
            return {
                'allocated': allocated,
                'reserved': reserved,
                'max_allocated': max_allocated
            }
        return None

    def get_gpu_utilization(self):
        """获取GPU利用率"""
        if torch.cuda.is_available() and self.device == 'cuda':
            # 注意：这个函数需要nvidia-ml-py或pynvml库
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                pynvml.nvmlShutdown()
                return util.gpu
            except:
                return -1  # 无法获取
        return -1

    def print_memory_status(self, epoch, step_name=""):
        """打印当前内存状态"""
        sys_percent = self.get_system_memory_usage()
        sys_used = self.get_system_memory_used()
        sys_total = self.get_system_memory_total()

        # 更新峰值
        if sys_percent > self.peak_system_memory:
            self.peak_system_memory = sys_percent

        status_str = f"[内存监控] {step_name} - "
        status_str += f"系统: {sys_used:.1f}/{sys_total:.1f} GB ({sys_percent:.1f}%)"

        if self.device == 'cuda' and torch.cuda.is_available():
            gpu_mem = self.get_gpu_memory_usage()
            if gpu_mem:
                if gpu_mem['allocated'] > self.peak_gpu_memory:
                    self.peak_gpu_memory = gpu_mem['allocated']
                status_str += f" | GPU: {gpu_mem['allocated']:.2f}/{gpu_mem['reserved']:.2f} GB"

        # 添加警告
        if sys_percent > self.memory_warning_threshold:
            status_str += f" ⚠️ 系统内存超过{self.memory_warning_threshold}%!"

        print(status_str)
        return sys_percent

    def get_peak_memory(self):
        """获取峰值内存使用"""
        return {
            'peak_system_memory': self.peak_system_memory,
            'peak_gpu_memory': self.peak_gpu_memory
        }

    def reset_peak(self):
        """重置峰值记录"""
        self.peak_system_memory = 0
        self.peak_gpu_memory = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


def check_data_worker_memory(dataset_path):
    """检查数据集大小和预估内存需求"""
    try:
        data = np.load(dataset_path)
        data_size_gb = data.nbytes / (1024 ** 3)
        print(f"\n数据集信息:")
        print(f"  文件: {dataset_path}")
        print(f"  大小: {data_size_gb:.2f} GB")
        print(f"  形状: {data.shape}")

        # 预估内存需求
        system_ram = psutil.virtual_memory().total / (1024 ** 3)
        print(f"\n内存预估:")
        print(f"  系统总内存: {system_ram:.1f} GB")
        print(f"  数据集大小: {data_size_gb:.2f} GB")

        # 不同worker数量的内存需求
        for workers in [0, 2, 4, 6, 8]:
            if workers == 0:
                est_memory = data_size_gb * 1.2  # 主进程 + 开销
            else:
                est_memory = data_size_gb * (workers + 1) * 1.2  # workers + 主进程 + 开销

            if est_memory < system_ram * 0.8:
                status = "✅ 安全"
            elif est_memory < system_ram:
                status = "⚠️ 紧张"
            else:
                status = "❌ 危险"

            print(f"  workers={workers}: ~{est_memory:.1f} GB {status}")

        return data_size_gb
    except Exception as e:
        print(f"无法分析数据集: {e}")
        return None


# -----------------------------
# 设置随机种子（用于验证集固定噪声）
# -----------------------------
def set_seed(seed=42):
    """固定随机种子，确保可重复性"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# 1. SNR计算函数（原始域，单位dB）
# -----------------------------
def calculate_snr(clean_signal, noisy_signal):
    """计算信噪比 (dB)"""
    if torch.is_tensor(clean_signal):
        clean_signal = clean_signal.cpu().numpy()
    if torch.is_tensor(noisy_signal):
        noisy_signal = noisy_signal.cpu().numpy()

    clean_signal = clean_signal.flatten()
    noisy_signal = noisy_signal.flatten()

    signal_power = np.mean(clean_signal ** 2)
    noise = noisy_signal - clean_signal
    noise_power = np.mean(noise ** 2)

    if noise_power < 1e-10:
        return 100.0
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


# -----------------------------
# 2. 相对MSE损失函数（经典DDPM在原始域使用）
# -----------------------------
def relative_mse_loss(pred_clean, target_clean):
    """
    相对MSE损失 = MSE / Signal Power = ||pred - target||² / ||target||²
    在原始域中计算，值越小表示重建质量越好
    """
    pred_flat = pred_clean.view(pred_clean.shape[0], -1)
    target_flat = target_clean.view(target_clean.shape[0], -1)

    mse = torch.mean((pred_flat - target_flat) ** 2, dim=1)
    signal_power = torch.mean(target_flat ** 2, dim=1)

    rel_loss = mse / (signal_power + 1e-8)
    return torch.mean(rel_loss)


def peak_aware_loss(pred_clean, target_clean, alpha=1.5):
    """
    峰值感知损失：对信号幅度较大的区域（脉冲位置）给予更高权重
    让模型更关注脉冲峰值的精确重建，减少弱脉冲丢失
    
    使用平方权重，对峰值区域给予更强的关注：
    weight = 1.0 + alpha * |target|²
    相比线性权重，平方权重能更好地区分高幅度脉冲和低幅度噪声
    
    Args:
        alpha: 峰值加权的强度，越大越关注峰值区域
    """
    pred_flat = pred_clean.view(pred_clean.shape[0], -1)
    target_flat = target_clean.view(target_clean.shape[0], -1)
    
    # 平方权重：对峰值区域给予更强的关注
    # 例如：|target|=0.8 → weight=1+1.5*0.64=1.96（峰值区域）
    #       |target|=0.1 → weight=1+1.5*0.01=1.015（低幅度区域）
    weight = 1.0 + alpha * (torch.abs(target_flat) ** 2)
    
    # 加权MSE
    mse = (pred_flat - target_flat) ** 2
    weighted_mse = torch.mean(weight * mse)
    
    return weighted_mse



def peak_correlation_loss(pred_clean, target_clean, polarity_weight=1.2):
    """
    峰值相关性损失：鼓励预测信号与目标信号的峰值位置对齐
    通过计算局部区域的互相关系数来实现
    
    加入局部极性惩罚，防止个别脉冲极性反了
    在脉冲位置（高能量区域）逐点检查极性一致性
    
    Args:
        polarity_weight: 极性惩罚的权重（默认1.2），越大越强制极性一致
    """
    pred_flat = pred_clean.view(pred_clean.shape[0], -1)
    target_flat = target_clean.view(target_clean.shape[0], -1)
    
    # 对每个样本计算余弦相似度（关注形状而非幅度）
    pred_norm = pred_flat / (torch.norm(pred_flat, dim=1, keepdim=True) + 1e-8)
    target_norm = target_flat / (torch.norm(target_flat, dim=1, keepdim=True) + 1e-8)
    
    correlation = torch.sum(pred_norm * target_norm, dim=1)
    # 转化为损失（1 - 相关性）
    corr_loss = 1.0 - torch.mean(correlation)
    
    # 局部极性惩罚：在脉冲位置逐点检查极性一致性
    # 用滑动窗口计算局部能量，找到脉冲位置
    window_size = 31  # 脉冲宽度约30个采样点
    padding = window_size // 2
    # 计算局部能量（滑动窗口内的平方和）
    target_padded = torch.nn.functional.pad(target_flat.unsqueeze(1), (padding, padding), mode='reflect')
    local_energy = torch.nn.functional.avg_pool1d(target_padded ** 2, kernel_size=window_size, stride=1).squeeze(1)
    # 能量阈值：取最大能量的10%
    energy_threshold = local_energy.max(dim=1, keepdim=True).values * 0.1
    pulse_mask = (local_energy > energy_threshold).float()
    
    # 在脉冲位置检查符号一致性
    pred_sign = torch.sign(pred_flat)
    target_sign = torch.sign(target_flat)
    # 符号一致时为1，不一致时为0
    sign_match = (pred_sign == target_sign).float()
    # 只在脉冲位置计算加权平均
    polarity_loss = 1.0 - torch.sum(sign_match * pulse_mask) / (torch.sum(pulse_mask) + 1e-8)
    
    # 组合：原始相关性损失 + 局部极性惩罚
    total_loss = corr_loss + polarity_weight * polarity_loss
    
    return total_loss


def combined_loss(pred_clean, target_clean, pred_noise=None, target_noise=None,
                  lambda_mse=2.5, lambda_rel=0.4, lambda_peak=1.8, lambda_corr=1.8,
                  use_noise_loss=False):

    """
    组合损失函数（增强版：加入峰值感知损失）
    lambda_mse: MSE权重（默认2.5）
    lambda_rel: 相对MSE权重（默认0.4）
    lambda_peak: 峰值感知损失权重（默认1.8）
    lambda_corr: 峰值相关性损失权重（默认1.8）
    """
    mse_loss_val = nn.functional.mse_loss(pred_clean, target_clean)
    rel_loss_val = relative_mse_loss(pred_clean, target_clean)
    peak_loss_val = peak_aware_loss(pred_clean, target_clean, alpha=1.5)
    corr_loss_val = peak_correlation_loss(pred_clean, target_clean, polarity_weight=1.2)

    total_loss = (lambda_mse * mse_loss_val +
                  lambda_rel * rel_loss_val +
                  lambda_peak * peak_loss_val +
                  lambda_corr * corr_loss_val)

    if use_noise_loss and pred_noise is not None and target_noise is not None:
        noise_loss = nn.functional.mse_loss(pred_noise, target_noise)
        total_loss = total_loss + 0.1 * noise_loss

    return total_loss, mse_loss_val, rel_loss_val, peak_loss_val, corr_loss_val


# -----------------------------
# 3. 逆向扩散去噪函数（经典DDPM）
# -----------------------------
def denoise_for_eval(model, noisy_signal, t_start, params, device):
    """
    经典DDPM逆向去噪，用于评估SNR
    """
    betas = params['betas']
    alphas = params['alphas']
    alphas_cumprod = params['alphas_cumprod']
    alphas_cumprod_prev = params['alphas_cumprod_prev']
    posterior_variance = params['posterior_variance']

    x = noisy_signal.clone()
    timesteps = len(betas)
    start_t = min(t_start, timesteps - 1)

    with torch.no_grad():
        for t_idx in range(start_t, 0, -1):
            t_tensor = torch.full((x.shape[0],), t_idx - 1, device=device, dtype=torch.long)

            predicted_noise = model(x, t_tensor)

            alpha = alphas[t_idx - 1].view(-1, *([1] * (x.dim() - 1)))
            alpha_bar = alphas_cumprod[t_idx - 1].view(-1, *([1] * (x.dim() - 1)))
            alpha_bar_prev = alphas_cumprod_prev[t_idx - 1].view(-1, *([1] * (x.dim() - 1)))
            beta = betas[t_idx - 1].view(-1, *([1] * (x.dim() - 1)))

            pred_x0 = (x - torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)
            pred_x0 = torch.clamp(pred_x0, -1.2, 1.2)

            # 注意：BPSK信号本身有正有负，无法从信号本身判断极性是否反了
            # 因此不在推理时做极性校正，而是通过训练时的损失函数来防止极性翻转
            if t_idx > 1:
                coeff1 = (torch.sqrt(alpha_bar_prev) * beta) / (1 - alpha_bar)
                coeff2 = (torch.sqrt(alpha) * (1 - alpha_bar_prev)) / (1 - alpha_bar)
                mu = coeff1 * pred_x0 + coeff2 * x
                var = posterior_variance[t_idx - 1].view(-1, *([1] * (x.dim() - 1)))
                noise = torch.randn_like(x)
                x = mu + torch.sqrt(var) * noise
            else:
                x = pred_x0

    return x


# -----------------------------
# 4. 1D UNet模型（经典DDPM版本）
# -----------------------------
class SimpleUNet1D_Classic(nn.Module):
    """
    改进版1D UNet：减少下采样次数（3次→2次），保留更多脉冲细节
    下采样：L → L/2 → L/4（原来是 L/8）
    上采样：L/4 → L/2 → L
    """
    def __init__(self, in_channels=1, out_channels=1, dropout_rate=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate

        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),
            nn.Dropout(dropout_rate)
        )

        # 编码器1：in_channels → 64，不下采样（保留原始分辨率）
        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Dropout1d(dropout_rate),
            nn.Conv1d(64, 64, kernel_size=7, padding=3),
            nn.GroupNorm(8, 64),
            nn.SiLU()
        )
        self.pool1 = nn.MaxPool1d(2)  # L → L/2

        # 编码器2：64 → 128
        self.enc2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.GroupNorm(16, 128),
            nn.SiLU(),
            nn.Dropout1d(dropout_rate),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.GroupNorm(16, 128),
            nn.SiLU()
        )
        self.pool2 = nn.MaxPool1d(2)  # L/2 → L/4

        # 中间层（瓶颈）：128 → 256（原来用512，减少参数量）
        self.mid = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256),
            nn.SiLU(),
            nn.Dropout1d(dropout_rate),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256),
            nn.SiLU()
        )

        # 上采样1：256 → 128，L/4 → L/2
        self.up2 = nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.GroupNorm(16, 128),
            nn.SiLU(),
            nn.Dropout1d(dropout_rate),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(16, 128),
            nn.SiLU()
        )

        # 上采样2：128 → 64，L/2 → L
        self.up1 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec1 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Dropout1d(dropout_rate),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU()
        )

        # 输出层
        self.out = nn.Conv1d(64, out_channels, kernel_size=7, padding=3)

        # 时间嵌入投影（调整通道数）
        self.time_proj1 = nn.Linear(128, 64)
        self.time_proj2 = nn.Linear(128, 128)
        self.time_proj_mid = nn.Linear(128, 256)

    def forward(self, x, t):
        t_emb = self.time_embed(t.float().unsqueeze(-1))

        # 编码器1（原始分辨率）
        e1 = self.enc1(x)
        e1 = e1 + self.time_proj1(t_emb).view(-1, 64, 1)
        p1 = self.pool1(e1)  # L → L/2

        # 编码器2（L/2分辨率）
        e2 = self.enc2(p1)
        e2 = e2 + self.time_proj2(t_emb).view(-1, 128, 1)
        p2 = self.pool2(e2)  # L/2 → L/4

        # 中间层（L/4分辨率）
        m = self.mid(p2)
        m = m + self.time_proj_mid(t_emb).view(-1, 256, 1)

        # 上采样1：L/4 → L/2
        d2 = self.up2(m)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        # 上采样2：L/2 → L（原始分辨率）
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out(d1)
        return out



# -----------------------------
# 5. EMA类
# -----------------------------
class EMA:
    def __init__(self, beta=0.995):
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=100):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


# -----------------------------
# 6. Beta调度
# -----------------------------
def linear_beta_schedule(beta_start, beta_end, timesteps):
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def compute_diffusion_params(betas, device):
    betas = betas.to(device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.)
    posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'alphas_cumprod_prev': alphas_cumprod_prev,
        'posterior_variance': posterior_variance,
    }


# -----------------------------
# 7. 检查点保存与恢复（暂停/继续功能）
# -----------------------------
CHECKPOINT_PATH = "./saved_models_classic/training_checkpoint.pth"

def save_checkpoint(epoch, model, optimizer, scheduler, ema_model=None,
                    train_losses=None, val_losses=None, lr_history=None,
                    epochs=None, snr_improvements=None, snr_epochs=None,
                    memory_usages=None, best_val_loss=None, best_snr_improvement=None,
                    best_epoch=None, loss_early_stopping=None,
                    memory_monitor=None, params=None, is_pause=False):
    """
    保存完整的训练检查点，支持暂停保存和定期保存
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses or [],
        'val_losses': val_losses or [],
        'lr_history': lr_history or [],
        'epochs': epochs or [],
        'snr_improvements': snr_improvements or [],
        'snr_epochs': snr_epochs or [],
        'memory_usages': memory_usages or [],
        'best_val_loss': best_val_loss if best_val_loss is not None else float('inf'),
        'best_snr_improvement': best_snr_improvement if best_snr_improvement is not None else -float('inf'),
        'best_epoch': best_epoch if best_epoch is not None else -1,
        'is_pause': is_pause,
    }

    if ema_model is not None:
        checkpoint['ema_state_dict'] = ema_model.state_dict()

    if loss_early_stopping is not None:
        checkpoint['loss_early_stopping'] = {
            'counter': loss_early_stopping.counter,
            'best_value': loss_early_stopping.best_value,
        }

    if memory_monitor is not None:
        checkpoint['memory_monitor'] = memory_monitor.get_peak_memory()

    if params is not None:
        # 只保存CPU上的参数（避免GPU内存占用）
        cpu_params = {}
        for k, v in params.items():
            if torch.is_tensor(v):
                cpu_params[k] = v.cpu()
            else:
                cpu_params[k] = v
        checkpoint['diffusion_params'] = cpu_params

    torch.save(checkpoint, CHECKPOINT_PATH)
    status = "⏸️ 暂停保存" if is_pause else "💾 检查点保存"
    print(f"\n  {status}到: {CHECKPOINT_PATH} (epoch {epoch})")
    return CHECKPOINT_PATH


def load_checkpoint(model, optimizer=None, scheduler=None, ema_model=None,
                    device='cuda', checkpoint_path=None):
    """
    加载训练检查点，恢复训练状态
    返回: (start_epoch, checkpoint_data) 或 (1, None) 如果没有检查点
    """
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_PATH

    if not os.path.exists(checkpoint_path):
        print(f"\n⚠️ 未找到检查点文件: {checkpoint_path}")
        print("   将从epoch 1开始全新训练")
        return 1, None

    print(f"\n📂 加载检查点: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✅ 模型权重加载成功")

        # 加载优化器状态
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"  ✅ 优化器状态加载成功")

        # 加载调度器状态
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"  ✅ 学习率调度器状态加载成功")

        # 加载EMA模型
        if ema_model is not None and 'ema_state_dict' in checkpoint:
            ema_model.load_state_dict(checkpoint['ema_state_dict'])
            print(f"  ✅ EMA模型权重加载成功")

        start_epoch = checkpoint.get('epoch', 0) + 1
        is_pause = checkpoint.get('is_pause', False)

        print(f"  📊 恢复信息:")
        print(f"     - 恢复epoch: {checkpoint.get('epoch', 0)}")
        print(f"     - 将从epoch {start_epoch}继续训练")
        print(f"     - 暂停状态: {'是' if is_pause else '否'}")
        if 'best_val_loss' in checkpoint:
            print(f"     - 最佳验证损失: {checkpoint['best_val_loss']:.6f}")
        if 'best_snr_improvement' in checkpoint:
            print(f"     - 最佳SNR提升: {checkpoint['best_snr_improvement']:.2f} dB")

        return start_epoch, checkpoint

    except Exception as e:
        print(f"  ❌ 加载检查点失败: {e}")
        print(f"    将从头开始训练")
        return 1, None


# -----------------------------
# 8. 早停类
# -----------------------------
class EarlyStopping:
    """早停基类"""
    def __init__(self, patience=100, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = None
        self.mode = 'min'  # 'min' 表示越小越好, 'max' 表示越大越好

    def check(self, current_value):
        if self.best_value is None:
            self.best_value = current_value
            return False

        if self.mode == 'min':
            improved = current_value < self.best_value - self.min_delta
        else:  # 'max' 模式
            improved = current_value > self.best_value + self.min_delta

        if improved:
            self.best_value = current_value
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class LossEarlyStopping(EarlyStopping):
    """基于验证损失的早停（越小越好）"""
    def __init__(self, patience=50, min_delta=1e-4):
        super().__init__(patience=patience, min_delta=min_delta)
        self.mode = 'min'
        self.name = 'Val_Loss'



# -----------------------------
# 8. 可视化去噪结果
# -----------------------------
def visualize_denoising(model, test_loader, params, device,
                        num_samples=5, save_path="./denoising_results.png"):
    """
    抽取num_samples个样本，对比去噪前后的信号
    """
    model.eval()

    # 评估时间步（中等噪声水平）
    eval_t = 200

    # 存储结果
    results = []

    with torch.no_grad():
        samples_collected = 0
        for batch_clean in test_loader:
            batch_clean = batch_clean.to(device)

            for i in range(batch_clean.shape[0]):
                if samples_collected >= num_samples:
                    break
                clean_sample = batch_clean[i:i + 1]

                # 经典DDPM加噪公式
                alpha_bar = params['alphas_cumprod'][eval_t].view(-1, 1, 1)
                noise = torch.randn_like(clean_sample)
                noisy = torch.sqrt(alpha_bar) * clean_sample + torch.sqrt(1 - alpha_bar) * noise

                # 去噪
                denoised = denoise_for_eval(model, noisy, eval_t, params, device)

                # 计算SNR
                input_snr = calculate_snr(clean_sample[0].cpu(), noisy[0].cpu())
                output_snr = calculate_snr(clean_sample[0].cpu(), denoised[0].cpu())

                results.append({
                    'clean': clean_sample[0, 0].cpu().numpy(),
                    'noisy': noisy[0, 0].cpu().numpy(),
                    'denoised': denoised[0, 0].cpu().numpy(),
                    'input_snr': input_snr,
                    'output_snr': output_snr,
                    'improvement': output_snr - input_snr
                })

                samples_collected += 1

            if samples_collected >= num_samples:
                break

    # 绘制结果
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 3 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for idx, res in enumerate(results):
        # 只显示前2000个点
        clean_plot = res['clean'][:2000]
        noisy_plot = res['noisy'][:2000]
        denoised_plot = res['denoised'][:2000]

        # 干净信号
        axes[idx, 0].plot(clean_plot, 'g-', linewidth=1)
        axes[idx, 0].set_title(f'Sample {idx + 1}: Clean Signal')
        axes[idx, 0].set_xlabel('Sample')
        axes[idx, 0].set_ylabel('Amplitude')
        axes[idx, 0].grid(True, alpha=0.3)

        # 带噪信号
        axes[idx, 1].plot(noisy_plot, 'r-', linewidth=1)
        axes[idx, 1].set_title(f'Sample {idx + 1}: Noisy Signal (SNR={res["input_snr"]:.2f}dB)')
        axes[idx, 1].set_xlabel('Sample')
        axes[idx, 1].set_ylabel('Amplitude')
        axes[idx, 1].grid(True, alpha=0.3)

        # 去噪后信号
        axes[idx, 2].plot(denoised_plot, 'b-', linewidth=1)
        axes[idx, 2].set_title(
            f'Sample {idx + 1}: Denoised Signal (SNR={res["output_snr"]:.2f}dB, Imp={res["improvement"]:+.2f}dB)')
        axes[idx, 2].set_xlabel('Sample')
        axes[idx, 2].set_ylabel('Amplitude')
        axes[idx, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"\n✅ 去噪结果可视化已保存到: {save_path}")

    # 打印统计信息
    print("\n" + "=" * 60)
    print("去噪结果统计（5个样本平均）")
    print("=" * 60)
    avg_input_snr = np.mean([r['input_snr'] for r in results])
    avg_output_snr = np.mean([r['output_snr'] for r in results])
    avg_improvement = np.mean([r['improvement'] for r in results])
    print(f"  平均输入SNR: {avg_input_snr:.2f} dB")
    print(f"  平均输出SNR: {avg_output_snr:.2f} dB")
    print(f"  平均SNR提升: +{avg_improvement:.2f} dB")
    print("=" * 60)

    return results


# -----------------------------
# 9. 训练函数
# -----------------------------
def train(
        clean_path="uwb_signals_time_clean.npy",
        batchsize=4,
        total_epoch=10,
        val_ratio=0.1,
        test_ratio=0.1,
        lr=2e-4,
        beta_start=0.0001,
        beta_end=0.02,
        timesteps=1000,
        use_cosine_schedule=True,
        gradient_clip=1.0,
        use_ema=True,
        ema_beta=0.995,
        save_every=10,
        device='cuda',
        snr_eval_freq=10,
        lambda_mse=2.0,
        lambda_rel=0.5,
        dropout_rate=0.1,
        num_workers=4,  # 新增参数，默认4个工作进程
        memory_check_freq=10,  # 内存检查频率（每N个epoch检查一次）
        resume_from=None  # 从检查点恢复训练，传入检查点路径或True（使用默认路径）
):
    device = torch.device(device)

    # 初始化内存监控器
    memory_monitor = MemoryMonitor(device=device, memory_warning_threshold=85)

    print("=" * 60)
    print("经典DDPM训练（加性噪声 + 混合损失）")
    print(f"损失权重: MSE={lambda_mse}, RelMSE={lambda_rel}")
    print(f"Dropout率: {dropout_rate}")
    print(f"EMA激活步数: 100")
    print(f"学习率调度: CosineAnnealingLR (T_max=500, eta_min=1e-6)")

    print(f"DataLoader工作进程数: {num_workers}")  # 显示工作进程数
    print(f"内存检查频率: 每{memory_check_freq}个epoch")
    print("=" * 60)

    # 检查文件
    print(f"\n检查数据文件...")
    print(f"  干净数据: {clean_path}")

    if not os.path.exists(clean_path):
        print(f"❌ 错误：找不到干净数据文件!")
        return None, None
    print("✅ 文件存在")

    # 分析数据集大小
    data_size_gb = check_data_worker_memory(clean_path)
    if data_size_gb and num_workers > 0:
        estimated_memory = data_size_gb * (num_workers + 1) * 1.2
        system_ram = psutil.virtual_memory().total / (1024 ** 3)
        if estimated_memory > system_ram * 0.9:
            print(f"\n⚠️ 警告: 当前num_workers={num_workers}可能导致内存不足!")
            print(f"   预估内存需求: {estimated_memory:.1f} GB")
            print(f"   系统总内存: {system_ram:.1f} GB")
            print(f"   建议降低num_workers或减小batchsize")

    # 加载数据（无变换）
    print("\n加载数据...")

    train_dataset = Dataset_UWB(
        clean_path=clean_path,
        split='train',
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        pad_size=128
    )

    norm_stats = train_dataset.get_norm_stats()

    val_dataset = Dataset_UWB(
        clean_path=clean_path,
        split='val',
        norm_stats=norm_stats,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        pad_size=128
    )

    test_dataset = Dataset_UWB(
        clean_path=clean_path,
        split='test',
        norm_stats=norm_stats,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        pad_size=128
    )

    # 修改点1: train_loader 增加 num_workers 和 pin_memory
    train_loader = DataLoader(
        train_dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=num_workers,  # 使用传入的num_workers参数
        pin_memory=True,  # 加速CPU到GPU的数据传输
        persistent_workers=True if num_workers > 0 else False  # 复用worker进程
    )

    # 修改点2: val_loader 增加 num_workers 和 pin_memory
    val_loader = DataLoader(
        val_dataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=num_workers,  # 使用传入的num_workers参数
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    # 修改点3: test_loader 增加 num_workers 和 pin_memory
    test_loader = DataLoader(
        test_dataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=num_workers,  # 使用传入的num_workers参数
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    print(f"\n数据加载器:")
    print(f"  训练集: {len(train_dataset)} 样本, {len(train_loader)} batches")
    print(f"  验证集: {len(val_dataset)} 样本, {len(val_loader)} batches")
    print(f"  测试集: {len(test_dataset)} 样本, {len(test_loader)} batches")

    # 打印初始内存状态
    memory_monitor.print_memory_status(0, "数据加载完成")

    # 初始化模型
    print("\n初始化模型...")
    model = SimpleUNet1D_Classic(in_channels=1, out_channels=1, dropout_rate=dropout_rate).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,}")

    if use_ema:
        ema_model = copy.deepcopy(model)
        ema = EMA(beta=ema_beta)
        print("✅ EMA已启用")
    else:
        ema_model = None
        print("⚠️ EMA已禁用")

    # 优化器 - 添加权重衰减正则化
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # 学习率调度器：初始时不启动，当损失下降小于0.1时才开始
    scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
    scheduler_started = False  # 标记调度器是否已启动
    prev_val_loss = None       # 上一轮的验证损失


    # Beta调度
    if use_cosine_schedule:
        betas = cosine_beta_schedule(timesteps)
        print("\n使用余弦beta调度")
    else:
        betas = linear_beta_schedule(beta_start, beta_end, timesteps)
        print(f"\n使用线性beta调度")

    betas = betas.to(device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    params = compute_diffusion_params(betas, device)

    # 训练准备
    os.makedirs("./saved_models_classic", exist_ok=True)
    os.makedirs("./logs_classic", exist_ok=True)

    log_file = open("./logs_classic/training_log.txt", "w", encoding='utf-8')
    log_file.write(
        "Epoch,Train_Loss,Train_MSE,Train_RelMSE,Train_Peak,Train_Corr,"
        "Val_Loss,Val_MSE,Val_RelMSE,Val_Peak,Val_Corr,"
        "LR,Input_SNR,Output_SNR,SNR_Improvement,Memory_Percent\n")

    # 创建图表（交互模式，每一轮更新数据，每10轮刷新一次窗口）
    plt.ion()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.canvas.draw()
    fig.canvas.flush_events()

    train_losses = []
    val_losses = []
    lr_history = []
    snr_improvements = []
    snr_epochs = []
    epochs = []
    memory_usages = []  # 记录内存使用

    line_train, = ax1.plot([], [], 'b-', label='Train')
    line_val, = ax1.plot([], [], 'r-', label='Validation')
    ax1.set_title("Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    lr_line, = ax2.plot([], [], 'g-')
    ax2.set_title("Learning Rate")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("LR")
    ax2.grid(True)

    snr_line, = ax3.plot([], [], 'm-', linewidth=2)
    ax3.set_title("SNR Improvement")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("SNR Improvement (dB)")
    ax3.grid(True)

    # 添加内存使用图表
    memory_line, = ax4.plot([], [], 'r-', linewidth=2)
    ax4.set_title("System Memory Usage")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Memory Usage (%)")
    ax4.set_ylim(0, 100)
    ax4.grid(True)
    ax4.axhline(y=85, color='orange', linestyle='--', label='Warning Threshold')
    ax4.axhline(y=95, color='red', linestyle='--', label='Critical Threshold')
    ax4.legend()

    plt.tight_layout()

    # ========== 检查点恢复逻辑 ==========
    start_epoch = 1
    best_val_loss = float('inf')
    best_snr_improvement = -float('inf')
    best_epoch = -1
    loss_early_stopping = LossEarlyStopping(patience=20, min_delta=0.05)

    # 如果指定了resume_from，尝试从检查点恢复
    if resume_from is not None:
        checkpoint_path = resume_from if isinstance(resume_from, str) else None
        loaded_epoch, checkpoint_data = load_checkpoint(
            model, optimizer, scheduler, ema_model,
            device=device, checkpoint_path=checkpoint_path
        )

        if checkpoint_data is not None:
            start_epoch = loaded_epoch

            # 恢复历史数据
            train_losses = checkpoint_data.get('train_losses', [])
            val_losses = checkpoint_data.get('val_losses', [])
            lr_history = checkpoint_data.get('lr_history', [])
            epochs = checkpoint_data.get('epochs', [])
            snr_improvements = checkpoint_data.get('snr_improvements', [])
            snr_epochs = checkpoint_data.get('snr_epochs', [])
            memory_usages = checkpoint_data.get('memory_usages', [])

            # 恢复最佳值
            best_val_loss = checkpoint_data.get('best_val_loss', float('inf'))
            best_snr_improvement = checkpoint_data.get('best_snr_improvement', -float('inf'))
            best_epoch = checkpoint_data.get('best_epoch', -1)

            # 恢复早停状态
            es_loss = checkpoint_data.get('loss_early_stopping')
            if es_loss is not None:
                loss_early_stopping.counter = es_loss.get('counter', 0)
                loss_early_stopping.best_value = es_loss.get('best_value')

            # 恢复EMA步数
            if use_ema and 'ema_state_dict' in checkpoint_data:
                # EMA的step需要根据已训练的epoch数来估算
                ema.step = max(100, start_epoch * len(train_loader))

            print(f"\n✅ 检查点恢复完成！将从epoch {start_epoch}继续训练")
            print(f"   已记录 {len(epochs)} 个epoch的训练历史")
        else:
            print("   检查点加载失败或不存在，将从头开始训练")

    # 训练循环
    print("\n开始训练...")
    print("=" * 60)
    print("💡 提示: 按 Ctrl+C 可暂停训练并保存检查点，下次运行加 resume=True 继续")
    print("=" * 60)

    for epoch in range(start_epoch, total_epoch + 1):
        model.train()
        epoch_train_loss = 0
        epoch_train_mse = 0
        epoch_train_rel = 0
        epoch_train_peak = 0
        epoch_train_corr = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch:3d} [Train]")

        for batch_clean in train_bar:
            batch_clean = batch_clean.to(device, non_blocking=True)  # 使用 non_blocking 加速

            t = torch.randint(0, timesteps, (batch_clean.shape[0],), device=device)

            # 经典DDPM加噪公式
            alpha_bar = alphas_cumprod[t].view(-1, 1, 1)
            noise = torch.randn_like(batch_clean)
            x_t = torch.sqrt(alpha_bar) * batch_clean + torch.sqrt(1 - alpha_bar) * noise

            # 模型预测噪声
            pred_noise = model(x_t, t)

            # 从预测噪声恢复干净信号
            alpha_bar_t = alphas_cumprod[t].view(-1, 1, 1)
            pred_clean = (x_t - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
            pred_clean = torch.clamp(pred_clean, -1, 1)

            # 注意：训练阶段不做极性校正（保持计算图完整）
            # 极性校正仅在验证/推理阶段进行
            # 损失函数中的 peak_correlation_loss 已包含极性惩罚项

            # 组合损失（使用用户指定的权重）
            total_loss, mse_loss_val, rel_loss_val, peak_loss_val, corr_loss_val = combined_loss(
                pred_clean, batch_clean,
                lambda_mse=lambda_mse,
                lambda_rel=lambda_rel,
                lambda_peak=1.8,
                lambda_corr=1.8
            )

            optimizer.zero_grad()
            total_loss.backward()

            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()

            if use_ema:
                ema.step_ema(ema_model, model)

            epoch_train_loss += total_loss.item()
            epoch_train_mse += mse_loss_val.item()
            epoch_train_rel += rel_loss_val.item()
            epoch_train_peak += peak_loss_val.item()
            epoch_train_corr += corr_loss_val.item()
            train_bar.set_postfix({"loss": f"{total_loss.item():.6f}", "peak": f"{peak_loss_val.item():.4f}", "corr": f"{corr_loss_val.item():.4f}"})

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_train_mse = epoch_train_mse / len(train_loader)
        avg_train_rel = epoch_train_rel / len(train_loader)
        avg_train_peak = epoch_train_peak / len(train_loader)
        avg_train_corr = epoch_train_corr / len(train_loader)

        # 验证阶段
        model.eval()
        epoch_val_loss = 0
        epoch_val_mse = 0
        epoch_val_rel = 0
        epoch_val_peak = 0
        epoch_val_corr = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch:3d} [Val]")

        with torch.no_grad():
            set_seed(42 + epoch)

            for batch_clean in val_bar:
                batch_clean = batch_clean.to(device, non_blocking=True)  # 使用 non_blocking 加速

                t = torch.randint(0, timesteps, (batch_clean.shape[0],), device=device)
                alpha_bar = alphas_cumprod[t].view(-1, 1, 1)
                noise = torch.randn_like(batch_clean)
                x_t = torch.sqrt(alpha_bar) * batch_clean + torch.sqrt(1 - alpha_bar) * noise

                if use_ema and ema_model is not None:
                    pred_noise = ema_model(x_t, t)
                else:
                    pred_noise = model(x_t, t)

                alpha_bar_t = alphas_cumprod[t].view(-1, 1, 1)
                pred_clean = (x_t - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
                pred_clean = torch.clamp(pred_clean, -1, 1)

                # 注意：BPSK信号本身有正有负，无法从信号本身判断极性是否反了
                # 因此不在验证时做极性校正，而是通过训练时的损失函数来防止极性翻转
                total_loss, mse_loss_val, rel_loss_val, peak_loss_val, corr_loss_val = combined_loss(
                    pred_clean, batch_clean,
                    lambda_mse=lambda_mse,
                    lambda_rel=lambda_rel,
                    lambda_peak=1.8,
                    lambda_corr=1.8
                )

                epoch_val_loss += total_loss.item()
                epoch_val_mse += mse_loss_val.item()
                epoch_val_rel += rel_loss_val.item()
                epoch_val_peak += peak_loss_val.item()
                epoch_val_corr += corr_loss_val.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_mse = epoch_val_mse / len(val_loader)
        avg_val_rel = epoch_val_rel / len(val_loader)
        avg_val_peak = epoch_val_peak / len(val_loader)
        avg_val_corr = epoch_val_corr / len(val_loader)

        # 学习率调度：当损失下降小于0.1时才开始启动调度器
        if not scheduler_started:
            if prev_val_loss is not None:
                loss_drop = prev_val_loss - avg_val_loss
                if loss_drop < 0.1:  # 损失下降小于0.1，启动调度器
                    scheduler_started = True
                    print(f"  [LR Scheduler] 损失下降({loss_drop:.4f}) < 0.1，启动学习率调度")
            prev_val_loss = avg_val_loss
        
        if scheduler_started:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        lr_history.append(current_lr)
        epochs.append(epoch)

        # 内存监控
        if epoch % memory_check_freq == 0 or epoch == 1:
            mem_percent = memory_monitor.print_memory_status(epoch, f"Epoch {epoch} 完成")
            memory_usages.append(mem_percent)

            # 更新内存图表
            if len(memory_usages) > 0:
                memory_line.set_xdata(range(1, len(memory_usages) + 1))
                memory_line.set_ydata(memory_usages)
                ax4.relim()
                ax4.autoscale_view()

            # 如果内存超过90%，强制垃圾回收
            if mem_percent > 90:
                print(f"  ⚠️ 内存使用率过高({mem_percent:.1f}%)，执行强制垃圾回收...")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                memory_monitor.print_memory_status(epoch, "垃圾回收后")

        # SNR评估
        avg_input_snr = 0
        avg_output_snr = 0
        avg_snr_improvement = 0

        if epoch % snr_eval_freq == 0 and epoch > 0:
            print(f"\n  计算SNR指标...")

            model_for_eval = ema_model if (use_ema and ema_model is not None) else model

            input_snrs = []
            output_snrs = []
            improvements = []

            eval_ts = [100, 150,200,250,300]
            num_eval_samples = 5

            with torch.no_grad():
                set_seed(100 + epoch)

                samples_evaluated = 0
                for batch_clean in val_loader:
                    if samples_evaluated >= num_eval_samples:
                        break

                    batch_clean = batch_clean.to(device, non_blocking=True)

                    for i in range(min(batch_clean.shape[0], num_eval_samples - samples_evaluated)):
                        clean_sample = batch_clean[i:i + 1]

                        for eval_t in eval_ts:
                            alpha_bar = alphas_cumprod[eval_t].view(-1, 1, 1)
                            noise = torch.randn_like(clean_sample)
                            noisy = torch.sqrt(alpha_bar) * clean_sample + torch.sqrt(1 - alpha_bar) * noise

                            denoised = denoise_for_eval(model_for_eval, noisy, eval_t, params, device)

                            input_snr = calculate_snr(clean_sample[0].cpu(), noisy[0].cpu())
                            output_snr = calculate_snr(clean_sample[0].cpu(), denoised[0].cpu())
                            improvement = output_snr - input_snr

                            input_snrs.append(input_snr)
                            output_snrs.append(output_snr)
                            improvements.append(improvement)

                        samples_evaluated += 1

            avg_input_snr = np.mean(input_snrs) if input_snrs else 0
            avg_output_snr = np.mean(output_snrs) if output_snrs else 0
            avg_snr_improvement = np.mean(improvements) if improvements else 0

            snr_improvements.append(avg_snr_improvement)
            snr_epochs.append(epoch)

            if len(snr_epochs) > 0:
                snr_line.set_xdata(snr_epochs)
                snr_line.set_ydata(snr_improvements)
                ax3.relim()
                ax3.autoscale_view()

        # 写入日志（包含峰值指标和内存使用）
        current_mem = memory_monitor.get_system_memory_usage()
        log_file.write(
            f"{epoch},{avg_train_loss:.6f},{avg_train_mse:.6f},{avg_train_rel:.6f},"
            f"{avg_train_peak:.6f},{avg_train_corr:.6f},"
            f"{avg_val_loss:.6f},{avg_val_mse:.6f},{avg_val_rel:.6f},"
            f"{avg_val_peak:.6f},{avg_val_corr:.6f},"
            f"{current_lr:.2e},{avg_input_snr:.2f},{avg_output_snr:.2f},{avg_snr_improvement:.2f},{current_mem:.1f}\n")
        log_file.flush()

        # 更新图表（每一轮都在内存中更新数据）
        line_train.set_xdata(epochs)
        line_train.set_ydata(train_losses)
        line_val.set_xdata(epochs)
        line_val.set_ydata(val_losses)
        ax1.relim()
        ax1.autoscale_view()

        lr_line.set_xdata(epochs)
        lr_line.set_ydata(lr_history)
        ax2.relim()
        ax2.autoscale_view()

        # 每snr_eval_freq轮刷新一次绘图窗口（降低弹窗频率）
        if epoch % snr_eval_freq == 0:
            plt.pause(0.01)

        # 打印进度
        print(f"\nEpoch {epoch:3d}:")
        print(f"  Train Loss: {avg_train_loss:.6f} (MSE={avg_train_mse:.6f}, RelMSE={avg_train_rel:.6f}, Peak={avg_train_peak:.4f}, Corr={avg_train_corr:.4f})")
        print(f"  Val Loss:   {avg_val_loss:.6f} (MSE={avg_val_mse:.6f}, RelMSE={avg_val_rel:.6f}, Peak={avg_val_peak:.4f}, Corr={avg_val_corr:.4f})")
        print(f"  LR:         {current_lr:.2e}")
        if avg_snr_improvement != 0:
            print(f"  Input SNR:  {avg_input_snr:.2f} dB")
            print(f"  Output SNR: {avg_output_snr:.2f} dB")
            print(f"  SNR Imp:    {avg_snr_improvement:+.2f} dB")
        print(f"  Memory:     {current_mem:.1f}%")

    # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict() if use_ema else None,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'snr_improvement': avg_snr_improvement,
            }, "./saved_models_classic/best_model_by_loss.pth")
            print(f"  ✅ 保存最佳模型 (loss={avg_val_loss:.6f})")

        if avg_snr_improvement > best_snr_improvement:
            best_snr_improvement = avg_snr_improvement

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict() if use_ema else None,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'snr_improvement': avg_snr_improvement,
            }, "./saved_models_classic/best_model_by_snr.pth")
            print(f"  ✅ 保存最佳SNR模型 (SNR提升={avg_snr_improvement:.2f}dB)")

        if epoch % save_every == 0 or epoch == total_epoch:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict() if use_ema else None,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'snr_improvement': avg_snr_improvement,
            }, f"./saved_models_classic/model_epoch{epoch}.pth")

        # 早停检查
        if loss_early_stopping.check(avg_val_loss):
            print(f"\n⚠️ 早停于 epoch {epoch}")
            print(f"  原因: Val_Loss 连续 {loss_early_stopping.patience} 轮未改善 (最佳: {loss_early_stopping.best_value:.6f})")
            break


    plt.ioff()
    plt.savefig("./logs_classic/training_curves.png", dpi=150)
    log_file.close()

    # 打印最终内存统计
    print("\n" + "=" * 60)
    print("训练完成！内存统计:")
    peak_memory = memory_monitor.get_peak_memory()
    print(f"  峰值系统内存: {peak_memory['peak_system_memory']:.1f}%")
    if peak_memory['peak_gpu_memory'] > 0:
        print(f"  峰值GPU内存: {peak_memory['peak_gpu_memory']:.2f} GB")
    print(f"最佳验证损失: {best_val_loss:.6f} (epoch {best_epoch})")
    print(f"最佳SNR提升: {best_snr_improvement:.2f} dB")
    print("=" * 60)

    # 加载最佳模型进行综合评估与分析
    print("\n" + "=" * 60)
    print("加载最佳模型，进行综合评估与分析...")
    print("=" * 60)
    best_model = SimpleUNet1D_Classic(in_channels=1, out_channels=1, dropout_rate=dropout_rate).to(device)
    checkpoint = torch.load("./saved_models_classic/best_model_by_loss.pth", map_location=device)
    best_model.load_state_dict(checkpoint['model_state_dict'])
    best_model.eval()

    # ---- 对全部测试集进行去噪评估 ----
    eval_t = 300
    result_data = {
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'best_snr_improvement': best_snr_improvement,
        'clean_signals': [],
        'noisy_signals': [],
        'denoised_signals': [],
        'input_snrs': [],
        'output_snrs': [],
        'eval_t': eval_t,
    }

    all_mse = []
    all_correlation = []

    with torch.no_grad():
        set_seed(42)
        for batch_clean in test_loader:
            batch_clean = batch_clean.to(device, non_blocking=True)
            for i in range(batch_clean.shape[0]):
                clean_sample = batch_clean[i:i + 1]

                alpha_bar = alphas_cumprod[eval_t].view(-1, 1, 1)
                noise = torch.randn_like(clean_sample)
                noisy = torch.sqrt(alpha_bar) * clean_sample + torch.sqrt(1 - alpha_bar) * noise

                denoised = denoise_for_eval(best_model, noisy, eval_t, params, device)

                input_snr = calculate_snr(clean_sample[0].cpu(), noisy[0].cpu())
                output_snr = calculate_snr(clean_sample[0].cpu(), denoised[0].cpu())

                clean_np = clean_sample[0, 0].cpu().numpy()
                denoised_np = denoised[0, 0].cpu().numpy()

                result_data['clean_signals'].append(clean_np)
                result_data['noisy_signals'].append(noisy[0, 0].cpu().numpy())
                result_data['denoised_signals'].append(denoised_np)
                result_data['input_snrs'].append(input_snr)
                result_data['output_snrs'].append(output_snr)

                # 计算 MSE 和相关系数
                mse_val = np.mean((clean_np - denoised_np) ** 2)
                all_mse.append(mse_val)
                corr_val = np.corrcoef(clean_np, denoised_np)[0, 1]
                all_correlation.append(corr_val)

    result_data['clean_signals'] = np.array(result_data['clean_signals'])
    result_data['noisy_signals'] = np.array(result_data['noisy_signals'])
    result_data['denoised_signals'] = np.array(result_data['denoised_signals'])
    result_data['input_snrs'] = np.array(result_data['input_snrs'])
    result_data['output_snrs'] = np.array(result_data['output_snrs'])
    all_mse = np.array(all_mse)
    all_correlation = np.array(all_correlation)
    snr_improvements = result_data['output_snrs'] - result_data['input_snrs']

    num_samples = len(result_data['clean_signals'])

    # 保存 result.npy
    result_save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "result.npy")
    np.save(result_save_path, result_data)
    print(f"\n  ✅ result.npy 已保存到 {result_save_path}")

    # ---- 综合分析 ----
    print("\n" + "=" * 60)
    print("📊 最佳模型 (Epoch {}) 综合评估分析报告".format(best_epoch))
    print("=" * 60)

    print(f"\n[1] 基本参数")
    print(f"  最佳Epoch: {best_epoch}")
    print(f"  验证损失: {best_val_loss:.6f}")
    print(f"  评估时间步: t={eval_t}")
    print(f"  测试样本数: {num_samples}")
    print(f"  信号长度: {result_data['clean_signals'].shape[1]}")

    print(f"\n[2] SNR 分析 (dB)")
    print(f"  输入SNR  - 均值: {np.mean(result_data['input_snrs']):.2f}  |  中位数: {np.median(result_data['input_snrs']):.2f}  |  标准差: {np.std(result_data['input_snrs']):.2f}")
    print(f"  输出SNR  - 均值: {np.mean(result_data['output_snrs']):.2f}  |  中位数: {np.median(result_data['output_snrs']):.2f}  |  标准差: {np.std(result_data['output_snrs']):.2f}")
    print(f"  SNR提升  - 均值: {np.mean(snr_improvements):.2f}  |  中位数: {np.median(snr_improvements):.2f}  |  标准差: {np.std(snr_improvements):.2f}")
    print(f"            |  最小: {np.min(snr_improvements):.2f}  |  最大: {np.max(snr_improvements):.2f}")
    pos_count = np.sum(snr_improvements > 0)
    neg_count = np.sum(snr_improvements < 0)
    print(f"  SNR提升>0 样本: {pos_count}/{num_samples} ({pos_count/num_samples*100:.1f}%)")
    print(f"  SNR提升<0 样本: {neg_count}/{num_samples} ({neg_count/num_samples*100:.1f}%)")

    print(f"\n[3] MSE 分析")
    print(f"  MSE   - 均值: {np.mean(all_mse):.6f}  |  中位数: {np.median(all_mse):.6f}  |  标准差: {np.std(all_mse):.6f}")
    print(f"         |  最小: {np.min(all_mse):.6f}  |  最大: {np.max(all_mse):.6f}")

    print(f"\n[4] 相关系数分析")
    print(f"  相关系数 - 均值: {np.mean(all_correlation):.4f}  |  中位数: {np.median(all_correlation):.4f}  |  标准差: {np.std(all_correlation):.4f}")
    print(f"           |  最小: {np.min(all_correlation):.4f}  |  最大: {np.max(all_correlation):.4f}")
    high_corr = np.sum(all_correlation > 0.9)
    mid_corr = np.sum((all_correlation > 0.7) & (all_correlation <= 0.9))
    low_corr = np.sum(all_correlation <= 0.7)
    print(f"  相关系数 > 0.9: {high_corr}/{num_samples} ({high_corr/num_samples*100:.1f}%)")
    print(f"  相关系数 0.7~0.9: {mid_corr}/{num_samples} ({mid_corr/num_samples*100:.1f}%)")
    print(f"  相关系数 < 0.7: {low_corr}/{num_samples} ({low_corr/num_samples*100:.1f}%)")

    # SNR 分布区间统计
    print(f"\n[5] SNR提升分布")
    bins = [(-float('inf'), -5), (-5, 0), (0, 5), (5, 10), (10, 20), (20, float('inf'))]
    for lo, hi in bins:
        if lo == -float('inf'):
            label = f"  < {hi} dB"
            count = np.sum(snr_improvements < hi)
        elif hi == float('inf'):
            label = f"  >= {lo} dB"
            count = np.sum(snr_improvements >= lo)
        else:
            label = f"  {lo} ~ {hi} dB"
            count = np.sum((snr_improvements >= lo) & (snr_improvements < hi))
        print(f"{label}: {count}/{num_samples} ({count/num_samples*100:.1f}%)")

    # 输出SNR分布区间
    print(f"\n[6] 输出SNR分布")
    bins = [(-float('inf'), 0), (0, 5), (5, 10), (10, 15), (15, 20), (20, float('inf'))]
    for lo, hi in bins:
        if lo == -float('inf'):
            label = f"  < {hi} dB"
            count = np.sum(result_data['output_snrs'] < hi)
        elif hi == float('inf'):
            label = f"  >= {lo} dB"
            count = np.sum(result_data['output_snrs'] >= lo)
        else:
            label = f"  {lo} ~ {hi} dB"
            count = np.sum((result_data['output_snrs'] >= lo) & (result_data['output_snrs'] < hi))
        print(f"{label}: {count}/{num_samples} ({count/num_samples*100:.1f}%)")

    # 保存分析报告到文本文件
    report_path = "./logs_classic/analysis_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"最佳模型 (Epoch {best_epoch}) 综合评估分析报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"[1] 基本参数\n")
        f.write(f"  最佳Epoch: {best_epoch}\n")
        f.write(f"  验证损失: {best_val_loss:.6f}\n")
        f.write(f"  评估时间步: t={eval_t}\n")
        f.write(f"  测试样本数: {num_samples}\n\n")
        f.write(f"[2] SNR 分析 (dB)\n")
        f.write(f"  输入SNR  - 均值: {np.mean(result_data['input_snrs']):.2f}  中位数: {np.median(result_data['input_snrs']):.2f}  标准差: {np.std(result_data['input_snrs']):.2f}\n")
        f.write(f"  输出SNR  - 均值: {np.mean(result_data['output_snrs']):.2f}  中位数: {np.median(result_data['output_snrs']):.2f}  标准差: {np.std(result_data['output_snrs']):.2f}\n")
        f.write(f"  SNR提升  - 均值: {np.mean(snr_improvements):.2f}  中位数: {np.median(snr_improvements):.2f}  标准差: {np.std(snr_improvements):.2f}\n")
        f.write(f"           最小: {np.min(snr_improvements):.2f}  最大: {np.max(snr_improvements):.2f}\n")
        f.write(f"  SNR提升>0: {pos_count}/{num_samples} ({pos_count/num_samples*100:.1f}%)\n")
        f.write(f"  SNR提升<0: {neg_count}/{num_samples} ({neg_count/num_samples*100:.1f}%)\n\n")
        f.write(f"[3] MSE 分析\n")
        f.write(f"  MSE - 均值: {np.mean(all_mse):.6f}  中位数: {np.median(all_mse):.6f}  标准差: {np.std(all_mse):.6f}\n")
        f.write(f"       最小: {np.min(all_mse):.6f}  最大: {np.max(all_mse):.6f}\n\n")
        f.write(f"[4] 相关系数分析\n")
        f.write(f"  相关系数 - 均值: {np.mean(all_correlation):.4f}  中位数: {np.median(all_correlation):.4f}  标准差: {np.std(all_correlation):.4f}\n")
        f.write(f"           最小: {np.min(all_correlation):.4f}  最大: {np.max(all_correlation):.4f}\n")
        f.write(f"  > 0.9: {high_corr}/{num_samples} ({high_corr/num_samples*100:.1f}%)\n")
        f.write(f"  0.7~0.9: {mid_corr}/{num_samples} ({mid_corr/num_samples*100:.1f}%)\n")
        f.write(f"  < 0.7: {low_corr}/{num_samples} ({low_corr/num_samples*100:.1f}%)\n\n")
        f.write(f"[5] 逐样本详细数据\n")
        for i in range(num_samples):
            f.write(f"  样本{i:3d}: 输入SNR={result_data['input_snrs'][i]:.2f}  输出SNR={result_data['output_snrs'][i]:.2f}  "
                    f"SNR提升={snr_improvements[i]:+.2f}  MSE={all_mse[i]:.6f}  相关系数={all_correlation[i]:.4f}\n")
    print(f"\n  ✅ 分析报告已保存到 {report_path}")

    # 可视化去噪结果
    print(f"\n[7] 去噪可视化")
    visualize_denoising(
        model=best_model,
        test_loader=test_loader,
        params=params,
        device=device,
        num_samples=10,
        save_path="./denoising_results_classic.png"
    )

    print("\n" + "=" * 60)
    print("✅ 全部训练及评估完成！")
    print("=" * 60)

    return model, ema_model


# -----------------------------
# 主函数
# -----------------------------

if __name__ == "__main__":
    print("启动经典DDPM训练...")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")

    # ========== 关键参数设置 ==========
    LAMBDA_MSE = 2.5  # MSE权重
    LAMBDA_REL = 0.4  # 相对MSE权重
    DROPOUT_RATE = 0.1  # Dropout率
    TOTAL_EPOCH = 500 # 总训练轮数
    NUM_WORKERS = 1  # DataLoader工作进程数（根据内存调整，建议2-4）
    MEMORY_CHECK_FREQ = 10  # 每10个epoch检查一次内存
    # =================================

    print("\n" + "=" * 60)
    print("经典DDPM训练（加性噪声 + 混合损失）")
    print(f"损失权重: MSE={LAMBDA_MSE}, RelMSE={LAMBDA_REL}")
    print(f"Dropout率: {DROPOUT_RATE}")
    print(f"总训练轮数: {TOTAL_EPOCH}")
    print(f"DataLoader工作进程数: {NUM_WORKERS}")
    print("=" * 60)

    model, ema_model = train(
        clean_path="uwb_signals_time_clean.npy",
        batchsize=2,
        total_epoch=TOTAL_EPOCH,
        val_ratio=0.1,
        test_ratio=0.1,
        lr=2e-4,
        timesteps=1000,
        use_cosine_schedule=True,
        gradient_clip=1.0,
        use_ema=True,
        save_every=10,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        snr_eval_freq=10,
        lambda_mse=LAMBDA_MSE,
        lambda_rel=LAMBDA_REL,
        dropout_rate=DROPOUT_RATE,
        num_workers=NUM_WORKERS,  # 传入工作进程数
        memory_check_freq=MEMORY_CHECK_FREQ  # 内存检查频率
    )