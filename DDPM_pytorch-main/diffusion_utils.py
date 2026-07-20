# -----------------------------
# diffusion_utils.py - 共享扩散参数工具
# 消除 train.py / reverse_diffusion_process.py / interpolate.py / diffusion_process.py 中的重复定义
# -----------------------------
import torch
import torch.nn.functional as F
import math


def linear_beta_schedule(beta_start, beta_end, timesteps):
    """线性 beta 调度"""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """余弦 beta 调度（效果更好）"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def calculate_theoretical_snr(alpha_bar):
    """计算给定 ᾱ 时的理论信噪比 (dB)"""
    snr = 10 * torch.log10(alpha_bar / (1 - alpha_bar + 1e-8))
    return snr.item()


def compute_diffusion_params(betas, device):
    """预计算所有扩散参数（返回 dict）"""
    betas = betas.to(device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
    posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'alphas_cumprod_prev': alphas_cumprod_prev,
        'posterior_variance': posterior_variance,
        'sqrt_alphas_cumprod': torch.sqrt(alphas_cumprod),
        'sqrt_one_minus_alphas_cumprod': torch.sqrt(1 - alphas_cumprod)
    }
