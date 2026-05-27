import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import math


# -----------------------------
# Beta调度
# -----------------------------
def linear_beta_schedule(beta_start, beta_end, timesteps):
    """线性beta调度"""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """余弦beta调度"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


# -----------------------------
# 计算理论 SNR
# -----------------------------
def calculate_theoretical_snr(alpha_bar):
    """计算给定 ᾱ 时的理论信噪比 (dB)"""
    snr = 10 * torch.log10(alpha_bar / (1 - alpha_bar + 1e-8))
    return snr.item()


# -----------------------------
# 加性噪声前向扩散（经典DDPM公式）
# -----------------------------
def additive_diffusion(x_0, betas, t, device):
    """
    加性噪声前向扩散
    公式: x_t = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * ε
    """
    if not isinstance(x_0, torch.Tensor):
        x_0 = torch.tensor(x_0, dtype=torch.float32).to(device)
    else:
        x_0 = x_0.to(device)

    timesteps = len(betas)
    betas = betas.to(device)

    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    if t < 0 or t >= timesteps:
        raise ValueError(f"t must be in [0, {timesteps - 1}], got {t}")

    t_tensor = torch.tensor([t], dtype=torch.long, device=device)
    alpha_bar = alphas_cumprod[t_tensor].view(-1, *([1] * (x_0.dim() - 1)))

    eps = torch.randn_like(x_0)
    x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * eps

    return x_t, eps, alpha_bar


# -----------------------------
# 乘性噪声前向扩散
# -----------------------------
def multiplicative_diffusion(x_0, betas, t, device):
    """
    乘性噪声前向扩散
    公式: x_t = x_0 * (1 + ε), 其中 ε ~ N(0, 1-ᾱ_t)
    """
    if not isinstance(x_0, torch.Tensor):
        x_0 = torch.tensor(x_0, dtype=torch.float32).to(device)
    else:
        x_0 = x_0.to(device)

    timesteps = len(betas)
    betas = betas.to(device)

    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    if t < 0 or t >= timesteps:
        raise ValueError(f"t must be in [0, {timesteps - 1}], got {t}")

    t_tensor = torch.tensor([t], dtype=torch.long, device=device)
    alpha_bar = alphas_cumprod[t_tensor].view(-1, *([1] * (x_0.dim() - 1)))

    noise_std = torch.sqrt(1 - alpha_bar)
    eps = torch.randn_like(x_0) * noise_std
    x_t = x_0 * (1 + eps)

    return x_t, eps, alpha_bar


# -----------------------------
# 统一扩散接口
# -----------------------------
def diffusion(x_0, betas, t, device, noise_type='additive'):
    """
    统一扩散接口
    Args:
        noise_type: 'additive' 或 'multiplicative'
    """
    if noise_type == 'additive':
        return additive_diffusion(x_0, betas, t, device)
    elif noise_type == 'multiplicative':
        return multiplicative_diffusion(x_0, betas, t, device)
    else:
        raise ValueError(f"不支持的噪声类型: {noise_type}")


# -----------------------------
# 图像预处理
# -----------------------------
def load_and_preprocess_image(img_path, target_size=(128, 128), normalize_range='[-1,1]'):
    img = Image.open(img_path).convert('RGB')
    img.thumbnail(target_size, Image.Resampling.LANCZOS)

    new_img = Image.new('RGB', target_size, (0, 0, 0))
    new_img.paste(img, ((target_size[0] - img.size[0]) // 2,
                        (target_size[1] - img.size[1]) // 2))

    img_array = np.array(new_img, dtype=np.float32)

    if normalize_range == '[-1,1]':
        img_array = img_array / 127.5 - 1.0
    else:
        img_array = img_array / 255.0

    return img_array


def save_comparison_image(original, noised_images, t_values, output_path):
    """保存对比图像"""
    if original.max() <= 1.0:
        if original.min() >= 0:
            original_disp = original * 255
        else:
            original_disp = (original + 1) * 127.5
    else:
        original_disp = original

    images_disp = [np.clip(original_disp, 0, 255).astype(np.uint8)]

    for img, t in zip(noised_images, t_values):
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()

        if img.max() <= 1.0:
            if img.min() >= 0:
                img_disp = img * 255
            else:
                img_disp = (img + 1) * 127.5
        else:
            img_disp = img

        images_disp.append(np.clip(img_disp, 0, 255).astype(np.uint8))

    concatenated = np.concatenate(images_disp, axis=1)
    Image.fromarray(concatenated).save(output_path)
    print(f"对比图像已保存到: {output_path}")
    print(f"原始图像 + {len(t_values)}个加噪图像")
    print(f"时间步: {t_values}")


# -----------------------------
# 主函数
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description='DDPM前向扩散过程可视化')
    parser.add_argument("--img_path", type=str, default="resources/face.png",
                        help="输入图像路径或npy信号路径")
    parser.add_argument("--data_type", type=str, default="image",
                        choices=['image', 'signal_1d'],
                        help="数据类型: image(2D图像) 或 signal_1d(1D信号)")
    parser.add_argument("--t", type=str, default="750, 800,850,900,950",
                        help="要可视化的时间步，用逗号分隔")
    parser.add_argument("--timesteps", type=int, default=1000,
                        help="总扩散步数")
    parser.add_argument("--beta_start", type=float, default=0.0001,
                        help="beta起始值")
    parser.add_argument("--beta_end", type=float, default=0.02,
                        help="beta结束值")
    parser.add_argument("--schedule", type=str, default="linear",
                        choices=['linear', 'cosine'],
                        help="beta调度类型")
    parser.add_argument("--noise_type", type=str, default="additive",
                        choices=['additive', 'multiplicative'],
                        help="噪声类型: additive(加性) 或 multiplicative(乘性)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="计算设备")
    parser.add_argument("--output", type=str, default="diffusion_output.png",
                        help="输出图像路径")
    parser.add_argument("--img_size", type=int, nargs=2, default=[128, 128],
                        help="图像大小 H W")
    parser.add_argument("--signal_length", type=int, default=5000,
                        help="1D信号长度")

    args = parser.parse_args()

    # 解析时间步
    t_values = [int(t.strip()) for t in args.t.split(",")]
    t_values = [t for t in t_values if 0 <= t < args.timesteps]

    if not t_values:
        print("错误: 没有有效的时间步")
        return

    print(f"运行配置:")
    print(f"  - 设备: {args.device}")
    print(f"  - 数据类型: {args.data_type}")
    print(f"  - 总步数: {args.timesteps}")
    print(f"  - 时间步: {t_values}")
    print(f"  - 调度: {args.schedule}")
    print(f"  - 噪声类型: {args.noise_type}")

    # 加载数据
    print(f"加载数据: {args.img_path}")

    if args.data_type == 'signal_1d':
        # 加载1D信号
        x_0 = np.load(args.img_path)
        if x_0.ndim == 2:
            x_0 = x_0[0]
        if x_0.ndim == 1:
            x_0 = x_0[np.newaxis, np.newaxis, :]

        # 归一化到 [-1, 1]
        x_0 = (x_0 - x_0.min()) / (x_0.max() - x_0.min() + 1e-8) * 2 - 1
        x_0 = torch.tensor(x_0, dtype=torch.float32).to(args.device)
        print(f"信号形状: {x_0.shape}, 范围: [{x_0.min():.3f}, {x_0.max():.3f}]")
    else:
        # 加载图像
        x_0 = load_and_preprocess_image(
            args.img_path,
            target_size=tuple(args.img_size),
            normalize_range='[-1,1]'
        )
        x_0 = torch.tensor(x_0, dtype=torch.float32).to(args.device)
        print(f"图像形状: {x_0.shape}, 范围: [{x_0.min():.3f}, {x_0.max():.3f}]")

    # 创建beta调度
    if args.schedule == 'linear':
        betas = linear_beta_schedule(args.beta_start, args.beta_end, args.timesteps)
    else:
        betas = cosine_beta_schedule(args.timesteps)

    # 计算 alphas_cumprod 用于 SNR 计算
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    print(f"Beta范围: [{betas.min():.6f}, {betas.max():.6f}]")

    # 执行扩散
    noised_results = []
    for t in t_values:
        print(f"处理时间步 t={t}...")
        x_t, eps, alpha_bar = diffusion(x_0, betas, t, args.device, noise_type=args.noise_type)
        
        # 计算并显示理论 SNR
        if args.noise_type == 'additive':
            snr = calculate_theoretical_snr(alpha_bar)
            print(f"  -> 理论 SNR: {snr:.2f} dB")
        
        noised_results.append((x_t, eps))

    # 保存结果
    if args.data_type == 'signal_1d':
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(t_values) + 1, 1, figsize=(12, 3 * (len(t_values) + 1)))

        original_np = x_0[0, 0].cpu().numpy()
        axes[0].plot(original_np[:1000])
        axes[0].set_title(f'Original Signal (clean)')
        axes[0].set_xlabel('Sample')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)

        for i, (t, (x_t, eps)) in enumerate(zip(t_values, noised_results)):
            x_t_np = x_t[0, 0].cpu().numpy()
            axes[i + 1].plot(x_t_np[:1000])
            
            # 在标题中显示 SNR（如果是加性噪声）
            if args.noise_type == 'additive':
                alpha_bar = alphas_cumprod[t]
                snr = calculate_theoretical_snr(alpha_bar)
                axes[i + 1].set_title(f'Noised Signal at t={t} (SNR ≈ {snr:.1f} dB)')
            else:
                axes[i + 1].set_title(f'Noised Signal at t={t} (noise_type={args.noise_type})')
            
            axes[i + 1].set_xlabel('Sample')
            axes[i + 1].set_ylabel('Amplitude')
            axes[i + 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(args.output)
        plt.show()
        print(f"信号图已保存到: {args.output}")
    else:
        noised_images = [x_t for x_t, _ in noised_results]
        save_comparison_image(x_0.cpu().numpy(), noised_images, t_values, args.output)

    print("完成！")


if __name__ == "__main__":
    main()