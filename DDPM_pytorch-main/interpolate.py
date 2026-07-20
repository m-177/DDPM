import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import os
import math
import matplotlib.pyplot as plt


# -------------------------------
# Beta 调度
# -------------------------------
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
    """预计算所有扩散参数"""
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


# -------------------------------
# 加性噪声逆向扩散（经典DDPM公式）
# -------------------------------
def reverse_diffusion(model, y_t, start_step, params, device, save_intermediate=False):
    """
    加性噪声的逆向扩散过程（标准DDPM）
    公式: p(x_{t-1}|x_t) = N(x_{t-1}; μ, σ^2)
    """
    model.eval()
    intermediates = []

    betas = params['betas']
    alphas = params['alphas']
    alphas_cumprod = params['alphas_cumprod']
    alphas_cumprod_prev = params['alphas_cumprod_prev']
    posterior_variance = params['posterior_variance']

    y = y_t.clone()

    for t in tqdm(range(start_step, 0, -1), desc="Reverse diffusion"):
        t_tensor = torch.full((y.shape[0],), t - 1, device=device, dtype=torch.long)

        # 模型预测加性噪声
        predicted_noise = model(y, t_tensor)

        # 获取参数
        alpha = alphas[t - 1].view(-1, *([1] * (y.dim() - 1)))
        alpha_bar = alphas_cumprod[t - 1].view(-1, *([1] * (y.dim() - 1)))
        alpha_bar_prev = alphas_cumprod_prev[t - 1].view(-1, *([1] * (y.dim() - 1)))
        beta = betas[t - 1].view(-1, *([1] * (y.dim() - 1)))

        # 预测干净信号（加性噪声公式）
        pred_y0 = (y - torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)
        pred_y0 = torch.clamp(pred_y0, -1, 1)

        # 计算后验均值
        if t > 1:
            coeff1 = (torch.sqrt(alpha_bar_prev) * beta) / (1 - alpha_bar)
            coeff2 = (torch.sqrt(alpha) * (1 - alpha_bar_prev)) / (1 - alpha_bar)
            mu = coeff1 * pred_y0 + coeff2 * y
        else:
            mu = pred_y0

        # 采样
        if t > 1:
            variance = posterior_variance[t - 1].view(-1, *([1] * (y.dim() - 1)))
            noise = torch.randn_like(y)
            y = mu + torch.sqrt(variance) * noise
        else:
            y = mu

        if save_intermediate and t % 100 == 0:
            intermediates.append(y[0, 0].cpu().numpy())

    if save_intermediate:
        return y, intermediates
    return y


# -------------------------------
# 计算理论 SNR
# -------------------------------
def calculate_theoretical_snr(alpha_bar):
    """计算给定 ᾱ 时的理论信噪比 (dB)"""
    snr = 10 * torch.log10(alpha_bar / (1 - alpha_bar + 1e-8))
    return snr.item()


# -------------------------------
# 插值函数
# -------------------------------
def interpolate(model, sig1, sig2, interp_steps, params, device, n_interps=11):
    """
    在两个信号之间插值（经典DDPM加性噪声）
    """
    alphas_cumprod = params['alphas_cumprod']
    alpha_bar = alphas_cumprod[interp_steps - 1].view(-1, *([1] * (sig1.dim() - 1)))

    # 显示理论 SNR
    snr = calculate_theoretical_snr(alpha_bar)
    print(f"插值步数 {interp_steps} 对应的理论 SNR: {snr:.2f} dB")

    # 加性噪声正向扩散（经典DDPM）
    noise1 = torch.randn_like(sig1)
    noise2 = torch.randn_like(sig2)
    x_t1 = torch.sqrt(alpha_bar) * sig1 + torch.sqrt(1 - alpha_bar) * noise1
    x_t2 = torch.sqrt(alpha_bar) * sig2 + torch.sqrt(1 - alpha_bar) * noise2

    results = []
    for i in range(n_interps):
        lambd = i / (n_interps - 1)
        print(f"\n插值 {i + 1}/{n_interps}: lambda={lambd:.2f}")

        x_t = (1 - lambd) * x_t1 + lambd * x_t2

        y_t = reverse_diffusion(model, x_t, interp_steps, params, device, save_intermediate=False)

        result = tensor_to_signal(y_t[0])
        results.append(result)

    return results


def tensor_to_signal(tensor):
    """将tensor转换为numpy信号 [0,255]"""
    if tensor.dim() == 3:
        tensor = tensor[0, 0] if tensor.shape[1] == 1 else tensor[0]
    elif tensor.dim() == 2:
        tensor = tensor[0] if tensor.shape[0] == 1 else tensor
    elif tensor.dim() == 1:
        tensor = tensor

    signal = tensor.cpu().numpy()
    signal = (signal + 1) * 127.5
    signal = np.clip(signal, 0, 255).astype(np.uint8)
    return signal


def load_signal(path, device):
    """加载1D信号"""
    data = np.load(path)
    # 处理不同维度
    if data.ndim == 1:
        data = data[np.newaxis, np.newaxis, :]
    elif data.ndim == 2:
        data = data[np.newaxis, np.newaxis, :]
    elif data.ndim == 3 and data.shape[0] == 1:
        data = data
    else:
        raise ValueError(f"不支持的信号形状: {data.shape}")

    # 归一化到 [-1, 1]
    data = (data - data.min()) / (data.max() - data.min() + 1e-8) * 2 - 1
    return torch.tensor(data, dtype=torch.float32).to(device)


def plot_1d_results(results, output_dir, interp_steps):
    """绘制1D信号插值结果"""
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n))

    if n == 1:
        axes = [axes]

    for i, signal in enumerate(results):
        axes[i].plot(signal[:1000])
        lambd = i / (n - 1)
        axes[i].set_title(f'Interpolation {i} (lambda={lambd:.2f})')
        axes[i].set_xlabel('Sample')
        axes[i].set_ylabel('Amplitude')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"interp_1d_steps{interp_steps}.png")
    plt.savefig(output_path)
    plt.show()
    print(f"1D插值结果已保存到: {output_path}")


# -------------------------------
# 主函数
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DDPM逆向扩散和插值（经典加性噪声）')
    parser.add_argument("--data_type", type=str, default="uwb_1d",
                        choices=["uwb_1d", "uwb_2d"],
                        help="数据类型: uwb_1d(1D时域信号), uwb_2d(2D图像)")
    parser.add_argument("--img1_path", type=str, required=True,
                        help="第一张图像/信号路径")
    parser.add_argument("--img2_path", type=str, required=True,
                        help="第二张图像/信号路径")
    parser.add_argument("--interp_steps", type=int, default=500,
                        help="扩散步数（决定噪声强度，越大SNR越低）")
    parser.add_argument("--timesteps", type=int, default=1000,
                        help="总时间步")
    parser.add_argument("--beta_start", type=float, default=0.0001,
                        help="beta起始值")
    parser.add_argument("--beta_end", type=float, default=0.02,
                        help="beta结束值")
    parser.add_argument("--schedule", type=str, default="cosine",
                        choices=["linear", "cosine"],
                        help="beta调度（推荐cosine）")
    parser.add_argument("--weights", type=str, default="./saved_models/best_model_by_snr.pth",
                        help="模型权重路径")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="计算设备")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="输出目录")
    parser.add_argument("--n_interps", type=int, default=11,
                        help="插值数量")

    args = parser.parse_args()

    print("=" * 60)
    print("DDPM 逆向扩散和插值测试（经典加性噪声）")
    print("=" * 60)
    print(f"  - 设备: {args.device}")
    print(f"  - 数据类型: {args.data_type}")
    print(f"  - 总步数: {args.timesteps}")
    print(f"  - 插值步数: {args.interp_steps}")
    print(f"  - 模型权重: {args.weights}")
    print("=" * 60)

    # 创建beta调度
    if args.schedule == "linear":
        betas = linear_beta_schedule(args.beta_start, args.beta_end, args.timesteps)
    else:
        betas = cosine_beta_schedule(args.timesteps)

    params = compute_diffusion_params(betas, args.device)

    # 初始化模型
    print(f"\n加载模型...")

    # 导入模型类（使用 train.py 中的 SimpleUNet1D_Classic）
    from train import SimpleUNet1D_Classic

    model = SimpleUNet1D_Classic(in_channels=1, out_channels=1, dropout_rate=0).to(args.device)

    # 加载权重
    if os.path.exists(args.weights):
        checkpoint = torch.load(args.weights, map_location=args.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  加载模型权重 (epoch {checkpoint.get('epoch', 'unknown')})")
        else:
            model.load_state_dict(checkpoint)
            print(f"  加载模型权重")
    else:
        print(f"  ❌ 警告: 权重文件不存在: {args.weights}")
        print(f"  将使用随机初始化的模型")

    model.eval()

    # 加载数据
    print(f"\n加载信号1: {args.img1_path}")
    print(f"加载信号2: {args.img2_path}")

    if args.data_type == "uwb_1d":
        sig1 = load_signal(args.img1_path, args.device)
        sig2 = load_signal(args.img2_path, args.device)
        print(f"信号1形状: {sig1.shape}, 范围: [{sig1.min():.3f}, {sig1.max():.3f}]")
        print(f"信号2形状: {sig2.shape}, 范围: [{sig2.min():.3f}, {sig2.max():.3f}]")
    else:
        # 2D图像处理
        img1 = Image.open(args.img1_path).convert("L").resize((128, 128))
        img2 = Image.open(args.img2_path).convert("L").resize((128, 128))
        sig1 = torch.tensor(np.array(img1), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(args.device)
        sig2 = torch.tensor(np.array(img2), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(args.device)
        sig1 = (sig1 / 127.5) - 1.0
        sig2 = (sig2 / 127.5) - 1.0

    # 显示理论 SNR
    alpha_bar = params['alphas_cumprod'][args.interp_steps - 1]
    snr = calculate_theoretical_snr(alpha_bar)
    print(f"\n插值步数 {args.interp_steps} 对应的理论 SNR: {snr:.2f} dB")
    if snr < 0:
        print(f"  ⚠️ 当前 SNR 为负值，将测试负信噪比场景")

    # 插值生成
    print("\n开始插值生成...")
    results = interpolate(
        model, sig1, sig2,
        args.interp_steps, params,
        args.device, args.n_interps
    )

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)

    if args.data_type == "uwb_1d":
        plot_1d_results(results, args.output_dir, args.interp_steps)
    else:
        # 2D结果拼接保存
        concatenated = np.concatenate(results, axis=1)
        output_path = os.path.join(args.output_dir, f"interp_steps{args.interp_steps}.png")
        Image.fromarray(concatenated).save(output_path)
        print(f"2D插值结果已保存到: {output_path}")

    print(f"\n✅ 完成！结果保存在: {args.output_dir}")