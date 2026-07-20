import torch
import torch.nn.functional as F
from models import DenoiseUNet
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import os
import math
import matplotlib.pyplot as plt
from skimage.transform import resize


# -----------------------------
# Beta 调度
# -----------------------------
def linear_beta_schedule(beta_start, beta_end, timesteps):
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    余弦调度（效果更好）
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


# -----------------------------
# 预计算扩散参数
# -----------------------------
class DiffusionParams:
    def __init__(self, betas):
        self.betas = betas
        self.alphas = 1 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self


# -----------------------------
# 计算理论 SNR
# -----------------------------
def calculate_theoretical_snr(alpha_bar):
    """计算给定 ᾱ 时的理论信噪比 (dB)"""
    snr = 10 * torch.log10(alpha_bar / (1 - alpha_bar + 1e-8))
    return snr.item()


# -----------------------------
# 加性噪声反向扩散（经典DDPM）
# -----------------------------
def reverse_diffusion(model, x_t, params, device, save_intermediate=True,
                      intermediate_step=100, clamp_x0=True):
    """
    加性噪声的DDPM反向扩散过程（经典DDPM公式）
    """
    model.eval()
    intermediates = []

    betas = params.betas
    alphas = params.alphas
    alphas_cumprod = params.alphas_cumprod
    alphas_cumprod_prev = params.alphas_cumprod_prev
    posterior_variance = params.posterior_variance

    x = x_t.clone()
    timesteps = len(betas)

    print(f"开始反向扩散（经典DDPM加性噪声），总步数: {timesteps}")

    for t_idx in tqdm(range(timesteps - 1, -1, -1), desc="反向扩散"):
        t = torch.full((x.shape[0],), t_idx, device=device, dtype=torch.long)

        shape = [1] * (x.dim() - 1)
        beta_t = betas[t_idx].view(-1, *shape)
        alpha_t = alphas[t_idx].view(-1, *shape)
        alpha_bar_t = alphas_cumprod[t_idx].view(-1, *shape)
        alpha_bar_t_prev = alphas_cumprod_prev[t_idx].view(-1, *shape)

        with torch.no_grad():
            predicted_noise = model(x, t)

        # 经典DDPM公式：预测干净信号
        pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)

        if clamp_x0:
            pred_x0 = torch.clamp(pred_x0, -1.2, 1.2)

        # 注意：BPSK信号本身有正有负，无法从信号本身判断极性是否反了
        # 因此不在推理时做极性校正，而是通过训练时的损失函数来防止极性翻转
        # 计算后验均值
        coeff1 = (torch.sqrt(alpha_bar_t_prev) * beta_t) / (1 - alpha_bar_t)
        coeff2 = (torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev)) / (1 - alpha_bar_t)
        mu = coeff1 * pred_x0 + coeff2 * x

        if t_idx > 0:
            var = posterior_variance[t_idx].view(-1, *shape)
            noise = torch.randn_like(x)
            x = mu + torch.sqrt(var) * noise
        else:
            x = mu

        if save_intermediate and t_idx % intermediate_step == 0:
            img = tensor_to_image(x[0])
            intermediates.append((t_idx, img))

    return x, intermediates


def tensor_to_image(tensor):
    """将tensor转换为numpy图像 [0,255]，支持1D和2D"""
    if tensor.dim() == 1:
        img = tensor.cpu().numpy()
        img = (img + 1) * 127.5
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    elif tensor.dim() == 2:
        img = tensor.cpu().numpy()
        img = (img + 1) * 127.5
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    elif tensor.dim() == 3:
        img = tensor[0].cpu().numpy()
        img = (img + 1) * 127.5
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    else:
        img = tensor[0, 0].cpu().numpy()
        img = (img + 1) * 127.5
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img


def create_grid_image(intermediates, target_size=None):
    """将中间结果拼接成网格图"""
    images = [img for _, img in intermediates]
    n_images = len(images)
    grid_size = math.ceil(math.sqrt(n_images))

    if len(images[0].shape) == 1:
        h, w = 64, 64
        grid = np.zeros((h * grid_size, w * grid_size), dtype=np.uint8)

        for i, signal in enumerate(images):
            fig, ax = plt.subplots(figsize=(0.5, 0.5))
            ax.plot(signal)

            if signal.max() > signal.min():
                ax.set_ylim(signal.min(), signal.max())

            ax.axis('off')
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = img.mean(axis=2)
            plt.close(fig)

            row = i // grid_size
            col = i % grid_size
            grid[row * h:(row + 1) * h, col * w:(col + 1) * w] = img[:h, :w]

        return grid
    else:
        h, w = images[0].shape
        if target_size:
            h, w = target_size

        grid = np.zeros((h * grid_size, w * grid_size), dtype=np.uint8)

        for i, img in enumerate(images):
            if img.shape != (h, w):
                img = resize(img, (h, w), preserve_range=True).astype(np.uint8)

            row = i // grid_size
            col = i % grid_size
            grid[row * h:(row + 1) * h, col * w:(col + 1) * w] = img

        return grid


# -----------------------------
# 主函数
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DDPM图像/信号生成（经典加性噪声）')
    parser.add_argument("--timesteps", type=int, default=1000,
                        help="总扩散步数")
    parser.add_argument("--beta_start", type=float, default=0.0001,
                        help="beta起始值")
    parser.add_argument("--beta_end", type=float, default=0.02,
                        help="beta结束值")
    parser.add_argument("--schedule", type=str, default="cosine",
                        choices=['linear', 'cosine'],
                        help="beta调度类型（推荐cosine）")
    parser.add_argument("--weights", type=str, default="./saved_models/best_model_by_snr.pth",
                        help="模型权重路径")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="计算设备")
    parser.add_argument("--output", type=str, default="reverse_diffusion.png",
                        help="输出图像路径")
    parser.add_argument("--signal_length", type=int, default=5000,
                        help="信号长度（1D）")
    parser.add_argument("--img_size", type=int, nargs=2, default=[128, 128],
                        help="图像大小 H W（2D）")
    parser.add_argument("--data_dim", type=str, default="1d",
                        choices=['1d', '2d'],
                        help="数据维度: 1d(时域信号) 或 2d(图像/STFT)")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="生成样本数量")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--intermediate", action="store_true",
                        help="保存中间结果")
    parser.add_argument("--start_t", type=int, default=999,
                        help="起始时间步（越大噪声越强，SNR越低）")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("DDPM 反向扩散生成（经典加性噪声）")
    print("=" * 60)
    print(f"  - 设备: {args.device}")
    print(f"  - 总步数: {args.timesteps}")
    print(f"  - 起始步: {args.start_t}")
    print(f"  - 调度: {args.schedule}")
    print(f"  - 数据维度: {args.data_dim}")
    if args.data_dim == '1d':
        print(f"  - 信号长度: {args.signal_length}")
    else:
        print(f"  - 图像大小: {args.img_size}")
    print(f"  - 样本数量: {args.num_samples}")
    print("=" * 60)

    # 创建beta调度
    if args.schedule == "linear":
        betas = linear_beta_schedule(args.beta_start, args.beta_end, args.timesteps)
    else:
        betas = cosine_beta_schedule(args.timesteps)

    params = DiffusionParams(betas).to(args.device)

    # 显示理论 SNR
    if args.start_t < args.timesteps:
        alpha_bar = params.alphas_cumprod[args.start_t]
        snr = calculate_theoretical_snr(alpha_bar)
        print(f"\n起始时间步 {args.start_t} 对应的理论 SNR: {snr:.2f} dB")
        if snr < 0:
            print(f"  ⚠️ 当前 SNR 为负值，将测试负信噪比场景")

    # 初始化模型
    print(f"\n加载模型: {args.weights}")

    # 使用 DenoiseUNet 模型
    model = DenoiseUNet(
        in_channels=1,
        out_channels=1,
        ch=64,
        droprate=0.1,
        device=args.device,
        data_dim=args.data_dim
    ).to(args.device)

    try:
        checkpoint = torch.load(args.weights, map_location=args.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  加载模型权重 (epoch {checkpoint.get('epoch', 'unknown')})")
        else:
            model.load_state_dict(checkpoint)
            print("  加载模型权重")
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        exit(1)

    model.eval()

    # 生成样本
    all_results = []
    all_intermediates = []

    for sample_idx in range(args.num_samples):
        print(f"\n生成样本 {sample_idx + 1}/{args.num_samples}")

        # 根据数据维度初始化噪声
        if args.data_dim == '1d':
            x_t = torch.randn(1, 1, args.signal_length).to(args.device)
        else:
            x_t = torch.randn(1, 1, args.img_size[0], args.img_size[1]).to(args.device)

        print(f"初始噪声范围: [{x_t.min():.3f}, {x_t.max():.3f}]")

        # 反向扩散（从指定时间步开始）
        x_0, intermediates = reverse_diffusion(
            model, x_t, params, args.device,
            save_intermediate=args.intermediate,
            intermediate_step=100,
            clamp_x0=True
        )

        result = tensor_to_image(x_0[0])
        all_results.append(result)

        if args.intermediate:
            all_intermediates.append(intermediates)

        print(f"生成结果范围: [{result.min()}, {result.max()}]")

    # 保存结果
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)

    if args.num_samples > 1:
        if args.data_dim == '1d':
            print("多样本1D信号保存...")
            for i, sig in enumerate(all_results):
                output_path = args.output.replace('.png', f'_sample{i}.png')
                plt.figure(figsize=(10, 4))
                plt.plot(sig[:1000])
                plt.title(f'Sample {i}')
                plt.xlabel('Sample index')
                plt.ylabel('Amplitude')
                plt.grid(True)
                plt.savefig(output_path)
                plt.close()
                print(f"信号图已保存到: {output_path}")
        else:
            grid_size = math.ceil(math.sqrt(args.num_samples))
            h, w = args.img_size
            grid = np.zeros((h * grid_size, w * grid_size), dtype=np.uint8)

            for i, img in enumerate(all_results):
                row = i // grid_size
                col = i % grid_size
                grid[row * h:(row + 1) * h, col * w:(col + 1) * w] = img

            Image.fromarray(grid).save(args.output)
            print(f"网格图像已保存到: {args.output}")
    else:
        if args.data_dim == '1d':
            plt.figure(figsize=(12, 5))
            plt.plot(all_results[0][:1000])
            plt.title(f'Generated Signal (start_t={args.start_t})')
            plt.xlabel('Sample index')
            plt.ylabel('Amplitude')
            plt.grid(True)
            plt.savefig(args.output)
            plt.close()
            print(f"信号图已保存到: {args.output}")
        else:
            Image.fromarray(all_results[0]).save(args.output)
            print(f"图像已保存到: {args.output}")

    # 如果有中间结果
    if args.intermediate and args.num_samples == 1:
        grid = create_grid_image(all_intermediates[0], target_size=args.img_size)
        interp_output = args.output.replace('.png', '_intermediate.png')

        if args.data_dim == '1d':
            n = len(all_intermediates[0])
            n_cols = 5
            n_rows = (n + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

            for idx, (t, img) in enumerate(all_intermediates[0]):
                if idx < len(axes):
                    axes[idx].plot(img[:1000])
                    axes[idx].set_title(f't={t}')
                    axes[idx].axis('off')

            for idx in range(len(all_intermediates[0]), len(axes)):
                axes[idx].axis('off')

            plt.tight_layout()
            plt.savefig(interp_output)
            plt.close()
        else:
            Image.fromarray(grid).save(interp_output)

        print(f"中间结果已保存到: {interp_output}")

    print("\n✅ 完成！")