# -----------------------------
# eval_multi_t.py - 多时间步评估脚本
# 加载最佳模型，在不同噪声水平下评估去噪效果
# 使用方式: python eval_multi_t.py
# -----------------------------
import torch
import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg')  # 无头模式，不弹窗
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import (SimpleUNet1D_Classic, compute_diffusion_params,
                   cosine_beta_schedule, denoise_for_eval, calculate_snr, set_seed)
from Dataset import (Dataset_UWB, DEFAULT_SPLIT_SEED, DEFAULT_TEST_RATIO,
                     DEFAULT_VAL_RATIO, create_split_indices,
                     restore_split_metadata)
from torch.utils.data import DataLoader


def evaluate_at_t(model, test_loader, eval_t, params, device, save_signals=10):
    """在指定时间步 t 评估模型，同时保存前 N 个样本的信号数据用于可视化"""
    input_snrs, output_snrs, corr_vals = [], [], []
    saved_clean, saved_noisy, saved_denoised = [], [], []

    with torch.no_grad():
        set_seed(42)
        sample_idx = 0
        for batch_clean in test_loader:
            batch_clean = batch_clean.to(device)
            for i in range(batch_clean.shape[0]):
                clean_sample = batch_clean[i:i + 1]

                alpha_bar = params['alphas_cumprod'][eval_t].view(-1, 1, 1)
                noise = torch.randn_like(clean_sample)
                noisy = torch.sqrt(alpha_bar) * clean_sample + torch.sqrt(1 - alpha_bar) * noise

                denoised = denoise_for_eval(model, noisy, eval_t, params, device)

                in_snr = calculate_snr(clean_sample[0].cpu(), noisy[0].cpu())
                out_snr = calculate_snr(clean_sample[0].cpu(), denoised[0].cpu())
                input_snrs.append(in_snr)
                output_snrs.append(out_snr)

                clean_np = clean_sample[0, 0].cpu().numpy()
                denoised_np = denoised[0, 0].cpu().numpy()
                noisy_np = noisy[0, 0].cpu().numpy()
                c = np.corrcoef(clean_np, denoised_np)[0, 1]
                corr_vals.append(c if not np.isnan(c) else 0.0)

                if sample_idx < save_signals:
                    saved_clean.append(clean_np)
                    saved_noisy.append(noisy_np)
                    saved_denoised.append(denoised_np)
                sample_idx += 1

    input_snrs = np.array(input_snrs)
    output_snrs = np.array(output_snrs)
    improvements = output_snrs - input_snrs
    corr_vals = np.array(corr_vals)

    return {
        'eval_t': eval_t,
        'input_snr_mean': np.mean(input_snrs),
        'output_snr_mean': np.mean(output_snrs),
        'snr_improvement_mean': np.mean(improvements),
        'snr_improvement_std': np.std(improvements),
        'snr_improvement_min': np.min(improvements),
        'snr_improvement_max': np.max(improvements),
        'corr_mean': np.mean(corr_vals),
        'corr_median': np.median(corr_vals),
        'corr_gt_07': int(np.sum(corr_vals > 0.7)),
        'corr_lt_05': int(np.sum(corr_vals < 0.5)),
        'num_samples': len(corr_vals),
    }, saved_clean, saved_noisy, saved_denoised


def plot_multi_t_comparison(all_signals, eval_ts, save_dir):
    """
    每个 t 值一个窗口（PNG 文件），10 行 × 3 列：
      左列=干净信号  中列=带噪信号  右列=去噪信号
    只显示前 2000 个采样点（便于观察脉冲细节）
    """
    display_len = 2000
    num_samples = len(all_signals[0][0])  # 每个 t 保存的样本数

    for t_idx, eval_t in enumerate(eval_ts):
        clean_list, noisy_list, denoised_list = all_signals[t_idx]

        fig, axes = plt.subplots(num_samples, 3, figsize=(18, 2.2 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        col_titles = ['干净信号', '带噪信号', '去噪后']
        for col in range(3):
            axes[0, col].set_title(col_titles[col], fontsize=13, fontweight='bold')

        for row in range(num_samples):
            c = clean_list[row][:display_len]
            n = noisy_list[row][:display_len]
            d = denoised_list[row][:display_len]

            in_snr = calculate_snr(torch.tensor(c), torch.tensor(n)) if hasattr(torch, 'tensor') else 0

            axes[row, 0].plot(c, 'g-', linewidth=0.7)
            axes[row, 0].set_ylabel(f'样本{row+1}', fontsize=9)
            axes[row, 0].set_ylim(-1.2, 1.2)

            axes[row, 1].plot(n, 'r-', linewidth=0.5, alpha=0.7)
            axes[row, 1].set_ylim(-1.2, 1.2)

            axes[row, 2].plot(d, 'b-', linewidth=0.7)
            axes[row, 2].set_ylim(-1.2, 1.2)

            if row == 0:
                snr_text = f'输入SNR≈{in_snr:.1f}dB'
                axes[row, 1].text(0.02, 0.92, snr_text, transform=axes[row, 1].transAxes,
                                  fontsize=8, color='darkred',
                                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        for col in range(3):
            axes[-1, col].set_xlabel('采样点', fontsize=10)

        fig.suptitle(f't = {eval_t}（噪声等级 {eval_t}/1000）', fontsize=15, fontweight='bold', y=1.01)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"denoising_t{eval_t:04d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  ✅ 已保存: {save_path}")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ---- 加载模型 ----
    # 优先用消融实验的 old_weights 最佳模型，其次用默认路径
    candidates = [
        os.path.join(script_dir, "ablation_results", "old_weights", "saved_models_classic", "best_model_by_loss.pth"),
        os.path.join(script_dir, "saved_models_classic", "best_model_by_loss.pth"),
    ]
    model_path = None
    for p in candidates:
        if os.path.exists(p):
            model_path = p
            break
    if model_path is None:
        print(f"❌ 找不到模型文件，检查了: {candidates}")
        sys.exit(1)

    print(f"加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model = SimpleUNet1D_Classic(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  最佳 Epoch: {checkpoint.get('epoch', '?')}")

    # ---- 加载与训练时完全相同的测试集 ----
    data_path = os.path.join(script_dir, "uwb_signals_time_clean.npy")
    n_samples = len(np.load(data_path, mmap_mode='r'))
    if 'data_split' in checkpoint:
        split_indices, norm_stats = restore_split_metadata(
            checkpoint['data_split'], n_samples)
        print("  使用检查点中保存的原始数据划分")
    else:
        print("  警告: 旧检查点未保存数据划分，将使用默认参数重建；请确认与训练参数一致")
        split_indices = create_split_indices(
            n_samples, DEFAULT_VAL_RATIO, DEFAULT_TEST_RATIO, DEFAULT_SPLIT_SEED)
        train_dataset = Dataset_UWB(
            clean_path=data_path, indices=split_indices['train'], split='train')
        norm_stats = train_dataset.get_norm_stats()
    test_dataset = Dataset_UWB(
        clean_path=data_path, indices=split_indices['test'], split='test',
        norm_stats=norm_stats)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    split_seed = checkpoint.get('data_split', {}).get('seed', DEFAULT_SPLIT_SEED)
    print(f"  划分种子: {split_seed}")
    print(f"  测试样本数: {len(test_dataset)}")

    # ---- 扩散参数 ----
    timesteps = 1000
    betas = cosine_beta_schedule(timesteps).to(device)
    params = compute_diffusion_params(betas, device)

    # ---- 多 t 评估 ----
    eval_ts = [100, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400]
    print(f"\n评估时间步: {eval_ts}")
    print("=" * 85)

    results = []
    all_signals = []  # [(clean_list, noisy_list, denoised_list), ...] per t

    for eval_t in eval_ts:
        r, clean_sigs, noisy_sigs, denoised_sigs = evaluate_at_t(
            model, test_loader, eval_t, params, device, save_signals=10)
        results.append(r)
        all_signals.append((clean_sigs, noisy_sigs, denoised_sigs))

        alpha_bar_t = params['alphas_cumprod'][eval_t].item()
        print(f"  t={eval_t:4d} | "
              f"输入SNR={r['input_snr_mean']:+6.2f} dB | "
              f"输出SNR={r['output_snr_mean']:+6.2f} dB | "
              f"SNR提升={r['snr_improvement_mean']:+6.2f} dB | "
              f"相关系数={r['corr_mean']:.4f} | "
              f">0.7: {r['corr_gt_07']}/{r['num_samples']} | "
              f"ᾱ={alpha_bar_t:.4f}")

    # ---- 汇总表 ----
    print("\n" + "=" * 85)
    print("📊 汇总表")
    print("=" * 85)
    print(f"{'t':>6} {'输入SNR':>10} {'输出SNR':>10} {'SNR提升':>10} {'相关系数':>10} {'>0.7':>8} {'<0.5':>8}")
    print("-" * 85)
    for r in results:
        print(f"{r['eval_t']:>6} {r['input_snr_mean']:+9.2f} {r['output_snr_mean']:+9.2f} "
              f"{r['snr_improvement_mean']:+9.2f} {r['corr_mean']:>10.4f} "
              f"{r['corr_gt_07']:>5}/{r['num_samples']:<3} {r['corr_lt_05']:>5}/{r['num_samples']:<3}")

    # ---- 绘制去噪对比图 ----
    save_dir = os.path.join(script_dir, "logs_classic", "multi_t_viz_fine")
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n绘制去噪对比图 → {save_dir}")
    plot_multi_t_comparison(all_signals, eval_ts, save_dir)

    # ---- 保存汇总文件 ----
    save_path = os.path.join(script_dir, "logs_classic", "multi_t_eval_fine.txt")
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("多时间步评估结果\n")
        f.write(f"模型: {os.path.basename(model_path)} (Epoch {checkpoint.get('epoch', '?')})\n")
        f.write("=" * 85 + "\n")
        f.write(f"{'t':>6} {'输入SNR':>10} {'输出SNR':>10} {'SNR提升':>10} {'相关系数':>10} {'>0.7':>8} {'<0.5':>8}\n")
        f.write("-" * 85 + "\n")
        for r in results:
            f.write(f"{r['eval_t']:>6} {r['input_snr_mean']:+9.2f} {r['output_snr_mean']:+9.2f} "
                    f"{r['snr_improvement_mean']:+9.2f} {r['corr_mean']:>10.4f} "
                    f"{r['corr_gt_07']:>5}/{r['num_samples']:<3} {r['corr_lt_05']:>5}/{r['num_samples']:<3}\n")
        f.write("\n说明:\n")
        f.write("  t=100: 轻度噪声  t=300: 中度噪声  t=500: 重度噪声  t=700: 极重噪声\n")
        f.write("  ᾱ 越小 → 噪声越强。当前训练 t∈[0,1000) 均匀采样。\n")

    print(f"\n✅ 汇总表已保存到 {save_path}")
    print(f"✅ 对比图已保存到 {save_dir}/")
    print(f"   共 {len(eval_ts)} 个文件: " + ", ".join([f"denoising_t{t:04d}.png" for t in eval_ts]))
