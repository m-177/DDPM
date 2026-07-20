# -----------------------------
# baseline_methods.py - 传统方法对比（小波去噪、维纳滤波）
# 使用方式: python baseline_methods.py
# -----------------------------
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Dataset import (Dataset_UWB, DEFAULT_SPLIT_SEED, DEFAULT_TEST_RATIO,
                     DEFAULT_VAL_RATIO, create_split_indices)

# ---- 加载测试集（与训练时相同的确定性划分） ----
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uwb_signals_time_clean.npy")
n_samples = len(np.load(data_path, mmap_mode='r'))
split_indices = create_split_indices(
    n_samples, DEFAULT_VAL_RATIO, DEFAULT_TEST_RATIO, DEFAULT_SPLIT_SEED)
train_dataset = Dataset_UWB(
    clean_path=data_path, indices=split_indices['train'], split='train')
test_dataset = Dataset_UWB(
    clean_path=data_path, indices=split_indices['test'], split='test',
    norm_stats=train_dataset.get_norm_stats())

clean_signals = [test_dataset[i] for i in range(len(test_dataset))]
clean_signals = np.array(clean_signals)  # (N, 10120)

# ---- 模拟 t=300 的加噪（与训练时一致） ----
eval_t = 300
timesteps = 1000
# Cosine beta schedule
s = 0.008
steps = timesteps + 1
x = np.linspace(0, timesteps, steps)
alphas_cumprod = (np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2)
alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
alpha_bar = alphas_cumprod[eval_t]

np.random.seed(42)
noise = np.random.randn(*clean_signals.shape).astype(np.float32)
noisy_signals = np.sqrt(alpha_bar) * clean_signals + np.sqrt(1 - alpha_bar) * noise
clean_signals = clean_signals.squeeze(1)      # (100, 10120)
noisy_signals = noisy_signals.squeeze(1)       # (100, 10120)

def calc_per_sample(clean_set, denoised_set):
    """逐样本计算指标，返回均值和标准差"""
    snr_imps, corrs = [], []
    for i in range(len(clean_set)):
        c, d = clean_set[i], denoised_set[i]
        in_snr = 10 * np.log10(np.mean(c**2) / np.mean((noisy_signals[i] - c)**2 + 1e-10))
        out_snr = 10 * np.log10(np.mean(c**2) / np.mean((d - c)**2 + 1e-10))
        snr_imps.append(out_snr - in_snr)
        corrs.append(np.corrcoef(c, d)[0, 1] if np.std(d) > 1e-8 else 0)
    return {
        'snr_imp_mean': np.mean(snr_imps),
        'snr_imp_std': np.std(snr_imps),
        'snr_imp_min': np.min(snr_imps),
        'snr_imp_max': np.max(snr_imps),
        'corr_mean': np.mean(corrs),
        'corr_median': np.median(corrs),
        'corr_gt_07': int(np.sum(np.array(corrs) > 0.7)),
        'corr_lt_05': int(np.sum(np.array(corrs) < 0.5)),
        'num_samples': len(clean_set),
    }

# ---- 1. 小波去噪 ----
print("=" * 60)
print("小波去噪 (Wavelet Denoising)")
print("=" * 60)

try:
    import pywt
    best_result = None
    best_config = None

    for wavelet in ['db4', 'db6', 'db8', 'sym4', 'sym6', 'coif3']:
        for level in [4, 5, 6, 7]:
            denoised = []
            for i in range(len(noisy_signals)):
                coeffs = pywt.wavedec(noisy_signals[i], wavelet, level=level)
                # 使用 VisuShrink 通用阈值
                sigma_est = np.median(np.abs(coeffs[-1])) / 0.6745
                threshold = sigma_est * np.sqrt(2 * np.log(len(noisy_signals[i])))
                coeffs_thresh = [coeffs[0]]  # 逼近系数不阈值化
                for c in coeffs[1:]:
                    coeffs_thresh.append(pywt.threshold(c, threshold, mode='soft'))
                denoised.append(pywt.waverec(coeffs_thresh, wavelet)[:len(noisy_signals[i])])
            denoised = np.array(denoised)

            r = calc_per_sample(clean_signals, denoised)
            if best_result is None or r['snr_imp_mean'] > best_result['snr_imp_mean']:
                best_result = r
                best_config = (wavelet, level)

    wavelet, level = best_config
    print(f"  最优配置: {wavelet}, 分解层数 {level}")
    print(f"  SNR 提升: {best_result['snr_imp_mean']:.2f} ± {best_result['snr_imp_std']:.2f} dB")
    print(f"  相关系数: {best_result['corr_mean']:.4f}（中位数 {best_result['corr_median']:.4f}）")
    print(f"  >0.7: {best_result['corr_gt_07']}/{best_result['num_samples']}")
    print(f"  <0.5: {best_result['corr_lt_05']}/{best_result['num_samples']}")
    wavelet_result = best_result
except ImportError:
    print("  ❌ 未安装 PyWavelets，请运行: pip install PyWavelets")
    wavelet_result = None

# ---- 2. 维纳滤波 ----
print("\n" + "=" * 60)
print("维纳滤波 (Wiener Filter)")
print("=" * 60)

try:
    from scipy.signal import wiener
    denoised = []
    for i in range(len(noisy_signals)):
        d = wiener(noisy_signals[i], mysize=17)  # 窗口大小 17（≈ 1 个脉冲宽度）
        denoised.append(d)
    denoised = np.array(denoised)

    wiener_result = calc_per_sample(clean_signals, denoised)
    print(f"  SNR 提升: {wiener_result['snr_imp_mean']:.2f} ± {wiener_result['snr_imp_std']:.2f} dB")
    print(f"  相关系数: {wiener_result['corr_mean']:.4f}（中位数 {wiener_result['corr_median']:.4f}）")
    print(f"  >0.7: {wiener_result['corr_gt_07']}/{wiener_result['num_samples']}")
    print(f"  <0.5: {wiener_result['corr_lt_05']}/{wiener_result['num_samples']}")
except ImportError:
    print("  ❌ 未安装 scipy，请运行: pip install scipy")
    wiener_result = None

# ---- 3. 汇总对比 ----
print("\n" + "=" * 60)
print("📊 方法对比")
print("=" * 60)
print(f"{'方法':<20} {'SNR提升':>12} {'相关系数':>10} {'>0.7':>8} {'<0.5':>8}")
print("-" * 60)
print(f"{'带噪信号（无处理）':<20} {'0.00 dB':>12} {'—':>10} {'—':>8} {'—':>8}")
if wavelet_result:
    print(f"{'小波去噪':<20} {wavelet_result['snr_imp_mean']:+8.2f} dB {wavelet_result['corr_mean']:>10.4f} "
          f"{wavelet_result['corr_gt_07']:>5}/{wavelet_result['num_samples']:<3} "
          f"{wavelet_result['corr_lt_05']:>5}/{wavelet_result['num_samples']:<3}")
if wiener_result:
    print(f"{'维纳滤波':<20} {wiener_result['snr_imp_mean']:+8.2f} dB {wiener_result['corr_mean']:>10.4f} "
          f"{wiener_result['corr_gt_07']:>5}/{wiener_result['num_samples']:<3} "
          f"{wiener_result['corr_lt_05']:>5}/{wiener_result['num_samples']:<3}")
print(f"{'本文方法（DDPM）':<20} {'+13.59 dB':>12} {'0.809':>10} "
      f"{'  82/100':>8} {'   0/100':>8}")
print("-" * 60)

# ---- 4. 保存 ----
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs_classic", "baseline_comparison.txt")
with open(save_path, 'w') as f:
    f.write("传统方法对比结果 (t=300)\n")
    f.write("=" * 60 + "\n")
    f.write(f"{'方法':<20} {'SNR提升':>12} {'相关系数':>10} {'>0.7':>8} {'<0.5':>8}\n")
    f.write("-" * 60 + "\n")
    if wavelet_result:
        f.write(f"{'小波去噪':<20} {wavelet_result['snr_imp_mean']:+8.2f} dB {wavelet_result['corr_mean']:>10.4f} "
                f"{wavelet_result['corr_gt_07']:>5}/{wavelet_result['num_samples']:<3} "
                f"{wavelet_result['corr_lt_05']:>5}/{wavelet_result['num_samples']:<3}\n")
    if wiener_result:
        f.write(f"{'维纳滤波':<20} {wiener_result['snr_imp_mean']:+8.2f} dB {wiener_result['corr_mean']:>10.4f} "
                f"{wiener_result['corr_gt_07']:>5}/{wiener_result['num_samples']:<3} "
                f"{wiener_result['corr_lt_05']:>5}/{wiener_result['num_samples']:<3}\n")
    f.write(f"{'本文方法（DDPM）':<20} {'+13.59 dB':>12} {'0.809':>10} "
            f"{'  82/100':>8} {'   0/100':>8}\n")

print(f"\n✅ 结果已保存到 {save_path}")
