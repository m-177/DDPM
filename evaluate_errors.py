# -----------------------------
# evaluate_errors.py - 基于 result.npy 数据评估极性反转和波形丢失占比
# 
# 功能：
#   1. 读取 train.py 保存的 result.npy 文件（包含最佳epoch的干净信号、加噪信号、去噪信号）
#   2. 检测极性反转：检查去噪信号与干净信号的峰值符号是否相反
#   3. 检测波形丢失：检查去噪信号中脉冲峰值是否严重衰减或消失
#   4. 统计两种错误的占比，并详细记录每个错误的个数
#
# 使用方式：
#   python evaluate_errors.py
# -----------------------------

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


# -----------------------------
# 1. 极性反转检测（基于整个信号）
# -----------------------------
def detect_polarity_reversal_full_signal(clean_signal, denoised_signal,
                                          threshold_ratio=0.3,
                                          min_pulse_distance=100):
    """
    在整个信号上检测极性反转。
    
    判断逻辑：
    1. 找到干净信号中所有显著脉冲的峰值位置（幅度 > 最大幅度*0.1）
    2. 对每个脉冲峰值，检查去噪信号在该位置的符号是否与干净信号相反
    3. 如果符号相反且去噪信号幅度足够大（> 干净幅度 * threshold_ratio），判定为极性反转
    
    Returns:
        reversed_count: 发生极性反转的脉冲数
        total_pulses: 检测的总脉冲数
        reversed_positions: 发生反转的峰值位置列表
        details: 每个脉冲的详细信息列表
    """
    abs_clean = np.abs(clean_signal)
    max_amp = np.max(abs_clean)
    
    if max_amp < 0.01:
        return 0, 0, [], []

    # 找到所有显著峰值（幅度 > 最大幅度 * 0.1）
    peak_positions = []
    temp_signal = abs_clean.copy()
    min_peak_height = max_amp * 0.1

    while True:
        if np.max(temp_signal) < min_peak_height:
            break
        pos = np.argmax(temp_signal)
        peak_positions.append(pos)
        # 将该峰值附近区域置零，避免重复检测
        start = max(0, pos - min_pulse_distance // 2)
        end = min(len(temp_signal), pos + min_pulse_distance // 2)
        temp_signal[start:end] = 0

    reversed_count = 0
    reversed_positions = []
    details = []

    for pos in peak_positions:
        clean_val = clean_signal[pos]
        denoised_val = denoised_signal[pos]

        clean_sign = np.sign(clean_val)
        denoised_sign = np.sign(denoised_val)

        is_reversed = (clean_sign != denoised_sign and
                       abs(denoised_val) > abs(clean_val) * threshold_ratio)

        if is_reversed:
            reversed_count += 1
            reversed_positions.append(pos)

        details.append({
            'position': pos,
            'clean_value': float(clean_val),
            'denoised_value': float(denoised_val),
            'clean_sign': int(clean_sign),
            'denoised_sign': int(denoised_sign),
            'is_reversed': is_reversed
        })

    return reversed_count, len(peak_positions), reversed_positions, details


# -----------------------------
# 2. 波形丢失检测（基于整个信号）
# -----------------------------
def detect_waveform_loss_full_signal(clean_signal, denoised_signal,
                                      amplitude_threshold=0.5,
                                      energy_threshold=0.3, min_pulse_distance=100):
    """
    在整个信号上检测波形丢失。
    
    判断逻辑：
    1. 找到干净信号中所有显著脉冲的峰值位置（幅度 > 最大幅度*0.1）
    2. 对每个脉冲峰值，检查去噪信号的幅度和能量是否严重衰减
    - 幅度丢失：去噪信号峰值幅度低于干净峰值的 amplitude_threshold
    - 能量丢失：去噪信号在脉冲区域的能量低于干净信号的 energy_threshold
    
    Returns:
        lost_count: 发生波形丢失的脉冲数
        total_pulses: 检测的总脉冲数
        lost_positions: 发生丢失的峰值位置列表
        details: 每个脉冲的详细信息列表
    """
    abs_clean = np.abs(clean_signal)
    max_amp = np.max(abs_clean)
    
    if max_amp < 0.01:
        return 0, 0, [], []

    # 找到所有显著峰值（幅度 > 最大幅度 * 0.1）
    peak_positions = []
    temp_signal = abs_clean.copy()
    min_peak_height = max_amp * 0.1

    while True:
        if np.max(temp_signal) < min_peak_height:
            break
        pos = np.argmax(temp_signal)
        peak_positions.append(pos)
        start = max(0, pos - min_pulse_distance // 2)
        end = min(len(temp_signal), pos + min_pulse_distance // 2)
        temp_signal[start:end] = 0

    lost_count = 0
    lost_positions = []
    details = []

    for pos in peak_positions:
        clean_val = clean_signal[pos]
        denoised_val = denoised_signal[pos]

        window = 50
        start = max(0, pos - window)
        end = min(len(clean_signal), pos + window)

        clean_energy = float(np.sum(clean_signal[start:end] ** 2))
        denoised_energy = float(np.sum(denoised_signal[start:end] ** 2))

        amp_ratio = abs(denoised_val) / (abs(clean_val) + 1e-8)
        energy_ratio = denoised_energy / (clean_energy + 1e-8)

        amp_lost = amp_ratio < amplitude_threshold
        energy_lost = energy_ratio < energy_threshold

        is_lost = amp_lost or energy_lost

        if is_lost:
            lost_count += 1
            lost_positions.append(pos)

        # 确定丢失类型
        if amp_lost and energy_lost:
            loss_type = 'both'
        elif amp_lost:
            loss_type = 'amplitude'
        elif energy_lost:
            loss_type = 'energy'
        else:
            loss_type = 'none'

        details.append({
            'position': pos,
            'clean_value': float(clean_val),
            'denoised_value': float(denoised_val),
            'amp_ratio': float(amp_ratio),
            'energy_ratio': float(energy_ratio),
            'is_lost': is_lost,
            'loss_type': loss_type
        })

    return lost_count, len(peak_positions), lost_positions, details


# -----------------------------
# 3. 基于 result.npy 的评估函数
# -----------------------------
def evaluate_from_result(result_path="./saved_models_classic/result.npy",
                          amplitude_threshold=0.5,
                          energy_threshold=0.3,
                          polarity_threshold_ratio=0.3,
                          verbose=True):
    """
    从 result.npy 文件读取数据，评估极性反转和波形丢失的占比。
    
    基于整个信号检测所有显著脉冲，统计极性反转和波形丢失的错误占比。
    
    Args:
        result_path: result.npy 文件路径
        amplitude_threshold: 波形丢失的幅度阈值
        energy_threshold: 波形丢失的能量阈值
        polarity_threshold_ratio: 极性反转的幅度比例阈值
        verbose: 是否打印详细信息
    
    Returns:
        results: dict，包含所有统计结果
    """
    # 检查文件是否存在
    if not os.path.exists(result_path):
        print(f"❌ 错误: 找不到 result.npy 文件: {result_path}")
        print("   请先运行 train.py 训练，生成 result.npy 文件")
        return None

    # 加载数据
    print(f"\n加载数据: {result_path}")
    data = np.load(result_path, allow_pickle=True).item()

    clean_signals = data['clean_signals']
    noisy_signals = data['noisy_signals']
    denoised_signals = data['denoised_signals']
    input_snrs = data['input_snrs']
    output_snrs = data['output_snrs']
    best_epoch = data['best_epoch']
    eval_t = data['eval_t']

    total_samples = len(clean_signals)

    print(f"  最佳epoch: {best_epoch}")
    print(f"  评估时间步: t={eval_t}")
    print(f"  样本总数: {total_samples}")
    print(f"  平均输入SNR: {np.mean(input_snrs):.2f} dB")
    print(f"  平均输出SNR: {np.mean(output_snrs):.2f} dB")

    # 初始化统计结果
    results = {
        'best_epoch': int(best_epoch),
        'eval_t': int(eval_t),
        'total_samples': total_samples,
        'avg_input_snr': float(np.mean(input_snrs)),
        'avg_output_snr': float(np.mean(output_snrs)),
        'amplitude_threshold': amplitude_threshold,
        'energy_threshold': energy_threshold,
        'polarity_threshold_ratio': polarity_threshold_ratio,
        'polarity_reversal': {
            'sample_count': 0,        # 发生极性反转的样本数
            'total_pulses': 0,        # 检测的总脉冲数
            'reversed_pulses': 0,     # 反转的脉冲总数
            'sample_details': [],     # 每个样本的详细信息
            'pulse_details': [],      # 每个脉冲的详细信息
        },
        'waveform_loss': {
            'sample_count': 0,        # 发生波形丢失的样本数
            'total_pulses': 0,        # 检测的总脉冲数
            'lost_pulses': 0,         # 丢失的脉冲总数
            'lost_by_amplitude': 0,   # 因幅度衰减导致的丢失数
            'lost_by_energy': 0,      # 因能量衰减导致的丢失数
            'lost_by_both': 0,        # 两者都导致的丢失数
            'sample_details': [],     # 每个样本的详细信息
            'pulse_details': [],      # 每个脉冲的详细信息
        }
    }

    # 逐样本检测
    for idx in range(total_samples):
        clean_np = clean_signals[idx]
        denoised_np = denoised_signals[idx]

        # --- 极性反转检测（基于整个信号） ---
        rev_count, total_pulses_pr, rev_positions, rev_details = detect_polarity_reversal_full_signal(
            clean_np, denoised_np,
            threshold_ratio=polarity_threshold_ratio
        )

        results['polarity_reversal']['total_pulses'] += total_pulses_pr
        results['polarity_reversal']['reversed_pulses'] += rev_count

        if rev_count > 0:
            results['polarity_reversal']['sample_count'] += 1
            results['polarity_reversal']['sample_details'].append({
                'sample_idx': idx,
                'reversed_count': rev_count,
                'total_pulses': total_pulses_pr,
                'reversed_positions': [int(p) for p in rev_positions],
                'input_snr': float(input_snrs[idx]),
                'output_snr': float(output_snrs[idx]),
            })

        # 记录每个脉冲的详细信息
        for d in rev_details:
            d_copy = d.copy()
            d_copy['sample_idx'] = idx
            d_copy['input_snr'] = float(input_snrs[idx])
            d_copy['output_snr'] = float(output_snrs[idx])
            results['polarity_reversal']['pulse_details'].append(d_copy)

        # --- 波形丢失检测（基于整个信号） ---
        lost_count, total_pulses_wl, lost_positions, wl_details = detect_waveform_loss_full_signal(
            clean_np, denoised_np,
            amplitude_threshold=amplitude_threshold,
            energy_threshold=energy_threshold
        )

        results['waveform_loss']['total_pulses'] += total_pulses_wl
        results['waveform_loss']['lost_pulses'] += lost_count

        # 统计丢失类型
        for d in wl_details:
            if d['is_lost']:
                if d['loss_type'] == 'amplitude':
                    results['waveform_loss']['lost_by_amplitude'] += 1
                elif d['loss_type'] == 'energy':
                    results['waveform_loss']['lost_by_energy'] += 1
                elif d['loss_type'] == 'both':
                    results['waveform_loss']['lost_by_both'] += 1

        if lost_count > 0:
            results['waveform_loss']['sample_count'] += 1
            results['waveform_loss']['sample_details'].append({
                'sample_idx': idx,
                'lost_count': lost_count,
                'total_pulses': total_pulses_wl,
                'lost_positions': [int(p) for p in lost_positions],
                'input_snr': float(input_snrs[idx]),
                'output_snr': float(output_snrs[idx]),
            })

        # 记录每个脉冲的详细信息
        for d in wl_details:
            d_copy = d.copy()
            d_copy['sample_idx'] = idx
            d_copy['input_snr'] = float(input_snrs[idx])
            d_copy['output_snr'] = float(output_snrs[idx])
            results['waveform_loss']['pulse_details'].append(d_copy)

    # 打印统计结果
    if verbose:
        print("\n" + "=" * 70)
        print("去噪错误评估结果（基于 result.npy）")
        print("=" * 70)
        print(f"最佳epoch: {best_epoch}")
        print(f"评估时间步: t={eval_t}")
        print(f"评估样本数: {total_samples}")
        print(f"极性反转阈值: 幅度比例 > {polarity_threshold_ratio}")
        print(f"波形丢失阈值: 幅度 < {amplitude_threshold} 或 能量 < {energy_threshold}")
        print(f"平均 SNR: {np.mean(input_snrs):.2f} → {np.mean(output_snrs):.2f} dB")
        print("=" * 70)

        pr = results['polarity_reversal']
        wl = results['waveform_loss']

        # --- 极性反转统计 ---
        print(f"\n[Chart] 极性反转 (Polarity Reversal)")
        print("-" * 40)
        rev_sample_ratio = pr['sample_count'] / total_samples * 100
        rev_pulse_ratio = pr['reversed_pulses'] / max(pr['total_pulses'], 1) * 100
        print(f"  发生反转的样本数: {pr['sample_count']}/{total_samples} ({rev_sample_ratio:.2f}%)")
        print(f"  反转脉冲总数: {pr['reversed_pulses']}/{pr['total_pulses']} ({rev_pulse_ratio:.2f}%)")
        if pr['sample_details']:
            print(f"\n  反转样本详情:")
            for s in pr['sample_details']:
                print(f"    样本 {s['sample_idx']}: {s['reversed_count']}个脉冲反转 "
                      f"(位置: {s['reversed_positions']}, SNR: {s['input_snr']:.1f}→{s['output_snr']:.1f}dB)")

        # --- 波形丢失统计 ---
        print(f"\n[Chart] 波形丢失 (Waveform Loss)")
        print("-" * 40)
        lost_sample_ratio = wl['sample_count'] / total_samples * 100
        lost_pulse_ratio = wl['lost_pulses'] / max(wl['total_pulses'], 1) * 100
        print(f"  发生丢失的样本数: {wl['sample_count']}/{total_samples} ({lost_sample_ratio:.2f}%)")
        print(f"  丢失脉冲总数: {wl['lost_pulses']}/{wl['total_pulses']} ({lost_pulse_ratio:.2f}%)")
        print(f"  丢失类型分布:")
        print(f"    幅度衰减: {wl['lost_by_amplitude']} 个脉冲")
        print(f"    能量衰减: {wl['lost_by_energy']} 个脉冲")
        print(f"    两者兼有: {wl['lost_by_both']} 个脉冲")
        if wl['sample_details']:
            print(f"\n  丢失样本详情:")
            for s in wl['sample_details']:
                print(f"    样本 {s['sample_idx']}: {s['lost_count']}个脉冲丢失 "
                      f"(位置: {s['lost_positions']}, SNR: {s['input_snr']:.1f}→{s['output_snr']:.1f}dB)")

        # --- 综合统计 ---
        print(f"\n[Chart] 综合统计")
        print("-" * 40)
        # 同时发生两种错误的样本
        both_error_samples = set()
        for s in pr['sample_details']:
            both_error_samples.add(s['sample_idx'])
        for s in wl['sample_details']:
            both_error_samples.add(s['sample_idx'])
        total_error_samples = len(both_error_samples)
        error_ratio = total_error_samples / total_samples * 100
        print(f"  存在至少一种错误的样本: {total_error_samples}/{total_samples} ({error_ratio:.2f}%)")
        print(f"  完全正确的样本: {total_samples - total_error_samples}/{total_samples} "
              f"({(total_samples - total_error_samples) / total_samples * 100:.2f}%)")

        print("\n" + "=" * 70)

    return results


# -----------------------------
# 4. 可视化错误样本
# -----------------------------
def visualize_error_samples(clean_signal, noisy_signal, denoised_signal,
                             reversed_details=None, wl_details=None,
                             sample_idx=0, eval_t=300, save_dir="./error_vis"):
    """
    可视化发生错误的样本，标注极性反转和波形丢失的位置。
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8))

    x_axis = np.arange(len(clean_signal[:1000]))

    # 上半图：干净信号
    axes[0].plot(x_axis, clean_signal[:1000], 'g-', linewidth=1, label='Clean')
    axes[0].set_title(f'Sample {sample_idx}: Clean Signal')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 中间图：加噪信号
    axes[1].plot(x_axis, noisy_signal[:1000], 'r-', linewidth=1, label='Noisy')
    axes[1].set_title(f'Sample {sample_idx}: Noisy Signal (t={eval_t})')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # 下半图：去噪信号 + 标注错误
    axes[2].plot(x_axis, denoised_signal[:1000], 'b-', linewidth=1, label='Denoised')
    axes[2].set_title(f'Sample {sample_idx}: Denoised Signal (t={eval_t})')
    axes[2].set_xlabel('Sample')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    # 标注极性反转位置
    if reversed_details:
        for d in reversed_details:
            pos = d['position']
            if pos < 1000 and d['is_reversed']:
                axes[2].axvline(x=pos, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
                axes[2].annotate(f'Polarity\nReversal\n({d["clean_value"]:.2f}→{d["denoised_value"]:.2f})',
                                 xy=(pos, denoised_signal[pos]),
                                 xytext=(pos + 30, denoised_signal[pos] + 0.4),
                                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                                 fontsize=7, color='red', fontweight='bold',
                                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # 标注波形丢失位置
    if wl_details:
        for d in wl_details:
            pos = d['position']
            if pos < 1000 and d['is_lost']:
                axes[2].axvline(x=pos, color='orange', linestyle=':', alpha=0.7, linewidth=1.5)
                loss_desc = f'Amp:{d["amp_ratio"]:.1%} Eng:{d["energy_ratio"]:.1%}'
                axes[2].annotate(f'Waveform\nLoss\n({loss_desc})',
                                 xy=(pos, denoised_signal[pos]),
                                 xytext=(pos + 30, denoised_signal[pos] - 0.4),
                                 arrowprops=dict(arrowstyle='->', color='orange', lw=1.5),
                                 fontsize=7, color='orange', fontweight='bold',
                                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'error_sample{sample_idx}_t{eval_t}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()

    return save_path


# -----------------------------
# 5. 保存详细报告到文本文件
# -----------------------------
def save_report(results, save_path=None):
    if save_path is None:
        # 默认保存到 evaluate_errors.py 所在目录
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "error_report.txt")
    """将评估结果保存到文本文件"""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("去噪错误评估报告\n")
        f.write("=" * 70 + "\n")
        f.write(f"最佳epoch: {results['best_epoch']}\n")
        f.write(f"评估时间步: t={results['eval_t']}\n")
        f.write(f"评估样本数: {results['total_samples']}\n")
        f.write(f"检测脉冲数/样本: 自动检测所有显著脉冲\n")
        f.write(f"极性反转阈值: 幅度比例 > {results['polarity_threshold_ratio']}\n")
        f.write(f"波形丢失阈值: 幅度 < {results['amplitude_threshold']} 或 能量 < {results['energy_threshold']}\n")
        f.write(f"平均 SNR: {results['avg_input_snr']:.2f} → {results['avg_output_snr']:.2f} dB\n")
        f.write("=" * 70 + "\n")

        pr = results['polarity_reversal']
        wl = results['waveform_loss']

        # 极性反转
        f.write("\n📊 极性反转 (Polarity Reversal)\n")
        f.write("-" * 40 + "\n")
        rev_sample_ratio = pr['sample_count'] / results['total_samples'] * 100
        rev_pulse_ratio = pr['reversed_pulses'] / max(pr['total_pulses'], 1) * 100
        f.write(f"发生反转的样本数: {pr['sample_count']}/{results['total_samples']} ({rev_sample_ratio:.2f}%)\n")
        f.write(f"反转脉冲总数: {pr['reversed_pulses']}/{pr['total_pulses']} ({rev_pulse_ratio:.2f}%)\n")

        if pr['sample_details']:
            f.write(f"\n反转样本详情:\n")
            for s in pr['sample_details']:
                f.write(f"  样本 {s['sample_idx']}: {s['reversed_count']}个脉冲反转 "
                        f"(位置: {s['reversed_positions']}, SNR: {s['input_snr']:.1f}→{s['output_snr']:.1f}dB)\n")

        # 所有脉冲的极性反转详情
        f.write(f"\n所有脉冲极性检测详情:\n")
        for d in pr['pulse_details']:
            status = "✅" if not d['is_reversed'] else "❌反转"
            f.write(f"  样本{d['sample_idx']} 位置{d['position']}: "
                    f"干净={d['clean_value']:.4f}(符号={d['clean_sign']:+d}), "
                    f"去噪={d['denoised_value']:.4f}(符号={d['denoised_sign']:+d}) {status}\n")

        # 波形丢失
        f.write(f"\n📊 波形丢失 (Waveform Loss)\n")
        f.write("-" * 40 + "\n")
        lost_sample_ratio = wl['sample_count'] / results['total_samples'] * 100
        lost_pulse_ratio = wl['lost_pulses'] / max(wl['total_pulses'], 1) * 100
        f.write(f"发生丢失的样本数: {wl['sample_count']}/{results['total_samples']} ({lost_sample_ratio:.2f}%)\n")
        f.write(f"丢失脉冲总数: {wl['lost_pulses']}/{wl['total_pulses']} ({lost_pulse_ratio:.2f}%)\n")
        f.write(f"丢失类型分布:\n")
        f.write(f"  幅度衰减: {wl['lost_by_amplitude']} 个脉冲\n")
        f.write(f"  能量衰减: {wl['lost_by_energy']} 个脉冲\n")
        f.write(f"  两者兼有: {wl['lost_by_both']} 个脉冲\n")

        if wl['sample_details']:
            f.write(f"\n丢失样本详情:\n")
            for s in wl['sample_details']:
                f.write(f"  样本 {s['sample_idx']}: {s['lost_count']}个脉冲丢失 "
                        f"(位置: {s['lost_positions']}, SNR: {s['input_snr']:.1f}→{s['output_snr']:.1f}dB)\n")

        # 所有脉冲的波形丢失详情
        f.write(f"\n所有脉冲波形检测详情:\n")
        for d in wl['pulse_details']:
            if d['is_lost']:
                status = f"❌丢失({d['loss_type']})"
            else:
                status = "✅"
            f.write(f"  样本{d['sample_idx']} 位置{d['position']}: "
                    f"幅度比={d['amp_ratio']:.2%}, 能量比={d['energy_ratio']:.2%} {status}\n")

        # 综合统计
        f.write(f"\n📊 综合统计\n")
        f.write("-" * 40 + "\n")
        both_error_samples = set()
        for s in pr['sample_details']:
            both_error_samples.add(s['sample_idx'])
        for s in wl['sample_details']:
            both_error_samples.add(s['sample_idx'])
        total_error_samples = len(both_error_samples)
        error_ratio = total_error_samples / results['total_samples'] * 100
        f.write(f"存在至少一种错误的样本: {total_error_samples}/{results['total_samples']} ({error_ratio:.2f}%)\n")
        f.write(f"完全正确的样本: {results['total_samples'] - total_error_samples}/{results['total_samples']} "
                f"({(results['total_samples'] - total_error_samples) / results['total_samples'] * 100:.2f}%)\n")

        f.write("\n" + "=" * 70 + "\n")

    print(f"\n[File] 详细报告已保存到: {save_path}")
    return save_path


# -----------------------------
# 6. 主函数
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='基于 result.npy 评估去噪结果中的极性反转和波形丢失')
    parser.add_argument('--result_path', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "result.npy"),
                        help='result.npy 文件路径')
    parser.add_argument('--amplitude_threshold', type=float, default=0.5,
                        help='波形丢失的幅度阈值（默认0.5，即去噪峰值低于干净峰值50%判定为丢失）')
    parser.add_argument('--energy_threshold', type=float, default=0.3,
                        help='波形丢失的能量阈值（默认0.3，即去噪能量低于干净能量30%判定为丢失）')
    parser.add_argument('--polarity_threshold', type=float, default=0.3,
                        help='极性反转的幅度比例阈值（默认0.3）')
    parser.add_argument('--num_pulses', type=int, default=5,
                        help='每个样本检测的脉冲数（默认5）')
    parser.add_argument('--visualize', action='store_true',
                        help='是否可视化错误样本')
    parser.add_argument('--max_viz', type=int, default=5,
                        help='最大可视化样本数')
    parser.add_argument('--save_report', action='store_true', default=True,
                        help='是否保存详细报告到文本文件')

    args = parser.parse_args()

    print("=" * 60)
    print("去噪错误评估 - 极性反转 & 波形丢失")
    print("（基于 result.npy 数据）")
    print("=" * 60)
    print(f"数据文件: {args.result_path}")
    print(f"波形丢失阈值: 幅度<{args.amplitude_threshold} 或 能量<{args.energy_threshold}")
    print(f"极性反转阈值: 幅度比例>{args.polarity_threshold}")
    print("=" * 60)

    # 执行评估
    results = evaluate_from_result(
        result_path=args.result_path,
        amplitude_threshold=args.amplitude_threshold,
        energy_threshold=args.energy_threshold,
        polarity_threshold_ratio=args.polarity_threshold,
        verbose=True
    )

    if results is None:
        exit(1)

    # 保存详细报告
    if args.save_report:
        save_report(results)

    # 可视化错误样本
    if args.visualize:
        print("\n可视化错误样本...")
        data = np.load(args.result_path, allow_pickle=True).item()
        clean_signals = data['clean_signals']
        noisy_signals = data['noisy_signals']
        denoised_signals = data['denoised_signals']
        eval_t = data['eval_t']

        viz_count = 0
        for idx in range(len(clean_signals)):
            if viz_count >= args.max_viz:
                break

            clean_np = clean_signals[idx]
            noisy_np = noisy_signals[idx]
            denoised_np = denoised_signals[idx]

            # 检测错误（基于整个信号）
            rev_count, _, _, rev_details = detect_polarity_reversal_full_signal(
                clean_np, denoised_np,
                threshold_ratio=args.polarity_threshold
            )
            lost_count, _, _, wl_details = detect_waveform_loss_full_signal(
                clean_np, denoised_np,
                amplitude_threshold=args.amplitude_threshold,
                energy_threshold=args.energy_threshold
            )

            if rev_count > 0 or lost_count > 0:
                save_path = visualize_error_samples(
                    clean_np, noisy_np, denoised_np,
                    reversed_details=rev_details if rev_count > 0 else None,
                    wl_details=wl_details if lost_count > 0 else None,
                    sample_idx=viz_count,
                    eval_t=eval_t
                )
                print(f"  已保存: {save_path}")
                viz_count += 1

        if viz_count == 0:
            print("  未发现错误样本，无需可视化")

    print("\n[OK] 评估完成！")
