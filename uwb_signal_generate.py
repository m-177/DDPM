import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体（解决中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# -------------------------------
# 1️⃣ 参数设置
# -------------------------------
Fs = 20e9                          # 20GHz 采样率
num_signals = 1000                # 生成信号数
seq_length = 10120                 # 信号长度（与 8GHz/4048 点相同时长 0.506μs，且为8的倍数）

# 脉冲参数（与 uwb_data(1).py 一致）
num_pulses_range = (30, 60)        # 随机脉冲数范围
sigma_range = (0.15e-9, 0.4e-9)   # 高斯二阶导 σ 范围 (秒)
jitter_ratio = 0.45                # 脉冲位置抖动比例 (±45% PRI)

dt = 1 / Fs

print(f"采样率: {Fs/1e9:.0f} GHz")
print(f"信号长度: {seq_length} 点 ({seq_length/Fs*1e9:.1f} ns)")
print(f"脉冲数范围: {num_pulses_range[0]}~{num_pulses_range[1]}")
print(f"σ 范围: {sigma_range[0]*1e9:.2f}~{sigma_range[1]*1e9:.2f} ns")
print(f"脉冲抖动: ±{jitter_ratio*100:.0f}% PRI")

# -------------------------------
# 2️⃣ 高斯二阶导函数（与 uwb_data(1).py 一致）
# -------------------------------
def generate_gaussian_second_derivative(sigma):
    """
    生成高斯二阶导脉冲模板 (Ricker/Mexican Hat 小波)
    :param sigma: 脉冲宽度参数 (秒)
    :return: pulse (脉冲波形), t_idx (采样索引偏移)
    """
    t_limit = 6 * sigma
    t_idx = np.arange(-int(t_limit * Fs), int(t_limit * Fs) + 1)
    t_time = t_idx * dt
    pulse = (1 - (t_time / sigma)**2) * np.exp(-0.5 * (t_time / sigma)**2)
    return pulse.astype(np.float32), t_idx

# -------------------------------
# 3️⃣ 批量生成干净信号（随机脉冲流）
# -------------------------------
print(f"\n开始生成 {num_signals} 个UWB干净信号（随机脉冲流 + 高斯二阶导）...")

signals_time_clean = []

for idx in range(num_signals):
    # 初始化空白信号
    signal = np.zeros(seq_length, dtype=np.float32)

    # 随机确定脉冲数和脉冲宽度
    num_pulses = np.random.randint(num_pulses_range[0], num_pulses_range[1] + 1)
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])

    # 生成脉冲模板
    pulse_shape, t_idx = generate_gaussian_second_derivative(sigma)

    # 平均脉冲间隔（采样点数）
    avg_pri_pts = seq_length // (num_pulses + 1)

    # 逐个脉冲叠加
    for i in range(num_pulses):
        base_pos = (i + 1) * avg_pri_pts
        polarity = np.random.choice([-1, 1])
        jitter = np.random.randint(-int(avg_pri_pts * jitter_ratio),
                                   int(avg_pri_pts * jitter_ratio))
        actual_center = base_pos + jitter

        # 计算脉冲在信号数组中的放置位置（边缘裁剪处理）
        start_idx = actual_center + t_idx[0]
        end_idx = actual_center + t_idx[-1] + 1

        pulse_start = 0
        pulse_end = len(pulse_shape)

        if start_idx < 0:
            pulse_start = -start_idx
            start_idx = 0
        if end_idx > seq_length:
            pulse_end = pulse_end - (end_idx - seq_length)
            end_idx = seq_length

        if start_idx < end_idx and pulse_start < pulse_end:
            signal[start_idx:end_idx] += (polarity * pulse_shape[pulse_start:pulse_end])

    signals_time_clean.append(signal)

    if (idx + 1) % 2000 == 0:
        print(f"  已生成 {idx + 1}/{num_signals}")

# -------------------------------
# 4️⃣ 统一长度（8的倍数，已满足 seq_length=10120=8×1265）
# -------------------------------
signals_time_clean = np.array(signals_time_clean)
actual_length = signals_time_clean.shape[1]
uniform_length = ((actual_length + 7) // 8) * 8
print(f"\n原始长度: {actual_length}")
print(f"统一后长度: {uniform_length} (8的倍数: {uniform_length % 8 == 0})")

if actual_length < uniform_length:
    padded = np.zeros((num_signals, uniform_length), dtype=np.float32)
    padded[:, :actual_length] = signals_time_clean
    signals_time_clean = padded

# -------------------------------
# 5️⃣ 保存信号
# -------------------------------
np.save('uwb_signals_time_clean.npy', signals_time_clean)

print(f"\n✅ 保存完成!")
print(f"  形状: {signals_time_clean.shape}")
print(f"  内存: {signals_time_clean.nbytes / 1024**2:.1f} MB")

# -------------------------------
# 6️⃣ 绘制5个样本的时域波形图（无重叠）
# -------------------------------
np.random.seed(42)
sample_indices = np.random.choice(num_signals, 5, replace=False)
print(f"\n挑选的样本索引: {sample_indices}")

# 设置显示范围：前2000个采样点（对应100纳秒，便于观察细节）
display_samples = 2000
print(f"显示前 {display_samples} 个采样点（{display_samples/Fs*1e9:.1f} 纳秒）")

# 创建5个子图，垂直排列
fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

time_axis = np.arange(display_samples) / Fs * 1e9

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for i, idx in enumerate(sample_indices):
    signal = signals_time_clean[idx]

    axes[i].plot(time_axis, signal[:display_samples], color=colors[i], linewidth=0.8)
    axes[i].axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    axes[i].set_ylabel('幅度', fontsize=10)
    axes[i].grid(True, alpha=0.3)
    axes[i].set_title(f'样本 {idx}', fontsize=11, fontweight='bold', loc='left', pad=5)

    signal_display = signal[:display_samples]
    stats_text = f'max={signal_display.max():.3f} | min={signal_display.min():.3f} | std={signal_display.std():.3f}'
    axes[i].text(0.98, 0.95, stats_text, transform=axes[i].transAxes,
                 fontsize=8, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

axes[-1].set_xlabel('时间 [ns]', fontsize=11)

fig.suptitle(f'UWB信号时域波形（5个随机样本，前{display_samples}个采样点）', fontsize=14, fontweight='bold', y=0.995)

plt.tight_layout()
plt.subplots_adjust(hspace=0.4, top=0.95)

plt.savefig('uwb_5_samples_waveform.png', dpi=200, bbox_inches='tight', facecolor='white')
print(f"\n✅ 波形图已保存: uwb_5_samples_waveform.png")

plt.show()

# -------------------------------
# 7️⃣ 额外统计信息
# -------------------------------
print(f"\n📊 5个样本统计信息（基于显示的前{display_samples}个采样点）:")
for i, idx in enumerate(sample_indices):
    signal = signals_time_clean[idx]
    signal_display = signal[:display_samples]
    print(f"  样本 {idx}: 范围 [{signal_display.min():.4f}, {signal_display.max():.4f}], 标准差 {signal_display.std():.4f}")

# -------------------------------
# 8️⃣ 验证数据完整性
# -------------------------------
test_load = np.load('uwb_signals_time_clean.npy')
print(f"\n🔍 数据完整性验证:")
print(f"  保存形状: {signals_time_clean.shape}")
print(f"  加载形状: {test_load.shape}")
print(f"  数据一致: {np.array_equal(signals_time_clean, test_load)}")
print(f"  ✅ 信号长度是8的倍数: {uniform_length % 8 == 0}")

print("\n🎉 所有任务完成！")