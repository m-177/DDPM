# check_data_leakage.py
import numpy as np
import torch
from scipy import stats
import os


def check_data_leakage(clean_path, noisy_path):
    """
    检查数据泄露的完整脚本
    """
    print("=" * 60)
    print("UWB数据泄露检查")
    print("=" * 60)

    # 1. 加载数据
    print("\n1. 加载数据...")
    clean_data = np.load(clean_path)
    noisy_data = np.load(noisy_path)

    print(f"   干净数据形状: {clean_data.shape}")
    print(f"   噪声数据形状: {noisy_data.shape}")

    # 2. 添加通道维度（如果需要）
    if clean_data.ndim == 3:
        clean_data = clean_data[:, np.newaxis, :, :]
        noisy_data = noisy_data[:, np.newaxis, :, :]
        print(f"   添加通道后: {clean_data.shape}")

    # 3. 划分数据集（模拟训练过程）
    print("\n2. 模拟数据划分...")
    total_samples = len(clean_data)
    val_ratio = 0.2
    val_size = int(total_samples * val_ratio)
    train_size = total_samples - val_size

    # 固定种子确保可重复
    np.random.seed(42)
    indices = np.random.permutation(total_samples)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    train_clean = clean_data[train_idx]
    train_noisy = noisy_data[train_idx]
    val_clean = clean_data[val_idx]
    val_noisy = noisy_data[val_idx]

    print(f"   训练集: {len(train_clean)} 个样本")
    print(f"   验证集: {len(val_clean)} 个样本")

    # 4. 检查数据泄露
    print("\n3. 数据泄露检查:")
    print("-" * 40)

    # 4.1 检查数据范围
    print("\n3.1 数据范围对比:")
    print(f"   训练集干净: [{train_clean.min():.4f}, {train_clean.max():.4f}]")
    print(f"   验证集干净: [{val_clean.min():.4f}, {val_clean.max():.4f}]")
    print(f"   训练集噪声: [{train_noisy.min():.4f}, {train_noisy.max():.4f}]")
    print(f"   验证集噪声: [{val_noisy.min():.4f}, {val_noisy.max():.4f}]")

    # 如果范围几乎一样，可能是泄露
    range_diff_clean = abs((train_clean.max() - train_clean.min()) -
                           (val_clean.max() - val_clean.min()))
    if range_diff_clean < 0.01:
        print("   ⚠️  训练集和验证集干净信号范围几乎相同！")

    # 4.2 检查均值和标准差
    print("\n3.2 均值和标准差:")
    print(f"   训练集干净: mean={train_clean.mean():.4f}, std={train_clean.std():.4f}")
    print(f"   验证集干净: mean={val_clean.mean():.4f}, std={val_clean.std():.4f}")
    print(f"   训练集噪声: mean={train_noisy.mean():.4f}, std={train_noisy.std():.4f}")
    print(f"   验证集噪声: mean={val_noisy.mean():.4f}, std={val_noisy.std():.4f}")

    # 4.3 KS检验（分布相似性）
    print("\n3.3 KS检验（p>0.05表示分布相似）:")

    # 干净信号
    ks_stat, ks_p = stats.ks_2samp(
        train_clean.flatten(),
        val_clean.flatten()
    )
    print(f"   干净信号: statistic={ks_stat:.4f}, p-value={ks_p:.4f}")
    if ks_p > 0.05:
        print("   ⚠️  干净信号分布相似，可能有泄露！")

    # 噪声信号
    ks_stat, ks_p = stats.ks_2samp(
        train_noisy.flatten(),
        val_noisy.flatten()
    )
    print(f"   噪声信号: statistic={ks_stat:.4f}, p-value={ks_p:.4f}")
    if ks_p > 0.05:
        print("   ⚠️  噪声信号分布相似，可能有泄露！")

    # 4.4 检查是否有相同样本
    print("\n3.4 检查重复样本:")

    # 简化方法：用前100个值的哈希作为指纹
    def get_sample_hash(sample):
        # 取前100个值（展平后）作为指纹
        flat = sample.flatten()[:100]
        return hash(flat.tobytes())

    train_hashes = set()
    for i in range(len(train_clean)):
        train_hashes.add(get_sample_hash(train_clean[i]))

    val_hashes = set()
    for i in range(len(val_clean)):
        val_hashes.add(get_sample_hash(val_clean[i]))

    overlap = train_hashes.intersection(val_hashes)
    print(f"   训练集唯一样本数: {len(train_hashes)}")
    print(f"   验证集唯一样本数: {len(val_hashes)}")
    print(f"   重叠样本数: {len(overlap)}")

    if len(overlap) > 0:
        print(f"   ⚠️  发现{len(overlap)}个可能的重复样本！")

    # 5. 检查归一化影响
    print("\n4. 归一化影响检查:")
    print("-" * 40)

    # 错误方式：用全局min/max归一化
    print("\n4.1 如果用全局min/max归一化:")
    c_min_global = clean_data.min()
    c_max_global = clean_data.max()
    n_min_global = noisy_data.min()
    n_max_global = noisy_data.max()

    print(f"   全局参数 - 干净: [{c_min_global:.4f}, {c_max_global:.4f}]")
    print(f"   全局参数 - 噪声: [{n_min_global:.4f}, {n_max_global:.4f}]")

    # 正确方式：用训练集min/max归一化
    print("\n4.2 正确方式（用训练集参数）:")
    c_min_train = train_clean.min()
    c_max_train = train_clean.max()
    n_min_train = train_noisy.min()
    n_max_train = train_noisy.max()

    print(f"   训练集参数 - 干净: [{c_min_train:.4f}, {c_max_train:.4f}]")
    print(f"   训练集参数 - 噪声: [{n_min_train:.4f}, {n_max_train:.4f}]")

    # 比较两种方式对验证集的影响
    def normalize(x, min_val, max_val):
        return 2 * (x - min_val) / (max_val - min_val + 1e-8) - 1

    # 用全局参数归一化验证集
    val_clean_global_norm = normalize(val_clean, c_min_global, c_max_global)
    val_noisy_global_norm = normalize(val_noisy, n_min_global, n_max_global)

    # 用训练集参数归一化验证集
    val_clean_train_norm = normalize(val_clean, c_min_train, c_max_train)
    val_noisy_train_norm = normalize(val_noisy, n_min_train, n_max_train)

    print("\n4.3 验证集归一化结果对比:")
    print(f"   全局参数归一化: 干净 [{val_clean_global_norm.min():.4f}, {val_clean_global_norm.max():.4f}]")
    print(f"   训练集参数归一化: 干净 [{val_clean_train_norm.min():.4f}, {val_clean_train_norm.max():.4f}]")

    diff = np.abs(val_clean_global_norm - val_clean_train_norm).mean()
    print(f"   两种方式平均差异: {diff:.6f}")
    if diff < 0.01:
        print("   ⚠️  差异很小，说明验证集和训练集分布非常相似！")

    # 6. 最终结论
    print("\n" + "=" * 60)
    print("检查结论:")
    print("=" * 60)

    leakage_score = 0
    if ks_p > 0.05:
        leakage_score += 1
        print("❌ 分布相似性检验提示可能泄露")
    if len(overlap) > 0:
        leakage_score += 1
        print("❌ 发现重复样本")
    if diff < 0.01:
        leakage_score += 1
        print("❌ 归一化差异很小")

    if leakage_score >= 2:
        print("\n⚠️  很可能存在数据泄露！建议修正数据加载方式。")
    elif leakage_score == 1:
        print("\n⚠️  可能存在轻微数据泄露，建议检查。")
    else:
        print("\n✅ 没有发现明显的数据泄露。")

    print("\n建议修正方案:")
    print("1. 先划分数据集，再计算归一化参数")
    print("2. 只用训练集的min/max进行归一化")
    print("3. 验证集和测试集复用训练集的归一化参数")

    return leakage_score


if __name__ == "__main__":
    # 设置你的数据路径
    clean_path = "uwb_signals_stft_clean.npy"
    noisy_path = "uwb_signals_stft_noisy.npy"

    # 检查文件是否存在
    if not os.path.exists(clean_path):
        print(f"错误：找不到 {clean_path}")
        print("当前目录下的.npy文件:")
        for f in os.listdir('.'):
            if f.endswith('.npy'):
                print(f"  - {f}")
    elif not os.path.exists(noisy_path):
        print(f"错误：找不到 {noisy_path}")
    else:
        # 运行检查
        check_data_leakage(clean_path, noisy_path)