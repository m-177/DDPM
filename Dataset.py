import torch
from torch.utils.data import Dataset
import numpy as np
import os


# -----------------------------
# UWB Dataset - 1D时域信号版本（仅干净信号）
# 流程：加载 → 划分 → 归一化到[-1,1]
# 直接输出原始域信号，供经典DDPM使用
# -----------------------------
class Dataset_UWB(Dataset):
    def __init__(self, clean_path, split='train',
                 norm_stats=None, val_ratio=0.1, test_ratio=0.1):
        """
        Args:
            clean_path: 干净数据路径 (npy文件，形状: [N, L] 或 [N, 1, L])
            split: 'train', 'val', 'test'
            norm_stats: 归一化统计量 (min, max)，用于验证/测试集
            val_ratio: 验证集比例
            test_ratio: 测试集比例
        """

        # -----------------------------
        # 步骤1: 加载原始数据
        # -----------------------------
        clean = np.load(clean_path)

        # 添加通道维度（如果是2D [N, L] -> [N, 1, L]）
        if clean.ndim == 2:
            clean = clean[:, np.newaxis, :]
        elif clean.ndim == 3:
            pass
        else:
            raise ValueError(f"不支持的数据维度: {clean.ndim}")

        # -----------------------------
        # 步骤2: 划分数据集
        # -----------------------------
        n_samples = len(clean)
        indices = np.random.permutation(n_samples)

        val_end = int(val_ratio * n_samples)
        test_end = val_end + int(test_ratio * n_samples)

        if split == 'train':
            self.clean = clean[indices[test_end:]]
        elif split == 'val':
            self.clean = clean[indices[:val_end]]
        elif split == 'test':
            self.clean = clean[indices[val_end:test_end]]
        else:
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split}")

        # -----------------------------
        # 步骤3: 计算/使用归一化参数
        # -----------------------------
        if norm_stats is None:
            # 训练集：计算并保存归一化参数（只用训练集！）
            self.clean_min = self.clean.min()
            self.clean_max = self.clean.max()
            print(f"训练集归一化参数:")
            print(f"  - 干净数据: min={self.clean_min:.4f}, max={self.clean_max:.4f}")
        else:
            # 验证/测试集：使用训练集的参数
            self.clean_min, self.clean_max = norm_stats['clean']
            print(f"使用训练集归一化参数:")
            print(f"  - 干净数据: min={self.clean_min:.4f}, max={self.clean_max:.4f}")

        # -----------------------------
        # 步骤4: 归一化到 [-1, 1]
        # -----------------------------
        self.clean = 2 * (self.clean - self.clean_min) / (self.clean_max - self.clean_min + 1e-8) - 1

        print(f"UWB {split}集加载完成，共{len(self)}个样本")
        print(f"  - 数据形状: {self.clean.shape}")
        print(f"  - 数据范围: [{self.clean.min():.4f}, {self.clean.max():.4f}]")

    def get_norm_stats(self):
        """返回归一化参数，用于验证/测试集"""
        return {
            'clean': (self.clean_min, self.clean_max)
        }

    def __getitem__(self, idx):
        return torch.tensor(self.clean[idx], dtype=torch.float32)

    def __len__(self):
        return len(self.clean)


# -----------------------------
# 使用示例
# -----------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("测试Dataset - 经典DDPM版本（无变换）")
    print("=" * 60)

    # 创建训练集
    train_dataset = Dataset_UWB(
        clean_path='uwb_signals_time_clean.npy',
        split='train'
    )

    # 获取归一化参数
    norm_stats = train_dataset.get_norm_stats()

    # 创建验证集（使用训练集的参数）
    val_dataset = Dataset_UWB(
        clean_path='uwb_signals_time_clean.npy',
        split='val',
        norm_stats=norm_stats
    )

    # 创建测试集（使用训练集的参数）
    test_dataset = Dataset_UWB(
        clean_path='uwb_signals_time_clean.npy',
        split='test',
        norm_stats=norm_stats
    )

    print("\n" + "=" * 60)
    print("测试数据加载")
    print("=" * 60)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    for clean in train_loader:
        print(f"训练集Batch形状: {clean.shape}")
        print(f"训练集数据范围: [{clean.min():.3f}, {clean.max():.3f}]")
        break

    print("\n✅ 数据加载成功！可直接用于经典DDPM训练")