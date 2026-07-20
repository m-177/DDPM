import numbers

import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_SPLIT_SEED = 42
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1
SPLIT_NAMES = ('train', 'val', 'test')


def _validate_dataset_indices(indices, n_samples, name='indices'):
    """校验并返回一份独立的一维整数索引。"""
    array = np.asarray(indices)
    if array.ndim != 1:
        raise ValueError(f"{name} 必须是一维索引")
    if array.size == 0:
        raise ValueError(f"{name} 不能为空")
    if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
        raise TypeError(f"{name} 必须是整数索引")

    array = array.astype(np.int64, copy=True)
    if np.any(array < 0) or np.any(array >= n_samples):
        raise IndexError(f"{name} 包含越界索引，有效范围为 [0, {n_samples})")
    if np.unique(array).size != array.size:
        raise ValueError(f"{name} 内部包含重复索引")
    return array


def validate_split_indices(split_indices, n_samples):
    """验证 train/val/test 索引互斥且完整覆盖全部样本。"""
    if not isinstance(n_samples, numbers.Integral) or isinstance(n_samples, (bool, np.bool_)):
        raise TypeError("n_samples 必须是整数")
    if n_samples <= 0:
        raise ValueError("n_samples 必须大于 0")
    if not isinstance(split_indices, dict):
        raise TypeError("split_indices 必须是包含 train/val/test 的字典")

    missing = set(SPLIT_NAMES) - set(split_indices)
    extra = set(split_indices) - set(SPLIT_NAMES)
    if missing:
        raise ValueError(f"split_indices 缺少集合: {sorted(missing)}")
    if extra:
        raise ValueError(f"split_indices 包含未知集合: {sorted(extra)}")

    checked = {
        name: _validate_dataset_indices(split_indices[name], n_samples, f"{name} indices")
        for name in SPLIT_NAMES
    }

    train_set = set(checked['train'].tolist())
    val_set = set(checked['val'].tolist())
    test_set = set(checked['test'].tolist())
    if not train_set.isdisjoint(val_set):
        raise ValueError("训练集与验证集索引存在重叠")
    if not train_set.isdisjoint(test_set):
        raise ValueError("训练集与测试集索引存在重叠")
    if not val_set.isdisjoint(test_set):
        raise ValueError("验证集与测试集索引存在重叠")

    covered = train_set | val_set | test_set
    expected = set(range(n_samples))
    if covered != expected:
        missing_indices = sorted(expected - covered)
        raise ValueError(f"划分未完整覆盖全部样本，缺少索引: {missing_indices[:10]}")
    return True


def create_split_indices(n_samples, val_ratio=DEFAULT_VAL_RATIO,
                         test_ratio=DEFAULT_TEST_RATIO, seed=DEFAULT_SPLIT_SEED):
    """使用一次局部随机排列创建可复现、互斥且完整的三组索引。"""
    if not isinstance(n_samples, numbers.Integral) or isinstance(n_samples, (bool, np.bool_)):
        raise TypeError("n_samples 必须是整数")
    if n_samples <= 0:
        raise ValueError("n_samples 必须大于 0")

    for name, ratio in (('val_ratio', val_ratio), ('test_ratio', test_ratio)):
        if not isinstance(ratio, numbers.Real) or isinstance(ratio, (bool, np.bool_)):
            raise TypeError(f"{name} 必须是数值")
        if not 0 < ratio < 1:
            raise ValueError(f"{name} 必须位于 (0, 1) 区间")
    if val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio 与 test_ratio 之和必须小于 1")
    if not isinstance(seed, numbers.Integral) or isinstance(seed, (bool, np.bool_)):
        raise TypeError("seed 必须是整数")

    val_count = int(val_ratio * n_samples)
    test_count = int(test_ratio * n_samples)
    train_count = n_samples - val_count - test_count
    if min(train_count, val_count, test_count) <= 0:
        raise ValueError(
            f"当前样本数和比例会产生空集合: train={train_count}, "
            f"val={val_count}, test={test_count}"
        )

    permutation = np.random.default_rng(seed).permutation(n_samples)
    val_end = val_count
    test_end = val_end + test_count
    split_indices = {
        'train': permutation[test_end:].copy(),
        'val': permutation[:val_end].copy(),
        'test': permutation[val_end:test_end].copy(),
    }
    validate_split_indices(split_indices, n_samples)
    return split_indices


def create_split_metadata(split_indices, n_samples, norm_stats,
                          val_ratio=DEFAULT_VAL_RATIO,
                          test_ratio=DEFAULT_TEST_RATIO,
                          seed=DEFAULT_SPLIT_SEED):
    """创建可随模型检查点保存的数据划分元信息。"""
    validate_split_indices(split_indices, n_samples)
    if not isinstance(norm_stats, dict) or 'clean' not in norm_stats:
        raise ValueError("norm_stats 必须包含 clean: (min, max)")
    if len(norm_stats['clean']) != 2:
        raise ValueError("norm_stats['clean'] 必须是 (min, max)")

    clean_min, clean_max = norm_stats['clean']
    return {
        'version': 1,
        'n_samples': int(n_samples),
        'indices': {
            name: np.asarray(split_indices[name], dtype=np.int64).copy()
            for name in SPLIT_NAMES
        },
        'norm_stats': {'clean': (float(clean_min), float(clean_max))},
        'val_ratio': float(val_ratio),
        'test_ratio': float(test_ratio),
        'seed': int(seed),
    }


def restore_split_metadata(metadata, n_samples):
    """校验检查点中的划分元信息，并返回索引和训练集归一化统计量。"""
    if not isinstance(metadata, dict):
        raise TypeError("data_split 必须是字典")
    if metadata.get('n_samples') != int(n_samples):
        raise ValueError(
            "检查点与当前数据集样本数不一致: "
            f"checkpoint={metadata.get('n_samples')}, current={n_samples}"
        )
    if 'indices' not in metadata:
        raise ValueError("data_split 缺少 indices")

    split_indices = {
        name: _validate_dataset_indices(
            metadata['indices'][name], n_samples, f"checkpoint {name} indices"
        )
        for name in SPLIT_NAMES
    }
    validate_split_indices(split_indices, n_samples)

    norm_stats = metadata.get('norm_stats')
    if not isinstance(norm_stats, dict) or 'clean' not in norm_stats:
        raise ValueError("data_split 缺少训练集 norm_stats")
    if len(norm_stats['clean']) != 2:
        raise ValueError("data_split 中的 clean 统计量必须是 (min, max)")
    clean_min, clean_max = norm_stats['clean']
    return split_indices, {'clean': (float(clean_min), float(clean_max))}


# -----------------------------
# UWB Dataset - 1D时域信号版本（仅干净信号）
# 流程：加载 → 按统一索引选取 → 归一化到[-1,1]
# 直接输出原始域信号，供经典DDPM使用
# -----------------------------
class Dataset_UWB(Dataset):
    def __init__(self, clean_path, indices, split='train',
                 norm_stats=None, pad_size=128):
        """
        Args:
            clean_path: 干净数据路径 (npy文件，形状: [N, L] 或 [N, 1, L])
            indices: 由 create_split_indices 生成的当前集合原始样本索引
            split: 当前集合名称，'train'、'val' 或 'test'
            norm_stats: 归一化统计量 (min, max)，用于验证/测试集
            pad_size: 镜像padding大小（首尾各延拓点数，默认128）
        """
        if split not in SPLIT_NAMES:
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split}")
        if not isinstance(pad_size, numbers.Integral) or isinstance(pad_size, (bool, np.bool_)):
            raise TypeError("pad_size 必须是整数")
        if pad_size < 0:
            raise ValueError("pad_size 不能小于 0")

        clean = np.load(clean_path)
        if clean.ndim == 2:
            clean = clean[:, np.newaxis, :]
        elif clean.ndim != 3:
            raise ValueError(f"不支持的数据维度: {clean.ndim}")

        self.split = split
        self.indices = _validate_dataset_indices(indices, len(clean), f"{split} indices")
        self.clean = clean[self.indices]

        if norm_stats is None:
            if split != 'train':
                raise ValueError("验证集和测试集必须使用训练集的 norm_stats")
            self.clean_min = self.clean.min()
            self.clean_max = self.clean.max()
            print("训练集归一化参数:")
            print(f"  - 干净数据: min={self.clean_min:.4f}, max={self.clean_max:.4f}")
        else:
            if 'clean' not in norm_stats or len(norm_stats['clean']) != 2:
                raise ValueError("norm_stats 必须包含 clean: (min, max)")
            self.clean_min, self.clean_max = norm_stats['clean']
            print("使用训练集归一化参数:")
            print(f"  - 干净数据: min={self.clean_min:.4f}, max={self.clean_max:.4f}")

        self.clean = 2 * (self.clean - self.clean_min) / (self.clean_max - self.clean_min + 1e-8) - 1

        self.pad_size = int(pad_size)
        if self.pad_size > 0:
            self.clean = np.pad(
                self.clean,
                ((0, 0), (0, 0), (self.pad_size, self.pad_size)),
                mode='reflect'
            )
        self.original_length = self.clean.shape[-1] - 2 * self.pad_size

        print(f"UWB {split}集加载完成，共{len(self)}个样本")
        print(f"  - 原始样本索引数: {len(self.indices)}")
        print(f"  - 数据形状: {self.clean.shape}")
        print(f"  - 数据范围: [{self.clean.min():.4f}, {self.clean.max():.4f}]")
        print(f"  - 镜像padding: 首尾各{self.pad_size}点 "
              f"(原始长度{self.original_length}, 填充后{self.clean.shape[-1]})")

    def get_norm_stats(self):
        """返回归一化参数，用于验证/测试集。"""
        return {'clean': (self.clean_min, self.clean_max)}

    def __getitem__(self, idx):
        return torch.tensor(self.clean[idx], dtype=torch.float32)

    def __len__(self):
        return len(self.clean)


if __name__ == "__main__":
    print("=" * 60)
    print("测试Dataset - 统一可复现划分")
    print("=" * 60)

    data_path = 'uwb_signals_time_clean.npy'
    n_samples = len(np.load(data_path, mmap_mode='r'))
    split_indices = create_split_indices(n_samples)
    validate_split_indices(split_indices, n_samples)

    print(f"划分种子: {DEFAULT_SPLIT_SEED}")
    print("划分数量: " + ", ".join(
        f"{name}={len(split_indices[name])}" for name in SPLIT_NAMES
    ))
    print("✅ 三组索引互不重叠并完整覆盖全部样本")

    train_dataset = Dataset_UWB(
        clean_path=data_path,
        indices=split_indices['train'],
        split='train'
    )
    norm_stats = train_dataset.get_norm_stats()
    val_dataset = Dataset_UWB(
        clean_path=data_path,
        indices=split_indices['val'],
        split='val',
        norm_stats=norm_stats
    )
    test_dataset = Dataset_UWB(
        clean_path=data_path,
        indices=split_indices['test'],
        split='test',
        norm_stats=norm_stats
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    for clean in train_loader:
        print(f"训练集Batch形状: {clean.shape}")
        print(f"训练集数据范围: [{clean.min():.3f}, {clean.max():.3f}]")
        break

    print("\n✅ 数据加载成功！可直接用于经典DDPM训练")
