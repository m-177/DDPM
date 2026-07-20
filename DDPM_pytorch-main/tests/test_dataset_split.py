import os
import sys
import tempfile
import types
import unittest

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 本机只做代码验证且未安装 PyTorch；这些测试不调用张量输出，使用最小桩加载 Dataset 模块。
try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch_stub = types.ModuleType('torch')
    torch_utils_stub = types.ModuleType('torch.utils')
    torch_data_stub = types.ModuleType('torch.utils.data')

    class TorchDatasetStub:
        pass

    torch_stub.tensor = lambda value, dtype=None: np.asarray(value, dtype=np.float32)
    torch_stub.float32 = np.float32
    torch_data_stub.Dataset = TorchDatasetStub
    torch_utils_stub.data = torch_data_stub
    torch_stub.utils = torch_utils_stub
    sys.modules['torch'] = torch_stub
    sys.modules['torch.utils'] = torch_utils_stub
    sys.modules['torch.utils.data'] = torch_data_stub

from Dataset import (Dataset_UWB, create_split_indices, create_split_metadata,
                     restore_split_metadata, validate_split_indices)


class SplitIndicesTests(unittest.TestCase):
    def test_split_is_disjoint_and_complete(self):
        splits = create_split_indices(100, val_ratio=0.1, test_ratio=0.2, seed=42)

        train = set(splits['train'].tolist())
        val = set(splits['val'].tolist())
        test = set(splits['test'].tolist())
        self.assertTrue(train.isdisjoint(val))
        self.assertTrue(train.isdisjoint(test))
        self.assertTrue(val.isdisjoint(test))
        self.assertEqual(train | val | test, set(range(100)))
        self.assertTrue(validate_split_indices(splits, 100))

    def test_split_sizes_follow_ratios(self):
        splits = create_split_indices(101, val_ratio=0.1, test_ratio=0.2, seed=42)

        self.assertEqual(len(splits['train']), 71)
        self.assertEqual(len(splits['val']), 10)
        self.assertEqual(len(splits['test']), 20)

    def test_same_seed_repeats_same_split(self):
        first = create_split_indices(100, seed=17)
        second = create_split_indices(100, seed=17)

        for name in ('train', 'val', 'test'):
            np.testing.assert_array_equal(first[name], second[name])

    def test_different_seed_changes_split(self):
        first = create_split_indices(100, seed=17)
        second = create_split_indices(100, seed=18)

        self.assertTrue(any(
            not np.array_equal(first[name], second[name])
            for name in ('train', 'val', 'test')
        ))

    def test_split_does_not_change_global_numpy_state(self):
        np.random.seed(123)
        expected = np.random.random(5)
        np.random.seed(123)

        create_split_indices(100, seed=42)
        actual = np.random.random(5)

        np.testing.assert_array_equal(actual, expected)

    def test_invalid_ratios_and_empty_splits_are_rejected(self):
        invalid_cases = (
            {'val_ratio': 0.0, 'test_ratio': 0.1},
            {'val_ratio': 0.1, 'test_ratio': 0.0},
            {'val_ratio': -0.1, 'test_ratio': 0.1},
            {'val_ratio': 0.5, 'test_ratio': 0.5},
        )
        for case in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    create_split_indices(100, **case)

        with self.assertRaises(ValueError):
            create_split_indices(5, val_ratio=0.1, test_ratio=0.1)

    def test_invalid_split_indices_are_rejected(self):
        valid = create_split_indices(10, val_ratio=0.2, test_ratio=0.2, seed=1)

        missing_group = {'train': valid['train'], 'val': valid['val']}
        with self.assertRaises(ValueError):
            validate_split_indices(missing_group, 10)

        duplicated = {name: values.copy() for name, values in valid.items()}
        duplicated['train'][1] = duplicated['train'][0]
        with self.assertRaises(ValueError):
            validate_split_indices(duplicated, 10)

        overlapping = {name: values.copy() for name, values in valid.items()}
        overlapping['test'][0] = overlapping['train'][0]
        with self.assertRaises(ValueError):
            validate_split_indices(overlapping, 10)

        out_of_bounds = {name: values.copy() for name, values in valid.items()}
        out_of_bounds['val'][0] = 10
        with self.assertRaises(IndexError):
            validate_split_indices(out_of_bounds, 10)

    def test_checkpoint_metadata_restores_exact_split(self):
        splits = create_split_indices(100, val_ratio=0.1, test_ratio=0.2, seed=17)
        metadata = create_split_metadata(
            splits, 100, {'clean': (-2.0, 3.0)},
            val_ratio=0.1, test_ratio=0.2, seed=17)

        restored, norm_stats = restore_split_metadata(metadata, 100)

        for name in ('train', 'val', 'test'):
            np.testing.assert_array_equal(restored[name], splits[name])
        self.assertEqual(norm_stats, {'clean': (-2.0, 3.0)})

    def test_checkpoint_metadata_rejects_different_dataset_size(self):
        splits = create_split_indices(100, seed=17)
        metadata = create_split_metadata(
            splits, 100, {'clean': (-1.0, 1.0)}, seed=17)

        with self.assertRaises(ValueError):
            restore_split_metadata(metadata, 101)

    def test_checkpoint_metadata_rejects_overlapping_indices(self):
        splits = create_split_indices(100, seed=17)
        metadata = create_split_metadata(
            splits, 100, {'clean': (-1.0, 1.0)}, seed=17)
        metadata['indices']['val'][0] = metadata['indices']['train'][0]

        with self.assertRaises(ValueError):
            restore_split_metadata(metadata, 100)


class DatasetIndicesTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_path = os.path.join(self.temp_dir.name, 'signals.npy')
        data = np.stack([
            np.linspace(index * 10, index * 10 + 7, 8, dtype=np.float32)
            for index in range(20)
        ])
        np.save(self.data_path, data)
        self.raw_data = data
        self.splits = create_split_indices(20, val_ratio=0.2, test_ratio=0.2, seed=7)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_dataset_preserves_original_indices(self):
        dataset = Dataset_UWB(
            self.data_path,
            indices=self.splits['train'],
            split='train',
            pad_size=0,
        )

        np.testing.assert_array_equal(dataset.indices, self.splits['train'])
        self.assertEqual(len(dataset), len(self.splits['train']))

    def test_train_stats_come_only_from_train_indices(self):
        dataset = Dataset_UWB(
            self.data_path,
            indices=self.splits['train'],
            split='train',
            pad_size=0,
        )
        expected_data = self.raw_data[self.splits['train']]
        stats = dataset.get_norm_stats()['clean']

        self.assertEqual(stats[0], expected_data.min())
        self.assertEqual(stats[1], expected_data.max())

    def test_val_and_test_reuse_train_stats(self):
        train_dataset = Dataset_UWB(
            self.data_path,
            indices=self.splits['train'],
            split='train',
            pad_size=0,
        )
        train_stats = train_dataset.get_norm_stats()
        val_dataset = Dataset_UWB(
            self.data_path,
            indices=self.splits['val'],
            split='val',
            norm_stats=train_stats,
            pad_size=0,
        )
        test_dataset = Dataset_UWB(
            self.data_path,
            indices=self.splits['test'],
            split='test',
            norm_stats=train_stats,
            pad_size=0,
        )

        self.assertEqual(val_dataset.get_norm_stats(), train_stats)
        self.assertEqual(test_dataset.get_norm_stats(), train_stats)

    def test_dataset_rejects_invalid_indices_and_missing_stats(self):
        with self.assertRaises(ValueError):
            Dataset_UWB(self.data_path, indices=[], split='train', pad_size=0)
        with self.assertRaises(IndexError):
            Dataset_UWB(self.data_path, indices=[20], split='train', pad_size=0)
        with self.assertRaises(ValueError):
            Dataset_UWB(
                self.data_path,
                indices=self.splits['val'],
                split='val',
                pad_size=0,
            )


if __name__ == '__main__':
    unittest.main()
