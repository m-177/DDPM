# DDPM for UWB Signal Denoising

基于 [DDPM_pytorch](https://github.com/MingtaoGuo/DDPM_pytorch) (Mingtao Guo) 修改的扩散概率模型，专门用于 **UWB（超宽带）信号去噪**。

## 项目介绍

本项目将经典的 Denoising Diffusion Probabilistic Models (DDPM) 应用于 UWB 信号的去噪任务。与原始 DDPM 图像生成不同，本项目：

- **数据维度**：从 2D 图像改为 **1D 时域信号**
- **任务目标**：从图像生成改为 **信号去噪**（加性噪声）
- **损失函数**：设计了针对脉冲信号的 **混合损失函数**（MSE + 相对MSE + 峰值感知损失 + 峰值相关性损失）
- **模型结构**：设计了专用的 **1D UNet**，减少下采样次数以保留脉冲细节

### 核心特性

- ✅ 经典 DDPM 加性噪声前向/反向扩散
- ✅ 1D UNet 模型（SimpleUNet1D_Classic）
- ✅ 混合损失函数（峰值感知 + 极性惩罚）
- ✅ EMA（指数移动平均）训练
- ✅ 检查点保存/恢复（支持暂停继续训练）
- ✅ 早停机制（基于验证损失和 SNR 收敛）
- ✅ 内存监控与自动垃圾回收
- ✅ 多时间步 SNR 评估（`eval_multi_t.py`）
- ✅ 极性反转与波形丢失检测（`evaluate_errors.py`）
- ✅ 基线方法对比（小波、中值滤波等）
- ✅ 消融实验框架（`run_ablation.py`）

## 代码基本结构与重要文件

```
DDPM_pytorch-main/
├── train.py                      # 主训练脚本（经典DDPM + 混合损失 + 1D UNet）
├── models.py                     # 改进版 UNet 模型定义（含自注意力）
├── Dataset.py                    # UWB 数据集加载与划分（train/val/test）
├── diffusion_process.py          # 前向扩散过程（加性/乘性噪声）
├── reverse_diffusion_process.py  # 反向扩散生成过程
├── interpolate.py                # 信号插值生成
├── uwb_signal_generate.py        # UWB 干净信号生成（高斯二阶导脉冲）
├── eval_multi_t.py               # 多时间步评估（t=100~400）
├── evaluate_errors.py            # 极性反转 & 波形丢失检测
├── baseline_methods.py           # 基线去噪方法（小波/中值滤波等）
├── run_ablation.py               # 消融实验运行脚本
├── run_ablation.bat              # 消融实验批处理
├── diffusion_utils.py            # 共享扩散参数工具
├── requirements.txt              # Python 依赖
├── tests/                        # 单元测试
│   ├── __init__.py
│   └── test_dataset_split.py     # 数据集划分测试
├── logs_classic/                 # 训练日志与曲线图
├── saved_models_classic/         # 保存的模型权重
└── result.npy                    # 评估结果
```

### 重要文件说明

| 文件 | 说明 |
|------|------|
| `train.py` | 核心训练脚本，包含 SimpleUNet1D_Classic 模型、混合损失函数、EMA、检查点、早停、SNR 评估等 |
| `models.py` | 改进版 UNet 模型（ImprovedDenoiseUNet1D），含时间嵌入和自注意力 |
| `Dataset.py` | UWB 数据集类，支持确定性 train/val/test 划分、镜像 padding、归一化（训练集统计量） |
| `diffusion_process.py` | 前向扩散过程，支持加性噪声和乘性噪声 |
| `reverse_diffusion_process.py` | 反向扩散生成过程 |
| `eval_multi_t.py` | 加载模型，在多个时间步（t=100~400）评估去噪效果 |
| `evaluate_errors.py` | 检测去噪结果中的极性反转和波形丢失，支持自适应阈值 |
| `baseline_methods.py` | 传统去噪方法对比（小波去噪、维纳滤波等） |
| `run_ablation.py` | 消融实验，验证各损失分量的贡献 |
| `diffusion_utils.py` | 共享扩散参数工具（beta 调度、SNR 计算、参数预计算） |
| `uwb_signal_generate.py` | 基于高斯二阶导的 UWB 信号生成器 |

## 如何运行项目

### 环境配置

```bash
pip install -r requirements.txt
```

### 1. 生成 UWB 信号数据

```bash
python uwb_signal_generate.py
```
生成 `uwb_signals_time_clean.npy`（1000个 UWB 干净信号样本，10120 采样点，20GHz 采样率）。

### 2. 训练模型

```bash
python train.py
```

训练参数可在 `train.py` 的 `__main__` 部分调整：
- `LAMBDA_MSE = 2.5` — MSE 损失权重
- `LAMBDA_REL = 0.4` — 相对 MSE 损失权重
- `LAMBDA_PEAK = 1.8` — 峰值感知损失权重
- `LAMBDA_CORR = 1.8` — 峰值相关性损失权重
- `DROPOUT_RATE = 0.1` — Dropout 率
- `TOTAL_EPOCH = 600` — 总训练轮数上限
- `SNR_PATIENCE = 4` — SNR 收敛阈值
- `BATCH_SIZE = 8` — 批次大小

### 3. 多时间步评估

```bash
python eval_multi_t.py
```
在 t ∈ [100, 400] 范围内评估去噪性能，生成各时间步的对比图。

### 4. 错误分析

```bash
# 评估极性反转和波形丢失
python evaluate_errors.py

# 指定阈值
python evaluate_errors.py --amplitude_threshold 0.5 --polarity_threshold 0.3 --visualize
```

### 5. 前向扩散可视化

```bash
python diffusion_process.py --data_type signal_1d --img_path uwb_signals_time_clean.npy --t 200,400,600,800,950
```

### 6. 反向扩散生成

```bash
python reverse_diffusion_process.py --data_dim 1d --weights ./saved_models_classic/best_model_by_snr.pth
```

### 7. 消融实验

```bash
python run_ablation.py
# 或双击 run_ablation.bat
```

### 8. 运行测试

```bash
python -m pytest tests/ -v
```

## 数据集划分设计

本项目使用确定性随机划分（seed=42），确保可复现：

- **训练集**: 80%（~800 样本）
- **验证集**: 10%（~100 样本）
- **测试集**: 10%（~100 样本）

关键设计要点：
- 三组索引**互斥且完整覆盖**全部样本，通过 `validate_split_indices()` 严格校验
- 归一化统计量（min/max）**仅从训练集计算**，验证集和测试集复用
- 检查点保存 `data_split` 元数据，评估时精确恢复原始划分
- **无数据泄露**：测试集在整个训练过程中完全隔离

## 与原始项目的差异

本项目基于 [DDPM_pytorch](https://github.com/MingtaoGuo/DDPM_pytorch) 进行了大量修改：

1. **数据维度**：从 2D 图像 → 1D 时域信号
2. **任务目标**：从图像生成 → 信号去噪
3. **模型结构**：新增 SimpleUNet1D_Classic（2 次下采样，减少细节丢失），保留 ImprovedDenoiseUNet1D/2D
4. **损失函数**：新增 peak_aware_loss、peak_correlation_loss、combined_loss 混合损失
5. **训练流程**：新增 EMA、检查点恢复、早停（loss + SNR 收敛）、内存监控
6. **数据集**：新增 Dataset_UWB（确定性划分 + 校验 + checkpoint 元数据持久化）
7. **评估体系**：新增多时间步评估、极性反转检测、波形丢失检测、消融实验框架
8. **测试**：新增数据集划分单元测试

## 许可证

MIT License

## 参考

- [DDPM_pytorch (Original)](https://github.com/MingtaoGuo/DDPM_pytorch)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
