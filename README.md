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
- ✅ 1D UNet 模型（支持 1D 和 2D 数据）
- ✅ 混合损失函数（峰值感知 + 极性惩罚）
- ✅ EMA（指数移动平均）训练
- ✅ 检查点保存/恢复（支持暂停继续训练）
- ✅ 早停机制（基于验证损失和 SNR 提升）
- ✅ 内存监控与自动垃圾回收
- ✅ SNR 评估与可视化
- ✅ 信号插值生成

## 代码基本结构与重要文件

```
DDPM_pytorch-main/
├── train.py                      # 主训练脚本（经典DDPM + 混合损失）
├── models.py                     # 模型定义（1D/2D UNet + 自注意力）
├── Dataset.py                    # UWB 数据集加载与预处理
├── diffusion_process.py          # 前向扩散过程（加性/乘性噪声）
├── reverse_diffusion_process.py  # 反向扩散过程（信号生成）
├── interpolate.py                # 信号插值生成
├── uwb_signal_generate.py        # UWB 干净信号生成（高斯二阶导脉冲）
├── check_data_leakage.py         # 数据泄露检查工具
├── check_structure.py            # 代码结构检查工具
├── temp.py                       # CUDA 可用性测试
├── requirements.txt              # Python 依赖
├── uwb_signals_time_clean.npy    # 生成的干净 UWB 信号数据
├── logs_classic/                 # 训练日志与曲线图
│   ├── training_log.txt
│   └── training_curves.png
├── saved_models_classic/         # 保存的模型权重
│   ├── best_model_by_loss.pth    # 最佳验证损失模型
│   ├── best_model_by_snr.pth     # 最佳 SNR 提升模型
│   └── model_epoch*.pth          # 各 epoch 检查点
├── denoising_results_classic.png # 去噪结果可视化
├── uwb_5_samples_waveform.png    # UWB 信号波形示例
└── uwb_pulse_verification.png    # 脉冲验证图
```

### 重要文件说明

| 文件 | 说明 |
|------|------|
| `train.py` | 核心训练脚本，包含 SimpleUNet1D_Classic 模型、混合损失函数、EMA、检查点、早停等 |
| `models.py` | 改进版 UNet 模型（ImprovedDenoiseUNet1D/2D），含时间嵌入和自注意力 |
| `Dataset.py` | UWB 数据集类，支持训练/验证/测试集划分和归一化 |
| `diffusion_process.py` | 前向扩散过程，支持加性噪声和乘性噪声 |
| `reverse_diffusion_process.py` | 反向扩散生成过程 |
| `interpolate.py` | 信号插值工具 |
| `uwb_signal_generate.py` | 基于高斯二阶导的 UWB 信号生成器 |

## 如何运行项目

### 环境配置

```bash
# 安装依赖
pip install -r requirements.txt
```

### 1. 生成 UWB 信号数据

```bash
python uwb_signal_generate.py
```
生成 `uwb_signals_time_clean.npy`（1000个UWB干净信号样本）。

### 2. 训练模型

```bash
# 基本训练
python train.py

# 从检查点恢复训练
python train.py  # 修改 train() 调用，传入 resume_from=True
```

训练参数可在 `train.py` 的 `__main__` 部分调整：
- `LAMBDA_MSE = 2.5` — MSE 损失权重
- `LAMBDA_REL = 0.4` — 相对 MSE 损失权重
- `DROPOUT_RATE = 0.1` — Dropout 率
- `TOTAL_EPOCH = 500` — 总训练轮数
- `NUM_WORKERS = 1` — DataLoader 工作进程数

### 3. 前向扩散可视化

```bash
# 1D 信号扩散
python diffusion_process.py --data_type signal_1d --img_path uwb_signals_time_clean.npy --t 200,400,600,800,950

# 2D 图像扩散
python diffusion_process.py --data_type image --img_path resources/face.png --t 100,300,500,700,999
```

### 4. 反向扩散生成

```bash
# 1D 信号生成
python reverse_diffusion_process.py --data_dim 1d --weights ./saved_models_classic/best_model_by_snr.pth --num_samples 1

# 2D 图像生成
python reverse_diffusion_process.py --data_dim 2d --weights ./saved_models/best_model_by_snr.pth
```

### 5. 信号插值

```bash
python interpolate.py --data_type uwb_1d --img1_path signal1.npy --img2_path signal2.npy --weights ./saved_models_classic/best_model_by_snr.pth
```

### 6. 数据泄露检查

```bash
python check_data_leakage.py
```

## 与原始项目的差异

本项目基于 [DDPM_pytorch](https://github.com/MingtaoGuo/DDPM_pytorch) 进行了大量修改，主要变更：

1. **数据维度**：从 2D 图像 → 1D 时域信号
2. **任务目标**：从图像生成 → 信号去噪
3. **模型结构**：新增 SimpleUNet1D_Classic（减少下采样次数），保留 ImprovedDenoiseUNet1D/2D
4. **损失函数**：新增 peak_aware_loss、peak_correlation_loss、combined_loss 混合损失
5. **训练流程**：新增 EMA、检查点恢复、早停、内存监控、SNR 评估
6. **数据集**：新增 Dataset_UWB（支持训练/验证/测试集划分）
7. **信号生成**：新增 uwb_signal_generate.py（高斯二阶导脉冲生成）

## 许可证

MIT License

## 参考

- [DDPM_pytorch (Original)](https://github.com/MingtaoGuo/DDPM_pytorch)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
