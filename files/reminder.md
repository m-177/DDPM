# 项目重要信息备忘

## 代码仓库

- **GitHub 仓库地址**: https://github.com/m-177/DDPM
- **原始参考仓库**: https://github.com/MingtaoGuo/DDPM_pytorch

## 项目路径

- **本地项目根目录**: `C:\Users\王俊凯夫人\Desktop\uwb_signal_first_add`
- **代码目录**: `C:\Users\王俊凯夫人\Desktop\uwb_signal_first_add\uwb_signal_first_add\DDPM_pytorch-main\DDPM_pytorch-main`
- **备份文件目录**: `C:\Users\王俊凯夫人\Desktop\uwb_signal_first_add\uwb_signal_first_add\files`

## 项目说明

本项目基于 [DDPM_pytorch](https://github.com/MingtaoGuo/DDPM_pytorch) 修改，将 Denoising Diffusion Probabilistic Models (DDPM) 应用于 **UWB（超宽带）信号去噪**。

### 核心变更

- 数据维度：2D 图像 → 1D 时域信号
- 任务目标：图像生成 → 信号去噪
- 新增混合损失函数（峰值感知 + 极性惩罚）
- 新增 1D UNet 模型
- 新增训练检查点、EMA、早停、内存监控等功能

### 训练状态

- 已训练 180 个 epoch（模型保存在 `saved_models_classic/`）
- 最佳模型：`best_model_by_loss.pth` 和 `best_model_by_snr.pth`
- 训练日志：`logs_classic/training_log.txt`

## 重要命令

```bash
# 训练
python train.py

# 生成信号
python uwb_signal_generate.py

# 反向扩散生成
python reverse_diffusion_process.py --data_dim 1d --weights ./saved_models_classic/best_model_by_snr.pth

# 前向扩散可视化
python diffusion_process.py --data_type signal_1d --img_path uwb_signals_time_clean.npy --t 200,400,600,800,950
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `reminder.md` | 本文件，项目重要信息备忘 |
| `change.md` | 与原始代码的变更记录 |
| `todo.md` | 项目待办事项与后续计划 |
