# 与原始 DDPM_pytorch 的变更记录

> 原始仓库: https://github.com/MingtaoGuo/DDPM_pytorch
> 本仓库: https://github.com/m-177/DDPM

## 主要变更概览

### 1. 数据维度与任务目标

| 项目 | 原始代码 | 本代码 |
|------|---------|--------|
| 数据维度 | 2D 图像 (CIFAR-10, CelebA) | 1D 时域信号 (UWB) |
| 任务目标 | 图像生成 | 信号去噪 |
| 数据集 | CIFAR-10 / CelebA | 自定义 UWB 信号 |

### 2. 新增文件

| 文件 | 说明 |
|------|------|
| `Dataset.py` | UWB 数据集加载与预处理（支持训练/验证/测试集划分） |
| `uwb_signal_generate.py` | UWB 干净信号生成器（基于高斯二阶导脉冲） |
| `check_data_leakage.py` | 数据泄露检查工具 |
| `check_structure.py` | 代码结构检查工具 |
| `temp.py` | CUDA 可用性测试 |
| `check_cuda.txt` | CUDA 环境检查记录 |
| `loss.txt` | 损失值记录 |

### 3. 修改文件

#### train.py（大幅修改）

- **模型**：新增 `SimpleUNet1D_Classic`（1D UNet，减少下采样次数 3→2 次）
- **损失函数**：
  - 新增 `relative_mse_loss` — 相对 MSE 损失
  - 新增 `peak_aware_loss` — 峰值感知损失（平方权重）
  - 新增 `peak_correlation_loss` — 峰值相关性损失（含极性惩罚）
  - 新增 `combined_loss` — 混合损失函数（MSE + RelMSE + Peak + Corr）
- **训练流程**：
  - 新增 `EMA` 类 — 指数移动平均
  - 新增 `EarlyStopping` / `LossEarlyStopping` / `SNREarlyStopping` — 早停机制
  - 新增 `save_checkpoint` / `load_checkpoint` — 检查点保存与恢复
  - 新增 `MemoryMonitor` — 内存监控与自动垃圾回收
  - 新增 `check_data_worker_memory` — 数据集内存预估
  - 新增 SNR 评估与可视化
  - 新增 `CosineAnnealingLR` 学习率调度
  - 新增 `num_workers` 和 `pin_memory` 优化 DataLoader
  - 新增训练曲线实时绘图（4个子图：Loss、LR、SNR、Memory）

#### models.py（大幅修改）

- 新增 `TimeEmbedding` — 改进的时间嵌入（正弦+余弦位置编码）
- 新增 `SelfAttention1D` — 1D 自注意力模块
- 新增 `ImprovedResBlock1D` — 1D 改进残差块
- 新增 `ImprovedDenoiseUNet1D` — 1D 改进 UNet
- 保留 `ImprovedDenoiseUNet2D` — 2D 版本（兼容原始代码）
- 新增 `DenoiseUNet` — 统一接口（自动选择 1D/2D）

#### diffusion_process.py（大幅修改）

- 新增 `cosine_beta_schedule` — 余弦 beta 调度
- 新增 `calculate_theoretical_snr` — 理论 SNR 计算
- 新增 `additive_diffusion` — 加性噪声前向扩散
- 新增 `multiplicative_diffusion` — 乘性噪声前向扩散
- 新增 `diffusion` — 统一扩散接口
- 新增 1D 信号扩散可视化支持
- 新增命令行参数（`--data_type`, `--noise_type`, `--signal_length` 等）

#### reverse_diffusion_process.py（大幅修改）

- 新增 `DiffusionParams` 类 — 预计算扩散参数
- 新增 `calculate_theoretical_snr` — 理论 SNR 计算
- 新增 `reverse_diffusion` — 加性噪声反向扩散
- 新增 1D 信号生成支持
- 新增 `tensor_to_image` — 支持 1D/2D/3D tensor 转换
- 新增 `create_grid_image` — 中间结果网格图
- 新增命令行参数（`--data_dim`, `--signal_length`, `--num_samples` 等）

#### interpolate.py（大幅修改）

- 新增 `cosine_beta_schedule` — 余弦 beta 调度
- 新增 `compute_diffusion_params` — 扩散参数预计算
- 新增 `reverse_diffusion` — 加性噪声逆向扩散
- 新增 `calculate_theoretical_snr` — 理论 SNR 计算
- 新增 `interpolate` — 信号插值函数
- 新增 `load_signal` — 1D 信号加载
- 新增 `plot_1d_results` — 1D 插值结果绘图
- 新增命令行参数（`--data_type`, `--interp_steps`, `--n_interps` 等）

### 4. 保留的原始功能

- 2D 图像扩散/生成/插值功能（通过 `--data_dim 2d` 参数）
- MIT 许可证
- 基础 DDPM 公式实现

## 未来变更记录

> 后续对代码的更新请在此处记录，格式如下：

### [2026-05-28] 优化结果保存策略：训练完成后统一存最佳epoch结果并做分析

- **修改文件**：train.py
- **变更内容**：
  - 移除训练循环内每轮保存 result.npy 的逻辑（原在 `if avg_val_loss < best_val_loss` 内保存）
  - 改为全部训练完成后，加载最佳模型，对测试集统一评估并保存 result.npy
  - 新增综合分析报告（SNR分析、MSE分析、相关系数分析、SNR分布区间、逐样本详细数据）
  - 分析报告保存到 `./logs_classic/analysis_report.txt`
- **原因**：避免训练过程中频繁写入 result.npy，只在训练结束后对最佳模型做一次完整评估和分析

### [2026-05-28] 删除SNR早停，调整Loss早停参数

- **修改文件**：train.py
- **变更内容**：
  - 删除 `SNREarlyStopping` 类及其所有引用（早停仅保留 Loss 早停）
  - `LossEarlyStopping`: patience 30→20, min_delta 1e-4→0.05
  - `save_checkpoint` 移除 `snr_early_stopping` 参数
  - 训练循环中早停逻辑简化为仅检查 val_loss
- **原因**：SNR 早停在 10 轮评估一次时实际需要 300 轮才触发，形同虚设；Loss 早停 min_delta 过小导致波动频繁重置 counter

### [2026-05-28] 数据镜像 padding + 损失函数强化

- **修改文件**：Dataset.py, train.py
- **变更内容**：
  - Dataset.py: 新增 `pad_size` 参数（默认128），归一化后对时间轴做 `np.pad(mode='reflect')` 镜像反射填充，首尾各延拓128点
  - train.py: `combined_loss` 新增 `time_weighted_mse_loss`（峰值区域2倍权重），`lambda_corr` 默认值 1.8→3.0
  - train/val/test 三个 Dataset 实例统一设置 `pad_size=128`
- **原因**：缓解 UNet 边界效应导致的信号首尾极性反转和幅度衰减；强化波形形态约束提升相关系数

### [2026-05-28] t 采样偏置：训练集中在高噪声区域

- **修改文件**：train.py
- **变更内容**：
  - 训练和验证的 `t` 采样区间从 `[0, timesteps)` 改为 `[timesteps//2, timesteps)`（即 t ∈ [500, 1000)）
  - eval_t=300 和 eval_ts 保持不变，用于测试模型对未见过噪声水平的泛化能力
- **原因**：低 SNR 样本训练不足，偏置采样让模型花更多时间学习高噪声去噪，改善弱信号还原

### [2026-05-31] 训练参数优化：放宽早停、简化学习率调度、增加损失权重参数

- **修改文件**：train.py
- **变更内容**：
  - `LossEarlyStopping` patience: 30→500（让模型有更长时间充分训练）
  - 移除 `scheduler_started` 条件逻辑，`CosineAnnealingLR` 从第1轮直接步进
  - `scheduler` 的 `T_max` 从硬编码 `200` 改为 `total_epoch`（函数参数）
  - `train()` 函数新增 `lambda_peak` 和 `lambda_corr` 参数（默认值均为 1.8）
  - `combined_loss` 调用处不再硬编码 `lambda_peak=1.8, lambda_corr=1.8`，改用函数参数
  - 主函数新增 `LAMBDA_PEAK` 和 `LAMBDA_CORR` 配置项
- **原因**：35 epoch 即早停过早，模型远未收敛；`scheduler_started` 条件导致 LR 几乎未衰减；新增损失权重参数方便后续调参

### [2026-06-01] 实现训练暂停/恢复功能

- **修改文件**：train.py
- **变更内容**：
  - 新增 `import signal`，注册 SIGINT (Ctrl+C) 信号处理器
  - 第一次 Ctrl+C：设置 `training_interrupted` 标志，当前 epoch 结束后自动保存检查点
  - 第二次 Ctrl+C：强制退出（防止卡死）
  - 训练循环末尾新增暂停检查：检测到中断标志后调用 `save_checkpoint(is_pause=True)` 并 break
  - 训练结束后（plt/log_file 清理后）检测中断状态，若暂停则提前 return，跳过评估阶段
  - 主函数新增 `RESUME` 配置项（True=从检查点恢复，False=从头训练）
  - 恢复原始信号处理器，避免影响后续代码
- **原因**：之前训练循环缺少 KeyboardInterrupt 处理，Ctrl+C 直接崩溃丢失进度

### [2026-06-02] 梯度累积 + SNR 收敛 + 参数调整，解决 loss 震荡

- **修改文件**：train.py
- **变更内容**：
  - `train()` 新增 `gradient_accumulation_steps` 参数（默认 1，即不累积）
  - 训练循环改为梯度累积模式：每 `gradient_accumulation_steps` 步执行一次 optimizer.step()
  - loss 按累积步数缩放（`total_loss / steps`），确保梯度数值一致
  - `train()` 新增 `snr_patience` 参数（默认 None=不启用）
  - 新增 SNR 收敛跟踪：连续 `snr_patience` 次 eval 未超过历史最佳则自动收敛停止
  - SNR 收敛使用独立变量 `snr_best_for_convergence`，不与 `best_snr_improvement`（保存逻辑）冲突
  - 主函数参数调整：batch=2, grad_accum=4（有效 batch=8），lr 2e-4→1e-4
  - total_epoch 500→600，Loss早停 patience 500→200、min_epochs 300→150
  - 新增 SNR_PATIENCE=4（连续 4 次 eval=40 轮未改善则收敛）
- **原因**：batch=2 梯度方差大导致 loss 剧烈震荡，梯度累积缓解；SNR 收敛比 val_loss 更可靠地反映去噪质量

### [YYYY-MM-DD] 变更描述

- **修改文件**：xxx.py
- **变更内容**：
  - 变更点 1
  - 变更点 2
- **原因**：xxx
### [2026-07-20] 整理项目根目录配置

- **修改文件**：`.gitignore`、`change.md`
- **变更内容**：
  - 在项目根目录的 `.gitignore` 中加入 `.idea/`
  - 将 `change.md` 从外层目录移动到 Git 项目根目录
- **原因**：统一项目目录结构，使 IDE 配置不参与版本管理，并让代码变更记录纳入项目仓库

### [2026-07-20] 修复训练、验证和测试集随机划分泄露

- **修改文件**：`Dataset.py`、`train.py`、`eval_multi_t.py`、`baseline_methods.py`、`evaluate_errors.py`、`run_ablation.py`、`tests/test_dataset_split.py`
- **变更内容**：
  - 新增 `create_split_indices`，使用局部 `np.random.default_rng(seed)` 一次生成 train/val/test 索引
  - 新增 `validate_split_indices`，校验集合非空、索引合法、内部无重复、三组互斥并完整覆盖所有样本
  - `Dataset_UWB` 改为必须接收显式 `indices`，并将原始样本索引保存在 `dataset.indices`
  - 训练入口只创建一次索引划分，并让验证集和测试集继续复用训练集归一化统计量
  - 多时间步评估、传统基线、错误分析和所有消融实验统一使用 `split_seed=42` 及相同测试集
  - 新增 11 项单元测试，覆盖确定性、不同种子、比例数量、全局随机状态隔离、非法索引及归一化统计量复用
- **验证结果**：
  - `python -m unittest discover -s tests -p "test_dataset_split.py" -v`：11 项测试全部通过
  - 修改文件通过 `py_compile`，所有 `Dataset_UWB` 调用均显式传入 `indices`
- **原因**：旧实现每次实例化 Dataset 都重新随机排列，导致训练、验证和测试集合可能重叠，验证与测试指标可能被高估
- **旧实验影响**：基于旧划分得到的验证、测试、消融和传统方法对比结果不再适合作为正式结论；应在训练电脑上使用修复后的统一划分重新运行正式实验
