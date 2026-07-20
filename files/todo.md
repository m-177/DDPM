# 项目待办事项与后续计划

> 最后更新：2026-07-20

---

## ⚠️ 重要：Bug 修复导致历史数据需重跑

2026-07-20 发现并修复了 `denoise_for_eval` 的 off-by-one 错误和 clamp 不一致问题。**训练过程不受影响（模型权重有效）**，但所有基于该函数的评估指标均不准确，需重新跑评估脚本。

---

## 一、Bug 修复状态

| # | 问题 | 文件 | 状态 |
|---|------|------|:---:|
| P0 | `denoise_for_eval` 逆向扩散 off-by-one（`t_idx-1` → `t_idx`） | `train.py:331-351` | ✅ |
| P1 | Clamp 不一致 `[-1.2,1.2]` → `[-1,1]` | `train.py:343` | ✅ |
| P1 | Clamp 不一致 `[-1.2,1.2]` → `[-1,1]` | `reverse_diffusion_process.py:104` | ✅ |
| P1 | 模型架构不兼容（1D: `DenoiseUNet` → `SimpleUNet1D_Classic`） | `reverse_diffusion_process.py:273-288` | ✅ |
| P2 | `generate_result.py` 已删除（生成的是假数据） | — | ✅ |
| P2 | `requirements.txt` 格式不规范，缺少 PyWavelets、psutil 等 | `requirements.txt` | ✅ |
| P2 | `baseline_methods.py:150` 硬编码旧 DDPM 数据 `+13.59 dB` | `baseline_methods.py` | 🔲 待重跑后更新 |
| P1 | `psutil` 未列入 `requirements.txt`，新环境 `pip install -r` 会崩溃 | `requirements.txt` | ✅ |
| P2 | 训练循环未调用 `set_seed()`，两次相同参数训练结果不可复现 | `train.py` | ✅ |
| P2 | `save_report()` docstring 位置错误（在 `if` 块之后） | `evaluate_errors.py:613` | ✅ |
| P2 | `run_ablation.py` 与主训练参数不一致（`num_workers=0` vs `1`, `save_every=50` vs `10`） | `run_ablation.py` | ✅ |
| P1 | `run_ablation.py` `baseline` 实验标签"定版配置"但 MSE=1.5, Peak=3.5，实际定版为 MSE=2.5, Peak=1.8（`old_weights` 才是真基线） | `run_ablation.py:39-63` | ✅ |
| P2 | 扩散参数计算三份重复实现（`train.py` / `reverse_diffusion_process.py` / `interpolate.py`） | 四文件 → `diffusion_utils.py` | ✅ |
| P0 | `interpolate.py` `reverse_diffusion` off-by-one（`t-1` → `t`，6处） | `interpolate.py:69-95` | ✅ |
| P3 | `interpolate.py` 同样使用旧模型架构 `DenoiseUNet` | `interpolate.py` | 🔲 低优先级 |
| P3 | `generate_result.py` 已删除 | — | ✅ |
| P3 | `baseline_methods.py` 中 `calc_metrics` 函数定义后从未调用（死代码） | `baseline_methods.py:44` | ✅ |

---

## 二、需要重新做的工作（按顺序）

> 所有命令均在 `DDPM_pytorch-main/` 目录下运行

### 第 1 步：生成数据

```bash
python uwb_signal_generate.py
```

- 产出：`uwb_signals_time_clean.npy`（1000 条 × 10120 采样点）
- ⏱ < 1 分钟

---

### 第 2 步：训练模型

```bash
python train.py
```

- 产出：`saved_models_classic/` 下模型权重 + `logs_classic/` 下训练日志
- ⏱ 数小时（600 epoch 上限，SNR 收敛自动早停）

---

### 第 3 步：多时间步评估

```bash
python eval_multi_t.py
```

- 产出：`logs_classic/multi_t_eval_fine.txt` + `logs_classic/multi_t_viz_fine/` 下 12 张图
- 用途：更新 paper.md 的 §5.1（主结果表）和 §5.2（多时间步表）
- ⚠️ 跑完后将 t=300 的指标更新到 `baseline_methods.py:150-151`
- ⏱ ~5 分钟

---

### 第 4 步：错误分析

```bash
python evaluate_errors.py --visualize --max_viz 5
```

- 产出：`error_report.txt` + `error_vis/` 下错误样本图
- 用途：更新 paper.md 的 §6.1（极性反转）和 §6.3（波形丢失）
- ⏱ ~3 分钟

---

### 第 5 步：基线方法对比

```bash
python baseline_methods.py
```

> 前提：第 3 步完成后已将 t=300 指标更新到 `baseline_methods.py:150-151`

- 产出：`logs_classic/baseline_comparison.txt`
- 用途：更新 paper.md 的 §8（方法对比表）
- ⏱ ~2 分钟

---

### 第 6 步：消融实验（4 组）

```bash
python run_ablation.py
```

> 4 组 × 600 epoch 上限，SNR 收敛会自动早停。最耗时，建议最后跑或挂机。

- 产出：`ablation_results/{baseline,no_peak,no_corr,old_weights}/logs_classic/analysis_report.txt`
- 用途：更新 paper.md 的 §7（消融实验表）
- ⏱ 数小时

---

### 第 7 步：更新 paper.md

将所有新结果填入 `files/paper.md` 对应章节。

---

## 三、操作汇总

| 步骤 | 命令 | 时间 | 更新 paper.md |
|:---:|------|:---:|:---:|
| 1 | `python uwb_signal_generate.py` | <1min | — |
| 2 | `python train.py` | 数小时 | — |
| 3 | `python eval_multi_t.py` | ~5min | §5.1, §5.2 |
| — | 更新 `baseline_methods.py:150-151` | <1min | — |
| 4 | `python evaluate_errors.py --visualize` | ~3min | §6.1, §6.3 |
| 5 | `python baseline_methods.py` | ~2min | §8 |
| 6 | `python run_ablation.py` | 数小时 | §7 |
| 7 | 手动编辑 `files/paper.md` | ~10min | 全文 |

> 第 3、4、5 步互不依赖，可同时跑；第 6 步最耗时。

---

## 四、论文待完成

| 优先级 | 任务 | 说明 |
|:---:|------|------|
| 🔴 | 按 SNR 分桶分析 | 定位模型在各 SNR 区间的表现差异 |
| 🔴 | 模型架构图 | 手绘或 draw.io |
| 🟡 | 去噪前后频谱对比（FFT） | 补充频域视角 |
| 🟡 | 撰写方法部分 | DDPM 原理 + UNet 架构 + 损失函数 |
| 🟡 | 撰写实验部分 | 数据描述 + 主结果 + 消融 + 多 t + 错误分析 |

---

## 五、已完成

- [x] UWB 信号生成器 + 信号特性分析
- [x] 1D UNet 模型设计（SimpleUNet1D_Classic，~2M 参数）
- [x] 混合损失函数（MSE + RelMSE + Peak-Aware + Peak-Corr）
- [x] EMA + CosineAnnealingLR + 梯度裁剪
- [x] 训练暂停/恢复 + SNR 收敛自动停止 + Loss 早停
- [x] 双模型评估对比（best_by_loss vs best_by_snr）
- [x] 损失权重定版：MSE=2.5, RelMSE=0.4, Peak=1.8, Corr=1.8
- [x] 波形丢失分级评估（自适应阈值 + 形状双判 + 四级分级）
- [x] 多 t 精细评估（100~400，每 20 步）
- [x] 数据集划分验证（互斥 + 完整覆盖）
- [x] P0/P1 Bug 修复（off-by-one + clamp + 模型架构 + interpolate.py off-by-one）
- [x] 扩散参数统一提取到 `diffusion_utils.py`
- [x] `train()` 默认参数更新为定版配置
- [x] `run_ablation.py` 配置修正（baseline 权重 + COMMON 参数）
- [x] 主训练添加 `set_seed(42)` 可复现
- [x] `save_report()` docstring 修复
- [x] `generate_result.py` 添加 `__main__` 保护
- [x] `baseline_methods.py` 移除死代码 `calc_metrics()`
- [x] `requirements.txt` 标准化为 `package==version` 格式
- [x] Git 重构提交（DDPM_pytorch-main/）
- [x] README 更新

---

## 六、中期计划

- [ ] DDIM 加速采样（推理时间步 1000 → 50）
- [ ] 增大模型容量（Mid 256→512 或加 Self-Attention）
- [ ] 配置参数外置（YAML 配置文件）
- [ ] 真实 UWB 数据验证（`rx_fullframe_iq.mat`）
- [ ] 将 `SimpleUNet1D_Classic` 提取到 `models.py` 统一管理
- [ ] 清理残留文件（`.gi`、`denoising_results_classic.png`、`check_cuda.txt` 等）

---

## 注意事项

1. **定版配置**：batch=8, lr=2e-4, MSE=2.5, Peak=1.8, Corr=1.8
2. **消融已证实**：Corr 最关键，Peak 影响最佳样本，MSE 不应降权
3. **Bug 修复后**：所有评估脚本需重新运行，但模型不需要重新训练
4. **变更记录**：每次改动同步追加 `files/update_7`
5. **论文数据**：更新 `files/paper.md`
