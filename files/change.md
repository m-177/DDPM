# 变更记录

## 2026-07-20

### 修复: `denoise_for_eval` 逆向扩散 Off-by-One 错误 (P0)

- **修改文件**: `DDPM_pytorch-main/train.py` (第 331-351 行)
- **变更内容**: `denoise_for_eval` 函数中所有数组索引从 `t_idx - 1` 改为 `t_idx`
  - `t_tensor`: `t_idx - 1` → `t_idx`
  - `alphas[...]`, `alphas_cumprod[...]`, `alphas_cumprod_prev[...]`, `betas[...]`: `t_idx - 1` → `t_idx`
  - `posterior_variance[...]`: `t_idx - 1` → `t_idx`
- **原因**: 训练时 `t = torch.randint(0, timesteps, ...)` 使用 0-indexed 时间步（t ∈ [0, 999]），噪声水平由 `alphas_cumprod[t]` 决定，模型学习映射 `(x_t, t) → noise`。原代码在推理循环第一步就将 `t_idx - 1` 传入模型，导致模型接收的时间步比实际噪声水平少 1，300 步逆向扩散中每一步的 `ᾱ_t` 都使用了错误值，累积误差影响去噪质量。
- **影响范围**: 所有依赖 `denoise_for_eval` 的评估结果（`evaluate_checkpoint`, `visualize_denoising`, SNR 监控, `eval_multi_t.py`, `evaluate_errors.py`）均受影响，修复后需重新训练和评估。
