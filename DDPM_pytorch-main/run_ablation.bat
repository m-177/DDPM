@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo 消融实验 — 自动运行 4 组
echo ============================================================

REM ---- 第 1 组: baseline（改 train.py 参数后运行） ----
echo.
echo [1/4] baseline — 定版配置
python -c "
import train
train.LAMBDA_MSE=1.5; train.LAMBDA_REL=0.4; train.LAMBDA_PEAK=3.5; train.LAMBDA_CORR=1.8
train.TOTAL_EPOCH=600; train.BATCH_SIZE=8; train.LEARNING_RATE=2e-4
train.NUM_WORKERS=0; train.RESUME=False
model, ema = train.train(clean_path='uwb_signals_time_clean.npy', batchsize=8, total_epoch=600, val_ratio=0.1, test_ratio=0.1, lr=2e-4, timesteps=1000, use_cosine_schedule=True, gradient_clip=1.0, use_ema=True, save_every=50, device='cuda' if __import__('torch').cuda.is_available() else 'cpu', snr_eval_freq=10, lambda_mse=1.5, lambda_rel=0.4, lambda_peak=3.5, lambda_corr=1.8, dropout_rate=0.1, num_workers=0, memory_check_freq=50, gradient_accumulation_steps=1, snr_patience=4, resume_from=None)
"
if exist "logs_classic\analysis_report.txt" (
    if not exist "ablation_results\baseline" mkdir "ablation_results\baseline"
    xcopy /E /Y /Q "logs_classic" "ablation_results\baseline\logs_classic\"
    xcopy /E /Y /Q "saved_models_classic" "ablation_results\baseline\saved_models_classic\"
    echo ✅ baseline 完成
) else (
    echo ❌ baseline 失败
)

REM ---- 第 2 组: no_peak ----
echo.
echo [2/4] no_peak — 关闭 Peak-Aware
python -c "
import train
model, ema = train.train(clean_path='uwb_signals_time_clean.npy', batchsize=8, total_epoch=600, val_ratio=0.1, test_ratio=0.1, lr=2e-4, timesteps=1000, use_cosine_schedule=True, gradient_clip=1.0, use_ema=True, save_every=50, device='cuda' if __import__('torch').cuda.is_available() else 'cpu', snr_eval_freq=10, lambda_mse=1.5, lambda_rel=0.4, lambda_peak=0.0, lambda_corr=1.8, dropout_rate=0.1, num_workers=0, memory_check_freq=50, gradient_accumulation_steps=1, snr_patience=4, resume_from=None)
"
if exist "logs_classic\analysis_report.txt" (
    if not exist "ablation_results\no_peak" mkdir "ablation_results\no_peak"
    xcopy /E /Y /Q "logs_classic" "ablation_results\no_peak\logs_classic\"
    xcopy /E /Y /Q "saved_models_classic" "ablation_results\no_peak\saved_models_classic\"
    echo ✅ no_peak 完成
) else (
    echo ❌ no_peak 失败
)

REM ---- 第 3 组: no_corr ----
echo.
echo [3/4] no_corr — 关闭 Peak-Corr
python -c "
import train
model, ema = train.train(clean_path='uwb_signals_time_clean.npy', batchsize=8, total_epoch=600, val_ratio=0.1, test_ratio=0.1, lr=2e-4, timesteps=1000, use_cosine_schedule=True, gradient_clip=1.0, use_ema=True, save_every=50, device='cuda' if __import__('torch').cuda.is_available() else 'cpu', snr_eval_freq=10, lambda_mse=1.5, lambda_rel=0.4, lambda_peak=3.5, lambda_corr=0.0, dropout_rate=0.1, num_workers=0, memory_check_freq=50, gradient_accumulation_steps=1, snr_patience=4, resume_from=None)
"
if exist "logs_classic\analysis_report.txt" (
    if not exist "ablation_results\no_corr" mkdir "ablation_results\no_corr"
    xcopy /E /Y /Q "logs_classic" "ablation_results\no_corr\logs_classic\"
    xcopy /E /Y /Q "saved_models_classic" "ablation_results\no_corr\saved_models_classic\"
    echo ✅ no_corr 完成
) else (
    echo ❌ no_corr 失败
)

REM ---- 第 4 组: old_weights ----
echo.
echo [4/4] old_weights — 旧权重
python -c "
import train
model, ema = train.train(clean_path='uwb_signals_time_clean.npy', batchsize=8, total_epoch=600, val_ratio=0.1, test_ratio=0.1, lr=2e-4, timesteps=1000, use_cosine_schedule=True, gradient_clip=1.0, use_ema=True, save_every=50, device='cuda' if __import__('torch').cuda.is_available() else 'cpu', snr_eval_freq=10, lambda_mse=2.5, lambda_rel=0.4, lambda_peak=1.8, lambda_corr=1.8, dropout_rate=0.1, num_workers=0, memory_check_freq=50, gradient_accumulation_steps=1, snr_patience=4, resume_from=None)
"
if exist "logs_classic\analysis_report.txt" (
    if not exist "ablation_results\old_weights" mkdir "ablation_results\old_weights"
    xcopy /E /Y /Q "logs_classic" "ablation_results\old_weights\logs_classic\"
    xcopy /E /Y /Q "saved_models_classic" "ablation_results\old_weights\saved_models_classic\"
    echo ✅ old_weights 完成
) else (
    echo ❌ old_weights 失败
)

echo.
echo ============================================================
echo 全部完成！结果在 ablation_results\
echo ============================================================
dir /B ablation_results
pause
