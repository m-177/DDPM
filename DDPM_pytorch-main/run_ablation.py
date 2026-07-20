# -----------------------------
# run_ablation.py - 消融实验脚本
# 自动跑 4 组实验，结果直接存入 ablation_results/
# 使用方式: python run_ablation.py
# -----------------------------
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import train
from Dataset import DEFAULT_SPLIT_SEED

# ---- 共用参数 ----
COMMON = dict(
    clean_path="uwb_signals_time_clean.npy",
    batchsize=8,
    total_epoch=600,
    val_ratio=0.1,
    test_ratio=0.1,
    split_seed=DEFAULT_SPLIT_SEED,
    lr=2e-4,
    timesteps=1000,
    use_cosine_schedule=True,
    gradient_clip=1.0,
    use_ema=True,
    save_every=50,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    snr_eval_freq=10,
    dropout_rate=0.1,
    num_workers=0,
    memory_check_freq=50,
    gradient_accumulation_steps=1,
    snr_patience=4,
    resume_from=None,
)

# ---- 消融实验定义 ----
EXPERIMENTS = [
    {
        'name': 'baseline',
        'desc': '定版配置（基线）',
        'lambda_mse': 1.5, 'lambda_rel': 0.4,
        'lambda_peak': 3.5, 'lambda_corr': 1.8,
    },
    {
        'name': 'no_peak',
        'desc': '关闭 Peak-Aware 损失',
        'lambda_mse': 1.5, 'lambda_rel': 0.4,
        'lambda_peak': 0.0, 'lambda_corr': 1.8,
    },
    {
        'name': 'no_corr',
        'desc': '关闭 Peak-Corr 损失',
        'lambda_mse': 1.5, 'lambda_rel': 0.4,
        'lambda_peak': 3.5, 'lambda_corr': 0.0,
    },
    {
        'name': 'old_weights',
        'desc': '旧权重（MSE=2.5, Peak=1.8）',
        'lambda_mse': 2.5, 'lambda_rel': 0.4,
        'lambda_peak': 1.8, 'lambda_corr': 1.8,
    },
]

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(SCRIPT_DIR)
    ABLATION_DIR = os.path.join(SCRIPT_DIR, "ablation_results")
    os.makedirs(ABLATION_DIR, exist_ok=True)

    for exp in EXPERIMENTS:
        print("\n" + "=" * 70)
        print(f"🔬 消融实验: {exp['name']} — {exp['desc']}")
        print(f"   MSE={exp['lambda_mse']}, Rel={exp['lambda_rel']}, "
              f"Peak={exp['lambda_peak']}, Corr={exp['lambda_corr']}")
        print("=" * 70)

        # 每组实验输出到独立目录
        exp_output = os.path.join(ABLATION_DIR, exp['name'])

        model, ema_model = train(
            lambda_mse=exp['lambda_mse'],
            lambda_rel=exp['lambda_rel'],
            lambda_peak=exp['lambda_peak'],
            lambda_corr=exp['lambda_corr'],
            output_dir=exp_output,
            **COMMON
        )

        # 写实验配置记录
        with open(os.path.join(exp_output, "config.txt"), 'w') as f:
            f.write(f"实验: {exp['name']} — {exp['desc']}\n")
            f.write(f"MSE={exp['lambda_mse']}, Rel={exp['lambda_rel']}, "
                    f"Peak={exp['lambda_peak']}, Corr={exp['lambda_corr']}\n")
            for k, v in COMMON.items():
                f.write(f"{k}: {v}\n")

        print(f"\n✅ {exp['name']} 完成 → {exp_output}")

    # ---- 汇总 ----
    print("\n" + "=" * 70)
    print("📊 所有消融实验完成！")
    print("=" * 70)
    for exp in EXPERIMENTS:
        d = os.path.join(ABLATION_DIR, exp['name'])
        report = os.path.join(d, "logs_classic", "analysis_report.txt")
        status = "✅" if os.path.exists(report) else "❌"
        print(f"  {status} {exp['name']}: {report}")
