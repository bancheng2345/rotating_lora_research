# PC-LoRA Research Framework

This repository provides a lightweight, runnable research framework for studying Preserve-and-Capture LoRA (PC-LoRA), together with standard LoRA, OPLoRA, and LoRA+NSC baselines.

## Project Goal

The framework is designed to test the following hypothesis:

- `Preserve`: protect the dominant pretrained singular subspace of `W0` using OPLoRA-style left/right residual projections.
- `Capture`: move NSC-style null-space capture into training time so that LoRA `A` learns to capture task-relevant activations inside the residual space.

Implemented update form:

```text
ΔW = P_L B A P_R
P_L = I - U_k U_k^T
P_R = I - V_k V_k^T
```

Implemented losses:

```text
L = L_task + lambda_cap * L_cap + lambda_orth * L_orth
L_orth = ||A A^T - I||_F^2
```

The current MVP implements input residual capture `omega_A(P_R z)` and leaves output capture for a later extension.

## Environment

Default environment:

```bash
conda run -n torch2.7 python ...
```

The current codebase assumes `torch`, `transformers`, `datasets`, `PyYAML`, `pandas`, and `matplotlib` are installed in `torch2.7`.

Tests use Python `unittest`, so `pytest` is not required.

## GPU Selection

GPU selection is config-driven and does not hardcode an index in Python code.

Default config:

```yaml
runtime:
  conda_env: torch2.7
  gpu:
    mode: prefer_sxm4
    cuda_visible_devices: auto
    fallback_cuda_visible_devices: "0"
    override_env: CUDA_VISIBLE_DEVICES
```

Behavior:

- `scripts/select_gpu.py` calls `nvidia-smi`.
- If a GPU name contains `SXM4`, that GPU is preferred by default.
- If detection fails, `fallback_cuda_visible_devices` is used.
- You can force a device manually by setting `runtime.gpu.cuda_visible_devices: "1"` or another fixed value.
- You can also force a device directly from the command line with `--gpu 0` or `--gpu 1`.

The main place to change this is [configs/base.yaml](/data/zck/code/rotating_lora_research/configs/base.yaml:1).

Examples:

```bash
conda run -n torch2.7 python scripts/train.py \
  --base-config configs/base.yaml \
  --config configs/recipes/math500_llama32_1b_pc_lora_dry_run.yaml \
  --gpu 1

bash scripts/run_math500_dry_run.sh --gpu 0
```

## Repository Layout

Core source files:

- [src/methods/lora_base.py](/data/zck/code/rotating_lora_research/src/methods/lora_base.py:1): standard LoRA layer, adapter injection, state dict helpers
- [src/methods/oplora.py](/data/zck/code/rotating_lora_research/src/methods/oplora.py:1): OPLoRA wrapper
- [src/methods/pc_lora.py](/data/zck/code/rotating_lora_research/src/methods/pc_lora.py:1): PC-LoRA wrapper
- [src/methods/projections.py](/data/zck/code/rotating_lora_research/src/methods/projections.py:1): low-rank `P_L` / `P_R`
- [src/methods/capture_losses.py](/data/zck/code/rotating_lora_research/src/methods/capture_losses.py:1): `omega_A(z)` and orthogonality loss
- [src/utils/svd.py](/data/zck/code/rotating_lora_research/src/utils/svd.py:1): top-k SVD and cache management
- [src/utils/gpu_select.py](/data/zck/code/rotating_lora_research/src/utils/gpu_select.py:1): GPU auto-selection
- [src/training/trainer.py](/data/zck/code/rotating_lora_research/src/training/trainer.py:1): lightweight trainer
- [src/training/hooks.py](/data/zck/code/rotating_lora_research/src/training/hooks.py:1): regularization aggregation and mechanism logging
- [src/evaluation/eval_lm.py](/data/zck/code/rotating_lora_research/src/evaluation/eval_lm.py:1): causal LM evaluation

Config files:

- [configs/base.yaml](/data/zck/code/rotating_lora_research/configs/base.yaml:1)
- [configs/experiments/lora.yaml](/data/zck/code/rotating_lora_research/configs/experiments/lora.yaml:1)
- [configs/experiments/oplora.yaml](/data/zck/code/rotating_lora_research/configs/experiments/oplora.yaml:1)
- [configs/experiments/lora_nsc.yaml](/data/zck/code/rotating_lora_research/configs/experiments/lora_nsc.yaml:1)
- [configs/experiments/pc_lora_unweighted.yaml](/data/zck/code/rotating_lora_research/configs/experiments/pc_lora_unweighted.yaml:1)
- [configs/experiments/pc_lora_loss_weighted.yaml](/data/zck/code/rotating_lora_research/configs/experiments/pc_lora_loss_weighted.yaml:1)
- [configs/experiments/actcov_oplora.yaml](/data/zck/code/rotating_lora_research/configs/experiments/actcov_oplora.yaml:1)
- [configs/experiments/actcov_pc_lora.yaml](/data/zck/code/rotating_lora_research/configs/experiments/actcov_pc_lora.yaml:1)
- [configs/experiments/ablation_random_projection.yaml](/data/zck/code/rotating_lora_research/configs/experiments/ablation_random_projection.yaml:1)
- [configs/experiments/ablation_no_orth.yaml](/data/zck/code/rotating_lora_research/configs/experiments/ablation_no_orth.yaml:1)
- [configs/experiments/rank_sweep.yaml](/data/zck/code/rotating_lora_research/configs/experiments/rank_sweep.yaml:1)
- [configs/experiments/k_sweep.yaml](/data/zck/code/rotating_lora_research/configs/experiments/k_sweep.yaml:1)
- `configs/presets/models/`: local model path presets
- `configs/presets/datasets/`: dataset path/schema presets
- `configs/presets/runs/`: dry-run/full-run training schedule presets
- `configs/recipes/`: single-file end-to-end presets

## Implemented Methods

Supported MVP variants:

- `LoRA`
- `OPLoRA`
- `LoRA + NSC(z)`
- `PC-LoRA`: `OPLoRA + NSC(P_R z)`
- `PC-LoRA loss-weighted`: batch-loss-weighted `omega_A(P_R z)`
- `ActCov-OPLoRA`: replace the right SVD basis with precomputed activation-covariance eigenvectors
- `ActCov-PC-LoRA`: `ActCov-OPLoRA + NSC(P_R z)`

Implemented metric families:

- task metrics: `eval_loss`, `eval_accuracy`, `eval_exact_match`, `eval_pass_at_1`, `eval_metric_name`, `eval_metric_value`
- forgetting metrics: `forgetting_gap`, `retention_score`, `base_model_score`, `current_score`
- subspace metrics: `rho_k`, `delta_w_norm`, `q_delta_w_norm`
- capture metrics: `omega_full_mean/std`, `omega_residual_mean/std`, `captured_energy_full_mean`, `captured_energy_residual_mean`
- orthogonality/update metrics: `orth_error_A`, `orth_error_A_norm`, `A_norm`, `B_norm`, `raw_BA_norm`, `projected_delta_w_norm`
- efficiency metrics: `step_time_sec`, `samples_per_sec`, `tokens_per_sec`, `peak_gpu_memory_gb`, `svd_time_sec`, `capture_metric_time_sec`, `rho_metric_time_sec`

## Smoke Test

Quick smoke run:

```bash
cd /data/zck/code/rotating_lora_research
bash scripts/run_smoke.sh
```

Direct unit tests:

```bash
cd /data/zck/code/rotating_lora_research
conda run -n torch2.7 python -m unittest discover -s tests -v
```

Metric-focused tests:

```bash
conda run -n torch2.7 python scripts/run_metric_tests.py
```

## Layered Configs

`train.py`, `eval.py`, and `select_gpu.py` now support repeated `--config` flags. Configs are merged in order, and later files override earlier files.

Recommended layering:

1. `configs/base.yaml`
2. `configs/presets/models/...`
3. `configs/presets/datasets/...`
4. `configs/presets/runs/...`
5. `configs/experiments/...`

Example:

```bash
conda run -n torch2.7 python scripts/train.py \
  --base-config configs/base.yaml \
  --config configs/presets/models/llama32_1b_local.yaml \
  --config configs/presets/datasets/math500.yaml \
  --config configs/presets/runs/math500_dry_run.yaml \
  --config configs/experiments/pc_lora_unweighted.yaml \
  --gpu 1
```

If you prefer a single file, use a recipe config such as:

```bash
conda run -n torch2.7 python scripts/train.py \
  --base-config configs/base.yaml \
  --config configs/recipes/math500_llama32_1b_pc_lora_dry_run.yaml \
  --gpu 1
```

## Preserve / Capture Split

The current experimental rule is:

- Preserve uses frozen `W0` directions, normally SVD right/left bases, to protect pretrained subspaces.
- Capture uses downstream-task signals only as guidance for `A`, not as hard projection directions.
- Task ActivationCov or MWA directions should therefore be loaded through `capture_basis_source`, with `use_capture_alignment_loss: true`, rather than through `right_projection_source` unless running a negative ablation.

The residual joint-capture variant keeps SVD-right preserve, keeps residual NSC `omega_A(P_R z)`, and adds a residualized data-weight joint capture basis:

```yaml
lora:
  right_projection_source: svd
  use_right_projection: true
  use_capture_loss: true
  capture_space: residual
  use_capture_alignment_loss: true
  capture_basis_source: mwa
  capture_basis_dir: activation_cov_experiment/outputs/directions/llama31_8b_metamathqa_probe/mwa
  capture_basis_residualize_against_preserve: true
```

Run the matched MetaMathQA comparison:

```bash
cd /data/zck/code/rotating_lora_research
bash scripts/run_metamathqa_residual_joint_capture_8b.sh --gpu 1
```

Summarize the latest completed runs:

```bash
/home/zck/miniconda3/envs/torch2.7/bin/python scripts/summarize_metamathqa_mwa_compare.py \
  --notes-contains metamathqa_8b_residual_joint_capture_bsz8 \
  --out results/tables/metamathqa_residual_joint_capture.csv
```

## ActivationCov Projection

The main trainer can use directions produced by `activation_cov_experiment` as the right projection basis:

```yaml
lora:
  right_projection_source: activation_cov
  activation_cov_dir: activation_cov_experiment/outputs/directions/llama31_8b_commonsense_probe_gpu1/activation_cov
```

This mode is now treated as a negative ablation for task-derived ActivationCov:
task high-frequency directions should usually be captured, not removed from the
LoRA update path.

The current 8B probe directions cover LLaMA-3.1-8B layers `[0, 8, 16, 24]` and modules `q_proj/v_proj`, so the matched comparison configs restrict LoRA injection to those layers. Run the quick 8B comparison without expensive `rho_k` SVD metrics:

```bash
cd /data/zck/code/rotating_lora_research
bash scripts/run_commonsense_actcov_compare_8b.sh --gpu 1
```

Add `--with-rho` only when you explicitly want full `rho_k` mechanism logging; on 8B this performs CPU SVDs and can dominate short dry runs. Use `--full` to switch from the dry-run preset to `commonsense_170k_full_8b`.

## Main Experiments

Run the P0 main set:

```bash
cd /data/zck/code/rotating_lora_research
bash scripts/run_main_experiments.sh --gpu 1
```

For math experiments, use MetaMathQA by default. The reported task accuracy is
final-answer Exact Match: number of samples whose final answer exactly matches
the reference answer divided by total evaluated samples. PPL, eval loss, and
token accuracy are auxiliary diagnostics only.

```bash
bash scripts/run_metamathqa_residual_joint_capture_8b.sh --gpu 1
```

After training, run generation-based final-answer EM:

```bash
bash scripts/eval_metamathqa_mwa_compare_generation_8b.sh \
  --gpu 1 \
  --notes-contains metamathqa_8b_residual_joint_capture_bsz8 \
  --max-examples 5000
```

## Ablations and Sweeps

Ablation entrypoint:

```bash
bash scripts/run_ablation.sh
```

Rank sweep:

```bash
bash scripts/run_rank_sweep.sh
```

Projection-rank sweep:

```bash
bash scripts/run_k_sweep.sh
```

## Outputs

Each run resolves to:

```text
outputs/runs/{model}/{method}_{model}_{dataset}_r{rank}_k{k}_seed{seed}_{timestamp}/
```

Expected run artifacts:

- `config_resolved.yaml`
- `train_metrics.jsonl`
- `eval_metrics.jsonl`
- `forgetting_metrics.jsonl`
- `mechanism_metrics.jsonl`
- `efficiency_metrics.jsonl`
- `final_summary.json`
- `checkpoints/`
- `logs/`

## Metric System

The framework now writes five metric streams per run:

- `train_metrics.jsonl`
  Records optimization-time scalars such as `task_loss`, `cap_loss`, `orth_loss`, `learning_rate`, `step_time_sec`, and GPU memory.
- `eval_metrics.jsonl`
  Records unified evaluation rows with `step`, `epoch`, `split`, `dataset`, `domain`, `metric_name`, `metric_value`, `loss`, `num_samples`, and timestamp. Placeholder fields for `eval_accuracy`, `eval_exact_match`, and `eval_pass_at_1` are always present.
- `forgetting_metrics.jsonl`
  Records cross-domain forgetting rows with `train_domain`, `eval_domain`, `current_score`, `base_model_score`, `forgetting_gap`, and `retention_score`.
- `mechanism_metrics.jsonl`
  Records per-layer mechanism rows including `rho_k`, `omega_full_mean/std`, `omega_residual_mean/std`, `orth_error_A`, `A_norm`, `B_norm`, `raw_BA_norm`, and projected-update norms.
- `efficiency_metrics.jsonl`
  Records efficiency overhead such as `samples_per_sec`, `tokens_per_sec`, `peak_gpu_memory_gb`, `svd_time_sec`, `capture_metric_time_sec`, and `rho_metric_time_sec`.

Interpretation:

- `rho_k`
  Measures how much of the LoRA update falls back into the pretrained top-`k` left singular subspace. Lower is better when testing preserve behavior.
- `omega_full` / `omega_residual`
  Measure how poorly the current LoRA `A` row space captures full activations or residual activations. Lower means better capture.
- `orth_error_A`
  Measures how far `A A^T` deviates from the identity. Lower means the LoRA rows are closer to orthogonal.
- `forgetting_gap`
  Defined as `base_model_score - current_score` when a base reference exists.
- `retention_score`
  Defined as `current_score / base_model_score` when a base reference exists.

## Plotting and Tables

Single-run mechanism plots:

```bash
conda run -n torch2.7 python scripts/plot_mechanism.py \
  --run_dir outputs/runs/<model>/<run_name>
```

This writes figures to `outputs/runs/<model>/<run_name>/figures/`:

- `omega_full_vs_residual.png`
- `rho_k_curve.png`
- `orth_error_curve.png`
- `cap_loss_curve.png`
- `efficiency_curve.png`

Multi-run comparison:

```bash
conda run -n torch2.7 python scripts/plot_mechanism.py \
  --root outputs/runs \
  --compare_methods lora,oplora,lora_nsc,pc_lora
```

Efficiency-only plots:

```bash
conda run -n torch2.7 python scripts/plot_efficiency.py \
  --root outputs/runs \
  --compare_methods lora,oplora,lora_nsc,pc_lora
```

Rank sweep figure:

```bash
conda run -n torch2.7 python scripts/plot_rank_sweep.py \
  --run-dirs outputs/runs/run_1 outputs/runs/run_2
```

Projection-rank sweep figure:

```bash
conda run -n torch2.7 python scripts/plot_k_sweep.py \
  --run-dirs outputs/runs/run_1 outputs/runs/run_2
```

Metric summary for one run:

```bash
conda run -n torch2.7 python scripts/analyze_metrics.py \
  --run-dir outputs/runs/<model>/<run_name>
```

Aggregate tables across runs:

```bash
conda run -n torch2.7 python scripts/aggregate_results.py \
  --root outputs/runs \
  --out results/tables
```

Default figure/table targets:

- `results/figures/omega_curve.png`
- `results/figures/rho_curve.png`
- `results/figures/rank_sweep.png`
- `results/figures/k_sweep.png`
- `results/figures/efficiency_curve.png`
- `results/tables/main_results.csv`
- `results/tables/ablation_results.csv`
- `results/tables/mechanism_summary.csv`

## Fisher/CUR Projection Experiments

Two non-SVD projection sources are available for matched OPLoRA-style comparisons:

- `fisher`: empirical-Fisher input factor directions, approximating `E[||grad_y||^2 z z^T]` from a small calibration split.
- `cur`: CUR-style row/column selection from pretrained weights; the right basis uses the row space of important rows selected by row norm.

Build directions and run seed 0 on Llama-3.1-8B commonsense:

```bash
bash scripts/run_commonsense_fisher_cur_compare_8b.sh --gpu 0 --seeds 0
```

If directions already exist, skip rebuilding:

```bash
bash scripts/run_commonsense_fisher_cur_compare_8b.sh --gpu 0 --seeds 1,2 --skip-build
```

Direction files are written to:

- `outputs/directions/fisher/llama31_8b_commonsense_probe_layers/`
- `outputs/directions/cur/llama31_8b_commonsense_probe_layers/`

The training configs are:

- `configs/experiments/fisher_oplora_probe_layers.yaml`
- `configs/experiments/cur_oplora_probe_layers.yaml`

## Notes and Current Scope

- The MVP is intentionally lightweight and defaults to an offline synthetic setup so smoke tests do not depend on downloads.
- `grad_norm` capture weighting is left as an explicit future extension.
- Output capture `L_out` is not yet implemented.
- For real commonsense/math/code experiments, point the config at local model/data assets and launch the larger runs manually.
