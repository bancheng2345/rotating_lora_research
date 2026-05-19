# Activation Covariance Experiment

## Purpose
This isolated submodule tests whether activation covariance can extract more realistic high-information directions than weight-only SVD.

OPLoRA uses top singular directions of the pretrained weight matrix `W0`. That is a static geometric signal. This experiment asks whether the model's actual input activations define a better notion of "important directions":

- Weight SVD uses singular vectors of `W0`
- Activation covariance uses eigenvectors of `C_z = E[z z^T]`

If activation-covariance directions capture more real activation energy than SVD directions, then the pretrained weight geometry is not identical to the directions most used under real inputs.

## Why Activation Covariance Can Matter
- SVD only sees the parameter matrix.
- Activation covariance sees the input distribution the model is actually processing.
- If the two subspaces differ by large principal angles, but activation covariance captures more energy, that is evidence for activation-aware projections.

## Directory Layout
- `configs/`: experiment configs
- `scripts/`: end-to-end pipeline scripts
- `src/`: isolated utilities and metrics
- `outputs/activations/{run_name}/`: collected activation chunks and metadata
- `outputs/directions/{run_name}/activation_cov/`: activation covariance eigenvectors/eigenvalues
- `outputs/directions/{run_name}/weight_svd/`: weight-SVD singular directions
- `outputs/metrics/{run_name}/`: comparison JSON/CSV
- `outputs/figures/{run_name}/`: generated figures

## Smoke Test
Runs the full pipeline on a tiny scratch GPT-2 model:

```bash
cd /data/zck/code/rotating_lora_research
conda run -n torch2.7 bash activation_cov_experiment/scripts/run_smoke.sh
```

You can pin the GPU:

```bash
conda run -n torch2.7 bash activation_cov_experiment/scripts/run_smoke.sh --gpu 0
```

## Unit Tests
```bash
cd /data/zck/code/rotating_lora_research
conda run -n torch2.7 python activation_cov_experiment/scripts/run_unit_tests.py
```

## How To Change Model
Edit `activation_cov_experiment/configs/activation_cov_base.yaml` or stack another config such as:

```bash
conda run -n torch2.7 python activation_cov_experiment/scripts/collect_activations.py \
  --base-config activation_cov_experiment/configs/activation_cov_base.yaml \
  --config activation_cov_experiment/configs/llama_layer_probe.yaml
```

For the local Llama-3.1-8B commonsense probe:

```bash
cd /data/zck/code/rotating_lora_research
conda run -n torch2.7 bash activation_cov_experiment/scripts/run_llama31_8b_probe.sh --gpu 0
```

## How To Change Layers And Modules
Edit:
- `targets.modules`
- `targets.layers`
- `targets.layer_selection`

Examples:
- GPT-2 smoke: `modules: ["c_attn"]`
- LLaMA/Qwen: `modules: ["q_proj", "v_proj"]`

`layer_selection` supports:
- `explicit`
- `last_n`
- `all`

## Output Files
- `metadata.json` in `outputs/activations/...`: collected activation metadata and target list
- `activation_cov/*.pt`: activation covariance eigenvectors/eigenvalues
- `weight_svd/*.pt`: weight-SVD singular directions
- `direction_comparison.json`: structured comparison results
- `direction_comparison.csv`: flat table for analysis

## How To Interpret Results
### `capture_energy_vs_k.png`
- Higher is better.
- If ActivationCov is above WeightSVD, activation-aware directions explain more real activation energy.

### `principal_angles.png`
- Larger mean principal angle means the two subspaces are less aligned.
- Large angle plus higher ActivationCov capture is strong evidence that SVD misses activation-relevant directions.

### `residual_energy_vs_k.png`
- Lower is better.
- Lower residual after removing top-k means the chosen subspace better explains the observed activations.

### `layer_module_heatmap.png`
- Positive values mean ActivationCov captures more activation energy than SVD for that layer/module.
- Strong layer/module variation suggests a single projection rule may be suboptimal.

## Current Scope
This module is analysis-only. It does not modify the main PC-LoRA training pipeline.

## How It Could Connect Back To PC-LoRA Later
If activation covariance consistently outperforms SVD:
- replace SVD-derived `V_k` with activation-aware `Q_act`
- optionally mix weight and activation directions
- make projection rules layer-specific instead of global

Current code is isolated under `activation_cov_experiment/` and does not change the original training flow.
