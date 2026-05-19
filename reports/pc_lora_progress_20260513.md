# PC-LoRA / Preserve-and-Capture LoRA 进展更新

日期：2026-05-13

## 当前结论

- SVD-right / OPLoRA 仍是最稳定的 Preserve baseline。
- ActCov/Fisher/CUR 直接 hard projection 是负结果；ActCov 更适合 allocation/capture 辅助。
- Capture 改变了机制指标，但 PC-LoRA 尚未稳定胜出。
- 本版汇报任务性能、机制指标和效率指标，不只看 loss。
- 下一步应优先完成 MetaMathQA 5000 generation EM，再决定是否补 seed。

## Commonsense matched 8-adapter：任务指标

| 方法                      | seeds | Exact/EM ↑      | Eval loss ↓ | Token acc ↑ | 参数      |
| ----------------------- | ----- | --------------- | ----------- | ----------- | ------- |
| LoRA matched            | 0,1,2 | 0.6099 ± 0.0107 | 0.2813      | 0.8659      | 425,984 |
| SVD-right / OPLoRA      | 0,1,2 | 0.6147 ± 0.0022 | 0.2802      | 0.8675      | 425,984 |
| PC-LoRA matched         | 0     | 0.6034          | 0.2813      | 0.8641      | 425,984 |
| ActCov rank LoRA        | 0,1,2 | 0.6129 ± 0.0181 | 0.2779      | 0.8669      | 417,792 |
| ActCov rank + SVD-right | 0,1,2 | 0.6122 ± 0.0163 | 0.2739      | 0.8669      | 417,792 |
| ActCov rank PC-LoRA     | 0,1,2 | 0.6132 ± 0.0074 | 0.2743      | 0.8672      | 417,792 |

## Commonsense matched 8-adapter：机制指标

| 方法                      | omega_full ↓ | omega_resid ↓ | orth_A ↓ | rho_k      |
| ----------------------- | ------------ | ------------- | -------- | ---------- |
| LoRA matched            | 0.9822       | 0.9822        | 1.0155   | not logged |
| SVD-right / OPLoRA      | 0.9859       | 0.9834        | 1.0164   | not logged |
| PC-LoRA matched         | 0.9839       | 0.9806        | 1.0031   | not logged |
| ActCov rank LoRA        | 0.9847       | 0.9847        | 0.9855   | not logged |
| ActCov rank + SVD-right | 0.9870       | 0.9851        | 0.9891   | not logged |
| ActCov rank PC-LoRA     | 0.9863       | 0.9842        | 0.9852   | not logged |

## Commonsense full 64-adapter：任务指标

| 方法            | seeds | Exact/EM ↑      | Eval loss ↓ | Token acc ↑ | 参数        |
| ------------- | ----- | --------------- | ----------- | ----------- | --------- |
| LoRA full     | 0,1,2 | 0.7137 ± 0.0284 | 0.2167      | 0.9022      | 3,407,872 |
| LoRA+NSC full | 0,1,2 | 0.7101 ± 0.0288 | 0.2199      | 0.9010      | 3,407,872 |
| OPLoRA full   | 0,1,2 | 0.6745 ± 0.0779 | 0.2384      | 0.8890      | 3,407,872 |
| PC-LoRA full  | 0,1,2 | 0.6401 ± 0.1246 | 0.2529      | 0.8773      | 3,407,872 |

## Commonsense full 64-adapter：机制/效率指标

| 方法            | omega_full | omega_resid | orth_A | rho_k      | time  | GPU GB |
| ------------- | ---------- | ----------- | ------ | ---------- | ----- | ------ |
| LoRA full     | 0.9898     | 0.9913      | 1.1332 | 0.0156     | 1.49h | 17.80  |
| LoRA+NSC full | 0.9891     | 0.9906      | 1.4536 | not logged | 1.42h | 17.97  |
| OPLoRA full   | 0.9910     | 0.9902      | 1.2079 | 0.0000     | 1.43h | 17.80  |
| PC-LoRA full  | 0.9901     | 0.9893      | 1.4877 | not logged | 1.50h | 17.98  |

## MetaMathQA seed0：任务指标

| 方法       | PPL ↓    | Eval loss ↓ | Token acc ↑ | Teacher EM |
| -------- | -------- | ----------- | ----------- | ---------- |
| oplora   | 1.302077 | 0.263961    | 0.9112      | 0.0012     |
| lora_nsc | 1.302963 | 0.264641    | 0.9110      | 0.0004     |
| pc_lora  | 1.302431 | 0.264233    | 0.9111      | 0.0010     |
| lora     | 1.302493 | 0.264280    | 0.9111      | 0.0008     |

## MetaMathQA seed0：机制/效率指标

| 方法       | omega_full ↓ | omega_resid ↓ | orth_A ↓ | time  | GPU GB |
| -------- | ------------ | ------------- | -------- | ----- | ------ |
| oplora   | 0.9953       | 0.9950        | 0.5498   | 1.61h | 23.65  |
| lora_nsc | 0.9916       | 0.9926        | 0.5432   | 1.54h | 24.08  |
| pc_lora  | 0.9925       | 0.9921        | 0.5584   | 1.65h | 24.08  |
| lora     | 0.9947       | 0.9955        | 0.5379   | 1.46h | 23.65  |

## 新增研究方法：Data-Weight Joint Directions

ActivationCov 找到的是数据中方差最大的输入方向，可解释真实 activation energy，但这些方向不一定会被预训练权重 W0 强烈使用。SVD 找到的是 W0 的高能映射方向，但它只依赖权重矩阵本身，不考虑真实输入分布。为同时建模“数据中经常出现”和“权重会强烈映射”这两个因素，我们构造 data-weight joint high-information directions：

`M_WA = C_z^{1/2} W0^T W0 C_z^{1/2}`

其中 `C_z = E[z z^T]` 表示某层输入激活协方差，`W0^T W0` 表示该线性层输入方向被映射到输出的能量。取 `M_WA` 的 top-k eigenvectors 后映射回输入空间并正交化，得到联合方向 `Q_WA`。实验上比较三类方向：ActivationCov、SVD、M_WA。评价指标包括 `InputCapture = ||Q Q^T z||^2 / ||z||^2` 和 `OutputCapture = ||W0 Q Q^T z||^2 / ||W0 z||^2`。如果 `Q_WA` 在 OutputCapture 上优于 ActivationCov/SVD，说明真实模型行为中的高信息方向需要同时考虑数据分布和权重映射。

## M_WA 实验方案

1. 在同一模型、同一数据、同一 target layers/modules 上收集 activation `z`。
2. 估计 `C_z`，并对每个原始权重 `W0` 计算 `G_W = W0^T W0`。
3. 构造 `M_WA = C_z^{1/2} G_W C_z^{1/2}`，取 top-k 得到 joint directions。
4. 对 k = 4, 8, 16, 32, 64 做 sweep，比较 ActivationCov / SVD / M_WA。
5. 记录 InputCapture、OutputCapture、ResidualInputEnergy、ResidualOutputEnergy 和 principal angles。
6. 若 M_WA 明显提升 OutputCapture，再考虑把 Preserve 投影从 SVD 替换为 joint-aware projection。

## 结论变化

- 上版 PPT 中 ActCov rank PC-LoRA 只有 seed0/1，0.6174 ± 0.0020；补 seed2 后为 0.6132 ± 0.0074。
- Commonsense full setting 中 LoRA/LoRA+NSC 强于 OPLoRA/PC-LoRA，PC-LoRA 不稳定。
- MetaMathQA 训练期 PPL 中 OPLoRA 暂时第一；小样本 generation 预评估不纳入本版结论。
