# PC-LoRA / Preserve-and-Capture LoRA 进展汇报

日期：2026-05-07

## 当前结论

- SVD 更适合做 Preserve：保护预训练权重 W0 的主映射方向。
- ActivationCov / Fisher 直接 hard projection 会删除任务信号，效果不如 SVD。
- Activation-aware 信号更适合作为 Allocation/Capture 辅助信号。
- 当前最有希望的方法：SVD-Preserve + Activation-aware Allocation + PC residual Capture。

## Matched 8-adapter 结果

| 方法 | seeds | Exact mean ± std | Loss mean | 参数 | 结论 |
|---|---:|---:|---:|---:|---|
| LoRA matched | 0,1,2 | 0.6099 ± 0.0107 | 0.2813 | 425,984 | baseline |
| SVD-right / OPLoRA | 0,1,2 | 0.6147 ± 0.0022 | 0.2802 | 425,984 | 最稳定 |
| ActCov rank LoRA | 0,1,2 | 0.6129 ± 0.0181 | 0.2779 | 417,792 | loss 低但波动大 |
| ActCov rank + SVD-right | 0,1,2 | 0.6122 ± 0.0163 | 0.2739 | 417,792 | loss 最低但 EM 不稳 |
| ActCov rank PC-LoRA | 0,1 | 0.6174 ± 0.0020 | 0.2744 | 417,792 | 当前最有希望；缺 seed2 |

## 关键负结果

- ActCov hard projection：exact 约 0.55~0.58，明显差于 SVD-right。
- Fisher-right OPLoRA：seed0 exact 0.5524，说明 Fisher top 方向不适合投影删除。
- CUR-right OPLoRA：最后有效 eval 约 0.5994，低于 LoRA/SVD-right。
- PC-LoRA matched 固定 rank：seed0 exact 0.6034，说明 capture 需要和 allocation 结合。

## 下一步

1. 补 `actcov_rank_pc_lora` seed2。
2. 生成正式结果表和机制图。
3. 扩展层/模块/rank allocation。
4. 补 math/code/forgetting eval。
