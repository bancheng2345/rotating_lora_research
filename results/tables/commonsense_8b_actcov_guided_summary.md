| group | label | k | lambda_cap | adapters | eval_loss | ppl | acc | exact | align_score | omega_res |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| matched_8_adapter | LoRA matched | 16 | 0.0 | 8 | 0.2815 | 1.33 | 0.8643 | 0.6046 |  | 0.980398 |
| matched_8_adapter | SVD-right OPLoRA matched | 16 | 0.0 | 8 | 0.2799 | 1.32 | 0.8666 | 0.6124 |  | 0.981807 |
| matched_8_adapter | ActCov-guided LoRA lambda=0.05 | 16 | 0.05 | 8 | 0.2814 | 1.32 | 0.8630 | 0.6002 | 0.047142 | 0.997026 |
| matched_8_adapter | ActCov-guided LoRA lambda=0.20 | 16 | 0.2 | 8 | 0.2812 | 1.32 | 0.8643 | 0.6044 | 0.055235 | 0.997227 |
| matched_8_adapter | ActCov-OPLoRA k=4 | 4 | 0.0 | 8 | 0.2963 | 1.34 | 0.8545 | 0.5764 |  | 0.995560 |
| matched_8_adapter | ActCov-OPLoRA k=8 | 8 | 0.0 | 8 | 0.3054 | 1.36 | 0.8453 | 0.5500 |  | 0.995878 |
| matched_8_adapter | ActCov-OPLoRA k=16 | 16 | 0.0 | 8 | 0.3109 | 1.36 | 0.8497 | 0.5626 |  | 0.995877 |
| full_64_adapter_reference | LoRA full-layer reference | 16 | 0.0 | 64 | 0.2125 | 1.24 | 0.9061 | 0.7246 |  | 0.990266 |
| full_64_adapter_reference | OPLoRA full-layer reference | 16 | 0.0 | 64 | 0.2259 | 1.25 | 0.8989 | 0.7038 |  | 0.989633 |
