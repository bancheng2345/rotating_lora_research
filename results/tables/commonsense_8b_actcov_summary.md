| group | label | status | k | adapters | last_train_step | last_eval_step | eval_n | eval_loss | ppl | acc | exact | omega_res |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| matched_8_adapter | LoRA matched | complete | 16 | 8 | 300 | 300 | 5000 | 0.2815 | 1.33 | 0.8643 | 0.6046 | 0.980398 |
| matched_8_adapter | SVD-right OPLoRA matched | complete | 16 | 8 | 300 | 300 | 5000 | 0.2799 | 1.32 | 0.8666 | 0.6124 | 0.981807 |
| matched_8_adapter | ActCov-OPLoRA k=4 | complete | 4 | 8 | 300 | 300 | 5000 | 0.2963 | 1.34 | 0.8545 | 0.5764 | 0.995560 |
| matched_8_adapter | ActCov-OPLoRA k=8 | complete | 8 | 8 | 300 | 300 | 5000 | 0.3054 | 1.36 | 0.8453 | 0.5500 | 0.995878 |
| matched_8_adapter | ActCov-OPLoRA k=16 | complete | 16 | 8 | 300 | 300 | 5000 | 0.3109 | 1.36 | 0.8497 | 0.5626 | 0.995877 |
| matched_8_adapter_partial | ActCov-OPLoRA k=32 partial | partial | 32 | 8 | 200 | 150 | 5000 | 0.3395 | 1.40 | 0.8351 | 0.5242 |  |
| matched_8_adapter_partial | ActCov-PC-LoRA k=4 partial | partial | 4 | 8 | 240 | 200 | 5000 | 0.3130 | 1.37 | 0.8428 | 0.5456 |  |
| full_64_adapter_reference | LoRA full-layer reference | complete | 16 | 64 | 300 | 300 | 5000 | 0.2125 | 1.24 | 0.9061 | 0.7246 | 0.990266 |
| full_64_adapter_reference | OPLoRA full-layer reference | complete | 16 | 64 | 300 | 300 | 5000 | 0.2259 | 1.25 | 0.8989 | 0.7038 | 0.989633 |
