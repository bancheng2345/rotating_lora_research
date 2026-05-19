| method | lambda_align | teacher_best_em | generation_em | generation_correct | omega_residual | capture_alignment_score | peak_gpu_gb |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lora | 0.0000 | 0.7036 | 0.1968 | 984 | 0.9858 |  | 56.92 |
| svd_preserve_only | 0.0000 | 0.7122 | 0.2022 | 1011 | 0.9871 |  | 57.17 |
| pc_lora | 0.0000 | 0.7094 | 0.2032 | 1016 | 0.9771 |  | 23.32 |
| residual_mwa_pc_lora_lambda_0.01 | 0.0100 | 0.7118 | 0.2066 | 1033 | 0.9764 | 0.0363 | 23.32 |
| residual_mwa_pc_lora_lambda_0.05 | 0.0500 | 0.7088 | 0.2088 | 1044 | 0.9758 | 0.0556 | 23.32 |
| residual_mwa_pc_lora_lambda_0.1 | 0.1000 | 0.7092 | 0.2138 | 1069 | 0.9736 | 0.0982 | 23.32 |
