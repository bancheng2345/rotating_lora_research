| group | label | params | loss | ppl | acc | exact | time_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| matched_baseline | LoRA matched rank=8 all q/v probe layers | 425984 | 0.2815 | 1.33 | 0.8643 | 0.6046 | 2901.1 |
| matched_baseline | SVD-right OPLoRA matched rank=8 | 425984 | 0.2799 | 1.32 | 0.8666 | 0.6124 | 2764.4 |
| rank_selection | ActCov rank-selected LoRA q=4 v=14 | 417792 | 0.2759 | 1.32 | 0.8677 | 0.6148 | 1922.1 |
| rank_selection | ActCov rank-selected SVD-right q=4 v=14 | 417792 | 0.2724 | 1.31 | 0.8672 | 0.6136 | 1968.0 |
| negative_control | ActCov projection OPLoRA k=4 | 425984 | 0.2963 | 1.34 | 0.8545 | 0.5764 | 2836.3 |
| negative_control | ActCov-guided LoRA lambda=0.20 | 425984 | 0.2812 | 1.32 | 0.8643 | 0.6044 | 2133.8 |
| full_reference | LoRA full 64 adapters | 3407872 | 0.2125 | 1.24 | 0.9061 | 0.7246 | 6203.7 |
| full_reference | OPLoRA full 64 adapters | 3407872 | 0.2259 | 1.25 | 0.8989 | 0.7038 | 5251.8 |
