"""PC-LoRA layer and model injection entrypoint."""

from __future__ import annotations

from typing import Dict, List, Sequence

from torch import nn

from src.methods.lora_base import BaseLoRALinear, inject_lora_layers


class PCLoRALinear(BaseLoRALinear):
    """PC-LoRA shares OPLoRA forward structure; capture enters through the loss."""

    def __init__(self, base_layer, **kwargs):
        super().__init__(base_layer, method_name="pc_lora", **kwargs)


def apply_pc_lora(
    model: nn.Module,
    *,
    lora_config: Dict[str, object],
    target_modules: Sequence[str],
    svd_cache_dir: str,
    random_seed: int,
) -> List[Dict[str, object]]:
    """Injects PC-LoRA layers into the target modules."""

    return inject_lora_layers(
        model,
        adapter_cls=PCLoRALinear,
        lora_config=lora_config,
        target_modules=target_modules,
        svd_cache_dir=svd_cache_dir,
        random_seed=random_seed,
    )
