"""Lightweight trainer for LoRA / OPLoRA / PC-LoRA experiments."""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

DEFAULT_RUNTIME_CACHE_ROOT = Path.cwd() / "outputs" / "cache"
os.environ.setdefault("HF_HOME", str(DEFAULT_RUNTIME_CACHE_ROOT / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(DEFAULT_RUNTIME_CACHE_ROOT / "hf_datasets"))

from datasets import Dataset, load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    get_linear_schedule_with_warmup,
)

from src.evaluation.eval_lm import move_batch_to_device
from src.evaluation.eval_math import exact_match as math_final_answer_exact_match
from src.methods.lora_base import LoRALinear, inject_lora_layers, load_lora_state_dict, lora_state_dict
from src.methods.oplora import apply_oplora
from src.methods.pc_lora import apply_pc_lora
from src.training.hooks import (
    MechanismCollectionResult,
    aggregate_regularization_losses,
    adapter_layers,
    clear_adapter_caches,
    collect_mechanism_metrics,
    count_trainable_parameters,
    resolve_metric_layers,
)
from src.utils.logging import ensure_dir, save_json, save_yaml
from src.utils.metric_logger import MetricLoggers, create_metric_loggers, timestamp_now
from src.utils.metrics import RunningScalarStats, gpu_memory_stats, perplexity_from_loss
from src.utils.run_summary import save_final_summary


class SyntheticCausalLMDataset(TorchDataset):
    """Synthetic fixed-length token sequences for smoke tests."""

    def __init__(
        self,
        *,
        num_samples: int,
        seq_length: int,
        vocab_size: int,
        seed: int,
    ):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        self.input_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(num_samples, seq_length),
            generator=generator,
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        input_ids = self.input_ids[index]
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids.clone(),
        }


@dataclass
class TrainingArtifacts:
    """Filesystem paths and loggers for a single run."""

    run_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    metric_loggers: MetricLoggers


def set_seed(seed: int) -> None:
    """Sets Python and Torch RNG state."""

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_runtime_cache_dirs(config: Dict[str, Any]) -> None:
    """Redirects HF caches to writable workspace paths when configured."""

    cache_cfg = config.get("runtime", {}).get("cache", {})
    mapping = {
        "HF_HOME": cache_cfg.get("hf_home"),
        "HF_DATASETS_CACHE": cache_cfg.get("hf_datasets_cache"),
        "TRANSFORMERS_CACHE": cache_cfg.get("transformers_cache"),
    }
    for env_name, path in mapping.items():
        if not path:
            continue
        resolved = Path(str(path))
        if not resolved.is_absolute():
            resolved = Path.cwd() / resolved
        resolved.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(env_name, str(resolved))


def ensure_finite_scalar(
    name: str,
    value: torch.Tensor,
    *,
    step: int,
    micro_step: int,
    method: str,
) -> None:
    """Raises a clear error when a scalar tensor becomes NaN/Inf during training."""

    if bool(torch.isfinite(value.detach()).item()):
        return
    scalar = float(value.detach().to(dtype=torch.float32).cpu().item())
    raise RuntimeError(
        f"Non-finite {name} detected for method={method} at "
        f"micro_step={micro_step}, optimizer_step={step}: {scalar}"
    )


def slugify(value: str) -> str:
    """Converts a name/path into a filesystem-friendly slug."""

    chars = []
    for char in value:
        if char.isalnum():
            chars.append(char.lower())
        else:
            chars.append("_")
    slug = "".join(chars)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "unknown"


def resolve_model_slug(model_name: str) -> str:
    """Builds a stable, human-readable model slug for run directories."""

    raw_name = str(model_name).strip()
    if not raw_name:
        return "unknown_model"

    normalized_name = raw_name
    if raw_name.startswith(("/", ".")) or raw_name.count("/") > 1 or "\\" in raw_name:
        path = Path(raw_name)
        parts = path.parts
        if "snapshots" in parts:
            snapshot_index = parts.index("snapshots")
            if snapshot_index > 0:
                normalized_name = parts[snapshot_index - 1]
        else:
            normalized_name = path.name or raw_name

    normalized_name = normalized_name.replace("models--", "")
    normalized_name = normalized_name.replace("--", "/")
    return slugify(normalized_name)


def resolve_model_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    """Maps config dtype strings to torch dtypes, with CPU-safe fallback."""

    mapping = {
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {dtype_name}")
    dtype = mapping[dtype_name]
    if device.type == "cpu" and dtype != torch.float32:
        return torch.float32
    return dtype


def resolve_run_dir(config: Dict[str, Any]) -> Path:
    """Formats the output_dir template into a concrete run directory."""

    method = str(config["lora"]["method"])
    model_cfg = config["model"]
    dataset_cfg = config["training"]
    experiment_cfg = config["experiment"]
    model_name = str(
        model_cfg.get("name_or_path")
        or model_cfg.get("synthetic_name")
        or model_cfg.get("architecture", "model")
    )
    model_slug = resolve_model_slug(model_name)
    dataset_name = str(
        dataset_cfg.get("train_file")
        or dataset_cfg.get("dataset_name")
        or "dataset"
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    template = str(
        experiment_cfg.get(
            "output_dir",
            "outputs/runs/{model_dir}/{method}_{model}_{dataset}_r{rank}_k{k}_seed{seed}_{timestamp}",
        )
    )
    output_dir = template.format(
        method=slugify(method),
        model=model_slug,
        model_dir=model_slug,
        dataset=slugify(Path(dataset_name).stem),
        rank=int(config["lora"]["rank"]),
        k=int(config["lora"].get("projection_rank_k", 0)),
        seed=int(experiment_cfg.get("seed", 0)),
        timestamp=timestamp,
    )
    return Path(output_dir)


def create_artifacts(run_dir: Path) -> TrainingArtifacts:
    """Creates output directories and loggers for a run."""

    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    logs_dir = ensure_dir(run_dir / "logs")
    ensure_dir(run_dir)
    return TrainingArtifacts(
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
        metric_loggers=create_metric_loggers(run_dir),
    )


def _dataset_num_samples(dataloader) -> int | None:
    """Returns dataset length when available."""

    dataset = getattr(dataloader, "dataset", None)
    if dataset is None:
        return None
    try:
        return int(len(dataset))
    except TypeError:
        return None


def _batch_num_samples(batch: Dict[str, torch.Tensor]) -> int:
    """Returns the leading batch size for a moved batch."""

    input_ids = batch.get("input_ids")
    if isinstance(input_ids, torch.Tensor) and input_ids.ndim > 0:
        return int(input_ids.shape[0])
    first_tensor = next(value for value in batch.values() if isinstance(value, torch.Tensor))
    return int(first_tensor.shape[0]) if first_tensor.ndim > 0 else 1


def _batch_num_valid_tokens(batch: Dict[str, torch.Tensor]) -> int:
    """Counts labels contributing to the task loss in a batch."""

    labels = batch.get("labels")
    if isinstance(labels, torch.Tensor):
        return int(labels.ne(-100).sum().item())
    return 0


def _decode_label_tokens(tokenizer, token_ids: torch.Tensor) -> str:
    """Decodes non-negative label token ids for sample-level final-answer EM."""

    ids = [int(token_id) for token_id in token_ids.detach().cpu().tolist() if int(token_id) >= 0]
    if not ids:
        return ""
    return tokenizer.decode(ids, skip_special_tokens=True)


def _metric_from_score(score: float | None, base_score: float | None) -> tuple[float | None, float | None]:
    """Computes forgetting gap and retention score when a base score is available."""

    if score is None or base_score is None or base_score == 0:
        return None, None
    forgetting_gap = float(base_score - score)
    retention_score = float(score / base_score)
    return forgetting_gap, retention_score


@dataclass
class EvalSpec:
    """Configuration for one evaluation stream."""

    split: str
    dataset: str
    domain: str | None
    eval_type: str
    dataloader: DataLoader
    metric_name: str
    metric_mode: str
    task_type: str


def enable_gradient_checkpointing_for_adapters(model: torch.nn.Module) -> None:
    """Enables checkpointing and preserves input gradients for adapter tuning.

    Hugging Face decoder models often require input embeddings to produce
    gradient-carrying activations when gradient checkpointing is enabled.
    Without this, adapter parameters can receive no task gradient.
    """

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        return

    input_embeddings = None
    if hasattr(model, "get_input_embeddings"):
        input_embeddings = model.get_input_embeddings()
    if input_embeddings is None:
        return

    def make_outputs_require_grad(_, __, output):
        if isinstance(output, torch.Tensor):
            output.requires_grad_(True)
        return output

    hook_handle = input_embeddings.register_forward_hook(make_outputs_require_grad)
    setattr(model, "_pc_lora_input_require_grads_hook", hook_handle)


def build_scratch_llama(model_cfg: Dict[str, Any]) -> LlamaForCausalLM:
    """Builds a tiny Llama model from config for smoke/offline experiments."""

    config = LlamaConfig(
        vocab_size=int(model_cfg.get("vocab_size", 128)),
        hidden_size=int(model_cfg.get("hidden_size", 128)),
        intermediate_size=int(model_cfg.get("intermediate_size", 256)),
        num_hidden_layers=int(model_cfg.get("num_hidden_layers", 2)),
        num_attention_heads=int(model_cfg.get("num_attention_heads", 4)),
        num_key_value_heads=int(model_cfg.get("num_key_value_heads", 4)),
        max_position_embeddings=int(model_cfg.get("max_position_embeddings", 128)),
        pad_token_id=int(model_cfg.get("pad_token_id", 0)),
        bos_token_id=int(model_cfg.get("bos_token_id", 1)),
        eos_token_id=int(model_cfg.get("eos_token_id", 2)),
        attention_dropout=float(model_cfg.get("attention_dropout", 0.0)),
    )
    return LlamaForCausalLM(config)


def build_model_and_tokenizer(
    config: Dict[str, Any],
    *,
    device: torch.device,
) -> Tuple[torch.nn.Module, Optional[Any], str]:
    """Builds a causal LM and tokenizer from config."""

    model_cfg = config["model"]
    dtype = resolve_model_dtype(str(model_cfg.get("torch_dtype", "bf16")), device)
    if bool(model_cfg.get("init_from_scratch", False)):
        architecture = str(model_cfg.get("architecture", "llama"))
        if architecture != "llama":
            raise ValueError(f"Unsupported scratch architecture: {architecture}")
        model = build_scratch_llama(model_cfg)
        model.to(dtype=dtype)
        return model, None, str(model_cfg.get("synthetic_name", "scratch_llama"))

    name_or_path = model_cfg.get("name_or_path")
    if not name_or_path:
        raise ValueError("model.name_or_path is required when init_from_scratch is false.")
    local_files_only = bool(model_cfg.get("local_files_only", False))
    attn_impl = model_cfg.get("attn_impl")
    model_kwargs: Dict[str, Any] = {"local_files_only": local_files_only}
    if device.type == "cuda":
        model_kwargs["dtype"] = dtype
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl
    model = AutoModelForCausalLM.from_pretrained(name_or_path, **model_kwargs)
    if device.type == "cpu":
        model.to(dtype=dtype)

    tokenizer_name = model_cfg.get("tokenizer_name_or_path", name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        local_files_only=local_files_only,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return model, tokenizer, str(name_or_path)


def _fixed_length_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stacks fixed-length tensor examples into a batch."""

    return {
        key: torch.stack([example[key] for example in batch], dim=0)
        for key in batch[0]
    }


def build_synthetic_datasets(
    config: Dict[str, Any],
    *,
    vocab_size: int,
) -> Tuple[TorchDataset, TorchDataset]:
    """Builds deterministic synthetic train/eval datasets."""

    training_cfg = config["training"]
    experiment_cfg = config["experiment"]
    seq_length = int(training_cfg.get("seq_length", 64))
    train_samples = int(training_cfg.get("synthetic_train_samples", 128))
    eval_samples = int(training_cfg.get("synthetic_eval_samples", 32))
    seed = int(experiment_cfg.get("seed", 0))
    return (
        SyntheticCausalLMDataset(
            num_samples=train_samples,
            seq_length=seq_length,
            vocab_size=vocab_size,
            seed=seed,
        ),
        SyntheticCausalLMDataset(
            num_samples=eval_samples,
            seq_length=seq_length,
            vocab_size=vocab_size,
            seed=seed + 1,
        ),
    )


def _tokenize_text_dataset(
    dataset: Dataset,
    tokenizer,
    *,
    text_column: str,
    seq_length: int,
) -> Dataset:
    """Tokenizes text examples to fixed length for causal LM training."""

    def preprocess(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        tokenized = tokenizer(
            batch[text_column],
            truncation=True,
            padding="max_length",
            max_length=seq_length,
        )
        tokenized["labels"] = [list(ids) for ids in tokenized["input_ids"]]
        return tokenized

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing text dataset",
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


def _tokenize_supervised_dataset(
    dataset: Dataset,
    tokenizer,
    *,
    prompt_column: str,
    response_column: str,
    seq_length: int,
    prompt_template: str,
    response_template: str,
    train_on_prompt: bool,
    min_response_tokens: int,
) -> Dataset:
    """Tokenizes prompt/response supervision pairs for causal LM fine-tuning."""

    def preprocess(batch: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        input_ids_batch: List[List[int]] = []
        attention_masks_batch: List[List[int]] = []
        labels_batch: List[List[int]] = []

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer pad_token_id is required for supervised tokenization.")

        for prompt_value, response_value in zip(batch[prompt_column], batch[response_column]):
            prompt_text = prompt_template.format(
                prompt=prompt_value,
                question=prompt_value,
                answer=response_value,
                response=response_value,
            )
            response_text = response_template.format(
                prompt=prompt_value,
                question=prompt_value,
                answer=response_value,
                response=response_value,
            )
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
            if tokenizer.eos_token_id is not None:
                response_ids = list(response_ids) + [tokenizer.eos_token_id]

            min_response_budget = max(int(min_response_tokens), 1)
            response_budget = min(len(response_ids), min_response_budget)
            max_prompt_tokens = max(seq_length - response_budget, 0)
            prompt_ids = prompt_ids[:max_prompt_tokens]
            remaining_for_response = max(seq_length - len(prompt_ids), 0)
            response_ids = response_ids[:remaining_for_response]
            full_ids = list(prompt_ids) + list(response_ids)

            prompt_length = len(prompt_ids)
            labels = list(full_ids)
            if not train_on_prompt:
                labels[:prompt_length] = [-100] * prompt_length

            attention_mask = [1] * len(full_ids)
            pad_length = seq_length - len(full_ids)
            if pad_length > 0:
                full_ids = full_ids + [pad_token_id] * pad_length
                attention_mask = attention_mask + [0] * pad_length
                labels = labels + [-100] * pad_length

            input_ids_batch.append(full_ids)
            attention_masks_batch.append(attention_mask)
            labels_batch.append(labels)

        return {
            "input_ids": input_ids_batch,
            "attention_mask": attention_masks_batch,
            "labels": labels_batch,
        }

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing supervised prompt/response dataset",
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


def _finalize_input_id_dataset(dataset: Dataset, input_ids_column: str) -> Dataset:
    """Normalizes datasets that already contain integer token ids."""

    def preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
        input_ids = example[input_ids_column]
        attention_mask = example.get("attention_mask")
        if attention_mask is None:
            attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": list(input_ids),
        }

    processed = dataset.map(
        preprocess,
        remove_columns=dataset.column_names,
        desc="Preparing token-id dataset",
    )
    processed.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return processed


def build_hf_datasets(
    config: Dict[str, Any],
    *,
    tokenizer,
) -> Tuple[Dataset, Dataset]:
    """Builds datasets from json/jsonl files or datasets.load_dataset."""

    training_cfg = config["training"]
    seq_length = int(training_cfg.get("seq_length", 256))
    data_files: Dict[str, str] = {}
    if training_cfg.get("train_file"):
        data_files["train"] = str(training_cfg["train_file"])
    if training_cfg.get("eval_file"):
        data_files["validation"] = str(training_cfg["eval_file"])

    if data_files:
        dataset_dict = load_dataset("json", data_files=data_files)
        train_dataset = dataset_dict["train"] if "train" in dataset_dict else dataset_dict["validation"]
        eval_dataset = dataset_dict["validation"] if "validation" in dataset_dict else dataset_dict["train"]
    else:
        dataset_name = training_cfg.get("dataset_name")
        if not dataset_name:
            raise ValueError("training.dataset_name or training.train_file must be provided.")
        dataset_kwargs: Dict[str, Any] = {}
        if training_cfg.get("dataset_config_name"):
            dataset_kwargs["name"] = training_cfg["dataset_config_name"]
        train_dataset = load_dataset(dataset_name, split=str(training_cfg.get("train_split", "train")), **dataset_kwargs)
        eval_dataset = load_dataset(dataset_name, split=str(training_cfg.get("eval_split", "validation")), **dataset_kwargs)

    max_train_samples = training_cfg.get("max_train_samples")
    max_eval_samples = training_cfg.get("max_eval_samples")
    if max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(len(train_dataset), int(max_train_samples))))
    if max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(len(eval_dataset), int(max_eval_samples))))

    input_ids_column = str(training_cfg.get("input_ids_column", "input_ids"))
    text_column = str(training_cfg.get("text_column", "text"))
    prompt_column = str(training_cfg.get("prompt_column", "question"))
    response_column = str(training_cfg.get("response_column", "answer"))
    prompt_template = str(
        training_cfg.get(
            "prompt_template",
            "Question:\n{question}\n\nAnswer:\n",
        )
    )
    response_template = str(
        training_cfg.get(
            "response_template",
            "{answer}",
        )
    )
    train_on_prompt = bool(training_cfg.get("train_on_prompt", False))
    min_response_tokens = int(training_cfg.get("min_response_tokens", 16))
    if input_ids_column in train_dataset.column_names:
        return (
            _finalize_input_id_dataset(train_dataset, input_ids_column),
            _finalize_input_id_dataset(eval_dataset, input_ids_column),
        )
    if (
        prompt_column in train_dataset.column_names
        and response_column in train_dataset.column_names
    ):
        if tokenizer is None:
            raise ValueError("Tokenizer is required for prompt/response datasets.")
        return (
            _tokenize_supervised_dataset(
                train_dataset,
                tokenizer,
                prompt_column=prompt_column,
                response_column=response_column,
                seq_length=seq_length,
                prompt_template=prompt_template,
                response_template=response_template,
                train_on_prompt=train_on_prompt,
                min_response_tokens=min_response_tokens,
            ),
            _tokenize_supervised_dataset(
                eval_dataset,
                tokenizer,
                prompt_column=prompt_column,
                response_column=response_column,
                seq_length=seq_length,
                prompt_template=prompt_template,
                response_template=response_template,
                train_on_prompt=train_on_prompt,
                min_response_tokens=min_response_tokens,
            ),
        )
    if tokenizer is None:
        raise ValueError("Tokenizer is required for text datasets.")
    return (
        _tokenize_text_dataset(train_dataset, tokenizer, text_column=text_column, seq_length=seq_length),
        _tokenize_text_dataset(eval_dataset, tokenizer, text_column=text_column, seq_length=seq_length),
    )


def build_dataloaders(
    config: Dict[str, Any],
    *,
    tokenizer,
    vocab_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """Builds training and evaluation dataloaders."""

    training_cfg = config["training"]
    train_file = training_cfg.get("train_file")
    dataset_name = str(training_cfg.get("dataset_name", "synthetic_lm"))
    if not train_file and dataset_name.startswith("synthetic"):
        train_dataset, eval_dataset = build_synthetic_datasets(config, vocab_size=vocab_size)
        collate_fn = _fixed_length_collate
    else:
        train_dataset, eval_dataset = build_hf_datasets(config, tokenizer=tokenizer)
        collate_fn = _fixed_length_collate

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_cfg.get("batch_size", 4)),
        shuffle=True,
        collate_fn=collate_fn,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(training_cfg.get("eval_batch_size", training_cfg.get("batch_size", 4))),
        shuffle=False,
        collate_fn=collate_fn,
    )
    return train_loader, eval_loader


def build_eval_dataloader_from_spec(
    config: Dict[str, Any],
    *,
    tokenizer,
    spec: Dict[str, Any],
) -> DataLoader:
    """Builds an evaluation-only dataloader for forgetting or cross-domain checks."""

    training_cfg = dict(config["training"])
    merged_training = dict(training_cfg)
    merged_training.update(
        {
            "dataset_name": spec.get("dataset_name", training_cfg.get("dataset_name")),
            "dataset_config_name": spec.get("dataset_config_name", training_cfg.get("dataset_config_name")),
            "eval_file": spec.get("eval_file", training_cfg.get("eval_file")),
            "eval_split": spec.get("eval_split", training_cfg.get("eval_split", "validation")),
            "text_column": spec.get("text_column", training_cfg.get("text_column", "text")),
            "input_ids_column": spec.get("input_ids_column", training_cfg.get("input_ids_column", "input_ids")),
            "prompt_column": spec.get("prompt_column", training_cfg.get("prompt_column", "question")),
            "response_column": spec.get("response_column", training_cfg.get("response_column", "answer")),
            "prompt_template": spec.get("prompt_template", training_cfg.get("prompt_template")),
            "response_template": spec.get("response_template", training_cfg.get("response_template")),
            "train_on_prompt": spec.get("train_on_prompt", training_cfg.get("train_on_prompt", False)),
            "min_response_tokens": spec.get("min_response_tokens", training_cfg.get("min_response_tokens", 16)),
            "seq_length": spec.get("seq_length", training_cfg.get("seq_length", 256)),
            "max_eval_samples": spec.get("max_eval_samples", training_cfg.get("max_eval_samples")),
        }
    )
    if not merged_training.get("eval_file") and not merged_training.get("dataset_name"):
        raise ValueError("Forgetting eval spec requires eval_file or dataset_name.")

    eval_only_config = dict(config)
    eval_only_config["training"] = merged_training
    _, eval_dataset = build_hf_datasets(eval_only_config, tokenizer=tokenizer)
    return DataLoader(
        eval_dataset,
        batch_size=int(spec.get("eval_batch_size", merged_training.get("eval_batch_size", merged_training.get("batch_size", 4)))),
        shuffle=False,
        collate_fn=_fixed_length_collate,
    )


def apply_method_to_model(
    model: torch.nn.Module,
    config: Dict[str, Any],
    *,
    run_dir: Path,
) -> List[Dict[str, object]]:
    """Injects the requested LoRA-family method into the model."""

    lora_cfg = config["lora"]
    target_modules = config["model"]["target_modules"]
    svd_cache_dir = str(lora_cfg.get("svd_cache_dir", Path("outputs") / "cache" / "svd"))
    method = str(lora_cfg["method"])
    random_seed = int(config["experiment"].get("seed", 0))

    if method in {"lora", "lora_nsc", "actcov_guided_lora", "actcov_rank_lora", "task_actcov_capture_lora"}:
        return inject_lora_layers(
            model,
            adapter_cls=LoRALinear,
            lora_config=lora_cfg,
            target_modules=target_modules,
            svd_cache_dir=svd_cache_dir,
            random_seed=random_seed,
        )
    if method in {"oplora", "actcov_oplora", "actcov_rank_svd_right", "fisher_oplora", "cur_oplora", "svd_preserve_only"}:
        return apply_oplora(
            model,
            lora_config=lora_cfg,
            target_modules=target_modules,
            svd_cache_dir=svd_cache_dir,
            random_seed=random_seed,
        )
    if method in {
        "pc_lora",
        "pc_lora_weighted",
        "actcov_pc_lora",
        "actcov_rank_pc_lora",
        "mwa_pc_lora",
        "svd_preserve_task_capture",
        "svd_preserve_mwa_capture",
        "residual_mwa_pc_lora",
    }:
        return apply_pc_lora(
            model,
            lora_config=lora_cfg,
            target_modules=target_modules,
            svd_cache_dir=svd_cache_dir,
            random_seed=random_seed,
        )
    raise ValueError(f"Unsupported lora.method: {method}")


def checkpoint_payload(
    *,
    step: int,
    config: Dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
) -> Dict[str, Any]:
    """Builds a checkpoint payload with adapter-only weights."""

    return {
        "step": step,
        "config": config,
        "adapter_state_dict": lora_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }


def save_checkpoint(
    *,
    path: Path,
    step: int,
    config: Dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
) -> None:
    """Saves an adapter-only checkpoint."""

    payload = checkpoint_payload(
        step=step,
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_adapter_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
) -> Dict[str, Any]:
    """Loads adapter weights from a checkpoint file into a prepared model."""

    payload = torch.load(checkpoint_path, map_location="cpu")
    load_lora_state_dict(model, payload["adapter_state_dict"])
    return payload


class ExperimentTrainer:
    """End-to-end trainer with logging, eval, and adapter-only checkpoints."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        configure_runtime_cache_dirs(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = int(config["experiment"].get("seed", 0))
        self.metrics_config = dict(config.get("metrics", {}))
        self.forgetting_config = dict(config.get("forgetting_eval", {}))
        self.efficiency_config = dict(config.get("efficiency", {}))
        set_seed(self.seed)
        if self.device.type == "cuda" and bool(self.efficiency_config.get("track_gpu_memory", True)):
            torch.cuda.reset_peak_memory_stats()

        self.run_dir = resolve_run_dir(config)
        self.artifacts = create_artifacts(self.run_dir)
        self.config["experiment"]["resolved_run_dir"] = str(self.run_dir)

        self.model, self.tokenizer, self.model_name = build_model_and_tokenizer(
            config,
            device=self.device,
        )
        self.adapter_summary = apply_method_to_model(self.model, config, run_dir=self.run_dir)
        self.model.to(self.device)

        if bool(config["training"].get("gradient_checkpointing", False)):
            enable_gradient_checkpointing_for_adapters(self.model)

        self.train_loader, self.eval_loader = build_dataloaders(
            config,
            tokenizer=self.tokenizer,
            vocab_size=int(config["model"].get("vocab_size", 128)),
        )
        self.train_num_samples = _dataset_num_samples(self.train_loader)
        self.eval_num_samples = _dataset_num_samples(self.eval_loader)
        self.parameter_counts = count_trainable_parameters(self.model)
        self.optimizer = AdamW(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            lr=float(config["training"].get("learning_rate", 2e-4)),
            weight_decay=float(config["training"].get("weight_decay", 0.0)),
        )
        self.trainable_parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]

        self.grad_accum_steps = int(config["training"].get("grad_accum_steps", 1))
        self.max_steps = int(config["training"].get("max_steps", 10))
        self.max_grad_norm = config["training"].get("max_grad_norm")
        warmup_ratio = float(config["training"].get("warmup_ratio", 0.0))
        warmup_steps = int(self.max_steps * warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.max_steps,
        )
        self.lambda_cap = float(config["lora"].get("lambda_cap", 0.0))
        self.lambda_align = float(config["lora"].get("lambda_align", 0.0))
        self.lambda_orth = float(config["lora"].get("lambda_orth", 0.0))
        self.logging_steps = int(config["training"].get("logging_steps", 10))
        self.eval_steps = int(config["training"].get("eval_steps", self.max_steps))
        self.save_steps = int(config["training"].get("save_steps", self.max_steps))
        mechanism_steps = self.metrics_config.get("mechanism_steps")
        self.mechanism_steps = self.eval_steps if mechanism_steps in (None, 0) else int(mechanism_steps)
        self.best_eval_metric: Optional[float] = None
        self.train_domain = self.forgetting_config.get("train_domain") or self.config["training"].get("domain")
        self.base_model_scores = self._load_base_model_scores()
        self.eval_specs = self._build_eval_specs()
        self.initial_svd_cache_hits = sum(
            1 for layer in self.adapter_summary if layer.get("svd_cache_hit")
        )
        self.initial_svd_time_sec = float(
            sum(float(layer.get("svd_time_sec", 0.0) or 0.0) for layer in self.adapter_summary)
        )

        save_yaml(self.run_dir / "config_resolved.yaml", self.config)
        save_json(
            self.run_dir / "adapter_summary.json",
            {
                "adapter_layers": self.adapter_summary,
                "parameter_counts": self.parameter_counts,
            },
        )

    def _load_base_model_scores(self) -> Dict[str, float]:
        """Loads optional base-model forgetting reference scores."""

        path = self.forgetting_config.get("base_model_scores_path")
        if not path:
            return {}
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("forgetting_eval.base_model_scores_path must point to a JSON object.")
        return {str(key): float(value) for key, value in payload.items() if value is not None}

    def _build_eval_specs(self) -> List[EvalSpec]:
        """Builds the in-domain eval stream plus optional forgetting streams."""

        training_cfg = self.config["training"]
        primary_dataset = str(training_cfg.get("eval_file") or training_cfg.get("dataset_name") or "eval_dataset")
        primary_domain = training_cfg.get("eval_domain") or training_cfg.get("domain")
        primary_metric_name = str(training_cfg.get("eval_metric_name", "perplexity"))
        primary_metric_mode = str(training_cfg.get("eval_metric_mode", "min"))
        primary_task_type = str(training_cfg.get("task_type", "causal_lm"))
        specs = [
            EvalSpec(
                split=str(training_cfg.get("eval_split", "validation")),
                dataset=primary_dataset,
                domain=None if primary_domain is None else str(primary_domain),
                eval_type="in_domain",
                dataloader=self.eval_loader,
                metric_name=primary_metric_name,
                metric_mode=primary_metric_mode,
                task_type=primary_task_type,
            )
        ]
        if not bool(self.forgetting_config.get("enabled", True)):
            return specs

        for raw_spec in self.forgetting_config.get("eval_domains", []):
            if isinstance(raw_spec, str):
                spec_dict = {
                    "domain": raw_spec,
                    "dataset_name": raw_spec,
                    "metric_name": primary_metric_name,
                }
            elif isinstance(raw_spec, dict):
                spec_dict = dict(raw_spec)
            else:
                raise ValueError(f"Unsupported forgetting eval spec: {raw_spec!r}")
            dataloader = build_eval_dataloader_from_spec(
                self.config,
                tokenizer=self.tokenizer,
                spec=spec_dict,
            )
            specs.append(
                EvalSpec(
                    split=str(spec_dict.get("eval_split", "validation")),
                    dataset=str(spec_dict.get("eval_file") or spec_dict.get("dataset_name") or spec_dict.get("domain") or "forgetting_eval"),
                    domain=None if spec_dict.get("domain") is None else str(spec_dict.get("domain")),
                    eval_type="forgetting",
                    dataloader=dataloader,
                    metric_name=str(spec_dict.get("metric_name", primary_metric_name)),
                    metric_mode=str(spec_dict.get("metric_mode", primary_metric_mode)),
                    task_type=str(spec_dict.get("task_type", primary_task_type)),
                )
            )
        return specs

    def _estimate_epoch(self, step: int) -> float | None:
        """Estimates training epochs completed from optimizer steps."""

        if self.train_num_samples in (None, 0):
            return None
        batch_size = int(self.config["training"].get("batch_size", 1))
        samples_seen = step * batch_size * self.grad_accum_steps
        return float(samples_seen / max(self.train_num_samples, 1))

    def _iter_train_batches(self) -> Iterable[Dict[str, torch.Tensor]]:
        while True:
            for batch in self.train_loader:
                yield batch

    def _log_efficiency_record(self, payload: Dict[str, Any]) -> None:
        """Writes one efficiency record when enabled."""

        if not bool(self.metrics_config.get("log_efficiency", True)):
            return
        payload = {
            "timestamp": timestamp_now(),
            **payload,
        }
        self.artifacts.metric_loggers.efficiency.log(payload)

    def _base_model_score_for_spec(self, spec: EvalSpec) -> float | None:
        """Resolves an optional base-model reference score for a forgetting spec."""

        lookup_keys = [
            spec.dataset,
            spec.domain,
            f"{spec.domain}:{spec.dataset}" if spec.domain else None,
        ]
        for key in lookup_keys:
            if key and key in self.base_model_scores:
                return float(self.base_model_scores[key])
        return None

    def _collect_capture_accumulators(self) -> Dict[str, Dict[str, RunningScalarStats]]:
        """Initializes per-layer accumulators for omega/captured-energy metrics."""

        accumulators: Dict[str, Dict[str, RunningScalarStats]] = {}
        layers = adapter_layers(self.model)
        tracked_layers = resolve_metric_layers(
            layers,
            self.metrics_config.get("omega_capture_layers", "all"),
        )
        for layer in tracked_layers:
            accumulators[layer.layer_name] = {
                "omega_full": RunningScalarStats(),
                "omega_residual": RunningScalarStats(),
                "captured_energy_full": RunningScalarStats(),
                "captured_energy_residual": RunningScalarStats(),
                "total_energy": RunningScalarStats(),
                "cap_loss": RunningScalarStats(),
            }
        return accumulators

    def _update_capture_accumulators(
        self,
        accumulators: Dict[str, Dict[str, RunningScalarStats]],
    ) -> None:
        """Adds the current batch's cached adapter metrics into running accumulators."""

        if not accumulators:
            return
        for layer in adapter_layers(self.model):
            layer_stats = accumulators.get(layer.layer_name)
            if layer_stats is None:
                continue
            state = layer.capture_metric_state()
            layer_stats["omega_full"].update(state.omega_full)
            layer_stats["omega_residual"].update(state.omega_residual)
            layer_stats["captured_energy_full"].update(state.captured_energy_full)
            layer_stats["captured_energy_residual"].update(state.captured_energy_residual)
            total_energy = state.total_energy_full if state.total_energy_full is not None else state.total_energy_residual
            layer_stats["total_energy"].update(total_energy)
            if state.capture_loss is not None:
                layer_stats["cap_loss"].update(torch.tensor([float(state.capture_loss.detach().cpu().item())]))

    def _apply_capture_accumulators_to_records(
        self,
        records: List[Dict[str, object]],
        accumulators: Dict[str, Dict[str, RunningScalarStats]],
    ) -> None:
        """Overwrites last-batch omega fields with eval-set aggregated statistics."""

        for record in records:
            layer_name = str(record["layer_name"])
            layer_stats = accumulators.get(layer_name)
            if layer_stats is None:
                continue
            record.update(layer_stats["omega_full"].to_dict("omega_full"))
            record.update(layer_stats["omega_residual"].to_dict("omega_residual"))
            record["captured_energy_full_mean"] = layer_stats["captured_energy_full"].to_dict("captured_energy_full")["captured_energy_full_mean"]
            record["captured_energy_residual_mean"] = layer_stats["captured_energy_residual"].to_dict("captured_energy_residual")["captured_energy_residual_mean"]
            record["total_energy_mean"] = layer_stats["total_energy"].to_dict("total_energy")["total_energy_mean"]
            record["cap_loss"] = layer_stats["cap_loss"].to_dict("cap_loss")["cap_loss_mean"]

    def _evaluate_spec(
        self,
        spec: EvalSpec,
        *,
        step: int,
        collect_mechanism: bool,
        total_train_time_sec: float | None = None,
    ) -> Dict[str, Any]:
        """Evaluates one dataloader and logs eval/forgetting/mechanism metrics."""

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        total_correct_tokens = 0
        total_sequence_exact_matches = 0
        total_final_answer_exact_matches = 0
        total_final_answer_samples = 0
        total_samples = 0
        accumulators = self._collect_capture_accumulators() if collect_mechanism else {}
        is_math_eval = str(spec.domain or "").lower() == "math"

        for batch in spec.dataloader:
            batch = move_batch_to_device(batch, self.device)
            outputs = self.model(**batch)
            labels = batch["labels"]
            shift_logits = outputs.logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            valid_mask = shift_labels.ne(-100)
            valid_tokens = int(valid_mask.sum().item())
            if valid_tokens == 0:
                continue
            predictions = shift_logits.argmax(dim=-1)
            total_loss += float(outputs.loss.detach().cpu().item()) * valid_tokens
            total_tokens += valid_tokens
            total_correct_tokens += int(((predictions == shift_labels) & valid_mask).sum().item())
            sequence_exact_match_mask = ((predictions == shift_labels) | ~valid_mask).all(dim=-1) & valid_mask.any(dim=-1)
            total_sequence_exact_matches += int(sequence_exact_match_mask.sum().item())
            if is_math_eval and self.tokenizer is not None:
                for row_index in range(int(predictions.shape[0])):
                    row_mask = valid_mask[row_index]
                    if not bool(row_mask.any().item()):
                        continue
                    prediction_text = _decode_label_tokens(self.tokenizer, predictions[row_index][row_mask])
                    gold_text = _decode_label_tokens(self.tokenizer, shift_labels[row_index][row_mask])
                    total_final_answer_exact_matches += int(
                        math_final_answer_exact_match(prediction_text, gold_text)
                    )
                    total_final_answer_samples += 1
            total_samples += _batch_num_samples(batch)
            if collect_mechanism:
                self._update_capture_accumulators(accumulators)

        average_loss = total_loss / total_tokens if total_tokens > 0 else float("nan")
        token_accuracy = (total_correct_tokens / total_tokens) if total_tokens > 0 else None
        sequence_exact_match = (total_sequence_exact_matches / total_samples) if total_samples > 0 else None
        final_answer_exact_match = (
            total_final_answer_exact_matches / total_final_answer_samples
            if total_final_answer_samples > 0
            else None
        )
        exact_match = final_answer_exact_match if is_math_eval else sequence_exact_match
        accuracy = exact_match if is_math_eval else token_accuracy
        metric_name = spec.metric_name
        perplexity = perplexity_from_loss(average_loss)
        if metric_name == "perplexity":
            metric_value = perplexity
        elif metric_name == "accuracy":
            metric_value = exact_match if is_math_eval else accuracy
        elif metric_name in {"exact_match", "em"}:
            metric_value = exact_match
        else:
            metric_value = perplexity if spec.task_type == "causal_lm" else accuracy
        record = {
            "step": step,
            "epoch": self._estimate_epoch(step),
            "split": spec.split,
            "dataset": spec.dataset,
            "domain": spec.domain,
            "eval_dataset": spec.dataset,
            "eval_domain": spec.domain,
            "eval_type": spec.eval_type,
            "train_domain": self.train_domain,
            "metric_name": metric_name,
            "metric_mode": spec.metric_mode,
            "metric_value": metric_value,
            "eval_metric_name": metric_name,
            "eval_metric_value": metric_value,
            "loss": average_loss,
            "eval_loss": average_loss,
            "perplexity": perplexity,
            "eval_accuracy": accuracy,
            "eval_token_accuracy": token_accuracy,
            "eval_exact_match": exact_match,
            "eval_exact_match_count": total_final_answer_exact_matches if is_math_eval else total_sequence_exact_matches,
            "eval_exact_match_type": "final_answer" if is_math_eval else "teacher_forced_sequence",
            "eval_sequence_exact_match": sequence_exact_match,
            "eval_final_answer_exact_match": final_answer_exact_match,
            "eval_pass_at_1": None,
            "eval_metric": metric_value,
            "eval_tokens": total_tokens,
            "eval_num_samples": total_samples if total_samples > 0 else _dataset_num_samples(spec.dataloader),
            "num_samples": total_samples if total_samples > 0 else _dataset_num_samples(spec.dataloader),
            "timestamp": timestamp_now(),
        }
        if bool(self.metrics_config.get("log_eval", True)):
            self.artifacts.metric_loggers.eval.log(record)

        mechanism_result = None
        if collect_mechanism and bool(self.metrics_config.get("log_mechanism", True)):
            mechanism_result = collect_mechanism_metrics(
                self.model,
                step=step,
                method=str(self.config["lora"]["method"]),
                metrics_config=self.metrics_config,
                svd_cache_dir=str(self.config["lora"].get("svd_cache_dir", Path("outputs") / "cache" / "svd")),
            )
            self._apply_capture_accumulators_to_records(mechanism_result.records, accumulators)
            for mechanism_record in mechanism_result.records:
                mechanism_record["lambda_cap"] = self.lambda_cap
                mechanism_record["lambda_align"] = self.lambda_align
                mechanism_record["lambda_orth"] = self.lambda_orth
                self.artifacts.metric_loggers.mechanism.log(mechanism_record)

        if spec.eval_type == "forgetting" and bool(self.metrics_config.get("log_forgetting", True)):
            base_model_score = self._base_model_score_for_spec(spec)
            forgetting_gap, retention_score = _metric_from_score(metric_value, base_model_score)
            self.artifacts.metric_loggers.forgetting.log(
                {
                    "step": step,
                    "method": self.config["lora"]["method"],
                    "train_domain": self.train_domain,
                    "eval_domain": spec.domain,
                    "eval_dataset": spec.dataset,
                    "current_score": metric_value,
                    "base_model_score": base_model_score,
                    "forgetting_gap": forgetting_gap,
                    "retention_score": retention_score,
                    "timestamp": timestamp_now(),
                }
            )

        if mechanism_result is not None:
            self._log_efficiency_record(
                {
                    "step": step,
                    "phase": "eval",
                    "samples_per_sec": None,
                    "tokens_per_sec": None,
                    "step_time_sec": None,
                    "svd_cache_hit": mechanism_result.svd_cache_misses == 0 and mechanism_result.svd_cache_hits > 0,
                    "svd_cache_hit_count": mechanism_result.svd_cache_hits + self.initial_svd_cache_hits,
                    "svd_cache_miss_count": mechanism_result.svd_cache_misses,
                    "svd_time_sec": mechanism_result.svd_time_sec + self.initial_svd_time_sec,
                    "capture_metric_time_sec": mechanism_result.capture_metric_time_sec,
                    "rho_metric_time_sec": mechanism_result.rho_metric_time_sec,
                    "total_train_time_sec": total_train_time_sec,
                    **gpu_memory_stats(),
                }
            )

        clear_adapter_caches(self.model)

        return record

    def evaluate(
        self,
        *,
        step: int,
        total_train_time_sec: float | None = None,
        collect_mechanism: bool | None = None,
    ) -> Dict[str, Any]:
        """Runs evaluation streams and logs eval/forgetting/mechanism metrics."""

        primary_record: Optional[Dict[str, Any]] = None
        for spec in self.eval_specs:
            spec_collect_mechanism = (
                bool(collect_mechanism)
                if collect_mechanism is not None
                else (spec.eval_type == "in_domain")
            ) and spec.eval_type == "in_domain"
            record = self._evaluate_spec(
                spec,
                step=step,
                collect_mechanism=spec_collect_mechanism,
                total_train_time_sec=total_train_time_sec,
            )
            if spec.eval_type == "in_domain" and primary_record is None:
                primary_record = record

        self.model.train()
        if primary_record is None:
            raise RuntimeError("No in-domain evaluation record was produced.")
        metric_value = primary_record["metric_value"]
        metric_mode = primary_record.get("metric_mode", "min")
        if metric_value is None:
            return primary_record
        if self.best_eval_metric is None:
            self.best_eval_metric = float(metric_value)
            return primary_record
        is_better = metric_value < self.best_eval_metric if metric_mode == "min" else metric_value > self.best_eval_metric
        if is_better:
            self.best_eval_metric = float(primary_record["metric_value"])
        return primary_record

    def train(self) -> Dict[str, Any]:
        """Runs the configured optimization loop."""

        self.model.train()
        train_iterator = self._iter_train_batches()
        step = 0
        micro_step = 0
        self.optimizer.zero_grad(set_to_none=True)
        last_eval = None
        train_start_time = time.perf_counter()
        accumulated_samples = 0
        accumulated_tokens = 0
        optimizer_step_start = time.perf_counter()

        while step < self.max_steps:
            micro_step += 1
            microbatch = move_batch_to_device(next(train_iterator), self.device)
            accumulated_samples += _batch_num_samples(microbatch)
            accumulated_tokens += _batch_num_valid_tokens(microbatch)
            outputs = self.model(**microbatch)
            task_loss = outputs.loss
            capture_weighting = str(self.config["lora"].get("capture_weighting", "none"))
            loss_weight = None
            if capture_weighting == "loss":
                loss_weight = task_loss.detach().to(dtype=torch.float32)
                clip_max = self.config["lora"].get("capture_loss_weight_clip_max")
                if clip_max is not None:
                    loss_weight = torch.clamp(loss_weight, min=0.0, max=float(clip_max))
                normalize_by = self.config["lora"].get("capture_loss_weight_normalize_by")
                if normalize_by is None and clip_max is not None:
                    normalize_by = clip_max
                if normalize_by is not None:
                    normalize_by = float(normalize_by)
                    if normalize_by <= 0:
                        raise ValueError(
                            "lora.capture_loss_weight_normalize_by must be > 0 "
                            "when provided."
                        )
                    loss_weight = loss_weight / normalize_by
            cap_loss, orth_loss, align_loss = aggregate_regularization_losses(
                self.model,
                loss_weight=loss_weight,
            )
            total_loss = (
                task_loss
                + self.lambda_cap * cap_loss
                + self.lambda_align * align_loss
                + self.lambda_orth * orth_loss
            )
            method_name = str(self.config["lora"]["method"])
            ensure_finite_scalar(
                "task_loss",
                task_loss,
                step=step,
                micro_step=micro_step,
                method=method_name,
            )
            ensure_finite_scalar(
                "cap_loss",
                cap_loss,
                step=step,
                micro_step=micro_step,
                method=method_name,
            )
            ensure_finite_scalar(
                "orth_loss",
                orth_loss,
                step=step,
                micro_step=micro_step,
                method=method_name,
            )
            ensure_finite_scalar(
                "align_loss",
                align_loss,
                step=step,
                micro_step=micro_step,
                method=method_name,
            )
            ensure_finite_scalar(
                "total_loss",
                total_loss,
                step=step,
                micro_step=micro_step,
                method=method_name,
            )
            (total_loss / self.grad_accum_steps).backward()

            if micro_step % self.grad_accum_steps == 0:
                grad_norm_value = None
                if self.max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.trainable_parameters,
                        max_norm=float(self.max_grad_norm),
                    )
                    grad_norm_tensor = (
                        grad_norm
                        if isinstance(grad_norm, torch.Tensor)
                        else torch.tensor(float(grad_norm))
                    )
                    ensure_finite_scalar(
                        "grad_norm",
                        grad_norm_tensor,
                        step=step,
                        micro_step=micro_step,
                        method=method_name,
                    )
                    grad_norm_value = float(grad_norm_tensor.detach().cpu().item())
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                # Capture caches can retain large activation tensors until the
                # end of the loop. Clear them before eval to avoid combining a
                # large training microbatch with eval activations in memory.
                clear_adapter_caches(self.model)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                step += 1

                elapsed = time.perf_counter() - optimizer_step_start
                total_train_time_sec = time.perf_counter() - train_start_time
                if step % self.logging_steps == 0 or step == 1:
                    train_record = {
                        "step": step,
                        "epoch": self._estimate_epoch(step),
                        "task_loss": float(task_loss.detach().cpu().item()),
                        "total_loss": float(total_loss.detach().cpu().item()),
                        "cap_loss": float(cap_loss.detach().cpu().item()),
                        "align_loss": float(align_loss.detach().cpu().item()),
                        "orth_loss": float(orth_loss.detach().cpu().item()),
                        "learning_rate": float(self.scheduler.get_last_lr()[0]),
                        "grad_norm": grad_norm_value,
                        "step_time_sec": elapsed,
                        "samples_per_sec": accumulated_samples / elapsed if elapsed > 0 else None,
                        "tokens_per_sec": accumulated_tokens / elapsed if elapsed > 0 else None,
                        "timestamp": timestamp_now(),
                    }
                    if bool(self.metrics_config.get("log_train", True)):
                        self.artifacts.metric_loggers.train.log({**train_record, **gpu_memory_stats()})
                    self._log_efficiency_record(
                        {
                            "step": step,
                            "phase": "train",
                            "step_time_sec": elapsed,
                            "samples_per_sec": accumulated_samples / elapsed if elapsed > 0 else None,
                            "tokens_per_sec": accumulated_tokens / elapsed if elapsed > 0 else None,
                            "svd_cache_hit": self.initial_svd_cache_hits > 0,
                            "svd_cache_hit_count": self.initial_svd_cache_hits,
                            "svd_cache_miss_count": max(len(self.adapter_summary) - self.initial_svd_cache_hits, 0),
                            "svd_time_sec": self.initial_svd_time_sec,
                            "capture_metric_time_sec": None,
                            "rho_metric_time_sec": None,
                            "total_train_time_sec": total_train_time_sec,
                            **gpu_memory_stats(),
                        }
                    )

                if step % self.eval_steps == 0 or step == self.max_steps:
                    last_eval = self.evaluate(
                        step=step,
                        total_train_time_sec=total_train_time_sec,
                        collect_mechanism=(step % self.mechanism_steps == 0 or step == self.max_steps),
                    )

                if step % self.save_steps == 0 or step == self.max_steps:
                    save_checkpoint(
                        path=self.artifacts.checkpoints_dir / f"step_{step}.pt",
                        step=step,
                        config=self.config,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                    )

                accumulated_samples = 0
                accumulated_tokens = 0
                optimizer_step_start = time.perf_counter()

            clear_adapter_caches(self.model)

        summary = save_final_summary(self.run_dir)
        summary.update(
            {
                "model_name": self.model_name,
                "dataset_name": self.config["training"].get("train_file")
                or self.config["training"].get("dataset_name"),
                "seed": self.seed,
                "max_steps": self.max_steps,
                "last_eval": last_eval,
                "run_dir": str(self.run_dir),
                "parameter_counts": self.parameter_counts,
            }
        )
        save_json(self.run_dir / "final_summary.json", summary)
        return summary
