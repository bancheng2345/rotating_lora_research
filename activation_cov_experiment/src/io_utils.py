"""Config, IO, and model/data helpers for the activation covariance experiment."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel


ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Loads a YAML mapping."""

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config at {path}, got {type(data)!r}.")
    return data


def deep_merge_dicts(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges mapping values with update precedence."""

    merged = copy.deepcopy(base)
    for key, value in update.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _parse_override_value(raw_value: str) -> Any:
    lowered = raw_value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value


def apply_overrides(config: Dict[str, Any], overrides: Optional[Iterable[str]]) -> Dict[str, Any]:
    """Applies dotlist-style overrides."""

    updated = copy.deepcopy(config)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}', expected key=value.")
        key_path, raw_value = override.split("=", 1)
        value = _parse_override_value(raw_value)
        cursor = updated
        parts = key_path.split(".")
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = value
    return updated


def load_config_stack(
    config_paths: Sequence[str | Path],
    *,
    base_config_path: Optional[str | Path] = None,
    overrides: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Loads base config plus ordered overlay configs."""

    config: Dict[str, Any] = {}
    if base_config_path is not None:
        config = load_yaml(base_config_path)
    for config_path in config_paths:
        config = deep_merge_dicts(config, load_yaml(config_path))
    return apply_overrides(config, overrides)


def ensure_dir(path: str | Path) -> Path:
    """Creates and returns a directory path."""

    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    """Writes formatted JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_yaml(path: str | Path, payload: Dict[str, Any]) -> None:
    """Writes YAML with stable order."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def save_csv_rows(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    """Writes a list of dictionaries to CSV."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def resolve_dtype(dtype_name: str, *, device: torch.device) -> torch.dtype:
    """Maps config dtype names to torch dtypes with CPU-safe fallback."""

    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype_name}'.")
    dtype = mapping[dtype_name]
    if device.type == "cpu" and dtype != torch.float32:
        return torch.float32
    return dtype


def build_scratch_gpt2(model_cfg: Dict[str, Any]) -> GPT2LMHeadModel:
    """Builds a tiny GPT-2 model for smoke tests without network dependency."""

    config = GPT2Config(
        vocab_size=int(model_cfg.get("vocab_size", 259)),
        n_positions=int(model_cfg.get("max_position_embeddings", 128)),
        n_ctx=int(model_cfg.get("max_position_embeddings", 128)),
        n_embd=int(model_cfg.get("hidden_size", 64)),
        n_layer=int(model_cfg.get("num_hidden_layers", 2)),
        n_head=int(model_cfg.get("num_attention_heads", 4)),
        bos_token_id=int(model_cfg.get("bos_token_id", 1)),
        eos_token_id=int(model_cfg.get("eos_token_id", 2)),
        pad_token_id=int(model_cfg.get("pad_token_id", 0)),
    )
    return GPT2LMHeadModel(config)


def build_model_and_tokenizer(
    config: Dict[str, Any],
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, Any | None, str]:
    """Builds a causal LM and tokenizer, supporting scratch GPT-2 smoke tests."""

    model_cfg = config["model"]
    dtype = resolve_dtype(str(model_cfg.get("torch_dtype", "float32")), device=device)
    if bool(model_cfg.get("init_from_scratch", False)):
        architecture = str(model_cfg.get("architecture", "gpt2"))
        if architecture != "gpt2":
            raise ValueError(f"Unsupported scratch architecture '{architecture}'.")
        model = build_scratch_gpt2(model_cfg)
        model.to(device=device, dtype=dtype)
        return model, None, str(model_cfg.get("synthetic_name", "scratch_gpt2"))

    name_or_path = model_cfg.get("name_or_path")
    if not name_or_path:
        raise ValueError("model.name_or_path is required when init_from_scratch is false.")
    local_files_only = bool(model_cfg.get("local_files_only", False))
    model_kwargs: Dict[str, Any] = {"local_files_only": local_files_only}
    if device.type == "cuda":
        model_kwargs["dtype"] = dtype
    attn_impl = model_cfg.get("attn_impl")
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
    return model.to(device), tokenizer, str(name_or_path)


def _simple_text_to_ids(text: str, *, vocab_size: int, max_seq_len: int) -> List[int]:
    """Converts text to a deterministic small-vocab token sequence for scratch smoke tests."""

    usable_vocab = max(int(vocab_size) - 3, 1)
    token_ids = [1]
    for byte in text.encode("utf-8"):
        token_ids.append(3 + (byte % usable_vocab))
    token_ids.append(2)
    return token_ids[:max_seq_len]


def tokenize_text_batch(
    texts: Sequence[str],
    *,
    tokenizer,
    max_seq_len: int,
    model_vocab_size: int,
) -> Dict[str, torch.Tensor]:
    """Tokenizes a batch with HF tokenizer or a scratch deterministic fallback."""

    if tokenizer is not None:
        batch = tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }

    encoded = [_simple_text_to_ids(text, vocab_size=model_vocab_size, max_seq_len=max_seq_len) for text in texts]
    max_length = max(len(ids) for ids in encoded)
    input_ids = []
    attention_mask = []
    for ids in encoded:
        pad_length = max_length - len(ids)
        input_ids.append(ids + [0] * pad_length)
        attention_mask.append([1] * len(ids) + [0] * pad_length)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def load_text_samples(config: Dict[str, Any]) -> List[str]:
    """Loads text samples from inline list or local JSONL/TXT sources."""

    data_cfg = config["data"]
    source = str(data_cfg.get("source", "text_list"))
    max_samples = int(data_cfg.get("max_samples", 32))
    if source == "text_list":
        texts = [str(text) for text in data_cfg.get("text_list", [])]
        return texts[:max_samples]
    if source == "jsonl":
        jsonl_path = Path(str(data_cfg["jsonl_path"]))
        if not jsonl_path.is_absolute():
            jsonl_path = ROOT / jsonl_path
        text_key = str(data_cfg.get("text_key", "text"))
        texts: List[str] = []
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                value = payload.get(text_key)
                if value is None:
                    continue
                texts.append(str(value))
                if len(texts) >= max_samples:
                    break
        return texts
    if source == "txt":
        txt_path = Path(str(data_cfg["txt_path"]))
        if not txt_path.is_absolute():
            txt_path = ROOT / txt_path
        texts = [line.strip() for line in txt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return texts[:max_samples]
    raise ValueError(f"Unsupported data.source '{source}'.")


def resolve_output_root(config: Dict[str, Any]) -> Path:
    """Returns the root output directory for the isolated experiment."""

    root_dir = Path(str(config["output"].get("root_dir", "activation_cov_experiment/outputs")))
    if not root_dir.is_absolute():
        root_dir = ROOT / root_dir
    return root_dir


def activation_run_dir(config: Dict[str, Any]) -> Path:
    """Returns the activation output directory for the configured run."""

    return resolve_output_root(config) / "activations" / str(config["output"]["run_name"])


def direction_run_dir(config: Dict[str, Any]) -> Path:
    """Returns the directions output directory for the configured run."""

    return resolve_output_root(config) / "directions" / str(config["output"]["run_name"])


def metrics_run_dir(config: Dict[str, Any]) -> Path:
    """Returns the metrics output directory for the configured run."""

    return resolve_output_root(config) / "metrics" / str(config["output"]["run_name"])


def figures_run_dir(config: Dict[str, Any]) -> Path:
    """Returns the figures output directory for the configured run."""

    return resolve_output_root(config) / "figures" / str(config["output"]["run_name"])


def load_activation_metadata(path: str | Path) -> Dict[str, Any]:
    """Loads activation collection metadata."""

    return json.loads(Path(path).read_text(encoding="utf-8"))
