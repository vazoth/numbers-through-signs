from __future__ import annotations
import contextlib
import gc
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .config import ExperimentConfig, ModelSpec


def module_device_inventory(module: torch.nn.Module) -> dict[str, Any]:
    """Summarize tensor placement."""
    parameter_counts: dict[str, int] = {}
    buffer_counts: dict[str, int] = {}
    for parameter in module.parameters():
        key = str(parameter.device)
        parameter_counts[key] = parameter_counts.get(key, 0) + parameter.numel()
    for buffer in module.buffers():
        key = str(buffer.device)
        buffer_counts[key] = buffer_counts.get(key, 0) + buffer.numel()
    devices = sorted(set(parameter_counts) | set(buffer_counts))
    return {
        "parameter_elements_by_device": parameter_counts,
        "buffer_elements_by_device": buffer_counts,
        "devices": devices,
        "all_cuda_zero": bool(devices) and devices == ["cuda:0"],
    }


def require_all_cuda_zero(module: torch.nn.Module) -> dict[str, Any]:
    inventory = module_device_inventory(module)
    if not inventory["all_cuda_zero"]:
        raise RuntimeError(
            f"Measurement requires every parameter and buffer on cuda:0; observed devices: {inventory['devices']}"
        )
    return inventory


@dataclass(frozen=True)
class VerbalizerIds:
    true: tuple[int, ...]
    false: tuple[int, ...]
    true_text: tuple[str, ...]
    false_text: tuple[str, ...]


@dataclass(frozen=True)
class TokenBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    final_indices: torch.Tensor

    def model_inputs(self) -> dict[str, torch.Tensor]:
        return {"input_ids": self.input_ids, "attention_mask": self.attention_mask}


def batched(items: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def resolve_site_window(*, start_site: int, window: int | str, num_sites: int) -> tuple[int, ...]:
    if not 0 <= start_site < num_sites:
        raise IndexError(f"start_site outside 0..{num_sites - 1}")
    if window == "to-end":
        stop_site = num_sites
    elif isinstance(window, int) and (not isinstance(window, bool)) and (window > 0):
        stop_site = min(start_site + window, num_sites)
    else:
        raise ValueError("window must be a positive integer or 'to-end'")
    return tuple(range(start_site, stop_site))


def patch_hidden_output(
    output: Any, carrier_indices: torch.Tensor, source_vectors: torch.Tensor
) -> Any:
    tuple_output = isinstance(output, tuple)
    hidden = output[0] if tuple_output else output
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise TypeError(f"Expected [batch, sequence, hidden] tensor, got {type(hidden)}")
    patched = hidden.clone()
    rows = torch.arange(patched.shape[0], device=patched.device)
    local_indices = carrier_indices.to(device=patched.device)
    patched[rows, local_indices] = source_vectors.to(device=patched.device, dtype=patched.dtype)
    if tuple_output:
        return (patched, *output[1:])
    return patched


class ModelRuntime:
    def __init__(self, spec: ModelSpec, config: ExperimentConfig) -> None:
        self.spec = spec
        self.config = config
        requested_device = torch.device("cuda:0")
        self.placement = spec.placement
        if self.placement != "single":
            raise ValueError("placement must be single")
        if not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
            raise RuntimeError("Measurements require a BF16-capable CUDA GPU")
        self.dtype = torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(
            spec.id, revision=spec.revision, use_fast=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        load_kwargs: dict[str, Any] = {
            "revision": spec.revision,
            "dtype": self.dtype,
            "low_cpu_mem_usage": True,
            "attn_implementation": "sdpa",
        }
        self.model = AutoModelForCausalLM.from_pretrained(spec.id, **load_kwargs)
        self.model.to(requested_device)
        self.model.eval()
        self.device_inventory = require_all_cuda_zero(self.model)
        attention = getattr(self.model.config, "_attn_implementation", None)
        if attention != "sdpa":
            raise RuntimeError(f"Measurement requires SDPA attention, observed {attention!r}")
        if getattr(self.model, "is_quantized", False):
            raise RuntimeError("Measurement forbids quantized model weights")
        if getattr(self.model, "quantization_method", None) is not None:
            raise RuntimeError("Measurement detected a quantization method")
        if getattr(self.model, "hf_device_map", None):
            raise RuntimeError("Measurement forbids automatic device maps")
        self.layers = self._resolve_layers()
        self.embedding = self.model.get_input_embeddings()
        self.modules = (self.embedding, *tuple(self.layers))
        self.device = self.embedding.weight.device
        self.commit_hash = str(getattr(self.model.config, "_commit_hash", spec.revision))
        self.verbalizers = {
            task: self._resolve_verbalizers(labels)
            for task, labels in config.class_verbalizers.items()
        }

    def _resolve_layers(self) -> Sequence[torch.nn.Module]:
        candidates = (
            getattr(getattr(self.model, "model", None), "layers", None),
            getattr(getattr(getattr(self.model, "model", None), "model", None), "layers", None),
        )
        for candidate in candidates:
            if candidate is not None:
                return candidate
        raise AttributeError("Could not locate decoder layers on the loaded model")

    def _resolve_verbalizers(self, verbalizers: dict[str, tuple[str, ...]]) -> VerbalizerIds:
        resolved: dict[str, list[tuple[str, int]]] = {"true": [], "false": []}
        for label in ("true", "false"):
            for text in verbalizers[label]:
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                if len(ids) == 1:
                    resolved[label].append((text, ids[0]))
        if not resolved["true"] or not resolved["false"]:
            raise ValueError("No single-token true/false verbalizers remain after tokenization")
        return VerbalizerIds(
            true=tuple((token for _, token in resolved["true"])),
            false=tuple((token for _, token in resolved["false"])),
            true_text=tuple((text for text, _ in resolved["true"])),
            false_text=tuple((text for text, _ in resolved["false"])),
        )

    @property
    def num_sites(self) -> int:
        return len(self.modules)

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    def tokenize(self, prompts: Sequence[str]) -> TokenBatch:
        encoded = self.tokenizer(
            list(prompts), add_special_tokens=False, padding=True, return_tensors="pt"
        )
        attention_mask = encoded["attention_mask"].to(self.device)
        return TokenBatch(
            input_ids=encoded["input_ids"].to(self.device),
            attention_mask=attention_mask,
            final_indices=attention_mask.sum(dim=-1) - 1,
        )

    def carrier_indices(self, prefixes: Sequence[str], prompts: Sequence[str]) -> torch.Tensor:
        if len(prefixes) != len(prompts):
            raise ValueError("prefixes and prompts must have the same length")
        indices: list[int] = []
        for prefix, prompt in zip(prefixes, prompts, strict=True):
            prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            if prompt_ids[: len(prefix_ids)] != prefix_ids:
                raise ValueError("Prefix tokenization is not a prefix of the full prompt")
            if not prefix_ids:
                raise ValueError("Empty quantity prefix")
            indices.append(len(prefix_ids) - 1)
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def quantity_indices(self, prefixes: Sequence[str], prompts: Sequence[str]) -> torch.Tensor:
        carrier = self.carrier_indices(prefixes, prompts)
        if torch.any(carrier < 1):
            raise ValueError("Quantity prefix has no token before the carrier")
        return carrier - 1

    def validate_carrier(self, prefixes: Sequence[str]) -> dict[str, Any]:
        carrier_ids = []
        carrier_text = []
        for prefix in prefixes:
            ids = self.tokenizer.encode(prefix, add_special_tokens=False)
            carrier_ids.append(ids[-1])
            carrier_text.append(self.tokenizer.decode([ids[-1]]))
        unique_ids = sorted(set(carrier_ids))
        return {
            "unique_carrier_ids": unique_ids,
            "unique_carrier_text": sorted(set(carrier_text)),
            "is_single_shared_carrier": len(unique_ids) == 1,
        }

    @torch.inference_mode()
    def final_logits(self, prompts: Sequence[str]) -> np.ndarray:
        token_batch = self.tokenize(prompts)
        outputs = self.model(**token_batch.model_inputs())
        logits_device = outputs.logits.device
        rows = torch.arange(len(prompts), device=logits_device)
        logits = outputs.logits[rows, token_batch.final_indices.to(device=logits_device)]
        return logits.float().cpu().numpy()

    @torch.inference_mode()
    def prefix_representations(self, prefixes: Sequence[str]) -> torch.Tensor:
        token_batch = self.tokenize(prefixes)
        state_position = token_batch.final_indices - 1
        gathered: list[torch.Tensor] = []

        def capture(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                raise TypeError(f"Expected [batch, sequence, hidden] tensor, got {type(hidden)}")
            rows = torch.arange(len(prefixes), device=hidden.device)
            gathered.append(hidden[rows, state_position.to(device=hidden.device)].to(device="cpu"))

        handles = [module.register_forward_hook(capture) for module in self.modules]
        try:
            self.model(**token_batch.model_inputs(), use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        if len(gathered) != self.num_sites:
            raise RuntimeError(f"Captured {len(gathered)} sites, expected {self.num_sites}")
        return torch.stack(gathered, dim=1)

    @torch.inference_mode()
    def prompt_representations(
        self, prompts: Sequence[str], prefixes: Sequence[str]
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Capture full-prompt states and their matching final logits."""
        token_batch = self.tokenize(prompts)
        state_position = self.quantity_indices(prefixes, prompts)
        gathered: list[torch.Tensor] = []

        def capture(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                raise TypeError(f"Expected [batch, sequence, hidden] tensor, got {type(hidden)}")
            rows = torch.arange(len(prompts), device=hidden.device)
            gathered.append(hidden[rows, state_position.to(device=hidden.device)].to(device="cpu"))

        handles = [module.register_forward_hook(capture) for module in self.modules]
        try:
            outputs = self.model(**token_batch.model_inputs(), use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        if len(gathered) != self.num_sites:
            raise RuntimeError(f"Captured {len(gathered)} sites, expected {self.num_sites}")
        logits_device = outputs.logits.device
        rows = torch.arange(len(prompts), device=logits_device)
        logits = outputs.logits[rows, token_batch.final_indices.to(device=logits_device)]
        return (torch.stack(gathered, dim=1), logits.float().cpu().numpy())

    @contextlib.contextmanager
    def patch_sites(
        self, site_vectors: dict[int, torch.Tensor], carrier_indices: torch.Tensor
    ) -> Iterator[None]:
        invalid = [site for site in site_vectors if not 0 <= site < self.num_sites]
        if invalid:
            raise IndexError(f"Sites {invalid} outside 0..{self.num_sites - 1}")

        def make_hook(source_vectors: torch.Tensor) -> Any:

            def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
                if source_vectors.device != self.device:
                    raise RuntimeError("Intervention vectors must be staged on the model GPU")
                return patch_hidden_output(output, carrier_indices, source_vectors)

            return hook

        handles = [
            self.modules[site].register_forward_hook(make_hook(source_vectors))
            for site, source_vectors in sorted(site_vectors.items())
        ]
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    @torch.inference_mode()
    def patched_final_logits(
        self,
        prompts: Sequence[str],
        prefixes: Sequence[str],
        site: int,
        source_vectors: torch.Tensor,
    ) -> np.ndarray:
        token_batch = self.tokenize(prompts)
        state_position = self.quantity_indices(prefixes, prompts)
        prepared = source_vectors.to(device=self.device, dtype=self.dtype)
        with self.patch_sites({site: prepared}, state_position):
            outputs = self.model(**token_batch.model_inputs(), use_cache=False)
        logits_device = outputs.logits.device
        rows = torch.arange(len(prompts), device=logits_device)
        logits = outputs.logits[rows, token_batch.final_indices.to(device=logits_device)]
        return logits.float().cpu().numpy()

    @torch.inference_mode()
    def trajectory_patched_final_logits(
        self,
        prompts: Sequence[str],
        prefixes: Sequence[str],
        start_site: int,
        source_representations: torch.Tensor,
    ) -> np.ndarray:
        if source_representations.ndim != 3:
            raise ValueError("Expected source representations [batch, site, hidden]")
        if source_representations.shape[1] != self.num_sites:
            raise ValueError(
                f"Expected {self.num_sites} source sites, got {source_representations.shape[1]}"
            )
        return self.window_patched_final_logits(
            prompts, prefixes, start_site, "to-end", source_representations
        )

    @torch.inference_mode()
    def window_patched_final_logits(
        self,
        prompts: Sequence[str],
        prefixes: Sequence[str],
        start_site: int,
        window: int | str,
        source_representations: torch.Tensor,
    ) -> np.ndarray:
        if source_representations.ndim != 3:
            raise ValueError("Expected source representations [batch, site, hidden]")
        if source_representations.shape[1] != self.num_sites:
            raise ValueError(
                f"Expected {self.num_sites} source sites, got {source_representations.shape[1]}"
            )
        selected_sites = resolve_site_window(
            start_site=start_site, window=window, num_sites=self.num_sites
        )
        token_batch = self.tokenize(prompts)
        state_position = self.quantity_indices(prefixes, prompts)
        site_vectors = {
            site: source_representations[:, site].to(device=self.device, dtype=self.dtype)
            for site in selected_sites
        }
        with self.patch_sites(site_vectors, state_position):
            outputs = self.model(**token_batch.model_inputs(), use_cache=False)
        logits_device = outputs.logits.device
        rows = torch.arange(len(prompts), device=logits_device)
        logits = outputs.logits[rows, token_batch.final_indices.to(device=logits_device)]
        return logits.float().cpu().numpy()

    def manifest(self) -> dict[str, Any]:
        cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        return {
            "model_id": self.spec.id,
            "model_short_name": self.spec.short_name,
            "requested_revision": self.spec.revision,
            "resolved_revision": self.commit_hash,
            "dtype": str(self.dtype),
            "placement": self.placement,
            "hf_device_map": {
                str(key): str(value)
                for key, value in getattr(self.model, "hf_device_map", {}).items()
            },
            "device_inventory": self.device_inventory,
            "num_decoder_layers": len(self.layers),
            "num_intervention_sites": self.num_sites,
            "hidden_size": self.hidden_size,
            "verbalizers": {
                task: {
                    "true_text": values.true_text,
                    "false_text": values.false_text,
                    "true_ids": values.true,
                    "false_ids": values.false,
                }
                for task, values in self.verbalizers.items()
            },
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_device": cuda_name,
            "attention_implementation": getattr(self.model.config, "_attn_implementation", None),
        }

    def close(self) -> None:
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
