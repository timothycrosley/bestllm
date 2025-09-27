"""Model catalog definitions for bestllm."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal


PreferredDevice = Literal["cpu", "gpu"]


@dataclass(frozen=True)
class ModelProfile:
    """Describes a local model build and its resource requirements."""

    name: str
    parameter_size_b: float
    context_window: int
    min_ram_gb: float
    min_cpu_cores: int
    min_vram_gb: float | None
    preferred_device: PreferredDevice
    notes: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "parameter_size_b": self.parameter_size_b,
            "context_window": self.context_window,
            "min_ram_gb": self.min_ram_gb,
            "min_cpu_cores": self.min_cpu_cores,
            "min_vram_gb": self.min_vram_gb,
            "preferred_device": self.preferred_device,
            "notes": self.notes,
        }


DEFAULT_MODEL_PROFILES: List[ModelProfile] = [
    ModelProfile(
        name="llama3-70b-instruct-q4",
        parameter_size_b=70,
        context_window=8192,
        min_ram_gb=96,
        min_cpu_cores=24,
        min_vram_gb=48,
        preferred_device="gpu",
        notes="Top-tier quality when multiple high-end GPUs are present.",
    ),
    ModelProfile(
        name="llama3-8b-instruct-q4",
        parameter_size_b=8,
        context_window=8192,
        min_ram_gb=16,
        min_cpu_cores=4,
        min_vram_gb=10,
        preferred_device="gpu",
        notes="Balanced GPU option with strong general-purpose performance.",
    ),
    ModelProfile(
        name="mistral-7b-instruct-q4",
        parameter_size_b=7,
        context_window=8192,
        min_ram_gb=12,
        min_cpu_cores=4,
        min_vram_gb=8,
        preferred_device="gpu",
        notes="Reliable on mid-range GPUs with modest VRAM budgets.",
    ),
    ModelProfile(
        name="qwen2.5-7b-instruct-gguf",
        parameter_size_b=7,
        context_window=32768,
        min_ram_gb=16,
        min_cpu_cores=6,
        min_vram_gb=None,
        preferred_device="cpu",
        notes="Great for CPU-bound setups needing long context windows.",
    ),
    ModelProfile(
        name="phi-3-mini-4k-instruct",
        parameter_size_b=3.8,
        context_window=4096,
        min_ram_gb=8,
        min_cpu_cores=4,
        min_vram_gb=None,
        preferred_device="cpu",
        notes="Lightweight CPU baseline for compact machines.",
    ),
]


def iter_models() -> Iterable[ModelProfile]:
    return tuple(DEFAULT_MODEL_PROFILES)


__all__ = ["ModelProfile", "DEFAULT_MODEL_PROFILES", "iter_models", "PreferredDevice"]
