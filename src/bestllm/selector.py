"""Model selection logic for bestllm."""
from __future__ import annotations

from typing import Iterable, Sequence

from .hardware import HardwareSpecs
from .models import DEFAULT_MODEL_PROFILES, ModelProfile


class NoSuitableModelError(RuntimeError):
    """Raised when no model fits the supplied hardware constraints."""


def recommend_model_for_specs(
    specs: HardwareSpecs,
    catalog: Sequence[ModelProfile] | None = None,
) -> ModelProfile:
    """Return the highest-impact model that fits the provided hardware specs."""
    available = catalog if catalog is not None else DEFAULT_MODEL_PROFILES
    candidates = [model for model in available if _meets_requirements(model, specs)]
    if not candidates:
        raise NoSuitableModelError(
            "No local model fits the provided hardware characteristics."
        )
    return max(candidates, key=lambda model: _score_model(model, specs))


def _meets_requirements(model: ModelProfile, specs: HardwareSpecs) -> bool:
    if specs.total_ram_gb + 1e-6 < model.min_ram_gb:
        return False
    if specs.cpu_physical_cores < model.min_cpu_cores:
        return False
    if model.min_vram_gb is not None:
        if specs.gpu_vram_gb is None:
            return False
        if specs.gpu_vram_gb + 1e-6 < model.min_vram_gb:
            return False
    elif model.preferred_device == "gpu" and not specs.has_gpu:
        return False
    return True


def _score_model(model: ModelProfile, specs: HardwareSpecs) -> tuple[float, int, float]:
    gpu_bonus = 1.0 if model.preferred_device == "gpu" and specs.has_gpu else 0.0
    return (
        gpu_bonus,
        model.context_window,
        model.parameter_size_b,
    )


def describe_catalog(catalog: Iterable[ModelProfile] | None = None) -> list[dict[str, object]]:
    profiles = catalog if catalog is not None else DEFAULT_MODEL_PROFILES
    return [profile.as_dict() for profile in profiles]


__all__ = [
    "NoSuitableModelError",
    "recommend_model_for_specs",
    "describe_catalog",
]
