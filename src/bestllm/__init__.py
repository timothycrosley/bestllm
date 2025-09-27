"""Public API for the bestllm package."""
from __future__ import annotations

from .hardware import HardwareSpecs
from .models import DEFAULT_MODEL_PROFILES, ModelProfile
from .selector import NoSuitableModelError, recommend_model_for_specs


def best_local_llm() -> ModelProfile:
    """Inspect the current machine and return the best-fitting local model profile."""
    specs = HardwareSpecs.from_system()
    return recommend_model_for_specs(specs)


def main() -> None:
    """CLI entry point printing the recommended model to stdout."""
    try:
        specs = HardwareSpecs.from_system()
        recommendation = recommend_model_for_specs(specs)
    except NoSuitableModelError as exc:  # pragma: no cover - CLI guard
        print(f"bestllm: {exc}")
        raise SystemExit(1) from exc
    except RuntimeError as exc:  # pragma: no cover - CLI guard
        print(f"bestllm: {exc}")
        raise SystemExit(2) from exc

    print(
        "Recommended model: "
        f"{recommendation.name} "
        f"({recommendation.context_window:,} token context window)."
    )
    print(
        "Summary: "
        f"requires >= {recommendation.min_ram_gb}GB RAM, "
        f">= {recommendation.min_cpu_cores} CPU cores, "
        f"{recommendation.min_vram_gb or 'no'} GPU VRAM requirement."
    )
    if specs.has_gpu:
        print(
            f"Detected GPU VRAM: {specs.gpu_vram_gb}GB — using GPU-friendly profile."
        )
    else:
        print("Detected CPU-only environment — recommending CPU-optimized build.")


__all__ = [
    "HardwareSpecs",
    "ModelProfile",
    "DEFAULT_MODEL_PROFILES",
    "NoSuitableModelError",
    "best_local_llm",
    "recommend_model_for_specs",
]
