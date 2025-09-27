import pytest

from bestllm.hardware import HardwareSpecs
from bestllm.models import DEFAULT_MODEL_PROFILES
from bestllm.selector import (
    NoSuitableModelError,
    describe_catalog,
    recommend_model_for_specs,
)


def test_prefers_gpu_model_when_requirements_met() -> None:
    specs = HardwareSpecs(total_ram_gb=64, cpu_physical_cores=16, gpu_vram_gb=24)
    profile = recommend_model_for_specs(specs)
    assert profile.name == "llama3-8b-instruct-q4"


def test_falls_back_to_cpu_when_gpu_vram_low() -> None:
    specs = HardwareSpecs(total_ram_gb=32, cpu_physical_cores=8, gpu_vram_gb=4)
    profile = recommend_model_for_specs(specs)
    assert profile.preferred_device == "cpu"
    assert profile.name == "qwen2.5-7b-instruct-gguf"


def test_cpu_only_environment_selects_cpu_model() -> None:
    specs = HardwareSpecs(total_ram_gb=20, cpu_physical_cores=8, gpu_vram_gb=None)
    profile = recommend_model_for_specs(specs)
    assert profile.name == "qwen2.5-7b-instruct-gguf"


def test_raises_when_no_model_fits() -> None:
    specs = HardwareSpecs(total_ram_gb=6, cpu_physical_cores=2, gpu_vram_gb=None)
    with pytest.raises(NoSuitableModelError):
        recommend_model_for_specs(specs)


def test_describe_catalog_returns_serializable_structures() -> None:
    serialized = describe_catalog(DEFAULT_MODEL_PROFILES)
    assert serialized
    assert "name" in serialized[0]
    assert isinstance(serialized[0]["context_window"], int)
