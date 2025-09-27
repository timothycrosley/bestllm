import pytest

from bestllm.hardware import HardwareSpecs


def test_from_system_aggregates_detectors(monkeypatch) -> None:
    monkeypatch.setattr("bestllm.hardware._detect_total_ram_gb", lambda: 32.0)
    monkeypatch.setattr("bestllm.hardware._detect_cpu_core_count", lambda: 8)
    monkeypatch.setattr("bestllm.hardware._detect_gpu_vram_gb", lambda: 12.0)

    specs = HardwareSpecs.from_system()

    assert specs.total_ram_gb == 32.0
    assert specs.cpu_physical_cores == 8
    assert specs.gpu_vram_gb == 12.0
    assert specs.has_gpu is True


def test_from_system_raises_when_ram_unknown(monkeypatch) -> None:
    monkeypatch.setattr("bestllm.hardware._detect_total_ram_gb", lambda: None)

    with pytest.raises(RuntimeError):
        HardwareSpecs.from_system()


def test_has_gpu_false_when_below_threshold() -> None:
    specs = HardwareSpecs(total_ram_gb=16, cpu_physical_cores=4, gpu_vram_gb=0.5)
    assert specs.has_gpu is False


def test_cpu_only_environment_reports_no_gpu() -> None:
    specs = HardwareSpecs(total_ram_gb=16, cpu_physical_cores=4, gpu_vram_gb=None)
    assert specs.has_gpu is False
