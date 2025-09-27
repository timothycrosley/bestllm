"""Hardware introspection utilities for bestllm."""
from __future__ import annotations

from dataclasses import dataclass
import os
import shutil
import subprocess
from typing import Optional


@dataclass(frozen=True)
class HardwareSpecs:
    """Snapshot of the user's hardware relevant to local LLM selection."""

    total_ram_gb: float
    cpu_physical_cores: int
    gpu_vram_gb: Optional[float] = None

    @property
    def has_gpu(self) -> bool:
        """Return True if a GPU with at least 1GB of VRAM is available."""
        return (self.gpu_vram_gb or 0) >= 1.0

    @classmethod
    def from_system(cls) -> "HardwareSpecs":
        """Collect hardware information from the current machine."""
        total_ram = _detect_total_ram_gb()
        if total_ram is None:
            raise RuntimeError(
                "Unable to detect total system RAM; provide HardwareSpecs manually."
            )

        cpu_cores = _detect_cpu_core_count()
        gpu_vram = _detect_gpu_vram_gb()
        return cls(
            total_ram_gb=total_ram,
            cpu_physical_cores=cpu_cores,
            gpu_vram_gb=gpu_vram,
        )


def _detect_total_ram_gb() -> Optional[float]:
    for detector in (_psutil_total_ram_gb, _sysconf_total_ram_gb):
        value = detector()
        if value is not None:
            return value
    return None


def _psutil_total_ram_gb() -> Optional[float]:
    try:
        import psutil  # type: ignore
    except ImportError:  # pragma: no cover - psutil rarely present in CI
        return None

    try:
        total_bytes = psutil.virtual_memory().total
    except Exception:  # pragma: no cover - defensive
        return None
    return _bytes_to_gb(total_bytes)


def _sysconf_total_ram_gb() -> Optional[float]:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None
    if page_size <= 0 or page_count <= 0:
        return None
    return _bytes_to_gb(page_size * page_count)


def _detect_cpu_core_count() -> int:
    try:
        import psutil  # type: ignore
    except ImportError:
        psutil = None  # type: ignore

    if psutil is not None:
        try:
            cores = psutil.cpu_count(logical=False)
            if cores:
                return cores
        except Exception:  # pragma: no cover - defensive
            pass

    cores = os.cpu_count()
    if cores is None or cores < 1:
        return 1
    return cores


def _detect_gpu_vram_gb() -> Optional[float]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None

    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):  # pragma: no cover - defensive
        return None

    values = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            values.append(float(stripped))
        except ValueError:  # pragma: no cover - defensive
            return None

    if not values:
        return None
    return round(max(values), 1)


def _bytes_to_gb(value: int) -> float:
    return round(value / (1024 ** 3), 1)


__all__ = ["HardwareSpecs"]
