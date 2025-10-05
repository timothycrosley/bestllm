"""hardware.py

A set of light weight cross-platform functions to aid in determining the hardware available on the current running device.
This data can then be used to help determine what local llm should be recommended for the user's available hardware.
"""
from __future__ import annotations
from functools import lru_cache
import ctypes
import shutil
import os
import sys
import subprocess
import re
from typing import NamedTuple
from ctypes import Structure, byref, c_char, c_int, c_int64, c_size_t, c_void_p, POINTER, sizeof, c_ulonglong, c_uint64
from enum import Enum


@lru_cache(maxsize=1)
def specs() -> HardwareSpecs:
    """Collect hardware information from the current machine."""
    gpu, gpu_vram = local_llm_vram()
    return HardwareSpecs(
        total_ram_gb=ram() / 1024.0 if ram() else 0,
        cpu_physical_cores=cpu_core_count(),
        gpu_vram_gb=gpu_vram / 1024.0 if gpu_vram else 0,
        gpu=gpu,
        available_disk_space_gb=available_disk_space() / (1024**3) if available_disk_space() else None
    )


class GPU(Enum):
    """Represents the kind of GPU that was identified as likely being able accelerate local LLMs."""
    UNKNOWN = 0
    NVIDIA = 1
    AMD = 2
    APPLE_SILICON = 3


class HardwareSpecs(NamedTuple):
    """Snapshot of the user's hardware relevant to local LLM selection."""

    total_ram_gb: float
    cpu_physical_cores: int
    gpu: GPU = GPU.UNKNOWN
    gpu_vram_gb: float = 0.0
    available_disk_space_gb: float = 0.0

    @property
    def has_gpu(self) -> bool:
        """Return True if a GPU with at least 1GB of VRAM is available."""
        return (self.gpu_vram_gb or 0) >= 1.0


def available_disk_space() -> int:
    """Returns the available disk space in bytes on the current disk, or 0 if undetermined."""
    try:
        return shutil.disk_usage('.').free
    except Exception:
        return 0


def cpu_core_count() -> int:
    """Returns the number of logical CPU cores available, or 0 if undetermined."""
    return os.cpu_count() or 0


def ram() -> int:
    """Returns the total physical system RAM in MB, or 0 if unable to determine."""
    try:
        if os.name == 'nt':  # Windows
            kernel32 = ctypes.windll.kernel32
            mem_kb = c_ulonglong()
            if kernel32.GetPhysicallyInstalledSystemMemory(byref(mem_kb)):
                return mem_kb.value // 1024
        elif sys.platform == 'linux':  # Linux
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        return int(line.split()[1]) // 1024
        elif sys.platform == 'darwin':  # macOS
            libc = ctypes.CDLL('libc.dylib')
            val = c_uint64()
            size = c_size_t(sizeof(c_uint64))
            libc.sysctlbyname(b'hw.memsize', byref(val), byref(size), None, 0)
            return val.value // (1024 * 1024)
    except Exception:
        pass
    return 0


def local_llm_vram() -> tuple[GPU, int]:
    """Returns an identified GPU capable of running local models if one is found alongside its VRAM.
    If none is available OR VRAM cannot be determined, (GPU.UNKNOWN, 0) is returned.
    The amount of VRAM is returned in MB.
    """
    # Strategy 1: Try runtime libraries first (most accurate for VRAM)
    gpu_type, vram = _check_runtime_libraries()
    if gpu_type != GPU.UNKNOWN and vram > 0:
        return (gpu_type, vram)
    
    # Strategy 2: Use system tools (works without ROCm/CUDA installed)
    if os.name == 'nt':
        gpu_type, vram = _check_system_tools_windows()
    elif sys.platform == 'darwin':
        gpu_type, vram = _check_metal()
    else:  # Linux
        gpu_type, vram = _check_system_tools_linux()
    
    if gpu_type != GPU.UNKNOWN and vram > 0:
        return (gpu_type, vram)
    
    return (GPU.UNKNOWN, 0)


def _check_runtime_libraries() -> tuple[GPU, int]:
    """Attempt to detect GPU via runtime libraries (CUDA/ROCm/HIP)"""
    if os.name == 'nt':  # Windows
        cuda_libs = [f'cudart64_{v}.dll' for v in range(130, 79, -1)]
        rocm_libs = ['amdhip64.dll']
    else:  # Linux/Mac
        cuda_libs = ['libcudart.so']
        rocm_libs = ['libamdhip64.so']

    # Check CUDA
    for libname in cuda_libs:
        try:
            lib = ctypes.CDLL(libname)
            get_count, get_props = _configure_api(lib, 'cuda')
            count = c_int()
            err = lib[get_count](byref(count))
            if err != 0 or count.value == 0:
                continue
            total_vram = 0
            for i in range(count.value):
                prop = _DeviceProp()
                err = lib[get_props](byref(prop), i)
                if err == 0:
                    total_vram += prop.totalGlobalMem
            return (GPU.NVIDIA, total_vram // (1024 * 1024))
        except Exception:
            pass

    # Check ROCm/HIP
    for libname in rocm_libs:
        try:
            lib = ctypes.CDLL(libname)
            get_count, get_props = _configure_api(lib, 'hip')
            count = c_int()
            err = lib[get_count](byref(count))
            if err != 0 or count.value == 0:
                continue
            total_vram = 0
            for i in range(count.value):
                prop = _DeviceProp()
                err = lib[get_props](byref(prop), i)
                if err == 0:
                    total_vram += prop.totalGlobalMem
            return (GPU.AMD, total_vram // (1024 * 1024))
        except Exception:
            pass

    return (GPU.UNKNOWN, 0)


def _check_system_tools_linux() -> tuple[GPU, int]:
    """Use system tools to detect GPU on Linux"""
    
    # Try nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            vram_mb = sum(int(line.strip()) for line in result.stdout.strip().split('\n') if line.strip())
            return (GPU.NVIDIA, vram_mb)
    except Exception:
        pass

    # Try rocm-smi
    try:
        result = subprocess.run(['rocm-smi', '--showmeminfo', 'vram'], 
                              capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            matches = re.findall(r'Total Memory \(B\):\s+(\d+)', result.stdout)
            if matches:
                total_bytes = sum(int(m) for m in matches)
                return (GPU.AMD, total_bytes // (1024 * 1024))
    except Exception:
        pass

    # Check /sys filesystem for AMD GPUs (works without ROCm)
    try:
        if os.path.exists('/sys/class/drm'):
            for device in os.listdir('/sys/class/drm'):
                if device.startswith('card') and not device.endswith(('-HDMI-A-1', '-DP-1', '-eDP-1')):
                    device_path = f'/sys/class/drm/{device}/device'
                    if os.path.exists(device_path):
                        vendor_file = f'{device_path}/vendor'
                        if os.path.exists(vendor_file):
                            with open(vendor_file, 'r') as f:
                                vendor = f.read().strip()
                                if vendor == '0x1002':  # AMD vendor ID
                                    mem_info_file = f'{device_path}/mem_info_vram_total'
                                    if os.path.exists(mem_info_file):
                                        with open(mem_info_file, 'r') as f:
                                            vram_bytes = int(f.read().strip())
                                            return (GPU.AMD, vram_bytes // (1024 * 1024))
                                elif vendor == '0x10de':  # NVIDIA vendor ID
                                    # Try to get VRAM from sysfs for NVIDIA
                                    mem_info_file = f'{device_path}/mem_info_vram_total'
                                    if os.path.exists(mem_info_file):
                                        with open(mem_info_file, 'r') as f:
                                            vram_bytes = int(f.read().strip())
                                            return (GPU.NVIDIA, vram_bytes // (1024 * 1024))
    except Exception:
        pass

    return (GPU.UNKNOWN, 0)


def _check_system_tools_windows() -> tuple[GPU, int]:
    """Use system tools to detect GPU on Windows"""
    
    # Try nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            vram_mb = sum(int(line.strip()) for line in result.stdout.strip().split('\n') if line.strip())
            return (GPU.NVIDIA, vram_mb)
    except Exception:
        pass

    # Try WMIC for GPU detection
    try:
        result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name,AdapterRAM'],
                              capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if 'NVIDIA' in line or 'GeForce' in line:
                    match = re.search(r'(\d+)', line)
                    if match:
                        vram_bytes = int(match.group(1))
                        if vram_bytes > 1024:  # Likely in bytes
                            return (GPU.NVIDIA, vram_bytes // (1024 * 1024))
                elif 'AMD' in line or 'Radeon' in line:
                    match = re.search(r'(\d+)', line)
                    if match:
                        vram_bytes = int(match.group(1))
                        if vram_bytes > 1024:  # Likely in bytes
                            return (GPU.AMD, vram_bytes // (1024 * 1024))
    except Exception:
        pass

    return (GPU.UNKNOWN, 0)


def _check_metal() -> tuple[GPU, int]:
    """Check for Apple Silicon Metal support"""
    try:
        metal_lib = ctypes.CDLL('/System/Library/Frameworks/Metal.framework/Metal')
        metal_lib.MTLCreateSystemDefaultDevice.restype = c_void_p
        device = metal_lib.MTLCreateSystemDefaultDevice()
        if device:
            try:
                objc_lib = ctypes.CDLL('/usr/lib/libobjc.A.dylib')
                sel = objc_lib.sel_registerName(b"recommendedMaxWorkingSetSize")
                objc_lib.objc_msgSend.restype = c_int64
                objc_lib.objc_msgSend.argtypes = [c_void_p, c_void_p]
                vram_bytes = objc_lib.objc_msgSend(device, sel)
                if vram_bytes > 0:
                    return (GPU.APPLE_SILICON, vram_bytes // (1024 * 1024))
            except Exception:
                pass
    except Exception:
        pass

    return (GPU.UNKNOWN, 0)


class _DeviceProp(Structure):
    """Shared structure for CUDA/HIP device properties (minimal fields)."""
    _fields_ = [
        ("name", c_char * 256),
        ("totalGlobalMem", c_size_t),
    ]


def _configure_api(lib, prefix):
    get_device_count = prefix + 'GetDeviceCount'
    get_device_props = prefix + 'GetDeviceProperties'
    lib[get_device_count].restype = c_int
    lib[get_device_count].argtypes = [POINTER(c_int)]
    lib[get_device_props].restype = c_int
    lib[get_device_props].argtypes = [POINTER(_DeviceProp), c_int]
    return get_device_count, get_device_props
