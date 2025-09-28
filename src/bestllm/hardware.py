"""hardware.py

A set of light weight cross-plaform functions to aid in determining the hardware available on the current running device.
This data can then be used to help determine what local llm should be recommended for the user's available hardware.
"""
import ctypes
import shutil
import os
import sys
from ctypes import Structure, byref, c_char, c_int, c_int64, c_size_t, c_void_p, POINTER, sizeof, c_ulonglong, c_uint64
from enum import Enum


class GPU(Enum):
    """Represents the kind of GPU that was identified as likely being able accelerate local LLMs."""
    UNKNOWN = 0
    NVIDIA = 1
    AMD = 2
    APPLE_SILICON = 3


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


def local_llm_vram() -> (GPU, int):
    """Returns an identified GPU capable of running local models if one is found alongside its VRAM.
    If none is available (for instance no gpu found capable of acceleration) (NONE, 0) is returned.
    The amount of VRAM is returned in mb.
    """
    if os.name == 'nt':  # Windows
        cuda_libs = [f'cudart64_{v}.dll' for v in range(130, 79, -1)]  # Updated to include CUDA 13.0+
        rocm_libs = ['amdhip64.dll']
    else:  # Linux/Mac (posix)
        cuda_libs = ['libcudart.so']
        rocm_libs = ['libamdhip64.so']

    # Check CUDA or ROCm/HIP
    for libname in cuda_libs + rocm_libs:
        try:
            lib = ctypes.CDLL(libname)
            if libname in cuda_libs or 'cudart' in libname:
                print("NVIDIA GPU detected")
                try:
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
                except:
                    print("Unable to determine VRAM for NVIDIA GPU")
                    return (GPU.NVIDIA, 0)
            else:
                print("AMD GPU detected")
                try:
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
                except:
                    print("Unable to determine VRAM for AMD GPU")
                    return (GPU.AMD, 0)

        except OSError:
            pass

    # Check Metal on macOS
    if sys.platform == 'darwin':
        try:
            metal_lib = ctypes.CDLL('/System/Library/Frameworks/Metal.framework/Metal')
            metal_lib.MTLCreateSystemDefaultDevice.restype = c_void_p
            device = metal_lib.MTLCreateSystemDefaultDevice()
            if device:
                print("Apple Silicon detected")
                try:
                    objc_lib = ctypes.CDLL('/usr/lib/libobjc.A.dylib')
                    sel = objc_lib.sel_registerName(b"recommendedMaxVRAM")
                    objc_lib.objc_msgSend.restype = c_int64
                    objc_lib.objc_msgSend.argtypes = [c_void_p, c_void_p]
                    vram_bytes = objc_lib.objc_msgSend(device, sel)
                    if vram_bytes > 0:
                        return (GPU.APPLE, vram_bytes // (1024 * 1024))
                except:
                    print("Unable to determine universal memory for Apple Silicon")
                    return (GPU.APPLE, 0)

        except OSError:
            pass

    print("No GPU found capable of accelerating local LLMs")
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
