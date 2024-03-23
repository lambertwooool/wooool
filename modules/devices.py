import sys
import numpy as np
import psutil
import torch
from modules import mac_specific, xpu_specific

XFORMERS_ENABLE = None
PYTORCH_ATTENTION_ENABLE = None

def has_mps() -> bool:
    if sys.platform != "darwin":
        return False
    else:
        return mac_specific.has_mps

def has_xpu() -> bool:
    return xpu_specific.has_xpu

def get_device_type():
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    elif has_mps():
        return "mps"
    elif has_xpu():
        return "xpu"
    else:
        return "cpu"

def get_torch_device():
    return torch.device(get_device_type())

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(get_device_type()):
            vram_old = get_free_memory()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            vram_new = get_free_memory()
            print(f"[GC] Free {(vram_new - vram_old) / (1024 ** 3):.2f}G VRAM, {vram_new / (1024 ** 3):.2f}G Now")

    if has_mps():
        mac_specific.torch_mps_gc()

def is_xformers_enable() -> bool:
    global XFORMERS_ENABLE

    is_available = False

    if XFORMERS_ENABLE is not None:
        is_available = XFORMERS_ENABLE
    else:
        try:
            import xformers
            import xformers.ops
            is_available = True
            try:
                version = xformers.version.__version__
                print("XFormers Version:", version)
                if version.startswith("0.0.18"):
                    print()
                    print("WARNING: This version of xformers has a major bug where you will get black images when generating high resolution images.")
                    print("Please downgrade or upgrade xformers to a different version.")
                    print()
            except:
                pass
        except:
            pass
    
        XFORMERS_ENABLE = is_available

    return is_available

def is_pytorch_attention_enable() -> bool:
    global PYTORCH_ATTENTION_ENABLE

    device_type = get_device_type()
    is_available = False

    if PYTORCH_ATTENTION_ENABLE is not None:
        is_available = PYTORCH_ATTENTION_ENABLE
    else:
        if device_type.startswith("cuda"):
            torch_version = torch.version.__version__
            if int(torch_version[0]) >= 2:
                is_available = True
            # if torch.cuda.is_bf16_supported():
            #     VAE_DTYPE = torch.bfloat16
        elif device_type == "xpu":
            is_available = True
    
        PYTORCH_ATTENTION_ENABLE = is_available

    return is_available

def device_supports_non_blocking(device):
    if has_mps():
        return False #pytorch bug? mps doesn't support non blocking
    return True

def should_use_fp16(device=None):
    if device == None:
        device = get_torch_device()

    if is_typeof(device, ['cpu', 'mps']):
        return False
    
    return torch.cuda.is_bf16_supported()

def dtype(device=None, want_use_dtype=torch.float16):
    if should_use_fp16(device=device):
        if want_use_dtype in [torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz, torch.float32]:
            return want_use_dtype
        if want_use_dtype == torch.bfloat16 and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            return torch.float16
    return torch.float32

def dtype_size(dtype):
    dtype_size = 4
    if hasattr(dtype, "itemsize"):
        dtype_size = dtype.itemsize
    elif dtype == torch.float16 or dtype == torch.bfloat16:
        dtype_size = 2
    elif dtype == torch.float32:
        dtype_size = 4
    return dtype_size

def get_autocast_device(device):
    # https://github.com/lllyasviel/Fooocus/discussions/571
    # https://github.com/lllyasviel/Fooocus/issues/620
    result = ''
    if hasattr(device, 'type'):
        result = str(device.type)
    if 'cuda' in result:
        return 'cuda'
    else:
        return 'cpu'

def cast_to_device(tensor, device, dtype, copy=False):
    device_supports_cast = False
    if tensor.dtype == torch.float32 or tensor.dtype == torch.float16:
        device_supports_cast = True
    elif tensor.dtype == torch.bfloat16:
        if hasattr(device, 'type') and device.type.startswith("cuda"):
            device_supports_cast = True

    non_blocking = device_supports_non_blocking(device)

    if device_supports_cast:
        if copy:
            if tensor.device == device:
                return tensor.to(dtype, copy=copy, non_blocking=non_blocking)
            return tensor.to(device, copy=copy, non_blocking=non_blocking).to(dtype, non_blocking=non_blocking)
        else:
            return tensor.to(device, non_blocking=non_blocking).to(dtype, non_blocking=non_blocking)
    else:
        return tensor.to(device, dtype, copy=copy, non_blocking=non_blocking)

# None means no manual cast
def unet_manual_cast(weight_dtype, inference_device, supported_dtypes=[torch.float16, torch.bfloat16, torch.float32]):
    if weight_dtype == torch.float32:
        return None

    fp16_supported = should_use_fp16(inference_device)
    if fp16_supported and weight_dtype == torch.float16:
        return None

    if fp16_supported and torch.float16 in supported_dtypes:
        return torch.float16
    elif fp16_supported and torch.bfloat16 in supported_dtypes:
        return torch.bfloat16
    else:
        return torch.float32

def intermediate_device():
    return torch.device("cpu")

def is_typeof(device, types) -> bool:
    return hasattr(device, 'type') and \
            (device.type == types or device.type in list(types))

def get_memory(device=None, get_all=False):
    if device is None or get_all:
        device = get_torch_device()

    mem_total = psutil.virtual_memory().total    
    mem_free = psutil.virtual_memory().available

    if is_typeof(device, ['cpu', 'mps']):
        mem_total_torch = 0
        mem_total_device = mem_total
        mem_free_torch = 0
        mem_free_device = mem_free
    else:
        # stats = torch.cuda.memory_stats(device)
        # mem_total_torch = stats['reserved_bytes.all.current']
        mem_free_torch, mem_total_torch = torch.cuda.mem_get_info(device)
        mem_total_device = mem_total_torch
         # stats = torch.cuda.memory_stats(device)
        # mem_active = stats['active_bytes.all.current']
        # mem_reserved = stats['reserved_bytes.all.current']
        mem_free_device = mem_free_torch

    if get_all:
        return np.array([mem_total, mem_total_torch, mem_free, mem_free_torch])
    else:
        return np.array([mem_total_device, mem_free_device])

def get_total_memory(device=None, get_all=False):
    mem = list(get_memory(device, get_all))
    return mem[0] if len(mem) < 3 else mem[:2]

def get_free_memory(device=None, get_all=False):
    mem = get_memory(device, get_all)
    return mem[1] if len(mem) < 3 else mem[2:]

if is_pytorch_attention_enable():
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

TOTAL_RAM, TOTAL_VRAM, FREE_RAM, FREE_VRAM = get_memory(get_all=True) / (1024 ** 3)
print(f"Total VRAM {TOTAL_VRAM:.1f}G RAM {TOTAL_RAM:.1f}G / Free VRAM {FREE_VRAM:.1f}G RAM {FREE_RAM:.1f}G")

try:
    # OOM_EXCEPTION = torch.cuda.OutOfMemoryError
    OOM_EXCEPTION = MemoryError if not torch.cuda.is_available() else torch.cuda.OutOfMemoryError
except:
    OOM_EXCEPTION = Exception