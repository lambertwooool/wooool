import torch

def check_for_xpu() -> bool:
    try:
        import intel_extension_for_pytorch as ipex
        if torch.xpu.is_available():
            return True
    except:
        return False

has_xpu = check_for_xpu()