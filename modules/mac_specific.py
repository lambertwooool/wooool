import logging
import torch
from packaging import version

log = logging.getLogger(__name__)

# before torch version 1.13, has_mps is only available in nightly pytorch and macOS 12.3+,
# use check `getattr` and try it for compatibility.
# in torch version 1.13, backends.mps.is_available() and backends.mps.is_built() are introduced in to check mps availabilty,
# since torch 2.0.1+ nightly build, getattr(torch, 'has_mps', False) was deprecated, see https://github.com/pytorch/pytorch/pull/103279
def check_for_mps() -> bool:
    if version.parse(torch.__version__) <= version.parse("2.0.1"):
        if not getattr(torch, 'has_mps', False):
            return False
        try:
            torch.zeros(1).to(torch.device("mps"))
            return True
        except Exception:
            return False
    else:
        return torch.backends.mps.is_available() and torch.backends.mps.is_built()

has_mps = check_for_mps()

def torch_mps_gc() -> None:
    try:
        from torch.mps import empty_cache
        empty_cache()
    except Exception:
        log.warning("MPS garbage collection failed", exc_info=True)