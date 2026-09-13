from ._cuda_loader import preload_cuda_runtime

preload_cuda_runtime()

# Load NumPy interface
from ._internal_numpy import *

# When the package is built using Pytorch, we typically
# need to import torch first to properly load some libraries
# such as libc10.so
try:
    import torch
    # Load PyTorch interface
    from ._internal_torch import *
except:
    pass

__version__ = "0.1.0"


## 新增
# Differentiable SPLR OT Python API
try:
    from . import autograd, backward
    from .autograd import (
        SPLROT,
        splr_ot,
        splr_ot_batch,
        splr_ot_with_info,
    )
    from .backward import solve_backward

    # `torch` here is the native pybind submodule exported by
    # curegot._internal_torch.
    torch.splr_ot = splr_ot
    torch.splr_ot_with_info = splr_ot_with_info
    torch.splr_ot_batch = splr_ot_batch
    torch.SPLROT = SPLROT
    torch.solve_backward = solve_backward

except ImportError:
    # Preserve the original optional-PyTorch behavior of curegot:
    # NumPy-only installations should still be importable.
    pass