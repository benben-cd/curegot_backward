"""Compatibility shim (Stage 5).

The single production implementation moved into the installed package:
``curegot/autograd.py``. This shim re-exports it unchanged so existing
repo-root scripts (validate_stage4_autograd.py, benchmark_stage4_autograd
.py, ...) keep working. Object identity is preserved:
``splr_autograd.splr_ot is curegot.autograd.splr_ot``.

New code should use the package API instead:

    import curegot
    T = curegot.torch.splr_ot(M, a, b, eta=...)

Requires the Stage 5 editable install (pip install -e .). Do NOT add code
here.
"""

from curegot.autograd import (
    SPLROT,
    SPLROTFunction,
    splr_ot,
    splr_ot_batch,
    splr_ot_with_info,
)

__all__ = ["splr_ot", "splr_ot_with_info", "splr_ot_batch",
           "SPLROT", "SPLROTFunction"]
