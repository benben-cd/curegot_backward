"""Compatibility shim (Stage 5).

The single production implementation moved into the installed package:
``curegot/_backward_core.py``. This shim re-exports it unchanged so the
existing repo-root scripts (stage3b_validate.py, benchmark_stage3c*.py,
validate_stage4_autograd.py, ...) keep working WITHOUT maintaining a second
copy. Object identity is preserved: ``backward_core.pcg is
curegot._backward_core.pcg``.

Requires the Stage 5 editable install (pip install -e .) that provides the
curegot package. Do NOT add code here.
"""

from curegot._backward_core import (
    assemble_grad,
    pcg,
    q_matvec,
    rhs_Q_scale,
)

__all__ = ["q_matvec", "rhs_Q_scale", "assemble_grad", "pcg"]
