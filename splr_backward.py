"""Compatibility shim (Stage 5).

The single production implementation moved into the installed package:
``curegot/backward.py``. This shim re-exports it unchanged so existing
repo-root scripts keep working. Object identity is preserved:
``splr_backward.solve_backward is curegot.backward.solve_backward``.

Requires the Stage 5 editable install (pip install -e .). Do NOT add code
here.
"""

from curegot.backward import (
    _VALID_PRECONDITIONERS,
    select_preconditioner,
    solve_backward,
)

__all__ = ["solve_backward", "select_preconditioner",
           "_VALID_PRECONDITIONERS"]
