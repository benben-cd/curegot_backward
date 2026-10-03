"""Implicit gradient for a converged SPLR plan.

Solve Q(T) lambda = rhs_Q_scale(G, T), then assemble dL/dM.
The default native diagonal uses diag(A) of the context sparse matrix,
not diag(Q(T)). SPLR reuses its factorization and optional rank-2 update;
none selects plain CG. Performance depends on workload and tolerance.
"""

import torch

from ._backward_core import (
    q_matvec,
    rhs_Q_scale,
    assemble_grad,
    pcg,
)

__all__ = ["solve_backward", "select_preconditioner"]

_VALID_PRECONDITIONERS = ("diag", "splr", "none")


def select_preconditioner(ctx, preconditioner):
    """Map a preconditioner mode string to the apply callable (or None).

    Production semantics: "diag" is the NATIVE diagonal. Raises ValueError
    on any other value.
    """
    if preconditioner == "diag":
        if not bool(ctx.has_diagonal_preconditioner):
            raise RuntimeError(
                "SPLRBackwardContext: diagonal preconditioner not set up "
                "(context predates Stage 3C-3 or setup failed)"
            )
        return ctx.apply_diagonal_preconditioner
    if preconditioner == "splr":
        return ctx.apply_preconditioner
    if preconditioner == "none":
        return None
    raise ValueError(
        f"invalid preconditioner '{preconditioner}'; "
        f"expected one of: {' | '.join(_VALID_PRECONDITIONERS)}"
    )


def solve_backward(
    T,
    G,
    ctx,
    eta,
    *,
    preconditioner="diag",
    tol=1e-12,
    max_iter=2000,
):
    """Solve the backward pass and assemble the implicit gradient.

    Args:
        T:   [n, m] float64 CUDA tensor, the optimal transport plan from
             sinkhorn_splr_bwd (out["plan"]).
        G:   [n, m] float64 CUDA tensor, upstream gradient dL/dT.
        ctx: SPLRBackwardContext returned by sinkhorn_splr_bwd
             (out["context"]).
        eta: entropic regularization used in the forward pass.
        preconditioner: "diag" (default) | "splr" | "none".
        tol:     PCG relative residual tolerance (matches the validated
                 oracle tolerance).
        max_iter: PCG iteration cap.

    Returns:
        dict with
            "lambda":         [n + m - 1] reduced dual solution,
            "grad":           [n, m] implicit gradient dL/dM,
            "iterations":     PCG iterations used,
            "converged":      whether PCG met tol within max_iter,
            "preconditioner": the mode actually used.

    Note: the returned "lambda" is the reduced-gauge dual (beta_m = 0);
    "grad" already accounts for the gauge internally.
    """
    if T.dim() != 2:
        raise ValueError("T must be a 2D tensor")
    if G.shape != T.shape:
        raise ValueError("G must have the same shape as T")
    # Note: ctx validity is enforced natively (an invalid context can never
    # be returned by sinkhorn_splr_bwd, and the apply entry points raise on
    # misuse); no Python-side is_valid binding exists.
    n, m = T.shape

    precond = select_preconditioner(ctx, preconditioner)

    mu = rhs_Q_scale(G, T, n, m)
    if not bool(torch.any(mu)):
        # Degenerate upstream gradient: Q lambda = 0 has the exact solution
        # lambda = 0 (Q is SPD). Short-circuit instead of running PCG --
        # the iterative loop would otherwise hit its p^T A p <= 0 breakdown
        # guard on the zero start residual. This is exact, not an
        # approximation, and does not touch any nonzero-G path.
        lam = torch.zeros_like(mu)
        grad = assemble_grad(T, G, lam, eta, n, m)
        return {
            "lambda": lam,
            "grad": grad,
            "iterations": 0,
            "converged": True,
            "preconditioner": preconditioner,
        }
    matvec = lambda v: q_matvec(T, v, n, m)
    lam, iterations, converged = pcg(matvec, precond, mu, tol=tol,
                                     max_iter=max_iter)
    grad = assemble_grad(T, G, lam, eta, n, m)
    return {
        "lambda": lam,
        "grad": grad,
        "iterations": iterations,
        "converged": converged,
        "preconditioner": preconditioner,
    }
