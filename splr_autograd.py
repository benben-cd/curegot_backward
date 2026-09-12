#!/usr/bin/env python3
"""Stage 4: end-to-end PyTorch autograd integration for the SPLR solver.

Public API:

    from splr_autograd import splr_ot

    T = splr_ot(M, a, b, eta=0.05)      # like a normal PyTorch op
    loss = downstream_loss(T)
    loss.backward()                     # M.grad = implicit gradient dL/dM

Gradient scope (Stage 4, deliberately conservative):

    M:    differentiable (dL/dM via the validated implicit-differentiation
          path: reduced-Hessian PCG with the native diagonal preconditioner
          by default)
    a, b: constants -- requires_grad=True raises NotImplementedError
    eta:  constant scalar

Requirements: M/a/b are CUDA float64 tensors on the same device. AMP /
autocast is unsupported and raises. Double backward is unsupported
(@once_differentiable raises a clear error on create_graph=True).

Non-convergence policy (production-safe): forward non-convergence and
backward PCG non-convergence both raise RuntimeError rather than returning
silently bad values. The lower-level manual API
(curegot.torch.sinkhorn_splr_bwd + splr_backward.solve_backward) keeps its
flag-returning semantics for research use.

This module changes NO solver math: forward is curegot.torch.sinkhorn_splr_bwd,
backward is splr_backward.solve_backward (backward_core PCG). The default
backward preconditioner remains "diag" (Stage 3C frozen policy).
"""

import torch
from torch.autograd.function import once_differentiable

import curegot

from splr_backward import solve_backward, _VALID_PRECONDITIONERS

__all__ = ["splr_ot", "splr_ot_with_info", "SPLROTFunction"]


# ---------------------------------------------------------------------------
# Input contract
# ---------------------------------------------------------------------------

def _check_inputs(M, a, b):
    """Enforce the Stage 4 device/dtype/shape contract.

    Strict on purpose: the native wrapper would silently convert
    float32/CPU inputs to float64 CUDA and convert outputs back, which
    would mislead users about the gradient dtype/device (Stage 4 prompt,
    section 26). We raise instead.
    """
    for name, t in (("M", M), ("a", a), ("b", b)):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"splr_ot: {name} must be a torch.Tensor, "
                            f"got {type(t).__name__}")
        if not t.is_cuda:
            raise RuntimeError(
                f"splr_ot: {name} must be a CUDA tensor "
                f"(Stage 4 supports CUDA float64 only)"
            )
        if t.dtype != torch.float64:
            raise RuntimeError(
                f"splr_ot: {name} must be float64, got {t.dtype}; "
                "the native solver is float64-only and silent conversion "
                "would make the gradient dtype misleading"
            )
    if M.dim() != 2:
        raise ValueError(f"splr_ot: M must be 2D, got shape {tuple(M.shape)}")
    n, m = M.shape
    if a.dim() != 1 or a.shape[0] != n:
        raise ValueError(
            f"splr_ot: a must be 1D of length {n}, got shape {tuple(a.shape)}"
        )
    if b.dim() != 1 or b.shape[0] != m:
        raise ValueError(
            f"splr_ot: b must be 1D of length {m}, got shape {tuple(b.shape)}"
        )
    if not (M.device == a.device == b.device):
        raise ValueError(
            f"splr_ot: M/a/b must be on the same device, got "
            f"{M.device}, {a.device}, {b.device}"
        )


# ---------------------------------------------------------------------------
# autograd.Function
# ---------------------------------------------------------------------------

class SPLROTFunction(torch.autograd.Function):
    """Custom autograd Function wrapping forward + implicit backward.

    forward runs the native SPLR solver and stashes the plan and the
    pybind backward context on the autograd ctx; backward runs the shared
    solve_backward (backward_core PCG) with the chosen preconditioner.
    """

    @staticmethod
    def forward(ctx, M, a, b, eta, tol, max_iter, density,
                candidate_sinkhorn_iter, backward_preconditioner,
                backward_tol, backward_max_iter):
        kwargs = {
            "candidate_sinkhorn_iter": candidate_sinkhorn_iter,
            "context_mode": "adopt",
        }
        if density is not None:
            kwargs["density"] = density
        out = curegot.torch.sinkhorn_splr_bwd(
            M, a, b, eta, tol, max_iter, 0, **kwargs
        )
        if not bool(out["converged"]):
            # Production-safe default (Stage 4, section 33): never hand a
            # non-converged plan to a training loop.
            raise RuntimeError(
                f"splr_ot: forward solver did not converge "
                f"(niter={int(out['niter'])}, max_iter={max_iter}, "
                f"tol={tol}); relax tol or increase max_iter"
            )
        T = out["plan"]

        # save_for_backward is tensor-only; the pybind context is attached
        # as a plain attribute. Python refcounting keeps it alive until the
        # autograd graph (and hence this ctx) is freed -- verified by the
        # "forward locals released before backward" validation test.
        ctx.save_for_backward(T)
        ctx.backward_context = out["context"]
        ctx.eta = eta
        ctx.backward_preconditioner = backward_preconditioner
        ctx.backward_tol = backward_tol
        ctx.backward_max_iter = backward_max_iter
        return T

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (T,) = ctx.saved_tensors
        # Downstream gradients are not guaranteed contiguous; the shared
        # backward math does not require contiguity, but normalize cheaply
        # (no-op when already contiguous) to keep the contract simple.
        G = grad_output if grad_output.is_contiguous() \
            else grad_output.contiguous()
        res = solve_backward(
            T, G, ctx.backward_context, ctx.eta,
            preconditioner=ctx.backward_preconditioner,
            tol=ctx.backward_tol,
            max_iter=ctx.backward_max_iter,
        )
        if not res["converged"]:
            # Production-safe default (Stage 4, section 32): never return a
            # silently unconverged gradient into a training loop.
            raise RuntimeError(
                f"splr_ot: backward PCG did not converge "
                f"(iterations={res['iterations']}, "
                f"max_iter={ctx.backward_max_iter}, "
                f"tol={ctx.backward_tol}, "
                f"preconditioner='{ctx.backward_preconditioner}')"
            )
        grad_M = res["grad"]
        # One gradient per forward input (M, a, b, eta, tol, max_iter,
        # density, candidate_sinkhorn_iter, backward_preconditioner,
        # backward_tol, backward_max_iter) -- order must match exactly.
        return (grad_M, None, None, None, None, None, None, None, None,
                None, None)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def splr_ot(
    M,
    a,
    b,
    *,
    eta,
    tol=1e-6,
    max_iter=1000,
    density=None,
    candidate_sinkhorn_iter=3,
    backward_preconditioner="diag",
    backward_tol=1e-12,
    backward_max_iter=2000,
):
    """Differentiable entropic-OT plan via the SPLR solver.

    Args:
        M:   [n, m] CUDA float64 tensor (cost matrix); the only
             differentiable input.
        a:   [n] CUDA float64 source marginal (constant).
        b:   [m] CUDA float64 target marginal (constant).
        eta: entropic regularization (constant scalar, required).
        tol: forward solver tolerance. Default 1e-6 matches the native
             pybind default of sinkhorn_splr_bwd (audited; Stage 3
             benchmarks used the stricter 1e-8 explicitly).
        max_iter: forward iteration cap (native default 1000).
        density: sparsification budget; None -> native default
             min(10 / min(n, m), 1.0) (audited from the native wrapper).
        candidate_sinkhorn_iter: native default 3.
        backward_preconditioner: "diag" (default, native diagonal) |
             "splr" (cuDSS + rank-2) | "none" (plain CG, debug).
        backward_tol: backward PCG relative tolerance (Stage 3 validated
             value 1e-12).
        backward_max_iter: backward PCG cap (2000).

    Returns:
        T: [n, m] CUDA float64 tensor, the optimal transport plan, with
           grad_fn attached when M.requires_grad.

    Raises:
        RuntimeError: CUDA/float64 contract violation, AMP/autocast active,
             forward non-convergence, backward PCG non-convergence.
        NotImplementedError: a or b requires grad.
        ValueError: invalid preconditioner or shape mismatch.
    """
    if torch.is_autocast_enabled():
        raise RuntimeError(
            "splr_ot: AMP/autocast is unsupported (the solver is "
            "float64-only); disable autocast for this call"
        )
    _check_inputs(M, a, b)
    if a.requires_grad or b.requires_grad:
        raise NotImplementedError(
            "splr_ot: gradients w.r.t. a/b are not supported in Stage 4 "
            "(only dL/dM); detach them or use the lower-level API"
        )
    if backward_preconditioner not in _VALID_PRECONDITIONERS:
        raise ValueError(
            f"splr_ot: invalid backward_preconditioner "
            f"'{backward_preconditioner}'; expected one of: "
            f"{' | '.join(_VALID_PRECONDITIONERS)}"
        )
    return SPLROTFunction.apply(
        M, a, b,
        float(eta), float(tol), int(max_iter), density,
        int(candidate_sinkhorn_iter), backward_preconditioner,
        float(backward_tol), int(backward_max_iter),
    )


def splr_ot_with_info(
    M,
    a,
    b,
    *,
    eta,
    tol=1e-6,
    max_iter=1000,
    density=None,
    candidate_sinkhorn_iter=3,
):
    """Debug/introspection helper: plan + solver metadata.

    NOT differentiable (runs under no_grad and returns a detached plan);
    for gradients use splr_ot. Non-convergence is reported via the
    "converged" flag instead of raising -- inspecting such cases is the
    point of this helper. The backward context is intentionally dropped.
    """
    _check_inputs(M, a, b)
    kwargs = {
        "candidate_sinkhorn_iter": int(candidate_sinkhorn_iter),
        "context_mode": "adopt",
    }
    if density is not None:
        kwargs["density"] = density
    with torch.no_grad():
        out = curegot.torch.sinkhorn_splr_bwd(
            M, a, b, float(eta), float(tol), int(max_iter), 0, **kwargs
        )
    return {
        "plan": out["plan"].detach(),
        "dual": out["dual"].detach(),
        "niter": int(out["niter"]),
        "converged": bool(out["converged"]),
    }
