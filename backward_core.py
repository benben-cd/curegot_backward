#!/usr/bin/env python3
"""Shared backward (implicit-gradient) math for the SPLR solver.

This module holds the SINGLE implementation of the reduced-Hessian backward
math used by both production and validation:

    q_matvec       matrix-free Q(T) z in the reduced gauge (beta_m = 0)
    rhs_Q_scale    mu = [(G . T) 1_m, ((G . T)^T 1_n)_{-m}]  (stored scale)
    assemble_grad  dL/dM = (1/eta) * T . (lam_alpha 1^T + 1 lam_beta^T - G)
    pcg            preconditioned conjugate gradient

These implementations are the Stage 3A/3B oracle-validated ones, moved here
verbatim from stage3b_validate.py (Stage 3C-4 code-hygiene refactor: no
formula, operation-order, dtype, tolerance, stopping-rule, return-semantics
or gauge-convention changes). Dependency direction:

    backward_core.py
        ^-- splr_backward.py        (production entry point)
        ^-- stage3b_validate.py     (validates THIS implementation)
        ^-- benchmark scripts       (where they need the math helpers)

Validation-only helpers (make_problem, dense_Q, dense_from_csr, ...)
intentionally remain in stage3b_validate.py.
"""

import torch


def q_matvec(T, z, n, m):
    """Matrix-free Q(T) z in the reduced gauge."""
    Tred = T[:, :m - 1]
    v_alpha = z[:n]
    v_beta = z[n:]
    out_alpha = T.sum(dim=1) * v_alpha + Tred @ v_beta
    out_beta = Tred.t() @ v_alpha + T.sum(dim=0)[:m - 1] * v_beta
    return torch.cat([out_alpha, out_beta])


def rhs_Q_scale(G, T, n, m):
    """mu = [(G . T) 1_m, ((G . T)^T 1_n)_{-m}]  (stored scale, no 1/eta)."""
    GT = G * T
    return torch.cat([GT.sum(dim=1), GT.sum(dim=0)[:m - 1]])


def assemble_grad(T, G, lam, eta, n, m):
    """dL/dM = (1/eta) * T . (lam_alpha 1^T + 1 lam_beta^T - G)."""
    lam_alpha = lam[:n]
    lam_beta = torch.zeros(m, dtype=lam.dtype, device=lam.device)
    lam_beta[:m - 1] = lam[n:]
    return (T * (lam_alpha[:, None] + lam_beta[None, :] - G)) / eta


def pcg(matvec, precond, b, tol=1e-12, max_iter=2000):
    """Preconditioned conjugate gradient.

    matvec:  A x
    precond: M^{-1} r   (use None for plain CG)
    Returns (x, num_iter, converged).
    """
    x = torch.zeros_like(b)
    r = b - matvec(x)
    bnorm = torch.linalg.norm(r) if torch.linalg.norm(b) == 0 else torch.linalg.norm(b)
    z = r.clone() if precond is None else precond(r)
    p = z.clone()
    rz = torch.dot(r, z)
    for k in range(max_iter):
        w = matvec(p)
        pw = torch.dot(p, w)
        if pw <= 0:
            raise RuntimeError("PCG breakdown: p^T A p <= 0 (matrix not SPD?)")
        alpha = rz / pw
        x = x + alpha * p
        r = r - alpha * w
        rnorm = torch.linalg.norm(r)
        if rnorm < tol * bnorm:
            return x, k + 1, True
        z = r.clone() if precond is None else precond(r)
        rz_new = torch.dot(r, z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, max_iter, False
