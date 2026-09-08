"""
Stage 3B.1 validation script: L0 / L1 / L2 tests for the SPLR backward context.

Real mode (requires the patched curegot build on a CUDA machine):
    python stage3b_validate.py

Emulation mode (pure PyTorch, works on CPU; validates the script logic
itself, including the CSR export parsing and the PCG loop):
    python stage3b_validate.py --emulate

Mathematical setup
------------------
We solve the entropic-regularized optimal transport problem (EROT)
    min_T  <T, M> - eta * h(T)   s.t.  T 1_m = a,  T^T 1_n = b,
with the transport plan parameterized by the dual variables
    T_ij = exp((alpha_i + beta_j - M_ij) / eta).
Gauge: beta_m = 0. The reduced variable is z = (alpha, beta_{-m}) in
R^{n+m-1}. The reduced Hessian of the dual objective is
    H = (1/eta) * Q(T),
    Q(T) = [[diag(T 1_m),        T_{-m}            ],
            [T_{-m}^T,           diag((T^T 1_n)_{-m})]],
where T_{-m} denotes T with its last column removed.

The backward pass (implicit function theorem) computes, for an upstream
gradient G = dL/dT:
    rhs   = (1/eta) * [(G . T) 1_m,  ((G . T)^T 1_n)_{-m}],
    H lam = rhs,
    dL/dM = (1/eta) * T . (lam_alpha 1_m^T + 1_n lam_beta^T - G),
with lam_{beta_m} = 0. Multiplying the linear system by eta gives the
stored-scale system Q(T*) lam = mu with mu = eta * rhs, which is what the
PCG below solves.

The preconditioner is the approximate Hessian of the last forward
iteration, in the stored scale:
    B = A + (eta/ys) * y y^T - (A s)(A s)^T / (s^T A s),
where A = eta * H_Omega + (eta*shift) * I is the matrix actually
factorized by cuDSS, and (s, y) is the last secant pair.

Validation ladder
-----------------
L0: ctx.apply_preconditioner(r)  vs  dense solve of B z = r
    (B rebuilt from the exported CSR triple and secant materials)
L1: consistency: for random z0, r = B z0 (dense), check that
    apply_preconditioner(r) recovers z0
L2: full backward gradient: PCG on Q(T*) lam = mu with the context as
    preconditioner, compared against the exact dense solve
Fallback: run the forward with x0 at the optimum, so that the Newton loop
    is never entered; the context then comes from the fallback path
    (fresh factorization at the final point, no low-rank correction)
"""

import argparse
import math
import torch


# ---------------------------------------------------------------------------
# Basic oracles (dense, exact)
# ---------------------------------------------------------------------------

def make_problem(n, m, eta, seed, device):
    """Random cost matrix and strictly positive marginals."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    M = torch.rand(n, m, generator=gen, dtype=torch.float64).to(device)
    a = torch.rand(n, generator=gen, dtype=torch.float64).to(device) + 0.5
    b = torch.rand(m, generator=gen, dtype=torch.float64).to(device) + 0.5
    a = a / a.sum()
    b = b / b.sum()
    return M, a, b


def plan_from_dual(M, dual, eta, n, m):
    """T_ij = exp((alpha_i + beta_j - M_ij) / eta). dual = (alpha, beta)."""
    alpha = dual[:n]
    beta = dual[n:]
    logT = (alpha[:, None] + beta[None, :] - M) / eta
    return torch.exp(logT)


def dense_Q(T, n, m):
    """Explicit Q(T) = eta * H in the reduced gauge (beta_m = 0)."""
    Tred = T[:, :m - 1]
    rsum = T.sum(dim=1)                    # T 1_m
    csum = T.sum(dim=0)[:m - 1]            # (T^T 1_n)_{-m}
    Q = torch.zeros(n + m - 1, n + m - 1, dtype=T.dtype, device=T.device)
    Q[:n, :n] = torch.diag(rsum)
    Q[:n, n:] = Tred
    Q[n:, :n] = Tred.t()
    Q[n:, n:] = torch.diag(csum)
    return Q


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


# ---------------------------------------------------------------------------
# Rebuild the preconditioner B from the context exports
# ---------------------------------------------------------------------------

def dense_from_csr(values, colind, rowptr, dim, device):
    """Rebuild a dense symmetric matrix from the exported CSR triple.

    The cuDSS matrix uses CUDSS_MVIEW_LOWER, so the CSR stores the lower
    triangle (diagonal included). This function inserts every stored entry
    at both (i, j) and (j, i), which is correct whether the storage holds
    the lower triangle only or the full matrix.
    """
    values = values.to(torch.float64).cpu()
    colind = colind.to(torch.int64).cpu()
    rowptr = rowptr.to(torch.int64).cpu()
    A = torch.zeros(dim, dim, dtype=torch.float64)
    for i in range(dim):
        for k in range(int(rowptr[i]), int(rowptr[i + 1])):
            j = int(colind[k])
            A[i, j] = values[k]
            A[j, i] = values[k]
    return A.to(device)


def build_precond_from_ctx(ctx, device):
    """Rebuild B = A + (eta/ys) y y^T - (A s)(A s)^T / (s^T A s) densely."""
    dim = ctx.dim
    A = dense_from_csr(ctx.csr_values(), ctx.csr_colind(), ctx.csr_rowptr(),
                       dim, device)
    if not ctx.low_rank_enabled:
        return A
    s = ctx.secant_s().to(torch.float64)
    y = ctx.secant_y().to(torch.float64)
    eta = ctx.reg
    ys = ctx.ys
    As = A @ s
    B = A + (eta / ys) * torch.outer(y, y) - torch.outer(As, As) / torch.dot(s, As)
    return B


# ---------------------------------------------------------------------------
# Real-mode context: from the patched curegot module
# ---------------------------------------------------------------------------

def run_forward_real(M, a, b, eta, tol, max_iter, x0=None, verbose=0):
    import curegot
    kwargs = dict(density=0.5, return_validation=True)
    if x0 is not None:
        kwargs["x0"] = x0
    out = curegot.torch.sinkhorn_splr_bwd(
        M, a, b, eta, tol=tol, max_iter=max_iter, verbose=verbose, **kwargs
    )
    return out["plan"], out["dual"], out["niter"], out["converged"], out["context"]


# ---------------------------------------------------------------------------
# Emulation mode: pure-Python forward producing a context with the same API
# ---------------------------------------------------------------------------

class EmulatedContext:
    """Mimics curegot.torch.SPLRBackwardContext for CPU-side testing.

    A = eta * H_Omega + (eta * shift) * I, stored densely; the CSR exports
    contain the lower triangle, as in the cuDSS storage.
    """

    def __init__(self, A, s, y, invA_y, ys, reg, low_rank, ctx_point,
                 fallback):
        self._A = A
        self._s = s
        self._y = y
        self._u = invA_y
        self._ys = ys
        self._reg = reg
        self._low_rank = low_rank
        self._ctx_point = ctx_point
        self._fallback = fallback
        self._build_csr()

    # -- pybind-compatible API -------------------------------------------
    @property
    def reg(self):
        return self._reg

    @property
    def dim(self):
        return self._A.shape[0]

    @property
    def low_rank_enabled(self):
        return self._low_rank

    @property
    def is_fallback(self):
        return self._fallback

    @property
    def ys(self):
        return self._ys

    def csr_values(self):
        return self._csr_values

    def csr_colind(self):
        return self._csr_colind

    def csr_rowptr(self):
        return self._csr_rowptr

    def secant_s(self):
        if not self._low_rank:
            raise RuntimeError("low-rank correction is disabled")
        return self._s

    def secant_y(self):
        if not self._low_rank:
            raise RuntimeError("low-rank correction is disabled")
        return self._y

    def ctx_point(self):
        return self._ctx_point

    def apply_preconditioner(self, r):
        """z = B^{-1} r via the stored-scale Woodbury formula.

        Uses the same update rule as launch_low_rank_search_direc:
            q = A^{-1} r, u = A^{-1} y
            z = q - (s^T r / ys) u
                + [(1/reg + y^T u / ys)(s^T r / ys) - y^T q / ys] s
        """
        q = torch.linalg.solve(self._A, r)
        if not self._low_rank:
            return q
        sg_ys = torch.dot(self._s, r) / self._ys
        yinvAy = torch.dot(self._y, self._u)
        yinvAg = torch.dot(self._y, q)
        term1 = (1.0 / self._reg + yinvAy / self._ys) * sg_ys - yinvAg / self._ys
        return q + term1 * self._s - sg_ys * self._u

    # -- helpers ----------------------------------------------------------
    def _build_csr(self):
        """Export the lower triangle of A in CSR form (int32 / float64)."""
        dim = self.dim
        A = self._A.cpu()
        values, colind, rowptr = [], [], [0]
        for i in range(dim):
            for j in range(i + 1):
                if A[i, j] != 0.0:
                    values.append(float(A[i, j]))
                    colind.append(j)
            rowptr.append(len(values))
        self._csr_values = torch.tensor(values, dtype=torch.float64,
                                        device=self._A.device)
        self._csr_colind = torch.tensor(colind, dtype=torch.int32,
                                        device=self._A.device)
        self._csr_rowptr = torch.tensor(rowptr, dtype=torch.int32,
                                        device=self._A.device)


def dual_obj_grad(M, a, b, gamma, eta, n, m):
    """Dual objective f(gamma) = eta * sum(T) - <alpha, a> - <beta, b>
    and its reduced gradient g = (T 1 - a, (T^T 1 - b)_{-m})."""
    T = plan_from_dual(M, gamma, eta, n, m)
    f = eta * T.sum() - torch.dot(gamma[:n], a) - torch.dot(gamma[n:], b)
    g = torch.cat([T.sum(dim=1) - a, (T.sum(dim=0) - b)[:m - 1]])
    return f, g, T


def sparsified_hessian(T, eta, shift, density, n, m):
    """A = eta * H_Omega + (eta * shift) * I, with the global top-k pattern
    on T_{-m} plus the full diagonal, in the stored scale (A = eta * H)."""
    Tred = T[:, :m - 1]
    k = max(1, int(density * n * (m - 1)))
    k = min(k, n * (m - 1))
    thresh = torch.kthvalue(Tred.flatten(), n * (m - 1) - k + 1).values
    mask = Tred >= thresh

    dim = n + m - 1
    A = torch.zeros(dim, dim, dtype=torch.float64, device=T.device)
    rsum = T.sum(dim=1)
    csum = T.sum(dim=0)[:m - 1]
    A[range(n), range(n)] = rsum
    A[range(n, dim), range(n, dim)] = csum
    A[:n, n:] = Tred * mask
    A[n:, :n] = (Tred * mask).t()
    A += eta * shift * torch.eye(dim, dtype=torch.float64, device=T.device)
    # Note: the diagonal of A above is exactly eta * H_Ω + eta*shift*I,
    # since H = (1/eta) * Q and A = eta * H_Ω = Q restricted to the pattern
    return A


def emulate_forward(M, a, b, eta, tol, max_iter, density=0.5, shift_max=1e-3,
                    x0=None):
    """Simplified SPLR forward: quasi-Newton with B = A + rank-2 BFGS
    correction as the direction matrix, plus Armijo backtracking.

    Returns (plan, dual, niter, converged, EmulatedContext), mirroring the
    real interface. The context reproduces the 方案-1 semantics: the
    factorization and the secant pair come from the LAST completed
    iteration (at gamma_prev), not from the final point.
    """
    n, m = M.shape
    dim = n + m - 1

    # Initialization: beta = 0, alpha = optimal given beta
    if x0 is not None:
        gamma = x0.clone()
        beta_m = gamma[n + m - 1].clone()
        gamma[:n] += beta_m
        gamma[n:] -= beta_m
    else:
        gamma = torch.zeros(n + m, dtype=torch.float64, device=M.device)
        logK = torch.logsumexp(-M / eta, dim=1)
        gamma[:n] = eta * (torch.log(a) - logK)

    f, g, T = dual_obj_grad(M, a, b, gamma, eta, n, m)
    gnorm = torch.linalg.norm(g)
    shift = min(float(gnorm), shift_max)
    A = sparsified_hessian(T, eta, shift, density, n, m)

    gamma_prev = torch.zeros_like(gamma)
    g_prev = torch.zeros(dim, dtype=torch.float64, device=M.device)

    niter = 0
    converged = False
    # Materials of the last completed iteration
    last = None

    for it in range(max_iter):
        if gnorm < tol:
            converged = True
            break
        z = torch.cat([gamma[:n], gamma[n:n + m - 1]])
        z_prev = torch.cat([gamma_prev[:n], gamma_prev[n:n + m - 1]])
        s = z - z_prev
        y = g - g_prev
        ys = float(torch.dot(y, s))
        yy = float(torch.dot(y, y))
        low_rank = (it > 0) and (ys > 1e-6 * yy)

        # Direction: x = B^{-1} g, step = -eta * x
        q = torch.linalg.solve(A, g)
        if low_rank:
            u = torch.linalg.solve(A, y)
            sg_ys = torch.dot(s, g) / ys
            term1 = ((1.0 / eta + torch.dot(y, u) / ys) * sg_ys
                     - torch.dot(y, q) / ys)
            x = q + term1 * s - sg_ys * u
            invA_y = u
        else:
            x = q
            invA_y = torch.zeros(dim, dtype=torch.float64, device=M.device)
        d = -eta * x

        # Save history, then Armijo backtracking on the dual objective
        gamma_prev = gamma.clone()
        g_prev = g.clone()
        step = 1.0
        for _ in range(50):
            z_new = z + step * d
            gamma_new = gamma.clone()
            gamma_new[:n] = z_new[:n]
            gamma_new[n:n + m - 1] = z_new[n:]
            gamma_new[n + m - 1] = 0.0
            f_new, g_new, T_new = dual_obj_grad(M, a, b, gamma_new, eta, n, m)
            if f_new <= f - 1e-4 * step * abs(float(torch.dot(g, d))) + 1e-300:
                break
            step *= 0.5
        gamma, f, g, T = gamma_new, f_new, g_new, T_new
        gnorm = torch.linalg.norm(g)

        # Context materials of THIS iteration (factorization at the point
        # before the step, i.e. gamma_prev after the step)
        shift = min(float(gnorm), shift_max)
        last = dict(A=A, s=s, y=y, invA_y=invA_y, ys=ys, low_rank=low_rank,
                    ctx_point=torch.cat([gamma_prev[:n],
                                         gamma_prev[n:n + m - 1]]).clone())

        # Rebuild A at the new point (as the real forward does at iteration
        # end); it will be consumed by the NEXT iteration's factorization
        A = sparsified_hessian(T, eta, shift, density, n, m)
        niter = it + 1

    dual = gamma
    plan = plan_from_dual(M, dual, eta, n, m)

    if niter > 0:
        ctx = EmulatedContext(last["A"], last["s"], last["y"],
                              last["invA_y"], last["ys"], eta,
                              last["low_rank"], last["ctx_point"],
                              fallback=False)
    else:
        # Fallback: immediate convergence, factorize A at the final point,
        # no low-rank correction
        ctx = EmulatedContext(A, None, None, None, 0.0, eta, False,
                              torch.cat([gamma[:n], gamma[n:n + m - 1]]),
                              fallback=True)
    return plan, dual, niter, converged, ctx


# ---------------------------------------------------------------------------
# Validation levels
# ---------------------------------------------------------------------------

def run_L0(ctx, device, seed=0):
    """L0: apply_preconditioner(r) vs dense solve of B z = r."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    r = torch.randn(ctx.dim, generator=gen, dtype=torch.float64).to(device)
    B = build_precond_from_ctx(ctx, device)
    z_ctx = ctx.apply_preconditioner(r)
    z_ref = torch.linalg.solve(B, r)
    err = torch.linalg.norm(z_ctx - z_ref) / torch.linalg.norm(z_ref)
    return float(err)


def run_L1(ctx, device, seed=1):
    """L1: for random z0, r = B z0 (dense), check z = apply(r) recovers z0."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    z0 = torch.randn(ctx.dim, generator=gen, dtype=torch.float64).to(device)
    B = build_precond_from_ctx(ctx, device)
    r = B @ z0
    z = ctx.apply_preconditioner(r)
    err = torch.linalg.norm(z - z0) / torch.linalg.norm(z0)
    return float(err)


def run_L2(ctx, M, dual, G, eta, n, m, device):
    """L2: full backward gradient with PCG vs the exact dense solve."""
    T = plan_from_dual(M, dual, eta, n, m)
    mu = rhs_Q_scale(G, T, n, m)

    matvec = lambda v: q_matvec(T, v, n, m)
    precond = lambda r: ctx.apply_preconditioner(r)

    # Preconditioned PCG
    lam_pcg, it_pcg, conv_pcg = pcg(matvec, precond, mu)
    # Plain CG for comparison
    lam_cg, it_cg, conv_cg = pcg(matvec, None, mu)

    # Exact dense reference
    Q = dense_Q(T, n, m)
    lam_ref = torch.linalg.solve(Q, mu)

    err_lam = torch.linalg.norm(lam_pcg - lam_ref) / torch.linalg.norm(lam_ref)
    grad_pcg = assemble_grad(T, G, lam_pcg, eta, n, m)
    grad_ref = assemble_grad(T, G, lam_ref, eta, n, m)
    err_grad = torch.linalg.norm(grad_pcg - grad_ref) / torch.linalg.norm(grad_ref)

    return dict(
        err_lam=float(err_lam), err_grad=float(err_grad),
        it_pcg=it_pcg, conv_pcg=conv_pcg,
        it_cg=it_cg, conv_cg=conv_cg,
    )


def run_all(name, plan, dual, niter, converged, ctx, M, eta, n, m, device,
            forward=None):
    """Run the L0 / L1 / L2 ladder on one context."""
    print(f"\n=== {name} ===")
    print(f"forward: niter = {niter}, converged = {converged}")
    print(f"context: dim = {ctx.dim}, low_rank = {ctx.low_rank_enabled}, "
          f"fallback = {ctx.is_fallback}, reg = {ctx.reg}")
    if ctx.low_rank_enabled:
        print(f"context: ys = {ctx.ys:.6e}")

    err0 = run_L0(ctx, device)
    print(f"[L0] apply vs dense B solve:      rel err = {err0:.3e}")

    err1 = run_L1(ctx, device)
    print(f"[L1] consistency (r = B z0):      rel err = {err1:.3e}")

    gen = torch.Generator(device="cpu").manual_seed(2)
    G = torch.randn(n, m, generator=gen, dtype=torch.float64).to(device)
    res2 = run_L2(ctx, M, dual, G, eta, n, m, device)
    print(f"[L2] lambda (PCG vs dense):       rel err = {res2['err_lam']:.3e}")
    print(f"[L2] dL/dM (PCG vs dense):        rel err = {res2['err_grad']:.3e}")
    print(f"[L2] PCG iters: preconditioned = {res2['it_pcg']} "
          f"(converged={res2['conv_pcg']}), plain CG = {res2['it_cg']} "
          f"(converged={res2['conv_cg']})")

    ok = (err0 < 1e-8) and (err1 < 1e-8) and (res2["err_grad"] < 1e-8)
    print(f"[{name}] {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emulate", action="store_true",
                        help="pure-Python emulation mode (no CUDA module)")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--m", type=int, default=120)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.emulate:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        forward = emulate_forward
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("real mode requires CUDA")
        device = torch.device("cuda")
        forward = run_forward_real

    n, m, eta = args.n, args.m, args.eta
    M, a, b = make_problem(n, m, eta, args.seed, device)

    # --- Test 1: normal forward, context from the last iteration ---
    plan, dual, niter, converged, ctx = forward(
        M, a, b, eta, tol=args.tol, max_iter=args.max_iter
    )
    ok1 = run_all("Test 1 (normal)", plan, dual, niter, converged, ctx,
                  M, eta, n, m, device)

    # --- Test 2: fallback path (x0 at the optimum -> niter = 0) ---
    plan2, dual2, niter2, converged2, ctx2 = forward(
        M, a, b, eta, tol=args.tol, max_iter=args.max_iter, x0=dual
    )
    print(f"\n[fallback check] niter = {niter2}, "
          f"is_fallback = {ctx2.is_fallback}")
    ok2 = run_all("Test 2 (fallback)", plan2, dual2, niter2, converged2, ctx2,
                  M, eta, n, m, device)

    print("\n========================================")
    print(f"OVERALL: {'PASS' if (ok1 and ok2) else 'FAIL'}")


if __name__ == "__main__":
    main()
