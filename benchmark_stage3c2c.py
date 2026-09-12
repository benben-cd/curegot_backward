#!/usr/bin/env python3
"""Stage 3C-2C: preconditioner cost-iteration trade-off in the backward PCG.

Compares, on the SAME linear system Q(T*) lambda = mu per problem size:

  variant  CG    -- plain CG, no preconditioner (baseline)
  variant  diag  -- diagonal preconditioner z = inv_diag * r, where inv_diag
                    is extracted ONCE from the reference context's exported
                    CSR diagonal of the stored-scale matrix
                    A = eta*H_Omega + (eta*shift)*I (the matrix actually
                    factorized by cuDSS). No new matrix is invented; the
                    diagonal always exists in the CSR because of the shift.
  variant  SPLR  -- current preconditioner (sparsified Hessian factorization
                    + optional low-rank Woodbury), swept over density.

Fairness: one upstream gradient G per size (deterministic seed), one
reference forward (default density 0.10) defining T_ref and
mu = rhs_Q_scale(G, T_ref); every variant solves the same system with the
same PCG tolerance and zero initial guess. Per-density forwards only
PRODUCE the preconditioner context; their dual drift vs the reference is
recorded (systems agree to forward tolerance).

Timing (stage 3C-2C1 cleanup):
  * production-style (RANKING metric): end-to-end Python PCG wall without
    benchmark-inserted per-op sync. NOTE: this is NOT fully asynchronous --
    the PCG's scalar truth-value tests (pw <= 0, rnorm < tol*bnorm)
    implicitly synchronize host/device, and apply_preconditioner() itself
    synchronizes internally. The benchmark simply adds no EXTRA syncs;
    the metric is identical across variants and therefore fair.
  * diagnostic: per-op sync (same math), splits HVP / preconditioner /
    Python control. Explanation only.

  * shared HVP (3C-2C1): every variant on a size uses the SAME matvec
    v -> Q(T_ref) v, so variant-specific avg_hvp_us differences are pure
    measurement noise. The break-even analysis therefore uses one
    shared_hvp_us per size: a dedicated microbenchmark on the reference
    system (deterministic vector batch, warmup, >=100 synced samples,
    median). avg_hvp_us is kept as a diagnostic-only column and is NOT
    used for break-even.

  * forward timing (3C-2C1): each size starts with one DISCARDED warmup
    forward (absorbs CUDA/cuDSS lazy init and allocator warmup); every
    density's forward_context_ms is then the median over
    --forward-timing-repeats fresh forward/context constructions, keeping
    the LAST repetition's context for the PCG benchmark. The reference
    density follows the same protocol as every other density.

No production code is modified: the diagonal preconditioner reuses the
context's existing CSR export accessors.

Usage:
  python benchmark_stage3c2c.py --quick
  python benchmark_stage3c2c.py \
      --sizes 100x120 300x360 500x600 1000x1200 \
      --densities 0.01 0.02 0.05 0.10 0.20 0.50 \
      --repeat 5 --output stage3c2c_tradeoff.csv
"""

import argparse
import csv
import gc
import statistics
import time

import torch

# Reuse the verified Stage-3B matrix-free oracle. Do not reimplement.
from stage3b_validate import make_problem, dense_Q
from backward_core import (
    q_matvec,
    rhs_Q_scale,
    assemble_grad,
)

CSV_FIELDS = [
    "n", "m", "dim",
    "variant",                      # CG | diag | SPLR
    "requested_density", "realized_density", "nnz_A",
    "low_rank", "ys",
    "repeat_id",
    "niter_forward", "converged_forward",
    "forward_context_ms",               # median over fresh forward/context repeats
    "forward_context_min_ms", "forward_context_max_ms", "forward_timing_repeats",
    "dual_drift_vs_ref",
    "pcg_iterations", "pcg_converged",
    "total_pcg_wall_ms",            # production-style (ranking metric)
    "shared_hvp_us",                # one microbenchmark median per size
    "diag_hvp_ms", "diag_precond_ms", "diag_python_control_ms",
    "avg_hvp_us",                   # diagnostic-only; NOT used for break-even
    "avg_precond_us",
    "relative_residual",
    "lambda_rel_err_vs_dense", "grad_rel_err_vs_dense",
    "iteration_reduction_ratio", "wall_speedup_vs_CG",
    "measured_prec_us", "break_even_prec_us", "prec_over_break_even",
    "e2e_incremental_ms",
]


def _sync():
    torch.cuda.synchronize()


def rel_err(x, ref):
    denom = float(torch.linalg.norm(ref))
    num = float(torch.linalg.norm(x - ref))
    return num / denom if denom > 0 else num


def g_seed_for(n, m):
    """Deterministic per-problem upstream-gradient seed (one G per size,
    shared by every variant and density)."""
    return (1000003 * n + 10007 * m + 7) % (2**31 - 1)


def parse_size(s):
    try:
        n, m = s.lower().split("x")
        return int(n), int(m)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid size '{s}', expected NxM")


# ---------------------------------------------------------------------------
# PCG: ONE implementation, two timing modes (identical math)
# ---------------------------------------------------------------------------

def run_pcg(matvec, precond, b, tol, max_iter, sync_per_op):
    """PCG with exactly the same math as stage3b_validate.pcg /
    benchmark_stage3c1.timed_pcg.

    sync_per_op=True  -> diagnostic timing: CUDA sync around every matvec /
                         precond call; returns per-component times.
    sync_per_op=False -> production-style: sync only at start/end;
                         per-component times are meaningless (async) and
                         returned as 0.0; only total_ms is valid.
    """
    hvp_ms = 0.0
    precond_ms = 0.0

    def timed_matvec(v):
        nonlocal hvp_ms
        if not sync_per_op:
            return matvec(v)
        _sync()
        t0 = time.perf_counter()
        w = matvec(v)
        _sync()
        hvp_ms += (time.perf_counter() - t0) * 1e3
        return w

    def timed_precond(r):
        nonlocal precond_ms
        if not sync_per_op:
            return precond(r)
        _sync()
        t0 = time.perf_counter()
        z = precond(r)
        _sync()
        precond_ms += (time.perf_counter() - t0) * 1e3
        return z

    _sync()
    t_start = time.perf_counter()

    x = torch.zeros_like(b)
    r = b - timed_matvec(x)
    bnorm = torch.linalg.norm(r) if torch.linalg.norm(b) == 0 else torch.linalg.norm(b)
    z = r.clone() if precond is None else timed_precond(r)
    p = z.clone()
    rz = torch.dot(r, z)
    iters = 0
    converged = False
    for k in range(max_iter):
        w = timed_matvec(p)
        pw = torch.dot(p, w)
        if pw <= 0:
            raise RuntimeError("PCG breakdown: p^T A p <= 0 (matrix not SPD?)")
        alpha = rz / pw
        x = x + alpha * p
        r = r - alpha * w
        rnorm = torch.linalg.norm(r)
        if rnorm < tol * bnorm:
            iters = k + 1
            converged = True
            break
        z = r.clone() if precond is None else timed_precond(r)
        rz_new = torch.dot(r, z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    if not converged:
        iters = max_iter

    _sync()
    total_ms = (time.perf_counter() - t_start) * 1e3
    return x, iters, converged, total_ms, hvp_ms, precond_ms


# ---------------------------------------------------------------------------
# Forward / context helpers
# ---------------------------------------------------------------------------

def run_forward(curegot, M, a, b, *, eta, tol, max_iter, density,
                candidate_sinkhorn_iter, context_mode):
    """Timed forward + context creation (adopt mode by default)."""
    _sync()
    t0 = time.perf_counter()
    out = curegot.torch.sinkhorn_splr_bwd(
        M, a, b, eta,
        tol=tol, max_iter=max_iter, verbose=0,
        density=density,
        candidate_sinkhorn_iter=candidate_sinkhorn_iter,
        context_mode=context_mode,
    )
    _sync()
    ms = (time.perf_counter() - t0) * 1e3
    return out, ms


def run_forward_timed_median(curegot, M, a, b, *, eta, tol, max_iter, density,
                             candidate_sinkhorn_iter, context_mode, repeats):
    """`repeats` fresh forward/context constructions on the same problem and
    configuration. Returns (out, median_ms, min_ms, max_ms) where `out` is
    the LAST repetition's result -- its context is the only one kept alive
    and must be used (and released) by the caller. Earlier contexts are
    destroyed immediately so that cuDSS memory does not accumulate."""
    samples = []
    out = None
    for i in range(repeats):
        out_i, ms = run_forward(
            curegot, M, a, b, eta=eta, tol=tol, max_iter=max_iter,
            density=density,
            candidate_sinkhorn_iter=candidate_sinkhorn_iter,
            context_mode=context_mode,
        )
        samples.append(ms)
        if out is not None:
            del out
            gc.collect()
            torch.cuda.empty_cache()
        out = out_i
    return out, statistics.median(samples), min(samples), max(samples)


def measure_shared_hvp(matvec, dim, device, warmup, repeat, seed):
    """Dedicated HVP microbenchmark on the reference system: deterministic
    vector batch, explicit sync per sample, median single-call latency in
    microseconds. Shared by ALL variants of this size for the break-even
    analysis (the matvec is variant-independent)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    vbatch = torch.randn((repeat, dim), generator=gen,
                         dtype=torch.float64).to(device)
    for i in range(warmup):
        matvec(vbatch[i % repeat])
    _sync()
    samples = []
    for i in range(repeat):
        _sync()
        t0 = time.perf_counter()
        matvec(vbatch[i])
        _sync()
        samples.append((time.perf_counter() - t0) * 1e6)
    del vbatch
    return statistics.median(samples)


def ctx_meta(ctx):
    dim = int(ctx.dim)
    nnz_A = int(ctx.csr_rowptr()[-1].item())
    low_rank = bool(ctx.low_rank_enabled)
    ys = float(ctx.ys) if low_rank else float("nan")
    realized = nnz_A / (dim * (dim + 1) / 2.0)
    return dim, nnz_A, realized, low_rank, ys


def extract_inv_diag(ctx, dim):
    """inv_diag of the stored-scale matrix A from the context's exported CSR
    (lower triangle). Extracted ONCE per context; the PCG callback then only
    does an elementwise multiply. Raises if any row lacks a (positive)
    diagonal entry -- with the current shift this cannot happen."""
    rowptr = ctx.csr_rowptr().to(torch.int64)
    colind = ctx.csr_colind().to(torch.int64)
    values = ctx.csr_values()
    device = values.device
    counts = rowptr[1:] - rowptr[:-1]
    rowidx = torch.repeat_interleave(
        torch.arange(dim, device=device), counts
    )
    mask = colind == rowidx
    if int(mask.sum()) != dim:
        raise RuntimeError(
            f"CSR diagonal incomplete: {int(mask.sum())} entries for dim {dim}"
        )
    diag = values[mask]  # one per row, in row order
    if not bool((diag > 0).all()):
        raise RuntimeError("non-positive diagonal in stored-scale matrix A")
    return 1.0 / diag


# ---------------------------------------------------------------------------
# Variant runner
# ---------------------------------------------------------------------------

def run_variant(*, n, m, dim, variant, precond, matvec, mu, lam_ref, grad_ref,
                T_ref, G, eta, args, row_const):
    """Run one variant: production-style repeats + one diagnostic pass.
    Returns list of CSV rows (break-even fields filled later)."""
    rows = []
    lam_last = None
    for rep in range(args.repeat):
        lam, iters, conv, total_ms, _, _ = run_pcg(
            matvec, precond, mu, tol=args.pcg_tol,
            max_iter=args.pcg_max_iter, sync_per_op=False,
        )
        res_num = float(torch.linalg.norm(mu - matvec(lam)))
        res_den = float(torch.linalg.norm(mu))
        residual = res_num / res_den if res_den > 0 else res_num
        lam_err = rel_err(lam, lam_ref) if lam_ref is not None else ""
        if lam_err != "" and lam_err > 1e-8:
            raise RuntimeError(
                f"{variant}: lambda rel err vs dense {lam_err:.3e} > 1e-8"
            )
        row = {k: "" for k in CSV_FIELDS}
        row.update(row_const)
        row.update(
            variant=variant, repeat_id=rep,
            pcg_iterations=iters, pcg_converged=conv,
            total_pcg_wall_ms=total_ms,
            relative_residual=residual,
            lambda_rel_err_vs_dense=lam_err,
        )
        rows.append(row)
        lam_last = lam

    # grad check once (last lambda)
    if grad_ref is not None:
        grad = assemble_grad(T_ref, G, lam_last, eta, n, m)
        grad_err = rel_err(grad, grad_ref)
        if grad_err > 1e-8:
            raise RuntimeError(
                f"{variant}: grad rel err vs dense {grad_err:.3e} > 1e-8"
            )
        for row in rows:
            row["grad_rel_err_vs_dense"] = grad_err

    # diagnostic pass (per-op sync), once
    _, iters_d, _, total_d, hvp_d, precond_d = run_pcg(
        matvec, precond, mu, tol=args.pcg_tol,
        max_iter=args.pcg_max_iter, sync_per_op=True,
    )
    for row in rows:
        row["diag_hvp_ms"] = hvp_d
        row["diag_precond_ms"] = precond_d
        row["diag_python_control_ms"] = total_d - hvp_d - precond_d
        row["avg_hvp_us"] = hvp_d * 1e3 / max(iters_d, 1)
        row["avg_precond_us"] = precond_d * 1e3 / max(iters_d, 1)
        row["measured_prec_us"] = row["avg_precond_us"]
    return rows


# ---------------------------------------------------------------------------
# Per-size driver
# ---------------------------------------------------------------------------

def run_size(curegot, n, m, args, device):
    print(f"\n{'=' * 78}\n=== size {n}x{m} ===")
    M, a, b = make_problem(n, m, args.eta, args.seed, device)
    gen = torch.Generator(device="cpu").manual_seed(g_seed_for(n, m))
    G = torch.randn((n, m), generator=gen, dtype=torch.float64).to(device)

    # ---- 0. discarded warmup forward: absorbs CUDA/cuDSS lazy init and
    #         allocator warmup so the reference density is not penalized ----
    out_w, _ = run_forward(
        curegot, M, a, b, eta=args.eta, tol=args.tol, max_iter=args.max_iter,
        density=args.reference_density,
        candidate_sinkhorn_iter=args.candidate_sinkhorn_iter,
        context_mode=args.context_mode,
    )
    del out_w
    gc.collect()
    torch.cuda.empty_cache()

    # ---- 1. reference forward (median over fresh repeats; the LAST
    #         repetition's context is the official one) ----
    out_ref, fc_ref_ms, fc_ref_min, fc_ref_max = run_forward_timed_median(
        curegot, M, a, b, eta=args.eta, tol=args.tol, max_iter=args.max_iter,
        density=args.reference_density,
        candidate_sinkhorn_iter=args.candidate_sinkhorn_iter,
        context_mode=args.context_mode, repeats=args.forward_timing_repeats,
    )
    T_ref = out_ref["plan"]
    dual_ref = out_ref["dual"]
    niter_ref = int(out_ref["niter"])
    conv_ref = bool(out_ref["converged"])
    ctx_ref = out_ref["context"]
    if niter_ref == 0 or bool(ctx_ref.is_fallback):
        raise RuntimeError("reference forward hit the niter=0 fallback; "
                           "not a benchmark case")
    dim, nnz_ref, realized_ref, lr_ref, ys_ref = ctx_meta(ctx_ref)
    print(f"  reference forward (d={args.reference_density}): "
          f"niter={niter_ref} dim={dim} nnz_A={nnz_ref} "
          f"realized={realized_ref:.4f} low_rank={lr_ref} "
          f"forward median={fc_ref_ms:.1f}ms "
          f"[min {fc_ref_min:.1f}, max {fc_ref_max:.1f}, "
          f"{args.forward_timing_repeats} reps]")

    mu = rhs_Q_scale(G, T_ref, n, m)
    matvec = lambda v: q_matvec(T_ref, v, n, m)

    # ---- 2. shared HVP microbenchmark (one value per size) ----
    shared_hvp_us = measure_shared_hvp(
        matvec, dim, device, warmup=args.hvp_warmup, repeat=args.hvp_repeat,
        seed=g_seed_for(n, m),
    )
    print(f"  shared_hvp_us = {shared_hvp_us:.1f} "
          f"(warmup={args.hvp_warmup}, repeat={args.hvp_repeat})")

    # Dense oracle (only when affordable)
    lam_ref = grad_ref = None
    if dim <= args.max_dense_dim:
        Q = dense_Q(T_ref, n, m)
        lam_ref = torch.linalg.solve(Q, mu)
        grad_ref = assemble_grad(T_ref, G, lam_ref, args.eta, n, m)
        del Q
    else:
        print(f"  dim={dim} > max_dense_dim={args.max_dense_dim}: "
              f"dense oracle skipped")

    inv_diag = extract_inv_diag(ctx_ref, dim)

    all_rows = []

    ref_const = dict(
        niter_forward=niter_ref, converged_forward=conv_ref,
        forward_context_ms=fc_ref_ms,
        forward_context_min_ms=fc_ref_min, forward_context_max_ms=fc_ref_max,
        forward_timing_repeats=args.forward_timing_repeats,
        dual_drift_vs_ref=0.0, shared_hvp_us=shared_hvp_us,
    )

    # ---- variant: plain CG ----
    all_rows += run_variant(
        n=n, m=m, dim=dim, variant="CG", precond=None, matvec=matvec, mu=mu,
        lam_ref=lam_ref, grad_ref=grad_ref, T_ref=T_ref, G=G, eta=args.eta,
        args=args,
        row_const=dict(n=n, m=m, dim=dim, **ref_const),
    )

    # ---- variant: diagonal ----
    if not args.skip_diag:
        precond_diag = lambda r: inv_diag * r
        all_rows += run_variant(
            n=n, m=m, dim=dim, variant="diag", precond=precond_diag,
            matvec=matvec, mu=mu,
            lam_ref=lam_ref, grad_ref=grad_ref, T_ref=T_ref, G=G,
            eta=args.eta, args=args,
            row_const=dict(n=n, m=m, dim=dim, **ref_const),
        )

    # ---- variant: SPLR per density ----
    for density in args.densities:
        if abs(density - args.reference_density) < 1e-12:
            ctx = ctx_ref
            fc_ms, fc_min, fc_max = fc_ref_ms, fc_ref_min, fc_ref_max
            niter, conv = niter_ref, conv_ref
            drift = 0.0
            dim_d, nnz_d, realized_d, lr_d, ys_d = \
                dim, nnz_ref, realized_ref, lr_ref, ys_ref
        else:
            out_d, fc_ms, fc_min, fc_max = run_forward_timed_median(
                curegot, M, a, b, eta=args.eta, tol=args.tol,
                max_iter=args.max_iter, density=density,
                candidate_sinkhorn_iter=args.candidate_sinkhorn_iter,
                context_mode=args.context_mode,
                repeats=args.forward_timing_repeats,
            )
            ctx = out_d["context"]
            niter = int(out_d["niter"])
            conv = bool(out_d["converged"])
            if niter == 0 or bool(ctx.is_fallback):
                del ctx, out_d
                raise RuntimeError(
                    f"density {density}: niter=0 fallback; not a benchmark case"
                )
            drift = rel_err(out_d["dual"], dual_ref)
            dim_d, nnz_d, realized_d, lr_d, ys_d = ctx_meta(ctx)
            del out_d

        print(f"  SPLR d={density}: nnz_A={nnz_d} realized={realized_d:.4f} "
              f"low_rank={lr_d} dual_drift={drift:.2e} "
              f"forward median={fc_ms:.1f}ms [min {fc_min:.1f}, "
              f"max {fc_max:.1f}]")
        precond = lambda r: ctx.apply_preconditioner(r)
        all_rows += run_variant(
            n=n, m=m, dim=dim_d, variant="SPLR", precond=precond,
            matvec=matvec, mu=mu,
            lam_ref=lam_ref, grad_ref=grad_ref, T_ref=T_ref, G=G,
            eta=args.eta, args=args,
            row_const=dict(n=n, m=m, dim=dim_d,
                           requested_density=density,
                           realized_density=realized_d, nnz_A=nnz_d,
                           low_rank=lr_d, ys=ys_d,
                           niter_forward=niter, converged_forward=conv,
                           forward_context_ms=fc_ms,
                           forward_context_min_ms=fc_min,
                           forward_context_max_ms=fc_max,
                           forward_timing_repeats=args.forward_timing_repeats,
                           dual_drift_vs_ref=drift,
                           shared_hvp_us=shared_hvp_us),
        )
        if ctx is not ctx_ref:
            del precond, ctx
            gc.collect()
            torch.cuda.empty_cache()

    del ctx_ref, out_ref, inv_diag, T_ref, dual_ref, mu, M, a, b, G
    gc.collect()
    torch.cuda.empty_cache()
    return all_rows


# ---------------------------------------------------------------------------
# Derived metrics (break-even etc.), filled per size after all variants ran
# ---------------------------------------------------------------------------

def fill_derived(rows):
    """Fill iteration_reduction_ratio / wall_speedup_vs_CG / break-even
    fields, per (n, m) group. CG is the baseline.

    Break-even uses the SHARED HVP cost (one microbenchmark median per
    size), not the variant-specific diagnostic avg_hvp_us: the matvec is
    identical for every variant of a size, so variant-level differences
    are measurement noise (Stage 3C-2C1)."""
    groups = {}
    for r in rows:
        groups.setdefault((r["n"], r["m"]), []).append(r)
    for key in sorted(groups):
        grows = groups[key]
        cg = [r for r in grows if r["variant"] == "CG"]
        if not cg:
            continue
        ncg = statistics.median(r["pcg_iterations"] for r in cg)
        tcg = statistics.median(r["total_pcg_wall_ms"] for r in cg)
        shared_hvp = grows[0]["shared_hvp_us"]
        for r in grows:
            np_ = r["pcg_iterations"]
            tot = r["total_pcg_wall_ms"]
            r["iteration_reduction_ratio"] = ncg / np_ if np_ else ""
            r["wall_speedup_vs_CG"] = tcg / tot if tot else ""
            r["e2e_incremental_ms"] = tot + (r["forward_context_ms"] or 0.0)
            if r["variant"] == "CG":
                r["break_even_prec_us"] = ""
                r["prec_over_break_even"] = ""
            else:
                be = (ncg / np_ - 1.0) * shared_hvp if np_ else ""
                r["break_even_prec_us"] = be
                r["prec_over_break_even"] = (
                    r["measured_prec_us"] / be if be and be > 0 else ""
                )


# ---------------------------------------------------------------------------
# Lifetime stress
# ---------------------------------------------------------------------------

def run_lifetime_stress(curegot, args, device):
    n, m = args.sizes[0]
    density = args.densities[0]
    print(f"\n=== lifetime stress: {args.lifetime_iters}x "
          f"(create context -> CG + diag + SPLR PCG -> destroy) "
          f"on {n}x{m} d={density} ===")
    M, a, b = make_problem(n, m, args.eta, args.seed, device)
    gen = torch.Generator(device="cpu").manual_seed(g_seed_for(n, m))
    G = torch.randn((n, m), generator=gen, dtype=torch.float64).to(device)
    for it in range(args.lifetime_iters):
        out, _ = run_forward(
            curegot, M, a, b, eta=args.eta, tol=args.tol,
            max_iter=args.max_iter, density=density,
            candidate_sinkhorn_iter=args.candidate_sinkhorn_iter,
            context_mode=args.context_mode,
        )
        ctx = out["context"]
        T = out["plan"]
        dim = int(ctx.dim)
        mu = rhs_Q_scale(G, T, n, m)
        matvec = lambda v: q_matvec(T, v, n, m)
        inv_diag = extract_inv_diag(ctx, dim)
        for precond in (None, lambda r: inv_diag * r,
                        lambda r: ctx.apply_preconditioner(r)):
            lam, _, _, _, _, _ = run_pcg(
                matvec, precond, mu, tol=args.pcg_tol,
                max_iter=args.pcg_max_iter, sync_per_op=False,
            )
            if not bool(torch.isfinite(lam).all()):
                raise RuntimeError(f"iteration {it}: non-finite lambda")
        del ctx, out, T, mu, inv_diag, lam
        gc.collect()
        torch.cuda.empty_cache()
    print("lifetime stress: PASS (no double free / invalid pointer / "
          "illegal memory access / cuDSS error)")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _med(rows, key):
    vals = [r[key] for r in rows if r[key] != ""]
    return statistics.median(vals) if vals else float("nan")


def summarize(all_rows):
    groups = {}
    for r in all_rows:
        groups.setdefault((r["n"], r["m"]), []).append(r)

    for key in sorted(groups):
        grows = groups[key]
        n, m = key
        print("\n" + "=" * 96)
        print(f"{n}x{m}  (dim={grows[0]['dim']}, "
              f"shared_hvp={grows[0]['shared_hvp_us']:.1f}us)  -- medians "
              f"over repeats; total = production-style PCG wall")
        print("=" * 96)
        print(f"{'variant':>7} {'dens':>6} {'iter':>5} {'prec_us':>9} "
              f"{'total_ms':>9} {'spd_vs_CG':>9} "
              f"{'iter_red':>8} {'BE_prec_us':>10} {'prec/BE':>8} "
              f"{'e2e_ms':>9} {'fwd_ms':>8}")
        print("-" * 96)
        def vkey(r):
            d = r["requested_density"]
            return ({"CG": 0, "diag": 1, "SPLR": 2}[r["variant"]],
                    d if d != "" else -1.0)
        seen = set()
        for r in sorted(grows, key=vkey):
            vtag = (r["variant"], r["requested_density"])
            if vtag in seen:
                continue
            seen.add(vtag)
            sub = [x for x in grows
                   if x["variant"] == r["variant"]
                   and x["requested_density"] == r["requested_density"]]
            dens = (f"{r['requested_density']:.2f}"
                    if r["requested_density"] != "" else "-")
            be = _med(sub, "break_even_prec_us")
            pbe = _med(sub, "prec_over_break_even")
            print(f"{r['variant']:>7} {dens:>6} "
                  f"{_med(sub, 'pcg_iterations'):>5.0f} "
                  f"{_med(sub, 'avg_precond_us'):>9.1f} "
                  f"{_med(sub, 'total_pcg_wall_ms'):>9.2f} "
                  f"{_med(sub, 'wall_speedup_vs_CG'):>9.2f} "
                  f"{_med(sub, 'iteration_reduction_ratio'):>8.2f} "
                  f"{be:>10.1f} "
                  f"{pbe if pbe == pbe else float('nan'):>8.2f} "
                  f"{_med(sub, 'e2e_incremental_ms'):>9.2f} "
                  f"{_med(sub, 'forward_context_ms'):>8.1f}")
    print("\nColumns: iter_red = CG_iters/PCG_iters; "
          "BE_prec_us = (iter_red - 1) * shared_hvp_us;")
    print("prec/BE < 1 -> preconditioner theoretically worthwhile; "
          "> 1 -> too expensive for its iteration reduction.")
    print("e2e_ms = forward_context_ms (median of fresh repeats) + PCG total; "
          "fwd_ms = that median alone.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sizes", nargs="+", type=parse_size,
                   default=[(100, 120), (300, 360), (500, 600), (1000, 1200)],
                   help="problem sizes as NxM (add 2000x2400 if resources allow)")
    p.add_argument("--densities", nargs="+", type=float,
                   default=[0.01, 0.02, 0.05, 0.10, 0.20, 0.50],
                   help="SPLR density sweep (candidate sparsification budget)")
    p.add_argument("--reference-density", type=float, default=0.10,
                   help="density of the reference forward that defines the "
                        "shared system Q(T_ref) lambda = mu")
    p.add_argument("--eta", type=float, default=0.05)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--candidate-sinkhorn-iter", type=int, default=3)
    p.add_argument("--context-mode", default="adopt",
                   choices=["adopt", "snapshot"],
                   help="KEEP adopt (the Stage 3B.2-C2 production path)")
    p.add_argument("--pcg-tol", type=float, default=1e-12)
    p.add_argument("--pcg-max-iter", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeat", type=int, default=5,
                   help="production-style PCG repeats per variant")
    p.add_argument("--forward-timing-repeats", type=int, default=3,
                   help="fresh forward/context constructions per density; "
                        "forward_context_ms is the median, the LAST "
                        "repetition's context is kept for the PCG benchmark")
    p.add_argument("--hvp-warmup", type=int, default=20)
    p.add_argument("--hvp-repeat", type=int, default=100,
                   help="synced samples for the shared HVP microbenchmark")
    p.add_argument("--skip-diag", action="store_true")
    p.add_argument("--max-dense-dim", type=int, default=2500,
                   help="skip the dense lambda/grad oracle above this dim")
    p.add_argument("--lifetime-iters", type=int, default=20)
    p.add_argument("--output", default="stage3c2c_tradeoff.csv")
    p.add_argument("--quick", action="store_true",
                   help="smoke test: 100x120+500x600, densities "
                        "0.05+0.10+0.50, repeat=2, lifetime=3")
    args = p.parse_args()

    if args.quick:
        args.sizes = [(100, 120), (500, 600)]
        args.densities = [0.05, 0.10, 0.50]
        args.repeat = 2
        args.forward_timing_repeats = 2
        args.hvp_repeat = 30
        args.lifetime_iters = 3

    if not torch.cuda.is_available():
        raise RuntimeError("benchmark requires CUDA")
    device = torch.device("cuda")

    import curegot  # noqa: E402

    print(f"sizes={args.sizes} densities={args.densities} "
          f"reference_density={args.reference_density} eta={args.eta} "
          f"tol={args.tol} context_mode={args.context_mode}")
    print(f"pcg_tol={args.pcg_tol} pcg_max_iter={args.pcg_max_iter} "
          f"repeat={args.repeat} seed={args.seed} output={args.output}")

    all_rows = []
    csv_file = open(args.output, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    writer.writeheader()
    csv_file.flush()

    try:
        for (n, m) in args.sizes:
            rows = run_size(curegot, n, m, args, device)
            fill_derived(rows)
            for row in rows:
                writer.writerow(row)
                all_rows.append(row)
            csv_file.flush()
            cg = [r for r in rows if r["variant"] == "CG"]
            print(f"  CG: iters={_med(cg, 'pcg_iterations'):.0f} "
                  f"total={_med(cg, 'total_pcg_wall_ms'):.2f}ms")
    finally:
        csv_file.close()

    print(f"\nCSV written to {args.output} ({len(all_rows)} rows)")
    summarize(all_rows)
    run_lifetime_stress(curegot, args, device)


if __name__ == "__main__":
    main()
