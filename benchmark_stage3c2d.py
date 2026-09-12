#!/usr/bin/env python3
"""Stage 3C-2D: cheap structured preconditioner (paired 2x2 block-Jacobi)
vs diagonal vs CG vs reference SPLR points.

Protocol is IDENTICAL to Stage 3C-2C1 (same shared system per size, same
shared-HVP microbenchmark, same forward timing protocol with discarded
warmup + fresh-repeat median, same PCG math, same correctness guards) --
the helpers are imported from benchmark_stage3c2c.py, not reimplemented.

Variant audit note (Variant B, 2-block diagonal): the reduced Hessian
    Q = [[D_r, T_r], [T_r^T, D_c]]
has D_r = diag(T 1_m) and D_c = diag((T^T 1_n)_{-m}) -- both blocks are
ALREADY diagonal, so a 2-block diagonal preconditioner collapses exactly
to the ordinary diagonal. Variant B is therefore NOT implemented (it would
be a synonym of `diag`).

Variant C (block2): paired 2x2 block-Jacobi of Q(T_ref).
  * Pairing (deterministic, disjoint): for each alpha_i take the strongest
    beta coupling j*(i) = argmax_j |T_ij| (j over the m-1 reduced betas);
    process alphas in descending |T_ij*| (stable ties by index); a beta
    already taken skips the alpha (alpha falls back to diagonal); unmatched
    variables are diagonal. No variable appears in two blocks.
  * Block coefficients come from the EXACT system matrix entries
    a = (T 1_m)_i, d = ((T^T 1_n))_j, b = T_ij (T_ref, no CSR needed).
    A pair is accepted only if its determinant satisfies
    det = a*d - b*b > 1e-12 * a * d (positive-definiteness guard);
    otherwise the pair falls back to diagonal.
  * Everything (pair indices, inverse coefficients c00/c01/c11, and the
    diagonal fallback inv_diag_q) is precomputed ONCE at setup; each apply
    is gather -> 2x2 inverse -> scatter on top of a diagonal fallback.
    No sparse solve, no cuDSS, no C++/CUDA changes.

  NOTE: block2 uses exact Q entries from T_ref, while the `diag` baseline
  (unchanged from 3C-2C) uses the diagonal of the SHIFTED sparsified matrix
  A exported from the context. The difference is the eta*shift diagonal
  increment; it is documented here rather than harmonized, to keep Variant
  A untouched.

Usage:
  python benchmark_stage3c2d.py --quick
  python benchmark_stage3c2d.py \
      --sizes 100x120 300x360 500x600 1000x1200 \
      --repeat 5 --output stage3c2d_structured_precond.csv
"""

import argparse
import csv
import gc
import statistics
import time

import torch

from stage3b_validate import make_problem, dense_Q
from backward_core import (
    q_matvec,
    rhs_Q_scale,
    assemble_grad,
)
from benchmark_stage3c2c import (
    _sync,
    rel_err,
    g_seed_for,
    parse_size,
    run_pcg,
    run_forward,
    run_forward_timed_median,
    measure_shared_hvp,
    ctx_meta,
    extract_inv_diag,
    run_variant,
    _med,
)

CSV_FIELDS = [
    "n", "m", "dim",
    "variant",                      # CG | diag | block2 | SPLR
    "requested_density", "realized_density", "nnz_A",
    "low_rank", "ys",
    "repeat_id",
    "niter_forward", "converged_forward",
    "forward_context_ms",               # median over fresh forward repeats
    "forward_context_min_ms", "forward_context_max_ms", "forward_timing_repeats",
    "dual_drift_vs_ref",
    "shared_hvp_us",
    "precond_setup_ms", "num_pairs",
    "pcg_iterations", "pcg_converged",
    "total_pcg_wall_ms",            # production-style (ranking metric)
    "diag_hvp_ms", "diag_precond_ms", "diag_python_control_ms",
    "avg_hvp_us",                   # diagnostic-only
    "avg_precond_us",
    "relative_residual",
    "lambda_rel_err_vs_dense", "grad_rel_err_vs_dense",
    "iteration_reduction_ratio",
    "wall_speedup_vs_CG", "wall_speedup_vs_diag",
    "measured_prec_us", "break_even_prec_us", "prec_over_break_even",
    "setup_plus_pcg_ms",
    "e2e_incremental_ms",
]


# ---------------------------------------------------------------------------
# Variant C: paired 2x2 block-Jacobi (pure PyTorch, benchmark-only)
# ---------------------------------------------------------------------------

def build_block2(T_ref, n, m):
    """Precompute the paired 2x2 block-Jacobi preconditioner of Q(T_ref).

    Returns (precond_fn, setup_ms, num_pairs). Deterministic; disjoint
    pairs; unmatched variables use the exact diagonal of Q.
    """
    _sync()
    t0 = time.perf_counter()

    dim = n + m - 1
    device = T_ref.device
    rsum = T_ref.sum(dim=1)                 # Q_ii for alpha_i
    csum = T_ref.sum(dim=0)[: m - 1]        # Q_jj for beta_j (reduced)
    Tred = T_ref[:, : m - 1]

    # Strongest-edge pairing, greedy with disjoint betas.
    w, j = Tred.abs().max(dim=1)            # per-alpha strongest beta
    order = torch.argsort(-w, stable=True)  # descending weight, stable ties
    taken = torch.zeros(m - 1, dtype=torch.bool, device=device)
    ia_l, ib_l = [], []
    for i in order.tolist():
        jj = int(j[i])
        if not bool(taken[jj]):
            taken[jj] = True
            ia_l.append(i)
            ib_l.append(jj)
    ia = torch.tensor(ia_l, dtype=torch.int64, device=device)
    jb = torch.tensor(ib_l, dtype=torch.int64, device=device)  # local beta idx
    ib = jb + n  # global index of beta_j in the reduced (n + m - 1) vector

    inv_diag_q = 1.0 / torch.cat([rsum, csum])

    if ia.numel() > 0:
        a = rsum[ia]
        d = csum[jb]
        b = Tred[ia, jb]
        det = a * d - b * b
        # SPD guard: keep only safely positive-definite 2x2 blocks.
        ok = det > 1e-12 * (a * d)
        ia, ib = ia[ok], ib[ok]
        a, d, b, det = a[ok], d[ok], b[ok], det[ok]
        c00 = d / det
        c01 = -b / det
        c11 = a / det
    num_pairs = int(ia.numel())

    def precond_block2(r):
        z = inv_diag_q * r
        if num_pairs > 0:
            ri = r[ia]
            rj = r[ib]
            z[ia] = c00 * ri + c01 * rj
            z[ib] = c01 * ri + c11 * rj
        return z

    _sync()
    setup_ms = (time.perf_counter() - t0) * 1e3
    return precond_block2, setup_ms, num_pairs


# ---------------------------------------------------------------------------
# Derived metrics (extends 3C-2C1 with speedup_vs_diag and setup-aware cost)
# ---------------------------------------------------------------------------

def fill_derived(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["n"], r["m"]), []).append(r)
    for key in sorted(groups):
        grows = groups[key]
        cg = [r for r in grows if r["variant"] == "CG"]
        dg = [r for r in grows if r["variant"] == "diag"]
        if not cg:
            continue
        ncg = statistics.median(r["pcg_iterations"] for r in cg)
        tcg = statistics.median(r["total_pcg_wall_ms"] for r in cg)
        tdg = (statistics.median(r["total_pcg_wall_ms"] for r in dg)
               if dg else float("nan"))
        shared_hvp = grows[0]["shared_hvp_us"]
        for r in grows:
            np_ = r["pcg_iterations"]
            tot = r["total_pcg_wall_ms"]
            setup = r["precond_setup_ms"] or 0.0
            r["iteration_reduction_ratio"] = ncg / np_ if np_ else ""
            r["wall_speedup_vs_CG"] = tcg / tot if tot else ""
            r["wall_speedup_vs_diag"] = tdg / tot if tot else ""
            r["setup_plus_pcg_ms"] = setup + tot
            r["e2e_incremental_ms"] = \
                (r["forward_context_ms"] or 0.0) + setup + tot
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
# Per-size driver
# ---------------------------------------------------------------------------

def run_size(curegot, n, m, args, device):
    print(f"\n{'=' * 78}\n=== size {n}x{m} ===")
    M, a, b = make_problem(n, m, args.eta, args.seed, device)
    gen = torch.Generator(device="cpu").manual_seed(g_seed_for(n, m))
    G = torch.randn((n, m), generator=gen, dtype=torch.float64).to(device)

    # 0. discarded warmup forward (lazy-init absorption)
    out_w, _ = run_forward(
        curegot, M, a, b, eta=args.eta, tol=args.tol, max_iter=args.max_iter,
        density=args.reference_density,
        candidate_sinkhorn_iter=args.candidate_sinkhorn_iter,
        context_mode=args.context_mode,
    )
    del out_w
    gc.collect()
    torch.cuda.empty_cache()

    # 1. reference forward (median over fresh repeats; last ctx is official)
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
    print(f"  reference forward (d={args.reference_density}): niter={niter_ref} "
          f"dim={dim} nnz_A={nnz_ref} realized={realized_ref:.4f} "
          f"forward median={fc_ref_ms:.1f}ms [min {fc_ref_min:.1f}, "
          f"max {fc_ref_max:.1f}]")

    mu = rhs_Q_scale(G, T_ref, n, m)
    matvec = lambda v: q_matvec(T_ref, v, n, m)

    # 2. shared HVP microbenchmark
    shared_hvp_us = measure_shared_hvp(
        matvec, dim, device, warmup=args.hvp_warmup, repeat=args.hvp_repeat,
        seed=g_seed_for(n, m),
    )
    print(f"  shared_hvp_us = {shared_hvp_us:.1f}")

    # Dense oracle (only when affordable)
    lam_ref = grad_ref = None
    if dim <= args.max_dense_dim:
        Q = dense_Q(T_ref, n, m)
        lam_ref = torch.linalg.solve(Q, mu)
        grad_ref = assemble_grad(T_ref, G, lam_ref, args.eta, n, m)
        del Q

    # 3. cheap preconditioner setups (timed)
    _sync()
    t0 = time.perf_counter()
    inv_diag = extract_inv_diag(ctx_ref, dim)
    _sync()
    diag_setup_ms = (time.perf_counter() - t0) * 1e3

    precond_block2, block2_setup_ms, num_pairs = build_block2(T_ref, n, m)
    print(f"  block2: pairs={num_pairs}/{min(n, m - 1)} "
          f"setup={block2_setup_ms:.2f}ms  (diag setup={diag_setup_ms:.2f}ms)")

    ref_const = dict(
        niter_forward=niter_ref, converged_forward=conv_ref,
        forward_context_ms=fc_ref_ms,
        forward_context_min_ms=fc_ref_min, forward_context_max_ms=fc_ref_max,
        forward_timing_repeats=args.forward_timing_repeats,
        dual_drift_vs_ref=0.0, shared_hvp_us=shared_hvp_us,
    )

    all_rows = []

    # ---- CG ----
    all_rows += run_variant(
        n=n, m=m, dim=dim, variant="CG", precond=None, matvec=matvec, mu=mu,
        lam_ref=lam_ref, grad_ref=grad_ref, T_ref=T_ref, G=G, eta=args.eta,
        args=args,
        row_const=dict(n=n, m=m, dim=dim, precond_setup_ms=0.0, num_pairs=0,
                       **ref_const),
    )

    # ---- diag ----
    precond_diag = lambda r: inv_diag * r
    all_rows += run_variant(
        n=n, m=m, dim=dim, variant="diag", precond=precond_diag,
        matvec=matvec, mu=mu,
        lam_ref=lam_ref, grad_ref=grad_ref, T_ref=T_ref, G=G, eta=args.eta,
        args=args,
        row_const=dict(n=n, m=m, dim=dim, precond_setup_ms=diag_setup_ms,
                       num_pairs=0, **ref_const),
    )

    # ---- block2 ----
    all_rows += run_variant(
        n=n, m=m, dim=dim, variant="block2", precond=precond_block2,
        matvec=matvec, mu=mu,
        lam_ref=lam_ref, grad_ref=grad_ref, T_ref=T_ref, G=G, eta=args.eta,
        args=args,
        row_const=dict(n=n, m=m, dim=dim, precond_setup_ms=block2_setup_ms,
                       num_pairs=num_pairs, **ref_const),
    )

    # ---- SPLR reference points (default: 0.01 and 0.10) ----
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
              f"low_rank={lr_d} forward median={fc_ms:.1f}ms")
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
                           shared_hvp_us=shared_hvp_us,
                           precond_setup_ms=0.0, num_pairs=0),
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
# Lifetime stress
# ---------------------------------------------------------------------------

def run_lifetime_stress(curegot, args, device):
    n, m = args.sizes[0]
    density = args.densities[0]
    print(f"\n=== lifetime stress: {args.lifetime_iters}x "
          f"(create context -> CG + diag + block2 + SPLR PCG -> destroy) "
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
        precond_b2, _, npairs = build_block2(T, n, m)
        for precond in (None, lambda r: inv_diag * r, precond_b2,
                        lambda r: ctx.apply_preconditioner(r)):
            lam, _, _, _, _, _ = run_pcg(
                matvec, precond, mu, tol=args.pcg_tol,
                max_iter=args.pcg_max_iter, sync_per_op=False,
            )
            if not bool(torch.isfinite(lam).all()):
                raise RuntimeError(f"iteration {it}: non-finite lambda")
        del ctx, out, T, mu, inv_diag, precond_b2, lam
        gc.collect()
        torch.cuda.empty_cache()
    print("lifetime stress: PASS (no double free / invalid pointer / "
          "illegal memory access / cuDSS error)")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize(all_rows):
    groups = {}
    for r in all_rows:
        groups.setdefault((r["n"], r["m"]), []).append(r)

    for key in sorted(groups):
        grows = groups[key]
        n, m = key
        print("\n" + "=" * 110)
        print(f"{n}x{m}  (dim={grows[0]['dim']}, "
              f"shared_hvp={grows[0]['shared_hvp_us']:.1f}us)  -- medians "
              f"over repeats; total = production-style PCG wall")
        print("=" * 110)
        print(f"{'variant':>7} {'dens':>6} {'iter':>5} {'setup_ms':>9} "
              f"{'prec_us':>8} {'pcg_ms':>8} {'setup+pcg':>10} "
              f"{'spd/CG':>7} {'spd/diag':>8} {'prec/BE':>8}")
        print("-" * 110)

        def vkey(r):
            d = r["requested_density"]
            return ({"CG": 0, "diag": 1, "block2": 2, "SPLR": 3}[r["variant"]],
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
            pbe = _med(sub, "prec_over_break_even")
            print(f"{r['variant']:>7} {dens:>6} "
                  f"{_med(sub, 'pcg_iterations'):>5.0f} "
                  f"{_med(sub, 'precond_setup_ms'):>9.2f} "
                  f"{_med(sub, 'avg_precond_us'):>8.1f} "
                  f"{_med(sub, 'total_pcg_wall_ms'):>8.2f} "
                  f"{_med(sub, 'setup_plus_pcg_ms'):>10.2f} "
                  f"{_med(sub, 'wall_speedup_vs_CG'):>7.2f} "
                  f"{_med(sub, 'wall_speedup_vs_diag'):>8.2f} "
                  f"{pbe if pbe == pbe else float('nan'):>8.2f}")
    print("\nsuccess criteria: block2 wins if PCG total < diag, or "
          "setup+PCG < diag's;")
    print("prec/BE uses the shared HVP microbenchmark median "
          "(<1 worthwhile, >1 too expensive).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sizes", nargs="+", type=parse_size,
                   default=[(100, 120), (300, 360), (500, 600), (1000, 1200)])
    p.add_argument("--densities", nargs="+", type=float, default=[0.01, 0.10],
                   help="SPLR reference points only (no full sweep)")
    p.add_argument("--reference-density", type=float, default=0.10)
    p.add_argument("--eta", type=float, default=0.05)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--candidate-sinkhorn-iter", type=int, default=3)
    p.add_argument("--context-mode", default="adopt",
                   choices=["adopt", "snapshot"])
    p.add_argument("--pcg-tol", type=float, default=1e-12)
    p.add_argument("--pcg-max-iter", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--forward-timing-repeats", type=int, default=3)
    p.add_argument("--hvp-warmup", type=int, default=20)
    p.add_argument("--hvp-repeat", type=int, default=100)
    p.add_argument("--max-dense-dim", type=int, default=2500)
    p.add_argument("--lifetime-iters", type=int, default=20)
    p.add_argument("--output", default="stage3c2d_structured_precond.csv")
    p.add_argument("--quick", action="store_true",
                   help="100x120+500x600, repeat=3, fwd_reps=2, hvp_repeat=30, "
                        "lifetime=3")
    args = p.parse_args()

    if args.quick:
        args.sizes = [(100, 120), (500, 600)]
        args.repeat = 3
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
    print(f"repeat={args.repeat} forward_timing_repeats="
          f"{args.forward_timing_repeats} hvp_repeat={args.hvp_repeat} "
          f"seed={args.seed} output={args.output}")

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
    finally:
        csv_file.close()

    print(f"\nCSV written to {args.output} ({len(all_rows)} rows)")
    summarize(all_rows)
    run_lifetime_stress(curegot, args, device)


if __name__ == "__main__":
    main()
