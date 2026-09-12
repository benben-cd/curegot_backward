#!/usr/bin/env python3
"""Stage 3C-3: native diagonal preconditioner in the backward PCG path.

Compares, on the SAME shared system per size (same forward problem, same
reference T*, same G, same mu, same PCG tolerance, same zero initial
lambda):

  CG          plain CG (no preconditioner)
  python_diag Stage 3C-2C Python callback: z = inv_diag_py * r (reference)
  diag        NATIVE diagonal: ctx.apply_diagonal_preconditioner(r)
  SPLR        ctx.apply_preconditioner(r) (cuDSS + rank-2, production path)

Key validation: native diag must agree with the Python callback
elementwise (relative error <= 1e-12; in practice bitwise identical, both
compute 1/value * r in float64 on the same CSR values) and must not be
slower than the Python callback.

Timing protocol is the Stage 3C-2C1 one (discarded warmup forward, fresh-
repeat median for forward timing, one shared system per size, dual PCG
timing modes). The SPLR/cuDSS apply synchronizes internally; the native
diagonal apply is fully asynchronous on the PyTorch current stream (the
PCG's scalar truth tests synchronize implicitly). This difference is
intentional and documented, not harmonized.

Usage:
  python benchmark_stage3c3.py --quick
  python benchmark_stage3c3.py \
      --sizes 100x120 300x360 500x600 1000x1200 \
      --repeat 5 --output stage3c3_native_diag.csv
"""

import argparse
import csv
import gc
import statistics

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
    ctx_meta,
    extract_inv_diag,
    _med,
)

CSV_FIELDS = [
    "n", "m", "dim",
    "variant",                      # CG | python_diag | diag | SPLR
    "requested_density", "realized_density", "nnz_A",
    "low_rank", "ys",
    "repeat_id",
    "niter_forward", "converged_forward",
    "forward_context_ms",
    "forward_context_min_ms", "forward_context_max_ms", "forward_timing_repeats",
    "diag_setup_us",                # native extraction, median of repeats
    "native_vs_python_invdiag_relerr",
    "native_vs_python_apply_relerr",
    "pcg_iterations", "pcg_converged",
    "total_backward_ms",            # production-style PCG wall
    "diag_hvp_ms", "diag_precond_ms", "diag_python_control_ms",
    "avg_preconditioner_us",
    "relative_residual",
    "lambda_rel_err_vs_dense", "grad_rel_err_vs_dense",
    "speedup_vs_CG", "speedup_vs_python_diag", "speedup_vs_SPLR",
]

VARIANTS = ["CG", "python_diag", "diag", "SPLR"]


def make_precond(ctx, mode, inv_diag_py=None):
    """String-mode preconditioner selector (production API semantics).

    "diag" maps to the NATIVE implementation -- the Python callback
    ("python_diag") exists only as a benchmark reference.
    """
    if mode == "none":
        return None
    if mode == "diag":
        return lambda r: ctx.apply_diagonal_preconditioner(r)
    if mode == "python_diag":
        if inv_diag_py is None:
            raise ValueError("python_diag requires inv_diag_py")
        return lambda r: inv_diag_py * r
    if mode == "splr":
        return lambda r: ctx.apply_preconditioner(r)
    raise ValueError(
        f"invalid preconditioner mode '{mode}'; "
        "expected one of: none | diag | splr"
    )


def run_variant(n, m, dim, variant, precond, matvec, mu, lam_ref, grad_ref,
                T_ref, G, eta, args, row_const):
    """Production-style repeats + one diagnostic pass; correctness guards."""
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
            total_backward_ms=total_ms,
            relative_residual=residual,
            lambda_rel_err_vs_dense=lam_err,
        )
        rows.append(row)
        lam_last = lam

    if grad_ref is not None:
        grad = assemble_grad(T_ref, G, lam_last, eta, n, m)
        grad_err = rel_err(grad, grad_ref)
        if grad_err > 1e-8:
            raise RuntimeError(
                f"{variant}: grad rel err vs dense {grad_err:.3e} > 1e-8"
            )
        for row in rows:
            row["grad_rel_err_vs_dense"] = grad_err

    _, iters_d, _, total_d, hvp_d, precond_d = run_pcg(
        matvec, precond, mu, tol=args.pcg_tol,
        max_iter=args.pcg_max_iter, sync_per_op=True,
    )
    for row in rows:
        row["diag_hvp_ms"] = hvp_d
        row["diag_precond_ms"] = precond_d
        row["diag_python_control_ms"] = total_d - hvp_d - precond_d
        row["avg_preconditioner_us"] = precond_d * 1e3 / max(iters_d, 1)
    return rows


def fill_speedups(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["n"], r["m"]), []).append(r)
    for key in sorted(groups):
        grows = groups[key]
        med = {}
        for v in VARIANTS:
            sub = [r for r in grows if r["variant"] == v]
            med[v] = (statistics.median(r["total_backward_ms"] for r in sub)
                      if sub else float("nan"))
        for r in grows:
            tot = r["total_backward_ms"]
            for v, col in (("CG", "speedup_vs_CG"),
                           ("python_diag", "speedup_vs_python_diag"),
                           ("SPLR", "speedup_vs_SPLR")):
                r[col] = med[v] / tot if tot and med[v] == med[v] else ""


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
    out_ref, fc_ms, fc_min, fc_max = run_forward_timed_median(
        curegot, M, a, b, eta=args.eta, tol=args.tol, max_iter=args.max_iter,
        density=args.reference_density,
        candidate_sinkhorn_iter=args.candidate_sinkhorn_iter,
        context_mode=args.context_mode, repeats=args.forward_timing_repeats,
    )
    T_ref = out_ref["plan"]
    niter_ref = int(out_ref["niter"])
    conv_ref = bool(out_ref["converged"])
    ctx_ref = out_ref["context"]
    if niter_ref == 0 or bool(ctx_ref.is_fallback):
        raise RuntimeError("reference forward hit the niter=0 fallback; "
                           "not a benchmark case")
    dim, nnz_A, realized, low_rank, ys = ctx_meta(ctx_ref)
    print(f"  reference forward (d={args.reference_density}): "
          f"niter={niter_ref} dim={dim} nnz_A={nnz_A} "
          f"realized={realized:.4f} low_rank={low_rank} "
          f"forward median={fc_ms:.1f}ms")

    mu = rhs_Q_scale(G, T_ref, n, m)
    matvec = lambda v: q_matvec(T_ref, v, n, m)

    # 2. native vs Python diagonal consistency (elementwise, <= 1e-12)
    if not bool(ctx_ref.has_diagonal_preconditioner):
        raise RuntimeError("native diagonal preconditioner not set up "
                           "by the forward call")
    inv_diag_py = extract_inv_diag(ctx_ref, dim)
    inv_diag_nat = ctx_ref.inv_diag()
    invdiag_err = rel_err(inv_diag_nat, inv_diag_py)
    gen_r = torch.Generator(device="cpu").manual_seed(g_seed_for(n, m) + 1)
    r_test = torch.randn(dim, generator=gen_r, dtype=torch.float64).to(device)
    z_nat = ctx_ref.apply_diagonal_preconditioner(r_test)
    z_py = inv_diag_py * r_test
    apply_err = rel_err(z_nat, z_py)
    print(f"  native vs python: inv_diag rel err = {invdiag_err:.3e}, "
          f"apply rel err = {apply_err:.3e} (require <= 1e-12)")
    if invdiag_err > 1e-12 or apply_err > 1e-12:
        raise RuntimeError(
            f"native diagonal mismatch: inv_diag {invdiag_err:.3e}, "
            f"apply {apply_err:.3e}"
        )

    # 3. one-time setup cost (extraction kernel + validation; the d_inv_diag
    #    allocation itself is part of context construction)
    diag_setup_us = ctx_ref.benchmark_diag_setup(args.setup_repeats)
    print(f"  diag_setup_us = {diag_setup_us:.1f} "
          f"(median of {args.setup_repeats} validated extractions)")

    # Dense oracle (only when affordable)
    lam_ref = grad_ref = None
    if dim <= args.max_dense_dim:
        Q = dense_Q(T_ref, n, m)
        lam_ref = torch.linalg.solve(Q, mu)
        grad_ref = assemble_grad(T_ref, G, lam_ref, args.eta, n, m)
        del Q

    row_const = dict(
        n=n, m=m, dim=dim,
        requested_density=args.reference_density,
        realized_density=realized, nnz_A=nnz_A,
        low_rank=low_rank, ys=ys,
        niter_forward=niter_ref, converged_forward=conv_ref,
        forward_context_ms=fc_ms,
        forward_context_min_ms=fc_min, forward_context_max_ms=fc_max,
        forward_timing_repeats=args.forward_timing_repeats,
        diag_setup_us=diag_setup_us,
        native_vs_python_invdiag_relerr=invdiag_err,
        native_vs_python_apply_relerr=apply_err,
    )

    all_rows = []
    for variant in VARIANTS:
        mode = {"CG": "none", "python_diag": "python_diag",
                "diag": "diag", "SPLR": "splr"}[variant]
        precond = make_precond(ctx_ref, mode, inv_diag_py)
        all_rows += run_variant(
            n, m, dim, variant, precond, matvec, mu, lam_ref, grad_ref,
            T_ref, G, args.eta, args, row_const,
        )
        print(f"  {variant:<11} iters={all_rows[-1]['pcg_iterations']:>4} "
              f"total={all_rows[-1]['total_backward_ms']:.2f}ms "
              f"(median over {args.repeat} repeats in CSV)")

    del ctx_ref, out_ref, T_ref, mu, M, a, b, G, inv_diag_py, inv_diag_nat
    gc.collect()
    torch.cuda.empty_cache()
    return all_rows


def run_lifetime_stress(curegot, args, device):
    n, m = args.sizes[0]
    print(f"\n=== lifetime stress: {args.lifetime_iters}x (create context -> "
          f"200x native diag + 10x SPLR apply -> destroy) on {n}x{m} ===")
    M, a, b = make_problem(n, m, args.eta, args.seed, device)
    for it in range(args.lifetime_iters):
        out, _ = run_forward(
            curegot, M, a, b, eta=args.eta, tol=args.tol,
            max_iter=args.max_iter, density=args.reference_density,
            candidate_sinkhorn_iter=args.candidate_sinkhorn_iter,
            context_mode=args.context_mode,
        )
        ctx = out["context"]
        dim = int(ctx.dim)
        r = torch.randn(dim, dtype=torch.float64, device=device)
        for _ in range(200):
            z = ctx.apply_diagonal_preconditioner(r)
        for _ in range(10):
            z2 = ctx.apply_preconditioner(r)
        _sync()
        if not (bool(torch.isfinite(z).all())
                and bool(torch.isfinite(z2).all())):
            raise RuntimeError(f"iteration {it}: non-finite output")
        del ctx, out, r, z, z2
        gc.collect()
        torch.cuda.empty_cache()
    print("lifetime stress: PASS (no double free / invalid pointer / "
          "illegal memory access / race / NaN / Inf)")


def summarize(all_rows):
    groups = {}
    for r in all_rows:
        groups.setdefault((r["n"], r["m"]), []).append(r)
    for key in sorted(groups):
        grows = groups[key]
        n, m = key
        print("\n" + "=" * 104)
        print(f"{n}x{m}  (dim={grows[0]['dim']}, "
              f"diag_setup={grows[0]['diag_setup_us']:.1f}us)  -- medians "
              f"over repeats; total = production-style PCG wall")
        print("=" * 104)
        print(f"{'variant':>11} {'iter':>5} {'prec_us':>9} {'total_ms':>9} "
              f"{'spd/CG':>7} {'spd/pydiag':>10} {'spd/SPLR':>9}")
        print("-" * 104)
        for v in VARIANTS:
            sub = [r for r in grows if r["variant"] == v]
            if not sub:
                continue
            print(f"{v:>11} "
                  f"{_med(sub, 'pcg_iterations'):>5.0f} "
                  f"{_med(sub, 'avg_preconditioner_us'):>9.1f} "
                  f"{_med(sub, 'total_backward_ms'):>9.2f} "
                  f"{_med(sub, 'speedup_vs_CG'):>7.2f} "
                  f"{_med(sub, 'speedup_vs_python_diag'):>10.2f} "
                  f"{_med(sub, 'speedup_vs_SPLR'):>9.2f}")
    print("\nkey check: speedup_vs_python_diag(diag) >= 1 -- native must not "
          "be slower than the Python callback.")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sizes", nargs="+", type=parse_size,
                   default=[(100, 120), (300, 360), (500, 600), (1000, 1200)])
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
    p.add_argument("--setup-repeats", type=int, default=50)
    p.add_argument("--max-dense-dim", type=int, default=2500)
    p.add_argument("--lifetime-iters", type=int, default=20)
    p.add_argument("--output", default="stage3c3_native_diag.csv")
    p.add_argument("--quick", action="store_true",
                   help="100x120+500x600, repeat=3, fwd_reps=2, "
                        "setup_repeats=20, lifetime=3")
    args = p.parse_args()

    if args.quick:
        args.sizes = [(100, 120), (500, 600)]
        args.repeat = 3
        args.forward_timing_repeats = 2
        args.setup_repeats = 20
        args.lifetime_iters = 3

    if not torch.cuda.is_available():
        raise RuntimeError("benchmark requires CUDA")
    device = torch.device("cuda")

    import curegot  # noqa: E402

    print(f"sizes={args.sizes} reference_density={args.reference_density} "
          f"eta={args.eta} tol={args.tol} context_mode={args.context_mode}")
    print(f"repeat={args.repeat} setup_repeats={args.setup_repeats} "
          f"seed={args.seed} output={args.output}")

    all_rows = []
    csv_file = open(args.output, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    writer.writeheader()
    csv_file.flush()

    try:
        for (n, m) in args.sizes:
            rows = run_size(curegot, n, m, args, device)
            fill_speedups(rows)
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
