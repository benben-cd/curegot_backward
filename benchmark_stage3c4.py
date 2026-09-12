#!/usr/bin/env python3
"""Stage 3C-4: final production-selector benchmark (Stage 3C freeze).

Tests ONLY the production helper ``splr_backward.solve_backward``:

  default    solve_backward(..., preconditioner omitted)  -> must be diag
  diag       explicit preconditioner="diag"
  splr       explicit preconditioner="splr"
  none       explicit preconditioner="none" (plain CG)

Key equivalence validation: default == explicit diag, i.e. same PCG
iteration count and bitwise-identical lambda / gradient (same code path).
All modes are also checked against the dense oracle (lambda/grad rel err
<= 1e-8) and must converge.

Timing is end-to-end production backward wall (RHS assembly + PCG +
gradient assembly), production-style (no benchmark-inserted per-op sync;
the PCG's scalar truth tests synchronize implicitly, and the SPLR apply
synchronizes internally -- documented behavior, not harmonized).

Usage:
  python benchmark_stage3c4.py --quick
  python benchmark_stage3c4.py \
      --sizes 100x120 300x360 500x600 1000x1200 \
      --repeat 5 --output stage3c4_final_policy.csv
"""

import argparse
import csv
import gc
import statistics
import time

import torch

from stage3b_validate import make_problem, dense_Q
from benchmark_stage3c2c import (
    _sync,
    rel_err,
    g_seed_for,
    parse_size,
    run_forward,
    run_forward_timed_median,
    ctx_meta,
    _med,
)
from splr_backward import solve_backward

CSV_FIELDS = [
    "n", "m", "dim",
    "mode",                         # default | diag | splr | none
    "repeat_id",
    "niter_forward", "converged_forward",
    "forward_context_ms",
    "forward_context_min_ms", "forward_context_max_ms", "forward_timing_repeats",
    "low_rank", "ys",
    "pcg_iterations", "pcg_converged",
    "total_backward_ms",            # end-to-end production backward wall
    "lambda_rel_err_vs_dense", "grad_rel_err_vs_dense",
    "default_eq_diag_iterations",
    "default_eq_diag_lambda_bitwise",
    "default_eq_diag_grad_bitwise",
    "speedup_vs_none",
]

MODES = ["default", "diag", "splr", "none"]


def run_mode(mode, T_ref, G, ctx, eta, args):
    """One end-to-end production backward call, timed production-style."""
    kwargs = {} if mode == "default" else {"preconditioner": mode}
    _sync()
    t0 = time.perf_counter()
    out = solve_backward(T_ref, G, ctx, eta, tol=args.pcg_tol,
                         max_iter=args.pcg_max_iter, **kwargs)
    _sync()
    ms = (time.perf_counter() - t0) * 1e3
    return out, ms


def run_size(curegot, n, m, args, device):
    print(f"\n{'=' * 78}\n=== size {n}x{m} ===")
    M, a, b = make_problem(n, m, args.eta, args.seed, device)
    gen = torch.Generator(device="cpu").manual_seed(g_seed_for(n, m))
    G = torch.randn((n, m), generator=gen, dtype=torch.float64).to(device)

    # 0. discarded warmup forward
    out_w, _ = run_forward(
        curegot, M, a, b, eta=args.eta, tol=args.tol, max_iter=args.max_iter,
        density=args.reference_density,
        candidate_sinkhorn_iter=args.candidate_sinkhorn_iter,
        context_mode=args.context_mode,
    )
    del out_w
    gc.collect()
    torch.cuda.empty_cache()

    # 1. reference forward (median protocol; last ctx is official)
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
    print(f"  reference forward: niter={niter_ref} dim={dim} nnz_A={nnz_A} "
          f"low_rank={low_rank} forward median={fc_ms:.1f}ms")

    # Dense oracle (only when affordable)
    lam_ref = grad_ref = None
    if dim <= args.max_dense_dim:
        from backward_core import rhs_Q_scale, assemble_grad
        Q = dense_Q(T_ref, n, m)
        mu = rhs_Q_scale(G, T_ref, n, m)
        lam_ref = torch.linalg.solve(Q, mu)
        grad_ref = assemble_grad(T_ref, G, lam_ref, args.eta, n, m)
        del Q, mu

    rows = []
    default_out = None
    eq_flags = None  # (eq_iter, eq_lam, eq_grad), filled at diag rep 0
    for mode in MODES:
        for rep in range(args.repeat):
            out, ms = run_mode(mode, T_ref, G, ctx_ref, args.eta, args)
            if not out["converged"]:
                raise RuntimeError(f"{mode}: PCG did not converge "
                                   f"({out['iterations']} iters)")
            lam_err = (rel_err(out["lambda"], lam_ref)
                       if lam_ref is not None else "")
            if lam_err != "" and lam_err > 1e-8:
                raise RuntimeError(
                    f"{mode}: lambda rel err vs dense {lam_err:.3e} > 1e-8"
                )
            grad_err = (rel_err(out["grad"], grad_ref)
                        if grad_ref is not None else "")
            if grad_err != "" and grad_err > 1e-8:
                raise RuntimeError(
                    f"{mode}: grad rel err vs dense {grad_err:.3e} > 1e-8"
                )
            row = {k: "" for k in CSV_FIELDS}
            row.update(
                n=n, m=m, dim=dim, mode=mode, repeat_id=rep,
                niter_forward=niter_ref, converged_forward=conv_ref,
                forward_context_ms=fc_ms,
                forward_context_min_ms=fc_min, forward_context_max_ms=fc_max,
                forward_timing_repeats=args.forward_timing_repeats,
                low_rank=low_rank, ys=ys,
                pcg_iterations=out["iterations"],
                pcg_converged=out["converged"],
                total_backward_ms=ms,
                lambda_rel_err_vs_dense=lam_err,
                grad_rel_err_vs_dense=grad_err,
            )
            rows.append(row)
            if mode == "default" and rep == 0:
                default_out = out
            if mode == "diag" and rep == 0:
                # default == explicit diag: same iterations, bitwise lambda/grad
                eq_iter = out["iterations"] == default_out["iterations"]
                eq_lam = bool(torch.equal(out["lambda"],
                                          default_out["lambda"]))
                eq_grad = bool(torch.equal(out["grad"], default_out["grad"]))
                print(f"  default==diag: iterations {eq_iter}, "
                      f"lambda bitwise {eq_lam}, grad bitwise {eq_grad}")
                if not (eq_iter and eq_lam and eq_grad):
                    raise RuntimeError(
                        "default backward differs from explicit diag: "
                        f"iterations {eq_iter}, lambda {eq_lam}, grad {eq_grad}"
                    )
                eq_flags = (eq_iter, eq_lam, eq_grad)
        med_ms = statistics.median(
            r["total_backward_ms"] for r in rows if r["mode"] == mode
            and r["total_backward_ms"] != ""
        )
        print(f"  {mode:<8} iters={out['iterations']:>4} "
              f"wall median={med_ms:.2f}ms")

    # Stamp the default==diag equivalence flags on every default/diag row
    # (all repeats), now that the check has run.
    if eq_flags is not None:
        for r in rows:
            if r["mode"] in ("default", "diag"):
                (r["default_eq_diag_iterations"],
                 r["default_eq_diag_lambda_bitwise"],
                 r["default_eq_diag_grad_bitwise"]) = eq_flags

    del ctx_ref, out_ref, T_ref, M, a, b, G
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def fill_speedups(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["n"], r["m"]), []).append(r)
    for key in sorted(groups):
        grows = groups[key]
        none_rows = [r for r in grows if r["mode"] == "none"]
        if not none_rows:
            continue
        t_none = statistics.median(r["total_backward_ms"] for r in none_rows)
        for r in grows:
            tot = r["total_backward_ms"]
            r["speedup_vs_none"] = t_none / tot if tot else ""


def run_lifetime_stress(curegot, args, device):
    n, m = args.sizes[0]
    print(f"\n=== lifetime stress: {args.lifetime_iters}x (create context -> "
          f"default diag backward + explicit SPLR backward -> destroy) "
          f"on {n}x{m} ===")
    M, a, b = make_problem(n, m, args.eta, args.seed, device)
    gen = torch.Generator(device="cpu").manual_seed(g_seed_for(n, m))
    G = torch.randn((n, m), generator=gen, dtype=torch.float64).to(device)
    for it in range(args.lifetime_iters):
        out, _ = run_forward(
            curegot, M, a, b, eta=args.eta, tol=args.tol,
            max_iter=args.max_iter, density=args.reference_density,
            candidate_sinkhorn_iter=args.candidate_sinkhorn_iter,
            context_mode=args.context_mode,
        )
        ctx = out["context"]
        T = out["plan"]
        for kwargs in ({}, {"preconditioner": "splr"}):
            res = solve_backward(T, G, ctx, args.eta, tol=args.pcg_tol,
                                 max_iter=args.pcg_max_iter, **kwargs)
            if not (bool(torch.isfinite(res["lambda"]).all())
                    and bool(torch.isfinite(res["grad"]).all())):
                raise RuntimeError(f"iteration {it}: non-finite output")
        del ctx, out, T, res
        gc.collect()
        torch.cuda.empty_cache()
    print("lifetime stress: PASS (no double free / invalid pointer / "
          "illegal access / NaN / Inf)")


def summarize(all_rows):
    groups = {}
    for r in all_rows:
        groups.setdefault((r["n"], r["m"]), []).append(r)
    for key in sorted(groups):
        grows = groups[key]
        n, m = key
        print("\n" + "=" * 88)
        print(f"{n}x{m}  (dim={grows[0]['dim']})  -- medians over repeats; "
              f"wall = end-to-end production backward")
        print("=" * 88)
        print(f"{'mode':>8} {'iter':>5} {'wall_ms':>9} {'spd/none':>9}")
        print("-" * 88)
        for mode in MODES:
            sub = [r for r in grows if r["mode"] == mode]
            if not sub:
                continue
            print(f"{mode:>8} "
                  f"{_med(sub, 'pcg_iterations'):>5.0f} "
                  f"{_med(sub, 'total_backward_ms'):>9.2f} "
                  f"{_med(sub, 'speedup_vs_none'):>9.2f}")
    print("\nexpected ranking (from 3C-3 data): default == diag < splr, "
          "diag ~1.2-1.9x vs none on tested sizes.")


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
    p.add_argument("--max-dense-dim", type=int, default=2500)
    p.add_argument("--lifetime-iters", type=int, default=20)
    p.add_argument("--output", default="stage3c4_final_policy.csv")
    p.add_argument("--quick", action="store_true",
                   help="100x120+500x600, repeat=3, fwd_reps=2, lifetime=3")
    args = p.parse_args()

    if args.quick:
        args.sizes = [(100, 120), (500, 600)]
        args.repeat = 3
        args.forward_timing_repeats = 2
        args.lifetime_iters = 3

    if not torch.cuda.is_available():
        raise RuntimeError("benchmark requires CUDA")
    device = torch.device("cuda")

    import curegot  # noqa: E402

    print(f"sizes={args.sizes} reference_density={args.reference_density} "
          f"eta={args.eta} seed={args.seed} output={args.output}")

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
