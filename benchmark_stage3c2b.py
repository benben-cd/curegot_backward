"""
Stage 3C-2B: solve-only microbenchmark for the repeated cuDSS solve path.

Isolates the pure base solve A^{-1} r (same factorization, repeated RHS):
no low-rank helper, no PCG, no HVP, minimal Python involvement.

Motivation (Stage 3C-2A findings): the cuDSS GPU solve dominates the
preconditioner host wall (50% -> 88% with size); Python/PyBind overhead is
~1x and the Woodbury correction is 11-22 us/call.

Source-level audit result (linsolve.cu, see Stage3C2B delivery note):
SparseCholeskySolver::solve() is a SINGLE cudssExecute(CUDSS_PHASE_SOLVE)
call. Handle/config/data/matrix wrappers(A, b, x) and the CUDA stream are
all created once and never touched per solve. The CSV therefore contains
only stages that really exist:
    host_memcpy_in_us        cudaMemcpy d_rhs <- d_r (D2D)
    host_cudss_execute_us    host wall inside cudssExecute(PHASE_SOLVE)
                             (NOT pure launch overhead: cuDSS may block
                             the host until part/all of the GPU solve is
                             done -- measured, not assumed)
    host_sync_wait_us        final cudaStreamSynchronize
    host_memcpy_out_us       cudaMemcpy d_z <- d_sol (D2D)
    gpu_solve_us             CUDA events around the solve

Usage:
  python benchmark_stage3c2b.py --quick
  python benchmark_stage3c2b.py \
      --sizes 100x120 500x600 1000x1200 --densities 0.1 0.5 \
      --warmup 20 --repeat 200 --include-fallback \
      --output stage3c2b_solve_profile.csv
  # after an optimization, rerun with --label after and compare:
  python benchmark_stage3c2b.py --label after --compare stage3c2b_solve_profile.csv
"""

import argparse
import csv
import gc
import statistics
import time

import torch

from stage3b_validate import make_problem

CSV_FIELDS = [
    "n", "m", "dim",
    "requested_density", "nnz_A", "realized_density",
    "low_rank", "context_kind",
    "repeat_id",
    "python_profiled_wall_us", "host_wrapper_us",
    "host_total_us", "host_memcpy_in_us", "host_cudss_execute_us",
    "host_sync_wait_us", "host_memcpy_out_us",
    "gpu_solve_us",
    "cpp_per_call_us",
    "before_or_after",
    "output_rel_err",
]


def _sync():
    torch.cuda.synchronize()


def rel_err(x, ref):
    denom = float(torch.linalg.norm(ref))
    num = float(torch.linalg.norm(x - ref))
    return num / denom if denom > 0 else num


def g_seed_for(n, m, density):
    return (1000003 * n + 10007 * m + int(round(density * 1e6))) % (2**31 - 1)


def parse_size(s):
    try:
        n, m = s.lower().split("x")
        return int(n), int(m)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid size '{s}', expected NxM")


def create_context(curegot, M, a, b, *, eta, tol, max_iter, density,
                   candidate_sinkhorn_iter, context_mode, x0=None):
    """Forward once; returns (out, ctx). Not part of any timed region."""
    kwargs = dict(
        tol=tol, max_iter=max_iter, verbose=0,
        density=density,
        candidate_sinkhorn_iter=candidate_sinkhorn_iter,
        context_mode=context_mode,
        return_validation=True,
    )
    if x0 is not None:
        kwargs["x0"] = x0
    out = curegot.torch.sinkhorn_splr_bwd(M, a, b, eta, **kwargs)
    return out, out["context"]


def dense_reference_matrix(ctx, dim, device):
    """Rebuild the dense SPD matrix A from the context's exported CSR
    (lower triangle, base zero) for the correctness reference."""
    rowptr = ctx.csr_rowptr().to("cpu", torch.int64)
    colind = ctx.csr_colind().to("cpu", torch.int64)
    values = ctx.csr_values().to("cpu", torch.float64)
    A = torch.zeros(dim, dim, dtype=torch.float64)
    for i in range(dim):
        for p in range(int(rowptr[i]), int(rowptr[i + 1])):
            A[i, colind[p]] = values[p]
    # symmetric from the stored lower triangle
    A = A + A.t() - torch.diag(torch.diag(A))
    return A.to(device)


# ---------------------------------------------------------------------------
# Per-config profiling
# ---------------------------------------------------------------------------

def profile_config(curegot, M, a, b, density, args, device, context_kind):
    n, m = M.shape
    x0 = None
    if context_kind == "fallback_no_lowrank":
        # First forward to get the optimal dual; second forward with x0 at
        # the optimum converges in niter=0 -> fallback context, low-rank
        # disabled. The curvature guard is NOT touched.
        out0, ctx0 = create_context(
            curegot, M, a, b, eta=args.eta, tol=args.tol,
            max_iter=args.max_iter, density=density,
            candidate_sinkhorn_iter=args.candidate_sinkhorn_iter,
            context_mode=args.context_mode,
        )
        x0 = out0["dual"]
        del out0, ctx0
        gc.collect()
        torch.cuda.empty_cache()
    out, ctx = create_context(
        curegot, M, a, b, eta=args.eta, tol=args.tol, max_iter=args.max_iter,
        density=density,
        candidate_sinkhorn_iter=args.candidate_sinkhorn_iter,
        context_mode=args.context_mode, x0=x0,
    )
    dim = int(ctx.dim)
    low_rank = bool(ctx.low_rank_enabled)
    nnz_A = int(ctx.csr_rowptr()[-1].item())
    niter = int(out["niter"])
    converged = bool(out["converged"])
    if context_kind == "fallback_no_lowrank" and low_rank:
        print("  WARNING: fallback context unexpectedly has low_rank enabled")
    if context_kind == "normal" and niter == 0:
        print("  WARNING: niter=0 fallback context; low_rank is disabled")

    # Dense reference for the pure base solve A^{-1} r (one dense solve per
    # repeat batch is cheap relative to the forward; dim <= ~2200).
    A_dense = dense_reference_matrix(ctx, dim, device)

    # Deterministic RHS batch (one per repeat)
    gen = torch.Generator(device="cpu").manual_seed(g_seed_for(n, m, density))
    rhs_batch = torch.randn((args.repeat, dim), generator=gen,
                            dtype=torch.float64).to(device)

    # Warmup: uninstrumented C++ loop (no per-call event create/destroy)
    ctx.benchmark_base_solve_repeated(rhs_batch[0], args.warmup)
    _sync()

    rows = []
    for rep in range(args.repeat):
        r = rhs_batch[rep]

        _sync()
        t0 = time.perf_counter()
        prof = ctx.profile_base_solve(r)
        _sync()
        py_wall_us = (time.perf_counter() - t0) * 1e6

        z_ref = torch.linalg.solve(A_dense, r)
        err = rel_err(prof["output"], z_ref)
        if rep == 0 and err > 1e-12:
            raise RuntimeError(
                f"base-solve output mismatch vs dense reference: "
                f"rel err {err:.3e}"
            )

        row = {k: "" for k in CSV_FIELDS}
        row.update(
            n=n, m=m, dim=dim,
            requested_density=density, nnz_A=nnz_A,
            realized_density=nnz_A / (dim * (dim + 1) / 2.0),
            low_rank=low_rank, context_kind=context_kind,
            repeat_id=rep,
            python_profiled_wall_us=py_wall_us,
            host_wrapper_us=prof["host_wrapper_us"],
            host_total_us=prof["host_total_us"],
            host_memcpy_in_us=prof["host_memcpy_in_us"],
            host_cudss_execute_us=prof["host_cudss_execute_us"],
            host_sync_wait_us=prof["host_sync_wait_us"],
            host_memcpy_out_us=prof["host_memcpy_out_us"],
            gpu_solve_us=prof["gpu_solve_us"],
            before_or_after=args.label,
            output_rel_err=err,
        )
        rows.append(row)

    # Steady-state repeated solve: uninstrumented C++ loop on the same RHS.
    rep_out = ctx.benchmark_base_solve_repeated(rhs_batch[0], args.repeat)
    cpp_per_call_us = rep_out["cpp_per_call_us"]
    z_ref0 = torch.linalg.solve(A_dense, rhs_batch[0])
    repeated_err = rel_err(rep_out["output"], z_ref0)
    if repeated_err > 1e-12:
        raise RuntimeError(
            f"repeated base-solve output mismatch: rel err {repeated_err:.3e}"
        )
    for row in rows:
        row["cpp_per_call_us"] = cpp_per_call_us

    del ctx, out, A_dense, rhs_batch
    gc.collect()
    torch.cuda.empty_cache()
    return rows, dict(low_rank=low_rank, niter=niter, converged=converged)


# ---------------------------------------------------------------------------
# Lifetime stress
# ---------------------------------------------------------------------------

def run_lifetime_stress(curegot, args, device):
    n, m = args.sizes[0]
    density = args.densities[0]
    print(f"\n=== lifetime stress: {args.lifetime_iters}x "
          f"(create context -> {args.lifetime_solves}x repeated base solve "
          f"-> destroy) on {n}x{m} d={density} ===")
    M, a, b = make_problem(n, m, args.eta, args.seed, device)
    for it in range(args.lifetime_iters):
        out, ctx = create_context(
            curegot, M, a, b, eta=args.eta, tol=args.tol,
            max_iter=args.max_iter, density=density,
            candidate_sinkhorn_iter=args.candidate_sinkhorn_iter,
            context_mode=args.context_mode,
        )
        dim = int(ctx.dim)
        gen = torch.Generator(device="cpu").manual_seed(1234 + it)
        r = torch.randn(dim, generator=gen, dtype=torch.float64).to(device)
        rep = ctx.benchmark_base_solve_repeated(r, args.lifetime_solves)
        if not bool(torch.isfinite(rep["output"]).all()):
            raise RuntimeError(f"iteration {it}: non-finite repeated output")
        prof = ctx.profile_base_solve(r)
        if not bool(torch.isfinite(prof["output"]).all()):
            raise RuntimeError(f"iteration {it}: non-finite profiled output")
        del ctx, out, r, prof, rep
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


def _group_key(r):
    return (r["n"], r["m"], r["requested_density"], r["context_kind"],
            r["before_or_after"])


def summarize(all_rows):
    groups = {}
    for r in all_rows:
        groups.setdefault(_group_key(r), []).append(r)

    print("\n" + "=" * 100)
    print("PER-CONFIG SUMMARY (medians; host = host wall, GPU = CUDA event)")
    print("=" * 100)
    for key in sorted(groups):
        rows = groups[key]
        n, m, dens, kind, label = key
        lr = rows[0]["low_rank"]
        host_other = (_med(rows, "host_memcpy_in_us")
                      + _med(rows, "host_sync_wait_us")
                      + _med(rows, "host_memcpy_out_us"))
        print(f"\n{n}x{m} density={dens} kind={kind} [{label}] "
              f"low_rank={lr} (dim={rows[0]['dim']}, "
              f"nnz_A={rows[0]['nnz_A']}, "
              f"realized_density={rows[0]['realized_density']:.4f})")
        entries = [
            ("Python profiled wall", "python_profiled_wall_us"),
            ("host total (C++)", "host_total_us"),
            ("  memcpy_in (host)", "host_memcpy_in_us"),
            ("  cudssExecute (host wall)", "host_cudss_execute_us"),
            ("  final sync wait (host)", "host_sync_wait_us"),
            ("  memcpy_out (host)", "host_memcpy_out_us"),
            ("GPU solve (event)", "gpu_solve_us"),
            ("C++ per-call (loop)", "cpp_per_call_us"),
        ]
        for label2, k in entries:
            print(f"  {label2:>26}: {_med(rows, k):>10.1f} us")
        print(f"  {'host outside cudss API':>26}: {host_other:>10.1f} us")

    print("\n" + "=" * 100)
    print("DECISION-RELEVANT TABLE (medians)")
    print("=" * 100)
    print(f"{'n':>5} {'m':>5} {'dens':>5} {'kind':>20} {'label':>7} "
          f"{'host_tot':>9} {'cudss_h':>9} {'gpu_solve':>10} "
          f"{'cpp/call':>9} {'exec/gpu':>8}")
    print("-" * 100)
    for key in sorted(groups):
        rows = groups[key]
        n, m, dens, kind, label = key
        ht = _med(rows, "host_total_us")
        ch = _med(rows, "host_cudss_execute_us")
        gs = _med(rows, "gpu_solve_us")
        cp = _med(rows, "cpp_per_call_us")
        ratio = ch / gs if gs else float("nan")
        print(f"{n:>5} {m:>5} {dens:>5.2f} {kind:>20} {label:>7} "
              f"{ht:>9.1f} {ch:>9.1f} {gs:>10.1f} {cp:>9.1f} {ratio:>8.2f}")
    print("\nexec/gpu ~ 0 -> cudssExecute returns fast, host waits later "
          "(async)")
    print("exec/gpu ~ 1 -> cudssExecute blocks the host until the GPU solve "
          "is done")
    print("host outside cudss API = memcpy_in + sync_wait + memcpy_out")


def compare_csvs(before_path, after_rows):
    """Side-by-side medians: an earlier CSV (before) vs this run (after)."""
    before_rows = []
    with open(before_path, newline="") as f:
        for r in csv.DictReader(f):
            for k in ("n", "m", "dim", "nnz_A", "repeat_id"):
                r[k] = int(r[k])
            for k in ("requested_density", "realized_density",
                      "host_total_us", "host_cudss_execute_us",
                      "gpu_solve_us", "cpp_per_call_us"):
                r[k] = float(r[k]) if r[k] != "" else float("nan")
            r["low_rank"] = r["low_rank"] == "True"
            before_rows.append(r)
    bg, ag = {}, {}
    for r in before_rows:
        bg.setdefault((r["n"], r["m"], r["requested_density"],
                       r["context_kind"]), []).append(r)
    for r in after_rows:
        ag.setdefault((r["n"], r["m"], r["requested_density"],
                       r["context_kind"]), []).append(r)
    print("\n" + "=" * 100)
    print("BEFORE / AFTER (medians)")
    print("=" * 100)
    print(f"{'n':>5} {'m':>5} {'dens':>5} {'kind':>20} "
          f"{'host_b':>9} {'host_a':>9} {'exec_b':>9} {'exec_a':>9} "
          f"{'gpu_b':>9} {'gpu_a':>9} {'spd_host':>8} {'spd_gpu':>8}")
    print("-" * 100)
    for key in sorted(bg):
        if key not in ag:
            continue
        hb, ha = _med(bg[key], "host_total_us"), _med(ag[key], "host_total_us")
        eb = _med(bg[key], "host_cudss_execute_us")
        ea = _med(ag[key], "host_cudss_execute_us")
        gb, ga = _med(bg[key], "gpu_solve_us"), _med(ag[key], "gpu_solve_us")
        sh = hb / ha if ha else float("nan")
        sg = gb / ga if ga else float("nan")
        n, m, dens, kind = key
        print(f"{n:>5} {m:>5} {dens:>5.2f} {kind:>20} "
              f"{hb:>9.1f} {ha:>9.1f} {eb:>9.1f} {ea:>9.1f} "
              f"{gb:>9.1f} {ga:>9.1f} {sh:>8.2f} {sg:>8.2f}")
    print("\nGPU solve time is expected to be unchanged by host-overhead "
          "optimizations; if it changed, explain why.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sizes", nargs="+", type=parse_size,
                   default=[(100, 120), (500, 600), (1000, 1200)])
    p.add_argument("--densities", nargs="+", type=float, default=[0.1, 0.5])
    p.add_argument("--eta", type=float, default=0.05)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--candidate-sinkhorn-iter", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--context-mode", default="adopt",
                   choices=["adopt", "snapshot"],
                   help="context construction mode for the profiled context")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeat", type=int, default=200)
    p.add_argument("--include-fallback", action="store_true",
                   help="also benchmark a low_rank=False fallback context "
                        "(x0 at the optimum -> niter=0) per config; the "
                        "curvature guard is not touched")
    p.add_argument("--label", default="before", choices=["before", "after"],
                   help="before_or_after column value for this run")
    p.add_argument("--compare", default=None, metavar="CSV",
                   help="optional: path of a previous (before) CSV to print "
                        "side-by-side medians against this run")
    p.add_argument("--lifetime-iters", type=int, default=20)
    p.add_argument("--lifetime-solves", type=int, default=200)
    p.add_argument("--output", default="stage3c2b_solve_profile.csv")
    p.add_argument("--quick", action="store_true",
                   help="smoke test: 500x600, densities 0.1+0.5, warmup=5, "
                        "repeat=30, lifetime=3x30")
    args = p.parse_args()

    if args.quick:
        args.sizes = [(500, 600)]
        args.densities = [0.1, 0.5]
        args.warmup = 5
        args.repeat = 30
        args.lifetime_iters = 3
        args.lifetime_solves = 30

    if not torch.cuda.is_available():
        raise RuntimeError("profiling requires CUDA")
    device = torch.device("cuda")

    import curegot  # noqa: E402

    print(f"sizes={args.sizes} densities={args.densities} eta={args.eta} "
          f"tol={args.tol} context_mode={args.context_mode} "
          f"label={args.label}")
    print(f"warmup={args.warmup} repeat={args.repeat} "
          f"include_fallback={args.include_fallback} "
          f"lifetime={args.lifetime_iters}x{args.lifetime_solves} "
          f"output={args.output}")

    all_rows = []
    csv_file = open(args.output, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    writer.writeheader()
    csv_file.flush()

    try:
        for (n, m) in args.sizes:
            for density in args.densities:
                kinds = ["normal"]
                if args.include_fallback:
                    kinds.append("fallback_no_lowrank")
                for kind in kinds:
                    print(f"\n=== n={n} m={m} density={density} "
                          f"kind={kind} ===")
                    M, a, b = make_problem(n, m, args.eta, args.seed, device)
                    rows, info = profile_config(
                        curegot, M, a, b, density, args, device, kind
                    )
                    print(f"  low_rank={info['low_rank']} "
                          f"niter={info['niter']} "
                          f"converged={info['converged']} "
                          f"| host_total={_med(rows, 'host_total_us'):.1f}us "
                          f"cudss_host={_med(rows, 'host_cudss_execute_us'):.1f}us "
                          f"gpu_solve={_med(rows, 'gpu_solve_us'):.1f}us "
                          f"cpp/call={_med(rows, 'cpp_per_call_us'):.1f}us")
                    for row in rows:
                        writer.writerow(row)
                        all_rows.append(row)
                    csv_file.flush()
                    del M, a, b
                    gc.collect()
                    torch.cuda.empty_cache()
    finally:
        csv_file.close()

    print(f"\nCSV written to {args.output} ({len(all_rows)} rows)")

    summarize(all_rows)
    if args.compare:
        compare_csvs(args.compare, all_rows)
    run_lifetime_stress(curegot, args, device)


if __name__ == "__main__":
    main()
