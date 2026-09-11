#!/usr/bin/env python3
"""Stage 3C-2A: profile the apply_preconditioner call path.

Profiling stage, not an optimization stage. Uses the debug-only APIs
  ctx.profile_preconditioner(r)              -> per-call host/GPU breakdown
  ctx.benchmark_preconditioner_repeated(r,N) -> C++-loop per-call cost
against the production ctx.apply_preconditioner(r) to answer:
  1. where the time of one preconditioner call goes
     (memcpy_in / cuDSS solve / low-rank / sync wait / memcpy_out /
      PyBind+tensor wrapper);
  2. whether the Python/PyBind/tensor-wrapper per-call overhead is
     significant (Python-loop vs C++-loop ratio).

Timing domains are kept separate: host wall times (std::chrono on the C++
side, perf_counter on the Python side) and GPU event elapsed times are
reported as distinct columns.

Attribution note (Stage 3C-2A fix): launch_low_rank_search_direc() contains
an INTERNAL cudaStreamSynchronize between its dot kernel and its host-side
scalar merge. In low-rank mode the dot kernel is queued behind the cuDSS
solve on the same stream, so that internal sync (lowrank_sync_wait_host_us)
absorbs whatever GPU solve time is still outstanding; host_final_sync_wait_us
then only waits for the correction kernel. The true GPU solve time is
cudss_gpu_us regardless of which host call happens to wait for it.

Usage:
  python benchmark_stage3c2a.py --quick
  python benchmark_stage3c2a.py \
      --sizes 100x120 500x600 1000x1200 --densities 0.1 0.5 \
      --warmup 10 --repeat 100 \
      --output stage3c2a_preconditioner_profile.csv
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
    "low_rank", "ys", "niter_forward", "converged_forward",
    "repeat_id",
    "python_single_call_wall_us",
    "host_wrapper_us", "cpp_ctx_host_us",
    "host_memcpy_in_us", "host_solve_launch_us", "host_lowrank_total_us",
    "lowrank_setup_launch_host_us", "lowrank_sync_wait_host_us",
    "lowrank_host_scalar_us", "lowrank_correction_launch_host_us",
    "host_final_sync_wait_us", "host_memcpy_out_us",
    "cudss_gpu_us", "lowrank_reduce_gpu_us", "lowrank_correction_gpu_us",
    "python_repeated_total_ms", "cpp_repeated_total_ms",
    "python_per_call_us", "cpp_per_call_us",
    "python_vs_cpp_overhead_ratio",
    "cudss_fraction_of_cpp", "lowrank_fraction_of_cpp",
    "profiled_output_rel_err", "repeated_output_rel_err",
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
                   candidate_sinkhorn_iter, context_mode):
    """Forward once; returns (out, ctx). Not part of any timed region."""
    out = curegot.torch.sinkhorn_splr_bwd(
        M, a, b, eta,
        tol=tol, max_iter=max_iter, verbose=0,
        density=density,
        candidate_sinkhorn_iter=candidate_sinkhorn_iter,
        context_mode=context_mode,
    )
    return out, out["context"]


# ---------------------------------------------------------------------------
# Per-config profiling
# ---------------------------------------------------------------------------

def profile_config(curegot, M, a, b, density, args, device):
    n, m = M.shape
    out, ctx = create_context(
        curegot, M, a, b, eta=args.eta, tol=args.tol, max_iter=args.max_iter,
        density=density,
        candidate_sinkhorn_iter=args.candidate_sinkhorn_iter,
        context_mode=args.context_mode,
    )
    dim = int(ctx.dim)
    low_rank = bool(ctx.low_rank_enabled)
    ys = float(ctx.ys) if low_rank else float("nan")
    nnz_A = int(ctx.csr_rowptr()[-1].item())
    niter = int(out["niter"])
    converged = bool(out["converged"])
    fallback = bool(ctx.is_fallback)
    if niter == 0 or fallback:
        print(f"  WARNING: niter=0 fallback context; low_rank is disabled")

    # Deterministic RHS batch (one per repeat)
    gen = torch.Generator(device="cpu").manual_seed(g_seed_for(n, m, density))
    rhs_batch = torch.randn((args.repeat, dim), generator=gen,
                            dtype=torch.float64).to(device)

    # Warmup on the production API
    for i in range(args.warmup):
        ctx.apply_preconditioner(rhs_batch[i % args.repeat])
    _sync()

    rows = []
    profiled_err = ""
    for rep in range(args.repeat):
        r = rhs_batch[rep]

        # 1) Production API, Python-side single-call wall time
        _sync()
        t0 = time.perf_counter()
        z_ref = ctx.apply_preconditioner(r)
        _sync()
        py_wall_us = (time.perf_counter() - t0) * 1e6

        # 2) Profiled call (same r): host breakdown + GPU events
        prof = ctx.profile_preconditioner(r)
        if rep == 0:
            profiled_err = rel_err(prof["output"], z_ref)
            if profiled_err > 1e-12:
                raise RuntimeError(
                    f"profiled output mismatch: rel err {profiled_err:.3e}"
                )

        row = {k: "" for k in CSV_FIELDS}
        row.update(
            n=n, m=m, dim=dim,
            requested_density=density, nnz_A=nnz_A,
            realized_density=nnz_A / (dim * (dim + 1) / 2.0),
            low_rank=low_rank, ys=ys,
            niter_forward=niter, converged_forward=converged,
            repeat_id=rep,
            python_single_call_wall_us=py_wall_us,
            host_wrapper_us=prof["host_wrapper_us"],
            cpp_ctx_host_us=prof["host_total_us"],
            host_memcpy_in_us=prof["host_memcpy_in_us"],
            host_solve_launch_us=prof["host_solve_launch_us"],
            host_lowrank_total_us=prof["host_lowrank_total_us"],
            lowrank_setup_launch_host_us=prof["lowrank_setup_launch_host_us"],
            lowrank_sync_wait_host_us=prof["lowrank_sync_wait_host_us"],
            lowrank_host_scalar_us=prof["lowrank_host_scalar_us"],
            lowrank_correction_launch_host_us=
                prof["lowrank_correction_launch_host_us"],
            host_final_sync_wait_us=prof["host_final_sync_wait_us"],
            host_memcpy_out_us=prof["host_memcpy_out_us"],
            cudss_gpu_us=prof["gpu_cudss_solve_us"],
            lowrank_reduce_gpu_us=prof["lowrank_reduce_gpu_us"],
            lowrank_correction_gpu_us=prof["lowrank_correction_gpu_us"],
        )
        rows.append(row)

    # 3) Repeated-solve experiment: Python loop vs C++ loop.
    #    Python loop cycles through the RHS batch; the C++ loop repeats the
    #    same RHS (solve time is data-independent; documented limitation).
    N = args.repeat
    _sync()
    t0 = time.perf_counter()
    for i in range(N):
        ctx.apply_preconditioner(rhs_batch[i])
    _sync()
    python_repeated_total_ms = (time.perf_counter() - t0) * 1e3
    python_per_call_us = python_repeated_total_ms * 1e3 / N

    rep_out = ctx.benchmark_preconditioner_repeated(rhs_batch[0], N)
    cpp_repeated_total_ms = rep_out["cpp_repeated_total_us"] / 1e3
    cpp_per_call_us = rep_out["cpp_per_call_us"]

    # Correctness: last C++-loop output vs production API on the same RHS
    z_ref = ctx.apply_preconditioner(rhs_batch[0])
    repeated_err = rel_err(rep_out["output"], z_ref)
    if repeated_err > 1e-12:
        raise RuntimeError(
            f"repeated-benchmark output mismatch: rel err {repeated_err:.3e}"
        )

    ratio = python_per_call_us / cpp_per_call_us if cpp_per_call_us > 0 else ""
    for row in rows:
        host_total = row["cpp_ctx_host_us"]
        row.update(
            python_repeated_total_ms=python_repeated_total_ms,
            cpp_repeated_total_ms=cpp_repeated_total_ms,
            python_per_call_us=python_per_call_us,
            cpp_per_call_us=cpp_per_call_us,
            python_vs_cpp_overhead_ratio=ratio,
            cudss_fraction_of_cpp=
                (row["cudss_gpu_us"] / host_total) if host_total else "",
            lowrank_fraction_of_cpp=
                ((row["lowrank_reduce_gpu_us"]
                  + row["lowrank_correction_gpu_us"]) / host_total)
                if host_total else "",
            profiled_output_rel_err=profiled_err,
            repeated_output_rel_err=repeated_err,
        )

    del ctx, out, rhs_batch
    gc.collect()
    torch.cuda.empty_cache()
    return rows, dict(low_rank=low_rank, ys=ys, niter=niter,
                      converged=converged, fallback=fallback)


# ---------------------------------------------------------------------------
# Lifetime stress
# ---------------------------------------------------------------------------

def run_lifetime_stress(curegot, args, device):
    n, m = args.sizes[0]
    density = args.densities[0]
    print(f"\n=== lifetime stress: {args.lifetime_iters}x "
          f"(create context -> profile solves -> destroy) "
          f"on {n}x{m} d={density} ===")
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
        for _ in range(3):
            prof = ctx.profile_preconditioner(r)
            if not bool(torch.isfinite(prof["output"]).all()):
                raise RuntimeError(f"iteration {it}: non-finite profiled output")
        rep = ctx.benchmark_preconditioner_repeated(r, 5)
        if not bool(torch.isfinite(rep["output"]).all()):
            raise RuntimeError(f"iteration {it}: non-finite repeated output")
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


def summarize(all_rows):
    groups = {}
    for r in all_rows:
        groups.setdefault((r["n"], r["m"], r["requested_density"]),
                          []).append(r)

    print("\n" + "=" * 96)
    print("PER-CONFIG SUMMARY (medians; host = host wall, GPU = CUDA event)")
    print("=" * 96)
    for key in sorted(groups):
        rows = groups[key]
        n, m, dens = key
        lr = rows[0]["low_rank"]
        print(f"\n{n}x{m} density={dens} low_rank={lr} "
              f"(dim={rows[0]['dim']}, nnz_A={rows[0]['nnz_A']}, "
              f"realized_density={rows[0]['realized_density']:.4f})")
        entries = [
            ("Python single-call wall", "python_single_call_wall_us"),
            ("C++ ctx host total", "cpp_ctx_host_us"),
            ("  memcpy_in (host)", "host_memcpy_in_us"),
            ("  solve launch (host)", "host_solve_launch_us"),
            ("  lowrank total (host)", "host_lowrank_total_us"),
            ("    setup+dot launch", "lowrank_setup_launch_host_us"),
            ("    internal sync wait*", "lowrank_sync_wait_host_us"),
            ("    host scalar merge", "lowrank_host_scalar_us"),
            ("    corr launch (host)", "lowrank_correction_launch_host_us"),
            ("  final sync wait (host)", "host_final_sync_wait_us"),
            ("  memcpy_out (host)", "host_memcpy_out_us"),
            ("cuDSS GPU solve (event)", "cudss_gpu_us"),
            ("lowrank dot GPU (event)", "lowrank_reduce_gpu_us"),
            ("lowrank corr GPU (event)", "lowrank_correction_gpu_us"),
            ("wrapper empty+guard (host)", "host_wrapper_us"),
            ("Python per-call (loop)", "python_per_call_us"),
            ("C++ per-call (loop)", "cpp_per_call_us"),
        ]
        for label, k in entries:
            print(f"  {label:>28}: {_med(rows, k):>10.1f} us")
        ratio = rows[0]["python_vs_cpp_overhead_ratio"]
        fc = _med(rows, "cudss_fraction_of_cpp")
        fl = _med(rows, "lowrank_fraction_of_cpp")
        print(f"  {'Python/C++ per-call ratio':>28}: {ratio:>10.2f} x")
        print(f"  {'cuDSS GPU / ctx host total':>28}: {fc:>10.2f}")
        print(f"  {'lowrank GPU / ctx host total':>28}: {fl:>10.2f}")

    print("\n" + "=" * 96)
    print("DECISION-RELEVANT TABLE (medians)")
    print("=" * 96)
    print(f"{'n':>5} {'m':>5} {'dens':>5} {'lowR':>5} {'py_1call':>9} "
          f"{'cpp_host':>9} {'cudss_gpu':>10} {'lowr_gpu':>9} "
          f"{'lr_syncw':>9} {'py/cpp':>7} {'cudss_frac':>10}")
    print("-" * 96)
    for key in sorted(groups):
        rows = groups[key]
        n, m, dens = key
        lowr_gpu = _med(rows, "lowrank_reduce_gpu_us") + \
            _med(rows, "lowrank_correction_gpu_us")
        print(f"{n:>5} {m:>5} {dens:>5.2f} {str(rows[0]['low_rank']):>5} "
              f"{_med(rows, 'python_single_call_wall_us'):>9.1f} "
              f"{_med(rows, 'cpp_ctx_host_us'):>9.1f} "
              f"{_med(rows, 'cudss_gpu_us'):>10.1f} "
              f"{lowr_gpu:>9.1f} "
              f"{_med(rows, 'lowrank_sync_wait_host_us'):>9.1f} "
              f"{rows[0]['python_vs_cpp_overhead_ratio']:>7.2f} "
              f"{_med(rows, 'cudss_fraction_of_cpp'):>10.2f}")
    print("\n* internal sync wait: in low-rank mode this is where the host "
          "actually waits for")
    print("  the cuDSS solve (dot kernel queued behind it on the same stream); "
          "it is NOT")
    print("  pure synchronization overhead.")
    print("\nInterpretation guide:")
    print("  py/cpp >> 1        -> Python/PyBind/tensor overhead significant "
          "(Case A: native C++ PCG)")
    print("  cudss_frac ~ 1     -> cuDSS solve dominates "
          "(Case B: optimize solve path)")
    print("  lowrank GPU large  -> Woodbury correction cost significant "
          "(Case C: profile correction further)")
    print("  lr_syncw large     -> host stalls inside the low-rank helper "
          "waiting on GPU;")
    print("                        overlap candidates: s^T r and y^T u partial "
          "dots do not")
    print("                        depend on the solve output (future "
          "optimization only).")


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
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeat", type=int, default=100,
                   help="single-call repeats AND the repeated-loop length N")
    p.add_argument("--lifetime-iters", type=int, default=20)
    p.add_argument("--output", default="stage3c2a_preconditioner_profile.csv")
    p.add_argument("--quick", action="store_true",
                   help="smoke test: 500x600, densities 0.1+0.5, warmup=3, "
                        "repeat=20, lifetime=5")
    args = p.parse_args()

    if args.quick:
        args.sizes = [(500, 600)]
        args.densities = [0.1, 0.5]
        args.warmup = 3
        args.repeat = 20
        args.lifetime_iters = 5

    if not torch.cuda.is_available():
        raise RuntimeError("profiling requires CUDA")
    device = torch.device("cuda")

    import curegot  # noqa: E402

    print(f"sizes={args.sizes} densities={args.densities} eta={args.eta} "
          f"tol={args.tol} context_mode={args.context_mode}")
    print(f"warmup={args.warmup} repeat={args.repeat} "
          f"lifetime_iters={args.lifetime_iters} output={args.output}")

    all_rows = []
    coverage = set()
    csv_file = open(args.output, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    writer.writeheader()
    csv_file.flush()

    try:
        for (n, m) in args.sizes:
            for density in args.densities:
                print(f"\n=== n={n} m={m} density={density} ===")
                M, a, b = make_problem(n, m, args.eta, args.seed, device)
                rows, info = profile_config(
                    curegot, M, a, b, density, args, device
                )
                coverage.add(info["low_rank"])
                print(f"  low_rank={info['low_rank']} ys={info['ys']:.3e} "
                      f"niter={info['niter']} converged={info['converged']} "
                      f"| py_1call={_med(rows, 'python_single_call_wall_us'):.1f}us "
                      f"cpp_host={_med(rows, 'cpp_ctx_host_us'):.1f}us "
                      f"cudss_gpu={_med(rows, 'cudss_gpu_us'):.1f}us "
                      f"ratio={rows[0]['python_vs_cpp_overhead_ratio']:.2f}x")
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

    if coverage != {False, True}:
        print(f"WARNING: low-rank coverage is {coverage}; both True and "
              f"False are needed to separate base A^{{-1}}r cost from the "
              f"Woodbury incremental cost. Consider --seed/--eta overrides "
              f"(e.g. --seed 40 --eta 0.03 for a known low-rank case).")

    summarize(all_rows)
    run_lifetime_stress(curegot, args, device)


if __name__ == "__main__":
    main()
