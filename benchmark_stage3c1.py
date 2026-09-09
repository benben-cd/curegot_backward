#!/usr/bin/env python3
"""Stage 3C-1 benchmark harness: zero-extra-factorization (adopt) vs the
Stage-3B.1 snapshot + fresh-factorization baseline.

Measures, per (size, density, mode, repeat):
  * forward + context creation wall time (CUDA-synchronized);
  * backward PCG wall time, split into HVP / preconditioner / Python-loop
    components;
  * plain CG (no preconditioner) wall time and iterations;
  * adopt-vs-snapshot correctness guards (niter/converged exact equality,
    plan/dual <= 1e-10, lambda/grad <= 1e-8, PCG iters equality).

The backward math (q_matvec, rhs_Q_scale, assemble_grad, PCG update rule) is
imported from / identical to stage3b_validate.py. No dense oracle is built.

Usage:
  python benchmark_stage3c1.py \
      --sizes 100x120 300x360 500x600 1000x1200 \
      --densities 0.05 0.1 0.2 0.5 \
      --warmup 3 --repeat 10 --output stage3c1_benchmark.csv

  python benchmark_stage3c1.py --quick
"""

import argparse
import csv
import gc
import statistics
import sys
import time

import torch

# Reuse the verified Stage-3B matrix-free oracle. Do not reimplement.
from stage3b_validate import (
    make_problem,
    q_matvec,
    rhs_Q_scale,
    assemble_grad,
)

MODES = ("snapshot", "adopt")

CSV_FIELDS = [
    "n", "m", "dim",
    "requested_density", "nnz_A", "realized_density",
    "eta", "tol",
    "mode", "repeat_id",
    "status", "error_type", "error_message",
    "niter_forward", "converged_forward",
    "low_rank", "adopted", "fallback", "ys",
    "forward_context_ms",
    "backward_pcg_ms", "hvp_ms", "precond_ms", "python_loop_ms",
    "pcg_iters", "pcg_converged",
    "total_ms",
    "plain_cg_ms", "plain_cg_iters", "plain_cg_converged",
    "plan_rel_error_vs_other_mode",
    "dual_rel_error_vs_other_mode",
    "lambda_rel_error_vs_other_mode",
    "grad_rel_error_vs_other_mode",
]


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _sync():
    torch.cuda.synchronize()


def timed_pcg(matvec, precond, b, tol=1e-12, max_iter=2000):
    """PCG with exactly the same math as stage3b_validate.pcg (same x/r/z/p/
    rz/alpha/beta updates, same stopping criterion, same initialization),
    plus CUDA-synchronized wall-clock timing split into:
      hvp_ms     -- time inside matvec (q_matvec)
      precond_ms -- time inside precond (ctx.apply_preconditioner)
      total_ms   -- whole solve
    python_loop_ms is derived by the caller as total - hvp - precond.
    Returns (x, iters, converged, total_ms, hvp_ms, precond_ms).
    """
    hvp_ms = 0.0
    precond_ms = 0.0

    def timed_matvec(v):
        nonlocal hvp_ms
        _sync()
        t0 = time.perf_counter()
        w = matvec(v)
        _sync()
        hvp_ms += (time.perf_counter() - t0) * 1e3
        return w

    def timed_precond(r):
        nonlocal precond_ms
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


def run_forward_context(curegot, M, a, b, *, eta, tol, max_iter, density,
                        candidate_sinkhorn_iter, mode):
    """Time sinkhorn_splr_bwd (forward optimization + context export)."""
    _sync()
    t0 = time.perf_counter()
    out = curegot.torch.sinkhorn_splr_bwd(
        M, a, b, eta,
        tol=tol, max_iter=max_iter, verbose=0,
        density=density,
        candidate_sinkhorn_iter=candidate_sinkhorn_iter,
        context_mode=mode,
    )
    _sync()
    ms = (time.perf_counter() - t0) * 1e3
    return out, ms


# ---------------------------------------------------------------------------
# Per-mode benchmark
# ---------------------------------------------------------------------------

def run_mode_benchmark(curegot, M, a, b, G, *, eta, tol, max_iter, density,
                       candidate_sinkhorn_iter, mode, pcg_tol, pcg_max_iter):
    """One (mode, repeat) measurement. Returns (result_dict, handle_dict).

    The handle keeps the GPU tensors needed for the cross-mode correctness
    guard and the plain-CG baseline; the backward context itself is released
    before returning so that cuDSS memory does not accumulate.
    """
    n, m = M.shape
    out, fc_ms = run_forward_context(
        curegot, M, a, b, eta=eta, tol=tol, max_iter=max_iter,
        density=density, candidate_sinkhorn_iter=candidate_sinkhorn_iter,
        mode=mode,
    )
    plan = out["plan"]
    dual = out["dual"]
    niter = int(out["niter"])
    converged = bool(out["converged"])
    ctx = out["context"]

    adopted = bool(ctx.adopted_forward_state)
    fallback = bool(ctx.is_fallback)
    low_rank = bool(ctx.low_rank_enabled)
    dim = int(ctx.dim)
    ys = float(ctx.ys) if low_rank else float("nan")
    nnz_A = int(ctx.csr_rowptr()[-1].item())

    # Mode-metadata guards (Stage 3C-1 section 23). The true fallback path
    # (niter == 0) is not a valid performance benchmark case.
    if niter == 0 or fallback:
        del ctx, out
        raise RuntimeError(
            "immediate-convergence fallback (niter=0): not a benchmark case"
        )
    if mode == "adopt" and not adopted:
        del ctx, out
        raise RuntimeError("adopt mode did not adopt the forward factorization")
    if mode == "snapshot" and adopted:
        del ctx, out
        raise RuntimeError("snapshot mode unexpectedly adopted forward state")

    T = plan
    mu = rhs_Q_scale(G, T, n, m)
    matvec = lambda v: q_matvec(T, v, n, m)
    precond = lambda r: ctx.apply_preconditioner(r)

    lam, pcg_iters, pcg_conv, pcg_ms, hvp_ms, precond_ms = timed_pcg(
        matvec, precond, mu, tol=pcg_tol, max_iter=pcg_max_iter
    )
    grad = assemble_grad(T, G, lam, eta, n, m)

    # Release the context (destroys the cuDSS factorization) before the next
    # run; this cost is deliberately outside every timed region.
    del precond, ctx, out
    gc.collect()
    torch.cuda.empty_cache()

    result = dict(
        niter_forward=niter,
        converged_forward=converged,
        low_rank=low_rank,
        adopted=adopted,
        fallback=fallback,
        ys=ys,
        dim=dim,
        nnz_A=nnz_A,
        forward_context_ms=fc_ms,
        backward_pcg_ms=pcg_ms,
        hvp_ms=hvp_ms,
        precond_ms=precond_ms,
        python_loop_ms=pcg_ms - hvp_ms - precond_ms,
        pcg_iters=pcg_iters,
        pcg_converged=pcg_conv,
    )
    handle = dict(plan=plan, dual=dual, lam=lam, grad=grad,
                  matvec=matvec, mu=mu)
    return result, handle


# ---------------------------------------------------------------------------
# Correctness guard
# ---------------------------------------------------------------------------

def rel_err(x, ref):
    denom = float(torch.linalg.norm(ref))
    num = float(torch.linalg.norm(x - ref))
    return num / denom if denom > 0 else num


def cross_mode_guard(row_s, row_a, h_s, h_a):
    """Compare the snapshot and adopt results of the same repeat.

    Returns (error_dict, problems). error_dict is filled into both rows;
    a non-empty problems list invalidates both rows.
    """
    errs = {
        "plan_rel_error_vs_other_mode": rel_err(h_a["plan"], h_s["plan"]),
        "dual_rel_error_vs_other_mode": rel_err(h_a["dual"], h_s["dual"]),
        "lambda_rel_error_vs_other_mode": rel_err(h_a["lam"], h_s["lam"]),
        "grad_rel_error_vs_other_mode": rel_err(h_a["grad"], h_s["grad"]),
    }
    problems = []
    if row_s["niter_forward"] != row_a["niter_forward"]:
        problems.append(
            f"niter mismatch ({row_s['niter_forward']} vs {row_a['niter_forward']})"
        )
    if row_s["converged_forward"] != row_a["converged_forward"]:
        problems.append("converged mismatch")
    if errs["plan_rel_error_vs_other_mode"] > 1e-10:
        problems.append("plan rel err > 1e-10")
    if errs["dual_rel_error_vs_other_mode"] > 1e-10:
        problems.append("dual rel err > 1e-10")
    if errs["lambda_rel_error_vs_other_mode"] > 1e-8:
        problems.append("lambda rel err > 1e-8")
    if errs["grad_rel_error_vs_other_mode"] > 1e-8:
        problems.append("grad rel err > 1e-8")
    if row_s["pcg_iters"] != row_a["pcg_iters"]:
        problems.append(
            f"pcg_iters mismatch ({row_s['pcg_iters']} vs {row_a['pcg_iters']})"
        )
    return errs, problems


# ---------------------------------------------------------------------------
# Rows / CSV
# ---------------------------------------------------------------------------

def base_row(n, m, density, eta, tol, mode, repeat_id):
    row = {k: "" for k in CSV_FIELDS}
    row.update(
        n=n, m=m, requested_density=density, eta=eta, tol=tol,
        mode=mode, repeat_id=repeat_id, status="ok",
    )
    return row


def mark_invalid(row, error_type, error_message):
    row["status"] = "invalid"
    row["error_type"] = error_type
    row["error_message"] = str(error_message).replace("\n", " ")[:300]


def fill_row_from_result(row, res):
    d = res["dim"]
    tri = d * (d + 1) / 2.0
    row.update(
        dim=res["dim"],
        nnz_A=res["nnz_A"],
        realized_density=(res["nnz_A"] / tri) if tri > 0 else "",
        niter_forward=res["niter_forward"],
        converged_forward=res["converged_forward"],
        low_rank=res["low_rank"],
        adopted=res["adopted"],
        fallback=res["fallback"],
        ys=res["ys"],
        forward_context_ms=res["forward_context_ms"],
        backward_pcg_ms=res["backward_pcg_ms"],
        hvp_ms=res["hvp_ms"],
        precond_ms=res["precond_ms"],
        python_loop_ms=res["python_loop_ms"],
        pcg_iters=res["pcg_iters"],
        pcg_converged=res["pcg_converged"],
        total_ms=res["forward_context_ms"] + res["backward_pcg_ms"],
    )


def fmt(v, w=10, p=3):
    if v == "" or v is None:
        return " " * w
    if isinstance(v, bool):
        return str(v).rjust(w)
    if isinstance(v, int):
        return str(v).rjust(w)
    return f"{v:{w}.{p}f}"


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _median(rows, key):
    vals = [r[key] for r in rows if r[key] != ""]
    return statistics.median(vals) if vals else None


def summarize(all_rows):
    ok = [r for r in all_rows if r["status"] == "ok"]
    groups = {}
    for r in ok:
        key = (r["n"], r["m"], r["requested_density"], r["mode"])
        groups.setdefault(key, []).append(r)
    configs = sorted({(r["n"], r["m"], r["requested_density"]) for r in all_rows})

    print("\n" + "=" * 118)
    print("MAIN TABLE (medians over repeats, status=ok rows only)")
    print("=" * 118)
    header = (
        f"{'n':>5} {'m':>5} {'req_d':>6} {'real_d':>7} {'mode':>8} "
        f"{'fwd_ctx_ms':>11} {'pcg_ms':>10} {'hvp_ms':>10} {'pre_ms':>10} "
        f"{'pyloop_ms':>10} {'total_ms':>10} {'pcg_it':>7}"
    )
    print(header)
    print("-" * 118)
    med = {}
    for (n, m, dens, mode) in sorted(groups):
        rows = groups[(n, m, dens, mode)]
        e = dict(
            fc=_median(rows, "forward_context_ms"),
            pcg=_median(rows, "backward_pcg_ms"),
            hvp=_median(rows, "hvp_ms"),
            pre=_median(rows, "precond_ms"),
            pyloop=_median(rows, "python_loop_ms"),
            tot=_median(rows, "total_ms"),
            it=_median(rows, "pcg_iters"),
            rd=_median(rows, "realized_density"),
        )
        med[(n, m, dens, mode)] = e
        print(
            f"{n:>5} {m:>5} {dens:>6.2f} {e['rd']:>7.4f} {mode:>8} "
            f"{e['fc']:>11.3f} {e['pcg']:>10.3f} {e['hvp']:>10.3f} "
            f"{e['pre']:>10.3f} {e['pyloop']:>10.3f} {e['tot']:>10.3f} "
            f"{int(e['it']):>7}"
        )

    print("\n" + "=" * 100)
    print("ADOPT vs SNAPSHOT (medians)")
    print("=" * 100)
    print(
        f"{'n':>5} {'m':>5} {'req_d':>6} {'ctx_speedup':>12} "
        f"{'total_speedup':>14} {'pcg_it_snap':>12} {'pcg_it_adopt':>13} {'match':>6}"
    )
    print("-" * 100)
    for (n, m, dens) in configs:
        s = med.get((n, m, dens, "snapshot"))
        a = med.get((n, m, dens, "adopt"))
        if not s or not a:
            print(f"{n:>5} {m:>5} {dens:>6.2f}   (incomplete/invalid)")
            continue
        ctx_sp = s["fc"] / a["fc"] if a["fc"] else float("nan")
        tot_sp = s["tot"] / a["tot"] if a["tot"] else float("nan")
        match = "yes" if s["it"] == a["it"] else "NO"
        print(
            f"{n:>5} {m:>5} {dens:>6.2f} {ctx_sp:>12.3f} {tot_sp:>14.3f} "
            f"{int(s['it']):>12} {int(a['it']):>13} {match:>6}"
        )

    print("\n" + "=" * 100)
    print("PCG vs PLAIN CG (adopt rows, medians)")
    print("=" * 100)
    print(
        f"{'n':>5} {'m':>5} {'req_d':>6} {'pcg_it':>7} {'cg_it':>7} "
        f"{'pcg_ms':>10} {'cg_ms':>10} {'cg/pcg_time':>12} {'cg_conv':>8}"
    )
    print("-" * 100)
    for (n, m, dens) in configs:
        rows = groups.get((n, m, dens, "adopt"))
        if not rows:
            continue
        it = _median(rows, "pcg_iters")
        cit = _median(rows, "plain_cg_iters")
        pcg = _median(rows, "backward_pcg_ms")
        cg = _median(rows, "plain_cg_ms")
        cconv = rows[0]["plain_cg_converged"]
        ratio = (cg / pcg) if (cg and pcg) else float("nan")
        print(
            f"{n:>5} {m:>5} {dens:>6.2f} {int(it):>7} {int(cit):>7} "
            f"{pcg:>10.3f} {cg:>10.3f} {ratio:>12.2f} {str(cconv):>8}"
        )

    print("\n" + "=" * 100)
    print("BACKWARD TIME BREAKDOWN (adopt rows, median share of backward_pcg_ms)")
    print("=" * 100)
    print(f"{'n':>5} {'m':>5} {'req_d':>6} {'hvp%':>7} {'precond%':>9} {'pyloop%':>8}")
    print("-" * 100)
    for (n, m, dens) in configs:
        rows = groups.get((n, m, dens, "adopt"))
        if not rows:
            continue
        pcg = _median(rows, "backward_pcg_ms")
        hvp = _median(rows, "hvp_ms")
        pre = _median(rows, "precond_ms")
        py = _median(rows, "python_loop_ms")
        if not pcg:
            continue
        print(
            f"{n:>5} {m:>5} {dens:>6.2f} "
            f"{100.0 * hvp / pcg:>6.1f} {100.0 * pre / pcg:>8.1f} "
            f"{100.0 * py / pcg:>7.1f}"
        )
    print()


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def parse_size(s):
    try:
        n, m = s.lower().split("x")
        return int(n), int(m)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid size '{s}', expected NxM")


def g_seed_for(n, m, density):
    """Deterministic per-configuration upstream-gradient seed.

    Deliberately not Python's hash(), which is randomized across processes.
    """
    return (1000003 * n + 10007 * m + int(round(density * 1e6))) % (2**31 - 1)


def warmup_pass(curegot, M, a, b, G, mode, common, pcg_tol, pcg_max_iter):
    """One full measurement pass (forward+context, PCG, plain CG), discarded.

    Function scope guarantees the GPU handle is released on return.
    """
    try:
        _, h = run_mode_benchmark(curegot, M, a, b, G, mode=mode, **common)
        timed_pcg(h["matvec"], None, h["mu"], tol=pcg_tol,
                  max_iter=pcg_max_iter)
    except Exception as exc:
        print(f"  [warmup {mode}] {type(exc).__name__}: {exc}")
    gc.collect()
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sizes", nargs="+", type=parse_size,
                   default=[(100, 120), (300, 360), (500, 600), (1000, 1200)],
                   help="problem sizes as NxM (add 2000x2400 for the large case)")
    p.add_argument("--densities", nargs="+", type=float,
                   default=[0.05, 0.1, 0.2, 0.5])
    p.add_argument("--eta", type=float, default=0.05)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--candidate-sinkhorn-iter", type=int, default=3)
    p.add_argument("--pcg-tol", type=float, default=1e-12)
    p.add_argument("--pcg-max-iter", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeat", type=int, default=10)
    p.add_argument("--output", default="stage3c1_benchmark.csv")
    p.add_argument("--quick", action="store_true",
                   help="smoke test: 100x120+500x600, densities 0.1+0.5, "
                        "warmup=1, repeat=3")
    args = p.parse_args()

    if args.quick:
        args.sizes = [(100, 120), (500, 600)]
        args.densities = [0.1, 0.5]
        args.warmup = 1
        args.repeat = 3

    if not torch.cuda.is_available():
        raise RuntimeError("benchmark requires CUDA")
    device = torch.device("cuda")

    import curegot  # noqa: E402  (import after CUDA check for a clean error)

    print(f"sizes={args.sizes} densities={args.densities} eta={args.eta} "
          f"tol={args.tol} max_iter={args.max_iter} "
          f"candidate_sinkhorn_iter={args.candidate_sinkhorn_iter}")
    print(f"pcg_tol={args.pcg_tol} pcg_max_iter={args.pcg_max_iter} "
          f"seed={args.seed} warmup={args.warmup} repeat={args.repeat}")
    print(f"output={args.output}")

    all_rows = []
    csv_file = open(args.output, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    writer.writeheader()
    csv_file.flush()

    try:
        for (n, m) in args.sizes:
            for density in args.densities:
                print(f"\n=== n={n} m={m} density={density} ===")
                # One shared problem per (size, density); identical M/a/b/G
                # for both context modes.
                M, a, b = make_problem(n, m, args.eta, args.seed, device)
                gen = torch.Generator(device="cpu").manual_seed(
                    g_seed_for(n, m, density)
                )
                G = torch.randn((n, m), generator=gen,
                                dtype=torch.float64).to(device)

                common = dict(
                    eta=args.eta, tol=args.tol, max_iter=args.max_iter,
                    density=density,
                    candidate_sinkhorn_iter=args.candidate_sinkhorn_iter,
                    pcg_tol=args.pcg_tol, pcg_max_iter=args.pcg_max_iter,
                )

                # Warmup: full measurement pass per mode, discarded. Covers
                # CUDA context / cuDSS / allocator / torch-kernel warmup.
                for mode in MODES:
                    for _ in range(args.warmup):
                        warmup_pass(curegot, M, a, b, G, mode, common,
                                    args.pcg_tol, args.pcg_max_iter)

                for rep in range(args.repeat):
                    # Alternate the measurement order per repeat to reduce
                    # thermal/cache bias: even reps snapshot->adopt, odd
                    # reps adopt->snapshot.
                    order = MODES if rep % 2 == 0 else MODES[::-1]
                    rows, handles = {}, {}
                    for mode in order:
                        row = base_row(n, m, density, args.eta, args.tol,
                                       mode, rep)
                        try:
                            res, h = run_mode_benchmark(
                                curegot, M, a, b, G, mode=mode, **common
                            )
                            fill_row_from_result(row, res)
                            handles[mode] = h
                            if not res["pcg_converged"]:
                                mark_invalid(row, "pcg_not_converged",
                                             f"PCG did not converge in "
                                             f"{res['pcg_iters']} iters")
                        except Exception as exc:
                            mark_invalid(row, type(exc).__name__, exc)
                        rows[mode] = row

                    # Plain CG baseline: once per repeat, from the first
                    # available mode's (identical) T and mu.
                    plain = None
                    for mode in order:
                        if mode in handles:
                            h = handles[mode]
                            try:
                                (_, cg_iters, cg_conv, cg_ms, _, _
                                 ) = timed_pcg(h["matvec"], None, h["mu"],
                                               tol=args.pcg_tol,
                                               max_iter=args.pcg_max_iter)
                                plain = (cg_ms, cg_iters, cg_conv)
                            except Exception as exc:
                                print(f"  [plain CG rep {rep}] "
                                      f"{type(exc).__name__}: {exc}")
                            break
                    if plain is not None:
                        for mode in MODES:
                            rows[mode]["plain_cg_ms"] = plain[0]
                            rows[mode]["plain_cg_iters"] = plain[1]
                            rows[mode]["plain_cg_converged"] = plain[2]

                    # Cross-mode correctness guard (both modes must be ok).
                    if all(m in handles for m in MODES) and all(
                            rows[m]["status"] == "ok" for m in MODES):
                        errs, problems = cross_mode_guard(
                            rows["snapshot"], rows["adopt"],
                            handles["snapshot"], handles["adopt"],
                        )
                        for mode in MODES:
                            rows[mode].update(errs)
                        if problems:
                            for mode in MODES:
                                mark_invalid(rows[mode],
                                             "cross_mode_mismatch",
                                             "; ".join(problems))

                    for mode in MODES:
                        writer.writerow(rows[mode])
                        all_rows.append(rows[mode])
                    csv_file.flush()

                    st = {m: rows[m]["status"] for m in MODES}
                    print(f"  rep {rep:>2}: snapshot={st['snapshot']:>7} "
                          f"adopt={st['adopt']:>7}"
                          + ("" if all(v == "ok" for v in st.values()) else
                             f"  ({rows['snapshot']['error_type']} / "
                             f"{rows['adopt']['error_type']})"))

                    handles.clear()
                    gc.collect()
                    torch.cuda.empty_cache()

                del M, a, b, G
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        csv_file.close()

    print(f"\nCSV written to {args.output} ({len(all_rows)} rows)")
    summarize(all_rows)

    n_invalid = sum(1 for r in all_rows if r["status"] != "ok")
    if n_invalid:
        print(f"NOTE: {n_invalid}/{len(all_rows)} rows are invalid; "
              f"see status/error_type/error_message in the CSV.")


if __name__ == "__main__":
    main()
