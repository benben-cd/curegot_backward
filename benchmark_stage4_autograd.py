"""benchmark_stage4_autograd.py -- Stage 4: manual vs autograd integration overhead.

Compares the manual production path (sinkhorn_splr_bwd + solve_backward)
against the autograd path (splr_ot + loss.backward()). Both sides solve
identical math with the same default preconditioner ("diag"); the only
expected difference is Python-side autograd wrapping.

Methodology (harness-cleanup revision):

  * PRIMARY metric: paired end-to-end totals.
      manual_once()    fresh native forward + fresh context + solve_backward
                       every repetition (exactly mirrors autograd_once)
      autograd_once()  fresh M.requires_grad + splr_ot + loss.backward()
    manual_total_ms / autograd_total_ms are medians of paired, interleaved
    samples (rep 0: manual->autograd, rep 1: autograd->manual, ...) to
    resist clock/load drift.
      integration_overhead_ms      = autograd_total_ms - manual_total_ms
      integration_overhead_percent = 100 * overhead / manual_total_ms

  * DIAGNOSTIC columns only (NOT used for the overhead ranking):
      manual_forward_ms / manual_backward_ms (split medians; the backward
      diagnostic reuses one pre-created context), autograd_forward_ms /
      autograd_backward_ms.

  * Warmup: 3 manual_once + 3 autograd_once per size with synchronization,
    excluded from the sample set (CUDA allocator / autograd init / lazy
    cuDSS+CUDA state).

  * Correctness gate: before timing each size, one manual and one autograd
    gradient are compared; relative error must be <= 1e-10 and is reported
    in the CSV (grad_rel_err_manual_vs_autograd), so timing never compares
    different computations.

Sizes: 100x120, 300x360, 500x600, 1000x1200, repeat=5 (full);
--quick: 100x120 + 500x600, repeat=3.

Acceptance (do NOT rewrite native C++ autograd for wrapper overhead):
    |overhead| < 5%   good / effectively negligible
    5-10%             acceptable, no optimization
    >10%              investigate only if reproducible (rerun once first)
A negative overhead is measurement noise / scheduling variation, not a
claimed autograd speedup.

Requires CUDA. Writes stage4_autograd.csv.
"""

import argparse
import statistics
import time

import torch

import curegot
from splr_autograd import splr_ot
from splr_backward import solve_backward
from stage3b_validate import make_problem

DEVICE = torch.device("cuda")
ETA = 0.05
TOL = 1e-8
MAX_ITER = 200

SIZES_FULL = [(100, 120), (300, 360), (500, 600), (1000, 1200)]
SIZES_QUICK = [(100, 120), (500, 600)]

WARMUPS = 3


def _sync():
    torch.cuda.synchronize()


def _time_once(fn):
    """One synchronized wall-clock sample, milliseconds."""
    _sync()
    t0 = time.perf_counter()
    fn()
    _sync()
    return (time.perf_counter() - t0) * 1e3


def _time(fn, repeats):
    return statistics.median([_time_once(fn) for _ in range(repeats)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="stage4_autograd.csv")
    args = ap.parse_args()

    sizes = SIZES_QUICK if args.quick else SIZES_FULL
    repeats = 3 if args.quick else 5

    header = ("n,m,"
              "manual_forward_ms,manual_backward_ms,manual_total_ms,"
              "autograd_forward_ms,autograd_backward_ms,autograd_total_ms,"
              "integration_overhead_ms,integration_overhead_percent,"
              "grad_rel_err_manual_vs_autograd")
    lines = [header]
    print(header)

    for idx, (n, m) in enumerate(sizes):
        M, a, b = make_problem(n, m, ETA, 200 + idx, DEVICE)
        W = torch.randn((n, m), dtype=torch.float64, device=DEVICE)

        # --- end-to-end paths (fresh forward + fresh context per call) ---
        def manual_once():
            out = curegot.torch.sinkhorn_splr_bwd(
                M, a, b, ETA, TOL, MAX_ITER, 0,
                density=0.10, candidate_sinkhorn_iter=3,
                context_mode="adopt")
            if not bool(out["converged"]):
                raise RuntimeError(
                    f"manual forward did not converge (niter="
                    f"{int(out['niter'])})")
            res = solve_backward(out["plan"], W, out["context"], ETA,
                                 preconditioner="diag")
            if not res["converged"]:
                raise RuntimeError(
                    f"manual backward PCG did not converge "
                    f"(iterations={res['iterations']})")
            return res["grad"]

        def autograd_once():
            Ma = M.clone().requires_grad_(True)
            T = splr_ot(Ma, a, b, eta=ETA, tol=TOL, max_iter=MAX_ITER,
                        density=0.10)
            (W * T).sum().backward()
            return Ma.grad

        # --- correctness gate: timing must never compare different math ---
        g_manual = manual_once()
        g_auto = autograd_once()
        rel = float(torch.linalg.norm(g_manual - g_auto)
                    / torch.linalg.norm(g_auto))
        assert rel <= 1e-10, \
            f"manual vs autograd grad rel err {rel:.3e} > 1e-10"

        # --- warmup (excluded from samples) ---
        for _ in range(WARMUPS):
            manual_once()
            autograd_once()
        _sync()

        # --- paired / interleaved primary samples ---
        manual_total_samples = []
        autograd_total_samples = []
        for r in range(repeats):
            if r % 2 == 0:
                manual_total_samples.append(_time_once(manual_once))
                autograd_total_samples.append(_time_once(autograd_once))
            else:
                autograd_total_samples.append(_time_once(autograd_once))
                manual_total_samples.append(_time_once(manual_once))
        manual_total = statistics.median(manual_total_samples)
        autograd_total = statistics.median(autograd_total_samples)

        # --- diagnostic split timings (NOT the ranking metric) ---
        plan0, ctx0 = None, None
        out0 = curegot.torch.sinkhorn_splr_bwd(
            M, a, b, ETA, TOL, MAX_ITER, 0,
            density=0.10, candidate_sinkhorn_iter=3, context_mode="adopt")
        assert bool(out0["converged"])
        plan0, ctx0 = out0["plan"], out0["context"]

        def manual_forward():
            out = curegot.torch.sinkhorn_splr_bwd(
                M, a, b, ETA, TOL, MAX_ITER, 0,
                density=0.10, candidate_sinkhorn_iter=3,
                context_mode="adopt")
            assert bool(out["converged"])

        def manual_backward():  # reuses ctx0: diagnostic only
            solve_backward(plan0, W, ctx0, ETA, preconditioner="diag")

        def autograd_forward():
            Ma = M.clone().requires_grad_(True)
            splr_ot(Ma, a, b, eta=ETA, tol=TOL, max_iter=MAX_ITER,
                    density=0.10)

        Ma0 = M.clone().requires_grad_(True)
        T0 = splr_ot(Ma0, a, b, eta=ETA, tol=TOL, max_iter=MAX_ITER,
                     density=0.10)

        def autograd_backward():
            (W * T0).sum().backward(retain_graph=True)

        mf = _time(manual_forward, repeats)
        mb = _time(manual_backward, repeats)
        af = _time(autograd_forward, repeats)
        ab = _time(autograd_backward, repeats)

        overhead_ms = autograd_total - manual_total
        overhead_pct = 100.0 * overhead_ms / manual_total

        row = (f"{n},{m},"
               f"{mf:.3f},{mb:.3f},{manual_total:.3f},"
               f"{af:.3f},{ab:.3f},{autograd_total:.3f},"
               f"{overhead_ms:.3f},{overhead_pct:.2f},{rel:.3e}")
        print(row)
        lines.append(row)

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}")
    print("Acceptance: |overhead| < 5% good; 5-10% acceptable; "
          ">10% investigate only if reproducible (rerun once first).")
    print("Negative overhead = measurement noise, not an autograd speedup.")


if __name__ == "__main__":
    main()
