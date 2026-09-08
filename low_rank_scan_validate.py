import gc
import math
import torch
import curegot

from stage3b_validate import (
    make_problem,
    dense_from_csr,
    build_precond_from_ctx,
    run_L0,
    run_L1,
)


# ------------------------------------------------------------
# Forward wrapper used only for low-rank scanning
# ------------------------------------------------------------

def run_forward_scan(
    M,
    a,
    b,
    eta,
    tol,
    max_iter,
    candidate_sinkhorn_iter,
    verbose=0,
):
    out = curegot.torch.sinkhorn_splr_bwd(
        M,
        a,
        b,
        eta,
        tol=tol,
        max_iter=max_iter,
        verbose=verbose,
        density=0.5,
        candidate_sinkhorn_iter=candidate_sinkhorn_iter,
        return_validation=True,
    )

    return (
        out["plan"],
        out["dual"],
        out["niter"],
        out["converged"],
        out["context"],
    )


# ------------------------------------------------------------
# Reconstruct A and independent dense y^T A^{-1} y
# ------------------------------------------------------------

def reconstruct_A(ctx, device):
    return dense_from_csr(
        ctx.csr_values(),
        ctx.csr_colind(),
        ctx.csr_rowptr(),
        ctx.dim,
        device,
    )


def compute_yinvAy_dense(ctx, device):
    if not ctx.low_rank_enabled:
        return float("nan")

    A = reconstruct_A(ctx, device)
    y = ctx.secant_y().to(device=device, dtype=torch.float64)

    u = torch.linalg.solve(A, y)
    yinvAy = torch.dot(y, u).item()

    return yinvAy


# ------------------------------------------------------------
# Search for a genuine low-rank context
# ------------------------------------------------------------

def scan_low_rank_case(
    M,
    a,
    b,
    eta,
    tol,
    device,
    candidate_sinkhorn_iter,
):
    max_iter_list = [2, 3, 5, 10, 20, 30, 40, 60, 100]

    print()
    print(
        "=== scanning candidate_sinkhorn_iter =",
        candidate_sinkhorn_iter,
        "===",
    )

    for max_iter in max_iter_list:

        plan, dual, niter, converged, ctx = run_forward_scan(
            M,
            a,
            b,
            eta,
            tol,
            max_iter,
            candidate_sinkhorn_iter,
        )

        if ctx.low_rank_enabled:
            yinvAy = compute_yinvAy_dense(ctx, device)
        else:
            yinvAy = float("nan")

        print(
            f"max_iter={max_iter:3d} | "
            f"niter={niter:3d} | "
            f"converged={str(converged):5s} | "
            f"fallback={str(ctx.is_fallback):5s} | "
            f"low_rank={str(ctx.low_rank_enabled):5s} | "
            f"ys={ctx.ys:.6e} | "
            f"yinvAy={yinvAy:.6e}"
        )

        valid_low_rank = (
            ctx.low_rank_enabled
            and not ctx.is_fallback
            and ctx.ys > 0.0
            and math.isfinite(yinvAy)
            and yinvAy > 0.0
        )

        if valid_low_rank:
            print()
            print("FOUND genuine low-rank case")
            print(
                "candidate_sinkhorn_iter =",
                candidate_sinkhorn_iter,
            )
            print("max_iter =", max_iter)

            return {
                "plan": plan,
                "dual": dual,
                "niter": niter,
                "converged": converged,
                "ctx": ctx,
                "max_iter": max_iter,
                "candidate_sinkhorn_iter":
                    candidate_sinkhorn_iter,
            }

        del ctx

    return None


# ------------------------------------------------------------
# L0 / L1
# ------------------------------------------------------------

def validate_L0_L1(case, device):
    ctx = case["ctx"]

    print()
    print("=== Low-rank L0 / L1 ===")

    err0 = run_L0(ctx, device)
    err1 = run_L1(ctx, device)

    print(f"[L0] relative error = {err0:.3e}")
    print(f"[L1] relative error = {err1:.3e}")

    if err0 >= 1e-8:
        raise RuntimeError(
            f"L0 failed: relative error = {err0:.3e}"
        )

    if err1 >= 1e-8:
        raise RuntimeError(
            f"L1 failed: relative error = {err1:.3e}"
        )

    print("[L0/L1] PASS")


# ------------------------------------------------------------
# Repeated solve test
# ------------------------------------------------------------

def repeated_solve_test(
    case,
    device,
    nrepeat=50,
):
    ctx = case["ctx"]

    print()
    print("=== repeated apply_preconditioner ===")

    B = build_precond_from_ctx(ctx, device)

    torch.manual_seed(123)

    r = torch.randn(
        ctx.dim,
        dtype=torch.float64,
        device=device,
    )

    z_ref = torch.linalg.solve(B, r)

    ref_norm = torch.linalg.norm(z_ref).item()

    max_err = 0.0

    for i in range(nrepeat):

        z = ctx.apply_preconditioner(r)

        err = (
            torch.linalg.norm(z - z_ref).item()
            / max(ref_norm, 1e-30)
        )

        max_err = max(max_err, err)

        if err >= 1e-8:
            raise RuntimeError(
                f"Repeated solve failed at call {i}: "
                f"rel err = {err:.3e}"
            )

    print(
        f"Repeated solve PASS: "
        f"{nrepeat} calls, "
        f"max rel err = {max_err:.3e}"
    )


# ------------------------------------------------------------
# Lifetime stress test
# ------------------------------------------------------------

def lifetime_stress_test(
    M,
    a,
    b,
    eta,
    tol,
    device,
    max_iter,
    candidate_sinkhorn_iter,
    nouter=20,
    nsolve=5,
):
    print()
    print("=== context lifetime stress test ===")

    for i in range(nouter):

        _, _, niter, converged, ctx = run_forward_scan(
            M,
            a,
            b,
            eta,
            tol,
            max_iter,
            candidate_sinkhorn_iter,
        )

        if ctx.is_fallback:
            raise RuntimeError(
                f"iteration {i}: unexpected fallback"
            )

        if not ctx.low_rank_enabled:
            raise RuntimeError(
                f"iteration {i}: "
                "low_rank unexpectedly disabled"
            )

        r = torch.randn(
            ctx.dim,
            dtype=torch.float64,
            device=device,
        )

        for _ in range(nsolve):
            z = ctx.apply_preconditioner(r)

        torch.cuda.synchronize()

        del z
        del r
        del ctx

        gc.collect()
        torch.cuda.synchronize()

        print(
            f"iteration {i:02d} ok "
            f"(niter={niter}, converged={converged})"
        )

    print("Lifetime stress test PASS")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    device = torch.device("cuda")

    n = 100
    m = 120
    eta = 0.05
    tol = 1e-10
    seed = 42

    M, a, b = make_problem(
        n,
        m,
        eta,
        seed,
        device,
    )

    case = None

    # First isolate quasi-Newton behavior.
    # Then try the default Sinkhorn candidate path.
    for candidate_sinkhorn_iter in [0, 3]:

        case = scan_low_rank_case(
            M,
            a,
            b,
            eta,
            tol,
            device,
            candidate_sinkhorn_iter,
        )

        if case is not None:
            break

    if case is None:
        raise RuntimeError(
            "No low_rank_enabled=True case was found."
        )

    ctx = case["ctx"]

    print()
    print("========================================")
    print("Selected low-rank case")
    print("----------------------------------------")
    print(
        "candidate_sinkhorn_iter =",
        case["candidate_sinkhorn_iter"],
    )
    print("max_iter =", case["max_iter"])
    print("niter =", case["niter"])
    print("converged =", case["converged"])
    print("is_fallback =", ctx.is_fallback)
    print("low_rank_enabled =", ctx.low_rank_enabled)
    print("ys =", ctx.ys)

    yinvAy = compute_yinvAy_dense(ctx, device)
    print("dense y^T A^{-1} y =", yinvAy)

    print("========================================")

    # 1. Core low-rank correctness
    validate_L0_L1(case, device)

    # 2. Repeated use of exactly the same context
    repeated_solve_test(
        case,
        device,
        nrepeat=50,
    )

    # Release this context before lifetime stress.
    del ctx
    case["ctx"] = None

    gc.collect()
    torch.cuda.synchronize()

    # 3. Repeated construction / destruction
    lifetime_stress_test(
        M,
        a,
        b,
        eta,
        tol,
        device,
        max_iter=case["max_iter"],
        candidate_sinkhorn_iter=
            case["candidate_sinkhorn_iter"],
        nouter=20,
        nsolve=5,
    )

    print()
    print("========================================")
    print("LOW-RANK VALIDATION OVERALL: PASS")
    print("========================================")

    if not case["converged"]:
        print()
        print(
            "NOTE: selected case is non-converged. "
            "L0/L1 and lifetime validation are valid, "
            "but this case should not be treated as the "
            "final converged implicit-gradient semantic test."
        )


if __name__ == "__main__":
    main()