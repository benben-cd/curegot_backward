import gc
import math
import torch

from stage3b_validate import (
    make_problem,
    run_all,
)

from low_rank_scan_validate import (
    run_forward_scan,
    compute_yinvAy_dense,
)


def main():

    device = torch.device("cuda")

    n = 100
    m = 120
    max_iter = 200

    found = False

    # Prefer default production-like candidate path first.
    candidate_list = [3, 0]

    seed_list = list(range(40, 50))
    eta_list = [0.03, 0.05, 0.08, 0.10]
    tol_list = [1e-8, 1e-10]

    for candidate_sinkhorn_iter in candidate_list:

        print()
        print(
            "========================================"
        )
        print(
            "Scanning candidate_sinkhorn_iter =",
            candidate_sinkhorn_iter,
        )
        print(
            "========================================"
        )

        for seed in seed_list:

            for eta in eta_list:

                # Important:
                # construct the problem using the same eta
                # that will be passed to the solver.
                M, a, b = make_problem(
                    n,
                    m,
                    eta,
                    seed,
                    device,
                )

                for tol in tol_list:

                    plan, dual, niter, converged, ctx = (
                        run_forward_scan(
                            M,
                            a,
                            b,
                            eta,
                            tol,
                            max_iter,
                            candidate_sinkhorn_iter,
                            verbose=0,
                        )
                    )

                    print(
                        f"candidate={candidate_sinkhorn_iter} | "
                        f"seed={seed:2d} | "
                        f"eta={eta:.2f} | "
                        f"tol={tol:.0e} | "
                        f"niter={niter:3d} | "
                        f"converged={str(converged):5s} | "
                        f"low_rank={str(ctx.low_rank_enabled):5s} | "
                        f"fallback={str(ctx.is_fallback):5s} | "
                        f"adopted={str(ctx.adopted_forward_state):5s}"
                    )

                    valid = (
                        converged
                        and ctx.low_rank_enabled
                        and not ctx.is_fallback
                        and ctx.adopted_forward_state
                        and ctx.ys > 0.0
                    )

                    if valid:

                        yinvAy = compute_yinvAy_dense(
                            ctx,
                            device,
                        )

                        if (
                            math.isfinite(yinvAy)
                            and yinvAy > 0.0
                        ):

                            print()
                            print(
                                "FOUND converged low-rank case"
                            )
                            print(
                                "candidate_sinkhorn_iter =",
                                candidate_sinkhorn_iter,
                            )
                            print("seed =", seed)
                            print("eta =", eta)
                            print("tol =", tol)
                            print("niter =", niter)
                            print(
                                "adopted_forward_state =",
                                ctx.adopted_forward_state,
                            )
                            print("ys =", ctx.ys)
                            print(
                                "dense y^T A^{-1} y =",
                                yinvAy,
                            )

                            print()
                            print(
                                "Running full L0/L1/L2..."
                            )

                            ok = run_all(
                                "Converged low-rank test",
                                plan,
                                dual,
                                niter,
                                converged,
                                ctx,
                                M,
                                eta,
                                n,
                                m,
                                device,
                            )

                            if not ok:
                                raise RuntimeError(
                                    "Converged low-rank "
                                    "L0/L1/L2 failed"
                                )

                            found = True

                            del ctx
                            gc.collect()
                            torch.cuda.synchronize()

                            break

                    del ctx
                    gc.collect()

                if found:
                    break

            if found:
                break

        if found:
            break

    print()
    print(
        "========================================"
    )

    if found:
        print(
            "CONVERGED LOW-RANK VALIDATION: PASS"
        )
    else:
        print(
            "No converged low-rank case found "
            "in the current scan range."
        )
        print(
            "This is not automatically a failure: "
            "the curvature guard may legitimately "
            "disable the low-rank update near convergence."
        )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()