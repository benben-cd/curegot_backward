#!/usr/bin/env python3
"""Stage 4 validation: end-to-end autograd integration.

Covers (Stage 4 prompt, sections 15-35):
  1. requires_grad semantics (no-grad forward vs grad-enabled forward)
  2. linear-loss manual equivalence      (rel err <= 1e-10, bitwise reported)
  3. nonlinear losses (T**2, sin(T))     vs manual backward
  4. non-contiguous grad_output
  5. in-place safety (T and G unmodified by backward)
  6. context lifetime after forward locals are released
  7. multi-branch backward (G1 + G2 aggregation)
  8. multiple independent contexts
  9. out-of-order backward across graphs
 10. retain_graph repeated backward
 11. sampled finite difference (eps sweep)
 12. torch.autograd.gradcheck (best-effort, non-fatal)
 13. training-loop smoke (Adam, loss decreases overall)
 14. repeated forward/backward lifetime x100 + memory sanity
 15. error paths (contract, non-convergence, AMP, double backward)
 16. zero upstream gradient (exact short-circuit, no PCG breakdown)

Final output on success: STAGE 4 AUTOGRAD VALIDATION: PASS
"""

import gc
import sys

import torch

from stage3b_validate import make_problem
from splr_backward import solve_backward
from splr_autograd import splr_ot, splr_ot_with_info

import curegot

DEVICE = torch.device("cuda")
ETA = 0.05
TOL = 1e-8
MAX_ITER = 200

_PASSED = []


def report(name, ok=True, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        raise RuntimeError(f"validation failed: {name}")
    _PASSED.append(name)


def rel_err(x, ref):
    den = float(torch.linalg.norm(ref))
    num = float(torch.linalg.norm(x - ref))
    return num / den if den > 0 else num


def manual_forward(M, a, b, eta=ETA, tol=TOL, max_iter=MAX_ITER):
    out = curegot.torch.sinkhorn_splr_bwd(
        M, a, b, eta, tol=tol, max_iter=max_iter, verbose=0,
        density=0.10, candidate_sinkhorn_iter=3, context_mode="adopt",
    )
    assert bool(out["converged"]), "manual forward did not converge"
    return out


# ---------------------------------------------------------------------------
# 1. requires_grad semantics
# ---------------------------------------------------------------------------

def test_requires_grad_semantics():
    M, a, b = make_problem(8, 10, ETA, 0, DEVICE)
    T0 = splr_ot(M, a, b, eta=ETA, tol=TOL, max_iter=MAX_ITER, density=0.10)
    assert not T0.requires_grad and T0.grad_fn is None
    M1 = M.clone().requires_grad_(True)
    T1 = splr_ot(M1, a, b, eta=ETA, tol=TOL, max_iter=MAX_ITER, density=0.10)
    assert T1.requires_grad and T1.grad_fn is not None
    report("requires_grad semantics")


# ---------------------------------------------------------------------------
# 2. linear-loss manual equivalence (sections 15/16)
# ---------------------------------------------------------------------------

def test_linear_loss_equivalence():
    n, m = 8, 10
    M, a, b = make_problem(n, m, ETA, 1, DEVICE)
    W = torch.randn((n, m), dtype=torch.float64, device=DEVICE)

    # autograd path
    Ma = M.clone().requires_grad_(True)
    T = splr_ot(Ma, a, b, eta=ETA, tol=TOL, max_iter=MAX_ITER, density=0.10)
    loss = (W * T).sum()
    loss.backward()
    grad_auto = Ma.grad.clone()

    # manual path: same deterministic forward -> same T; G = W
    out = manual_forward(M, a, b)
    plan_eq = bool(torch.equal(T.detach(), out["plan"]))
    res = solve_backward(out["plan"], W, out["context"], ETA,
                         preconditioner="diag")
    assert res["converged"]
    grad_manual = res["grad"]

    err = rel_err(grad_auto, grad_manual)
    bitwise = bool(torch.equal(grad_auto, grad_manual))
    assert err <= 1e-10, f"rel err {err:.3e} > 1e-10"
    report("linear-loss manual equivalence",
           detail=f"rel err={err:.3e} bitwise={bitwise} plan_bitwise={plan_eq}")


# ---------------------------------------------------------------------------
# 3. nonlinear losses (section 19)
# ---------------------------------------------------------------------------

def _check_nonlinear(loss_fn, G_fn, name):
    n, m = 8, 10
    M, a, b = make_problem(n, m, ETA, 2, DEVICE)
    Ma = M.clone().requires_grad_(True)
    T = splr_ot(Ma, a, b, eta=ETA, tol=TOL, max_iter=MAX_ITER, density=0.10)
    loss_fn(T).backward()
    grad_auto = Ma.grad.clone()

    out = manual_forward(M, a, b)
    G = G_fn(out["plan"])
    res = solve_backward(out["plan"], G, out["context"], ETA,
                         preconditioner="diag")
    grad_manual = res["grad"]
    err = rel_err(grad_auto, grad_manual)
    assert err <= 1e-10, f"{name}: rel err {err:.3e}"
    report(f"nonlinear loss {name}", detail=f"rel err={err:.3e}")


def test_nonlinear_losses():
    _check_nonlinear(lambda T: (T ** 2).sum(), lambda T: 2 * T, "(T^2).sum")
    _check_nonlinear(lambda T: torch.sin(T).sum(), torch.cos, "sin(T).sum")


# ---------------------------------------------------------------------------
# 4. non-contiguous grad_output (section 9)
# ---------------------------------------------------------------------------

def test_noncontiguous_grad_output():
    n, m = 8, 10
    M, a, b = make_problem(n, m, ETA, 3, DEVICE)
    W2 = torch.randn((m, n), dtype=torch.float64, device=DEVICE)
    Ma = M.clone().requires_grad_(True)
    T = splr_ot(Ma, a, b, eta=ETA, tol=TOL, max_iter=MAX_ITER, density=0.10)
    # grad w.r.t. T arrives as W2.t(), which is non-contiguous
    loss = (W2 * T.t()).sum()
    loss.backward()
    grad_auto = Ma.grad.clone()

    out = manual_forward(M, a, b)
    res = solve_backward(out["plan"], W2.t(), out["context"], ETA,
                         preconditioner="diag")
    err = rel_err(grad_auto, res["grad"])
    assert err <= 1e-10, f"rel err {err:.3e}"
    report("non-contiguous grad_output", detail=f"rel err={err:.3e}")


# ---------------------------------------------------------------------------
# 5. in-place safety (section 29)
# ---------------------------------------------------------------------------

def test_inplace_safety():
    n, m = 8, 10
    M, a, b = make_problem(n, m, ETA, 4, DEVICE)
    Ma = M.clone().requires_grad_(True)
    T = splr_ot(Ma, a, b, eta=ETA, tol=TOL, max_iter=MAX_ITER, density=0.10)
    T_before = T.detach().clone()
    M_before = Ma.detach().clone()
    W = torch.randn((n, m), dtype=torch.float64, device=DEVICE)
    W_before = W.clone()
    (W * T).sum().backward()
    assert torch.equal(T.detach(), T_before), "T modified by backward"
    assert torch.equal(Ma.detach(), M_before), "M modified by backward"
    assert torch.equal(W, W_before), "upstream tensor modified by backward"
    report("in-place safety (T / M / upstream unmodified)")


# ---------------------------------------------------------------------------
# 6. context lifetime after forward locals released (section 6)
# ---------------------------------------------------------------------------

def test_context_lifetime():
    n, m = 8, 10
    M, a, b = make_problem(n, m, ETA, 5, DEVICE)
    Ma = M.clone().requires_grad_(True)

    def forward_only():
        # all forward locals (out dict, plan wrapper) die on return; only
        # the autograd ctx keeps the pybind backward context alive
        return splr_ot(Ma, a, b, eta=ETA, tol=TOL, max_iter=MAX_ITER,
                       density=0.10)

    T = forward_only()
    gc.collect()
    torch.cuda.synchronize()
    W = torch.randn((n, m), dtype=torch.float64, device=DEVICE)
    (W * T).sum().backward()
    assert Ma.grad is not None and bool(torch.isfinite(Ma.grad).all())
    report("context lifetime after forward locals released")


# ---------------------------------------------------------------------------
# 7. multi-branch backward (section 23)
# ---------------------------------------------------------------------------

def test_multi_branch():
    n, m = 8, 10
    M, a, b = make_problem(n, m, ETA, 6, DEVICE)
    W1 = torch.randn((n, m), dtype=torch.float64, device=DEVICE)
    W2 = torch.randn((n, m), dtype=torch.float64, device=DEVICE)
    Ma = M.clone().requires_grad_(True)
    T = splr_ot(Ma, a, b, eta=ETA, tol=TOL, max_iter=MAX_ITER, density=0.10)
    loss1 = (W1 * T).sum()
    loss2 = (W2 * T ** 2).sum()
    (loss1 + loss2).backward()
    grad_auto = Ma.grad.clone()

    out = manual_forward(M, a, b)
    G_combined = W1 + 2 * W2 * out["plan"]
    res = solve_backward(out["plan"], G_combined, out["context"], ETA,
                         preconditioner="diag")
    err = rel_err(grad_auto, res["grad"])
    assert err <= 1e-10, f"rel err {err:.3e}"
    report("multi-branch backward (G1+G2 aggregation)",
           detail=f"rel err={err:.3e}")


# ---------------------------------------------------------------------------
# 8. multiple independent contexts (section 24)
# ---------------------------------------------------------------------------

def test_multiple_contexts():
    n, m = 8, 10
    M1, a, b = make_problem(n, m, ETA, 7, DEVICE)
    M2 = M1 * 0.5 + 0.1
    W1 = torch.randn((n, m), dtype=torch.float64, device=DEVICE)
    W2 = torch.randn((n, m), dtype=torch.float64, device=DEVICE)
    Ma = M1.clone().requires_grad_(True)
    Mb = M2.clone().requires_grad_(True)
    T1 = splr_ot(Ma, a, b, eta=ETA, tol=TOL, max_iter=MAX_ITER, density=0.10)
    T2 = splr_ot(Mb, a, b, eta=ETA, tol=TOL, max_iter=MAX_ITER, density=0.10)
    loss = (W1 * T1).sum() + (W2 * T2).sum()
    loss.backward()

    out1 = manual_forward(M1, a, b)
    out2 = manual_forward(M2, a, b)
    r1 = solve_backward(out1["plan"], W1, out1["context"], ETA,
                        preconditioner="diag")
    r2 = solve_backward(out2["plan"], W2, out2["context"], ETA,
                        preconditioner="diag")
    e1 = rel_err(Ma.grad, r1["grad"])
    e2 = rel_err(Mb.grad, r2["grad"])
    assert e1 <= 1e-10 and e2 <= 1e-10, (e1, e2)
    report("multiple independent contexts",
           detail=f"err1={e1:.3e} err2={e2:.3e}")


# ---------------------------------------------------------------------------
# 9. out-of-order backward (section 25)
# ---------------------------------------------------------------------------

def test_out_of_order_backward():
    n, m = 8, 10
    M1, a, b = make_problem(n, m, ETA, 8, DEVICE)
    M2 = M1 * 0.7 - 0.05
    W1 = torch.randn((n, m), dtype=torch.float64, device=DEVICE)
    W2 = torch.randn((n, m), dtype=torch.float64, device=DEVICE)
    Ma = M1.clone().requires_grad_(True)
    Mb = M2.clone().requires_grad_(True)
    T1 = splr_ot(Ma, a, b, eta=ETA, tol=TOL, max_iter=MAX_ITER, density=0.10)
    T2 = splr_ot(Mb, a, b, eta=ETA, tol=TOL, max_iter=MAX_ITER, density=0.10)
    loss1 = (W1 * T1).sum()
    loss2 = (W2 * T2).sum()
    loss2.backward()   # reverse construction order
    loss1.backward()
    assert bool(torch.isfinite(Ma.grad).all())
    assert bool(torch.isfinite(Mb.grad).all())
    out1 = manual_forward(M1, a, b)
    r1 = solve_backward(out1["plan"], W1, out1["context"], ETA,
                        preconditioner="diag")
    assert rel_err(Ma.grad, r1["grad"]) <= 1e-10
    report("out-of-order backward across graphs")


# ---------------------------------------------------------------------------
# 10. retain_graph repeated backward (section 22)
# ---------------------------------------------------------------------------

def test_retain_graph():
    n, m = 8, 10
    M, a, b = make_problem(n, m, ETA, 9, DEVICE)
    W = torch.randn((n, m), dtype=torch.float64, device=DEVICE)
    Ma = M.clone().requires_grad_(True)
    T = splr_ot(Ma, a, b, eta=ETA, tol=TOL, max_iter=MAX_ITER, density=0.10)
    loss = (W * T).sum()
    loss.backward(retain_graph=True)
    g1 = Ma.grad.clone()
    loss.backward()
    g2 = Ma.grad.clone()
    # gradients accumulate: second call doubles
    assert torch.allclose(g2, 2 * g1, rtol=1e-12, atol=1e-15), \
        "repeated backward did not accumulate as expected"
    report("retain_graph repeated backward (accumulation correct)")


# ---------------------------------------------------------------------------
# 11. sampled finite difference (section 17)
# ---------------------------------------------------------------------------

def test_finite_difference():
    n, m = 5, 6
    # Harness-only accuracy regime (NOT the production default): tol=1e-8 /
    # max_iter=2000 matches the validated Stage 3 experiments. The earlier
    # 1e-9/500 choice made the FD *baseline forward itself* hit the iteration
    # cap (niter == max_iter) before any comparison -- a harness failure,
    # not a gradient mismatch. Nonconverged forwards remain forbidden here:
    # splr_ot raises on them by design.
    fd_tol, fd_max_iter = 1e-8, 2000
    # Deterministic problem seed. If this seed ever fails to converge at
    # 2000 iterations on a new setup, pick another fixed seed that converges
    # reliably and document it here -- do NOT relax fd_tol/fd_max_iter.
    FD_SEED = 10
    M, a, b = make_problem(n, m, ETA, FD_SEED, DEVICE)
    W = torch.randn((n, m), dtype=torch.float64, device=DEVICE)

    Ma = M.clone().requires_grad_(True)
    T = splr_ot(Ma, a, b, eta=ETA, tol=fd_tol, max_iter=fd_max_iter,
                density=0.10)
    (W * T).sum().backward()
    grad = Ma.grad.detach().clone()

    gen = torch.Generator(device="cpu").manual_seed(99)
    flat = torch.randperm(n * m, generator=gen)[:8].tolist()
    entries = [(f // m, f % m) for f in flat]

    def loss_at(Mp):
        Tp = splr_ot(Mp, a, b, eta=ETA, tol=fd_tol, max_iter=fd_max_iter,
                     density=0.10)
        return float((W * Tp).sum())

    fd_vec = torch.zeros(len(entries), dtype=torch.float64)
    g_vec = torch.tensor([grad[i, j] for i, j in entries],
                         dtype=torch.float64)
    best = None
    for eps in (1e-3, 3e-4, 1e-4, 3e-5):
        for k, (i, j) in enumerate(entries):
            Mp = M.clone(); Mp[i, j] += eps
            Mm = M.clone(); Mm[i, j] -= eps
            fd_vec[k] = (loss_at(Mp) - loss_at(Mm)) / (2 * eps)
        err = float(torch.linalg.norm(fd_vec - g_vec)
                    / torch.linalg.norm(g_vec))
        print(f"    eps={eps:<7}: normalized FD error = {err:.3e}")
        if best is None or err < best[1]:
            best = (eps, err)
    # solver-tolerance-limited; ~1e-4 expected, accept <= 1e-3 (section 17)
    assert best[1] <= 1e-3, f"best FD error {best[1]:.3e} > 1e-3"
    report("sampled finite difference",
           detail=f"best eps={best[0]} err={best[1]:.3e}")


# ---------------------------------------------------------------------------
# 12. gradcheck (best-effort, section 18)
# ---------------------------------------------------------------------------

def test_gradcheck_best_effort():
    n, m = 5, 6
    M, a, b = make_problem(n, m, ETA, 11, DEVICE)
    Mg = M.clone().requires_grad_(True)
    fn = lambda MM: splr_ot(MM, a, b, eta=ETA, tol=1e-9, max_iter=500,
                            density=0.10)
    try:
        ok = torch.autograd.gradcheck(fn, (Mg,), raise_exception=False)
    except Exception as e:  # noqa: BLE001
        print(f"    gradcheck raised: {type(e).__name__}: {e}")
        ok = False
    # Non-fatal by design: the forward is an iterative solver, so the
    # default gradcheck epsilon/rtol may be ill-suited (section 18). The
    # sampled finite-difference test above is the authoritative check.
    print(f"    torch.autograd.gradcheck returned: {ok} (non-fatal)")
    report("gradcheck best-effort (recorded, non-fatal)")


# ---------------------------------------------------------------------------
# 13. training-loop smoke (section 20)
# ---------------------------------------------------------------------------

def test_training_loop():
    n, m = 10, 12
    M0, a, b = make_problem(n, m, ETA, 12, DEVICE)
    M_target = M0 + 0.3 * torch.randn((n, m), dtype=torch.float64,
                                      device=DEVICE)
    T_target = splr_ot_with_info(M_target, a, b, eta=ETA, tol=TOL,
                                 max_iter=MAX_ITER, density=0.10)["plan"]
    M_param = torch.nn.Parameter(M0.clone())
    # lr=0.1 oscillates on this landscape (verified); 0.03 converges
    # smoothly within the 20-50 step budget (section 20).
    opt = torch.optim.Adam([M_param], lr=0.03)
    losses = []
    for step in range(40):
        opt.zero_grad()
        T = splr_ot(M_param, a, b, eta=ETA, tol=TOL, max_iter=MAX_ITER,
                    density=0.10)
        loss = ((T - T_target) ** 2).mean()
        loss.backward()
        assert bool(torch.isfinite(loss))
        assert bool(torch.isfinite(M_param.grad).all())
        opt.step()
        losses.append(float(loss.detach()))
    print(f"    loss[0]={losses[0]:.6e} loss[-1]={losses[-1]:.6e}")
    assert losses[-1] < losses[0], "loss did not decrease overall"
    report("training-loop smoke (40 Adam steps, loss decreased)")


# ---------------------------------------------------------------------------
# 14. repeated lifetime x100 + memory sanity (sections 21/30)
# ---------------------------------------------------------------------------

def test_repeated_lifetime():
    n, m = 20, 24
    M, a, b = make_problem(n, m, ETA, 13, DEVICE)
    W = torch.randn((n, m), dtype=torch.float64, device=DEVICE)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem_start = torch.cuda.memory_allocated()
    mem_after_warmup = None
    for it in range(100):
        Ma = M.clone().requires_grad_(True)
        T = splr_ot(Ma, a, b, eta=ETA, tol=TOL, max_iter=MAX_ITER,
                    density=0.10)
        loss = (W * T).sum()
        loss.backward()
        assert bool(torch.isfinite(Ma.grad).all()), f"iter {it}: non-finite"
        del Ma, T, loss
        if it == 4:
            torch.cuda.synchronize()
            mem_after_warmup = torch.cuda.memory_allocated()
        if it % 10 == 9:
            torch.cuda.synchronize()
    gc.collect()
    torch.cuda.synchronize()
    mem_end = torch.cuda.memory_allocated()
    growth = mem_end - mem_after_warmup
    print(f"    memory_allocated: start={mem_start} "
          f"after_warmup={mem_after_warmup} end={mem_end} "
          f"growth={growth} bytes")
    # caching allocator may retain reserved memory; allocated must not grow
    # linearly (section 30)
    assert growth <= (1 << 20), f"memory_allocated grew by {growth} bytes"
    report("repeated forward/backward lifetime x100 + memory sanity")


# ---------------------------------------------------------------------------
# 15. error paths (sections 4/28/31/32/33 + double backward)
# ---------------------------------------------------------------------------

def _expect(exc_types, fn, name):
    try:
        fn()
    except exc_types as e:
        print(f"    {name}: {type(e).__name__}: {str(e)[:90]}")
        return
    raise RuntimeError(f"{name}: expected {exc_types}, got no exception")


def test_error_paths():
    n, m = 8, 10
    M, a, b = make_problem(n, m, ETA, 14, DEVICE)

    _expect(ValueError,
            lambda: splr_ot(M, a, b, eta=ETA,
                            backward_preconditioner="python_diag"),
            "invalid preconditioner")
    _expect(ValueError,
            lambda: splr_ot(M, a, b[:-1], eta=ETA),
            "shape mismatch")
    _expect(RuntimeError,
            lambda: splr_ot(M.cpu(), a.cpu(), b.cpu(), eta=ETA),
            "CPU input")
    _expect(RuntimeError,
            lambda: splr_ot(M.float(), a.float(), b.float(), eta=ETA),
            "float32 input")
    _expect(NotImplementedError,
            lambda: splr_ot(M, a.clone().requires_grad_(True), b, eta=ETA),
            "a.requires_grad")
    _expect(NotImplementedError,
            lambda: splr_ot(M, a, b.clone().requires_grad_(True), eta=ETA),
            "b.requires_grad")

    # backward PCG non-convergence -> RuntimeError (not silent bad gradient)
    def pcg_nonconv():
        Ma = M.clone().requires_grad_(True)
        T = splr_ot(Ma, a, b, eta=ETA, tol=TOL, max_iter=MAX_ITER,
                    density=0.10, backward_max_iter=1)
        (T ** 2).sum().backward()
    _expect(RuntimeError, pcg_nonconv, "backward PCG non-convergence")

    # forward non-convergence -> RuntimeError
    _expect(RuntimeError,
            lambda: splr_ot(M, a, b, eta=ETA, tol=1e-14, max_iter=1,
                            density=0.10),
            "forward non-convergence")

    # AMP/autocast -> RuntimeError
    def amp():
        with torch.autocast("cuda"):
            splr_ot(M, a, b, eta=ETA)
    _expect(RuntimeError, amp, "autocast active")

    # double backward -> clean error (once_differentiable). torch>=2.x
    # semantics: the first grad(create_graph=True) returns an alias wired
    # to a DelayedError node; the error fires on the SECOND differentiation.
    # A nonlinear loss is required so grad_output requires grad (with a
    # linear loss PyTorch silently returns a non-differentiable gradient --
    # documented upstream limitation, see STAGE4_AUTOGRAD.md).
    def double_backward():
        Ma = M.clone().requires_grad_(True)
        T = splr_ot(Ma, a, b, eta=ETA, tol=TOL, max_iter=MAX_ITER,
                    density=0.10)
        loss = (T ** 2).sum()
        g = torch.autograd.grad(loss, Ma, create_graph=True)[0]
        g.sum().backward()
    _expect(RuntimeError, double_backward, "double backward (create_graph)")

    # exception safety: contexts from failed calls must not corrupt later
    # calls -- run a clean forward/backward afterwards
    Ma = M.clone().requires_grad_(True)
    T = splr_ot(Ma, a, b, eta=ETA, tol=TOL, max_iter=MAX_ITER, density=0.10)
    (T ** 2).sum().backward()
    assert bool(torch.isfinite(Ma.grad).all())
    report("error paths (9 cases) + post-error clean call")


# ---------------------------------------------------------------------------
# 16. zero upstream gradient (exact short-circuit)
# ---------------------------------------------------------------------------

def test_zero_upstream_gradient():
    n, m = 8, 10
    M, a, b = make_problem(n, m, ETA, 15, DEVICE)
    Ma = M.clone().requires_grad_(True)
    T = splr_ot(Ma, a, b, eta=ETA, tol=TOL, max_iter=MAX_ITER, density=0.10)
    # a loss whose gradient w.r.t. T is exactly zero (masked out)
    loss = (0.0 * T).sum()
    loss.backward()
    assert Ma.grad is not None
    assert bool(torch.all(Ma.grad == 0.0)), "zero G must give zero grad"
    # manual path agrees and reports 0 PCG iterations
    out = manual_forward(M, a, b)
    res = solve_backward(out["plan"], torch.zeros_like(out["plan"]),
                         out["context"], ETA)
    assert res["iterations"] == 0 and res["converged"]
    assert bool(torch.all(res["grad"] == 0.0))
    report("zero upstream gradient (exact short-circuit)")


# ---------------------------------------------------------------------------

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 4 validation requires CUDA")
    print("=" * 72)
    print("STAGE 4 AUTOGRAD VALIDATION")
    print("=" * 72)
    test_requires_grad_semantics()
    test_linear_loss_equivalence()
    test_nonlinear_losses()
    test_noncontiguous_grad_output()
    test_inplace_safety()
    test_context_lifetime()
    test_multi_branch()
    test_multiple_contexts()
    test_out_of_order_backward()
    test_retain_graph()
    test_finite_difference()
    test_gradcheck_best_effort()
    test_training_loop()
    test_repeated_lifetime()
    test_error_paths()
    test_zero_upstream_gradient()
    print("=" * 72)
    print(f"STAGE 4 AUTOGRAD VALIDATION: PASS ({len(_PASSED)} test groups)")


if __name__ == "__main__":
    main()
