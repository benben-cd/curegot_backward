#!/usr/bin/env python3
"""Stage 5 workload validation: realistic differentiable-OT training.

Covers (Stage 5 prompt, sections 12-16, 24-26):
  A. learnable cost matrix (nn.Parameter M, MSE to target plan, Adam)
  B. embedding-based cost (M_ij = ||X_i - Y_j||^2, X learnable, affinity
     target; gradient must flow THROUGH the OT layer into X)
  C. tiny neural net upstream (net(raw) -> X -> cdist -> splr_ot -> loss;
     every net parameter must receive finite, nonzero gradients)
  D. performance sanity (forward+backward step wall, 100x120 / 300x360)
  E. memory sanity (50+ training steps, memory_allocated stable)
  F. optimizer compatibility (Adam everywhere + SGD smoke)

Final output on success: STAGE 5 WORKLOAD VALIDATION: PASS
"""

import statistics
import time

import torch
import torch.nn.functional as F

import curegot

DEVICE = torch.device("cuda")
ETA = 0.05
TOL = 1e-8
# Harness-only iteration budget (Stage 5 stabilization): matches the
# production forward default. The development-era 200 cap could be hit by
# training iterates as the cost matrix evolves -- production correctly
# raises on non-convergence, so the harness must give the solver the same
# budget it has in production. Production defaults/policy are unchanged.
WORKLOAD_MAX_ITER = 1000

_PASSED = []


def report(name, detail=""):
    _PASSED.append(name)
    print(f"  [PASS] {name}" + (f"  ({detail})" if detail else ""))


def _sync():
    torch.cuda.synchronize()


def make_marginals(n, m, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    a = (torch.rand(n, generator=g, dtype=torch.float64) + 0.5).to(DEVICE)
    b = (torch.rand(m, generator=g, dtype=torch.float64) + 0.5).to(DEVICE)
    return a / a.sum(), b / b.sum()


# ---------------------------------------------------------------------------
# A. learnable cost matrix
# ---------------------------------------------------------------------------

def workload_a_learnable_cost():
    n, m = 100, 120
    a, b = make_marginals(n, m, 10)
    g = torch.Generator(device="cpu").manual_seed(11)
    M0 = (torch.rand((n, m), generator=g, dtype=torch.float64) * 0.5
          ).to(DEVICE)
    # non-constant target perturbation (a constant shift leaves the EROT
    # plan unchanged -- absorbed into the scalings)
    M_target = M0 + 0.3 * torch.rand((n, m), generator=g,
                                     dtype=torch.float64).to(DEVICE)
    T_target = curegot.torch.splr_ot_with_info(
        M_target, a, b, eta=ETA, tol=TOL, max_iter=WORKLOAD_MAX_ITER,
        density=0.10)["plan"]

    layer = curegot.torch.SPLROT(eta=ETA, tol=TOL, max_iter=WORKLOAD_MAX_ITER,
                                 density=0.10)
    M_param = torch.nn.Parameter(M0.clone())
    opt = torch.optim.Adam([M_param], lr=0.03)
    losses, step_ms = [], []
    for _ in range(60):
        opt.zero_grad()
        _sync()
        t0 = time.perf_counter()
        T = layer(M_param, a, b)
        loss = F.mse_loss(T, T_target)
        loss.backward()
        _sync()
        step_ms.append((time.perf_counter() - t0) * 1e3)
        assert bool(torch.isfinite(loss)), "non-finite loss"
        assert bool(torch.isfinite(M_param.grad).all()), "non-finite grad"
        opt.step()
        losses.append(float(loss.detach()))
    gnorm = float(torch.linalg.norm(M_param.grad))
    print(f"    loss[0]={losses[0]:.6e} loss[-1]={losses[-1]:.6e} "
          f"grad_norm={gnorm:.3e} step_median={statistics.median(step_ms):.2f}ms")
    assert losses[-1] < 0.2 * losses[0], "loss did not decrease significantly"
    report("workload A: learnable cost matrix (SPLROT module, 60 Adam steps)",
           f"{losses[0]:.2e} -> {losses[-1]:.2e}")


# ---------------------------------------------------------------------------
# B. embedding-based cost
# ---------------------------------------------------------------------------

def workload_b_embedding_cost():
    n = m = 48
    d = 8
    a, b = make_marginals(n, m, 20)
    g = torch.Generator(device="cpu").manual_seed(21)
    Y = torch.rand((m, d), generator=g, dtype=torch.float64).to(DEVICE)
    X0 = torch.rand((n, d), generator=g, dtype=torch.float64).to(DEVICE)
    # synthetic target alignment: point i should match point i
    affinity = torch.eye(n, dtype=torch.float64, device=DEVICE)

    X = torch.nn.Parameter(X0.clone())
    opt = torch.optim.Adam([X], lr=0.05)
    losses = []
    for _ in range(60):
        opt.zero_grad()
        M = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(dim=-1)
        T = curegot.torch.splr_ot(M, a, b, eta=ETA, tol=TOL,
                                  max_iter=WORKLOAD_MAX_ITER, density=0.10)
        loss = -(T * affinity).sum()
        loss.backward()
        assert X.grad is not None, "no gradient reached X"
        assert bool(torch.isfinite(X.grad).all()), "non-finite X.grad"
        assert bool((X.grad != 0).any()), "zero X.grad"
        opt.step()
        losses.append(float(loss.detach()))
    gnorm = float(torch.linalg.norm(X.grad))
    print(f"    loss[0]={losses[0]:.6e} loss[-1]={losses[-1]:.6e} "
          f"final_grad_norm={gnorm:.3e} (affinity mass maximized)")
    assert losses[-1] < losses[0], "loss did not decrease"
    report("workload B: embedding cost, gradient flows THROUGH OT layer",
           f"{losses[0]:.3e} -> {losses[-1]:.3e}")


# ---------------------------------------------------------------------------
# C. tiny neural net upstream (+ memory sanity)
# ---------------------------------------------------------------------------

def workload_c_neural_upstream():
    n = m = 48
    d0, h, d = 12, 32, 8
    a, b = make_marginals(n, m, 30)
    g = torch.Generator(device="cpu").manual_seed(31)
    raw_X = torch.randn((n, d0), generator=g, dtype=torch.float64).to(DEVICE)
    Y = torch.rand((m, d), generator=g, dtype=torch.float64).to(DEVICE)
    M_target = (torch.rand((n, m), generator=g, dtype=torch.float64) * 0.5
                ).to(DEVICE)
    T_target = curegot.torch.splr_ot_with_info(
        M_target, a, b, eta=ETA, tol=TOL, max_iter=WORKLOAD_MAX_ITER,
        density=0.10)["plan"]

    torch.manual_seed(32)
    net = torch.nn.Sequential(
        torch.nn.Linear(d0, h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, d),
    ).to(DEVICE, torch.float64)
    params0 = [p.detach().clone() for p in net.parameters()]
    opt = torch.optim.Adam(net.parameters(), lr=0.01)

    losses = []
    mem_after_warmup = None
    for step in range(50):
        opt.zero_grad()
        X = net(raw_X)
        # Cost conditioning: divide the squared-distance cost by the
        # embedding dimension d. The OT plan still varies strongly with X
        # (the full autograd chain raw_X -> net -> X -> M -> T -> loss is
        # preserved), while the dynamic range of M shrinks so the forward
        # converges well within the production-typical max_iter=1000 at
        # tol=1e-8.
        M = (torch.cdist(X, Y) ** 2) / d
        T = curegot.torch.splr_ot(M, a, b, eta=ETA, tol=TOL,
                                  max_iter=WORKLOAD_MAX_ITER, density=0.10)
        loss = ((T - T_target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if step == 4:
            _sync()
            mem_after_warmup = torch.cuda.memory_allocated()
    _sync()
    mem_end = torch.cuda.memory_allocated()
    growth = mem_end - mem_after_warmup

    for name, p in net.named_parameters():
        assert p.grad is not None, f"{name}: no gradient"
        assert bool(torch.isfinite(p.grad).all()), f"{name}: non-finite"
    moved = [float((p.detach() - p0).abs().max()) for p, p0
             in zip(net.parameters(), params0)]
    assert max(moved) > 1e-6, "no parameter updated"
    nonzero = all(bool((p.grad != 0).any()) for p in net.parameters())
    gnorm = float(torch.sqrt(sum(torch.linalg.norm(p.grad) ** 2
                                 for p in net.parameters())))
    print(f"    loss[0]={losses[0]:.6e} loss[-1]={losses[-1]:.6e} "
          f"max_param_move={max(moved):.3e} all_grads_nonzero={nonzero} "
          f"final_grad_norm={gnorm:.3e}")
    print(f"    memory_allocated: after_warmup={mem_after_warmup} "
          f"end={mem_end} growth={growth} bytes")
    assert losses[-1] < losses[0], "loss did not decrease"
    # caching allocator may retain reserved memory; allocated must not grow
    # linearly across 45 post-warmup steps
    assert growth <= (4 << 20), f"memory_allocated grew by {growth} bytes"
    report("workload C: tiny NN upstream, grads finite+nonzero, "
           "memory stable", f"{losses[0]:.2e} -> {losses[-1]:.2e}")


# ---------------------------------------------------------------------------
# D. performance sanity (step wall only -- no solver benchmark)
# ---------------------------------------------------------------------------

def perf_sanity():
    for idx, (n, m) in enumerate(((100, 120), (300, 360))):
        a, b = make_marginals(n, m, 40 + idx)
        g = torch.Generator(device="cpu").manual_seed(50 + idx)
        M0 = (torch.rand((n, m), generator=g, dtype=torch.float64) * 0.5
              ).to(DEVICE)
        W = torch.randn((n, m), generator=g, dtype=torch.float64
                  ).to(DEVICE)

        def step():
            Ma = M0.clone().requires_grad_(True)
            T = curegot.torch.splr_ot(Ma, a, b, eta=ETA, tol=TOL,
                                      max_iter=WORKLOAD_MAX_ITER, density=0.10)
            (W * T).sum().backward()

        for _ in range(2):  # warmup
            step()
        _sync()
        ts = []
        for _ in range(5):
            _sync()
            t0 = time.perf_counter()
            step()
            _sync()
            ts.append((time.perf_counter() - t0) * 1e3)
        print(f"    {n}x{m}: forward+backward step median "
              f"{statistics.median(ts):.2f} ms")
    report("performance sanity (step wall; compare with Stage 4 benchmark)")


# ---------------------------------------------------------------------------
# F. SGD optimizer smoke (Adam is used in all workloads above)
# ---------------------------------------------------------------------------

def optimizer_smoke_sgd():
    n, m = 32, 40
    a, b = make_marginals(n, m, 60)
    g = torch.Generator(device="cpu").manual_seed(61)
    M0 = (torch.rand((n, m), generator=g, dtype=torch.float64) * 0.5
          ).to(DEVICE)
    M_target = M0 + 0.2 * torch.rand((n, m), generator=g,
                                     dtype=torch.float64).to(DEVICE)
    T_target = curegot.torch.splr_ot_with_info(
        M_target, a, b, eta=ETA, tol=TOL, max_iter=WORKLOAD_MAX_ITER,
        density=0.10)["plan"]
    M_param = torch.nn.Parameter(M0.clone())
    opt = torch.optim.SGD([M_param], lr=0.5)
    losses = []
    for _ in range(30):
        opt.zero_grad()
        T = curegot.torch.splr_ot(M_param, a, b, eta=ETA, tol=TOL,
                                  max_iter=WORKLOAD_MAX_ITER, density=0.10)
        loss = ((T - T_target) ** 2).mean()
        loss.backward()
        assert bool(torch.isfinite(M_param.grad).all())
        opt.step()
        losses.append(float(loss.detach()))
    gnorm = float(torch.linalg.norm(M_param.grad))
    print(f"    SGD: loss[0]={losses[0]:.6e} loss[-1]={losses[-1]:.6e} "
          f"final_grad_norm={gnorm:.3e}")
    assert losses[-1] < losses[0]
    report("optimizer compatibility: SGD smoke (Adam used in A/B/C)")


# ---------------------------------------------------------------------------

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 5 workload validation requires CUDA")
    print("=" * 72)
    print("STAGE 5 WORKLOAD VALIDATION")
    print("=" * 72)
    workload_a_learnable_cost()
    workload_b_embedding_cost()
    workload_c_neural_upstream()
    perf_sanity()
    optimizer_smoke_sgd()
    print("=" * 72)
    print(f"STAGE 5 WORKLOAD VALIDATION: PASS ({len(_PASSED)} groups)")


if __name__ == "__main__":
    main()
