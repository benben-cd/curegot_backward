#!/usr/bin/env python3
"""Stage 5 API validation: production packaging + public API.

Covers (Stage 5 prompt, sections 9-11, 17-19, 27-33):
  1. import curegot (package, not cwd luck)
  2. curegot.torch.splr_ot / splr_ot_with_info / SPLROT / splr_ot_batch /
     solve_backward exist
  3. old curegot.torch.sinkhorn_splr still exists
  4. old curegot.torch.sinkhorn_splr_bwd (+ SPLRBackwardContext) still exist
  5. public splr_ot forward works (shape + marginals)
  6. backward works (M.grad populated)
  7. M.grad finite and matches the native manual path
  8. result matches the Stage 4 legacy shim (object identity + numerics)
  9. exposed solve_backward works
 10. invalid inputs preserve Stage 4 errors (+ Stage 5 scalar validation)
 11. fresh-process import from an unrelated cwd (subprocess)
 12. SPLROT module behavior (repr / state_dict / train-eval invariance)
 13. splr_ot_batch matches per-slice splr_ot and propagates gradients
 14. torch.compile audit (best-effort, non-fatal)
 15. TorchScript audit (best-effort, non-fatal; expected unsupported)

Final output on success: STAGE 5 API VALIDATION: PASS
"""

import os
import subprocess
import sys

import torch

import curegot

DEVICE = torch.device("cuda")
ETA = 0.05
TOL = 1e-8
MAX_ITER = 200

_PASSED = []


def report(name, detail=""):
    _PASSED.append(name)
    print(f"  [PASS] {name}" + (f"  ({detail})" if detail else ""))


def make_problem(n, m, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    M = (torch.rand((n, m), generator=g, dtype=torch.float64) * 0.5
         ).to(DEVICE)
    a = (torch.rand(n, generator=g, dtype=torch.float64) + 0.5).to(DEVICE)
    b = (torch.rand(m, generator=g, dtype=torch.float64) + 0.5).to(DEVICE)
    return M, a / a.sum(), b / b.sum()


def seeded_randn(shape, seed):
    """Deterministic standard-normal tensor on DEVICE (explicit generator)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=g, dtype=torch.float64).to(DEVICE)


def expect(exc, fn, name):
    try:
        fn()
    except exc as e:
        print(f"    {name}: {type(e).__name__}: {str(e)[:80]}")
        return
    raise RuntimeError(f"{name}: expected {exc}, got no exception")


# ---------------------------------------------------------------------------
# 1-4. import surface
# ---------------------------------------------------------------------------

def test_import_surface():
    assert hasattr(curegot, "__file__"), "curegot has no __file__"
    print(f"    curegot loaded from: {curegot.__file__}")
    for sym in ("splr_ot", "splr_ot_with_info", "SPLROT", "splr_ot_batch",
                "solve_backward"):
        assert hasattr(curegot.torch, sym), f"curegot.torch.{sym} missing"
    report("curegot.torch public API present")
    assert hasattr(curegot.torch, "sinkhorn_splr")
    report("legacy curegot.torch.sinkhorn_splr present")
    assert hasattr(curegot.torch, "sinkhorn_splr_bwd")
    assert hasattr(curegot.torch, "SPLRBackwardContext")
    report("legacy sinkhorn_splr_bwd + SPLRBackwardContext present")


# ---------------------------------------------------------------------------
# 5-7. forward / backward / finite gradient
# ---------------------------------------------------------------------------

def test_forward_backward():
    n, m = 40, 48
    M, a, b = make_problem(n, m, 0)
    Ma = M.clone().requires_grad_(True)
    T = curegot.torch.splr_ot(Ma, a, b, eta=ETA, tol=TOL, max_iter=MAX_ITER,
                              density=0.10)
    assert T.shape == (n, m) and T.dtype == torch.float64
    assert T.device == Ma.device, "plan device not preserved"
    assert torch.allclose(T.sum(dim=1), a, atol=1e-6), "row marginals off"
    assert torch.allclose(T.sum(dim=0), b, atol=1e-6), "col marginals off"
    report("public splr_ot forward works (shape + marginals)")

    W = seeded_randn((n, m), 100)
    (W * T).sum().backward()
    assert Ma.grad is not None and Ma.grad.shape == (n, m)
    report("backward works (M.grad populated)")

    assert bool(torch.isfinite(Ma.grad).all())
    # cross-check against the native manual path
    out = curegot.torch.sinkhorn_splr_bwd(
        M, a, b, ETA, TOL, MAX_ITER, 0,
        density=0.10, candidate_sinkhorn_iter=3, context_mode="adopt")
    assert bool(out["converged"])
    res = curegot.torch.solve_backward(out["plan"], W, out["context"], ETA)
    rel = float(torch.linalg.norm(Ma.grad - res["grad"])
                / torch.linalg.norm(res["grad"]))
    bitwise = bool(torch.equal(Ma.grad, res["grad"]))
    assert rel <= 1e-10, f"public vs manual grad rel err {rel:.3e}"
    report("M.grad finite + matches manual path",
           f"rel={rel:.3e} bitwise={bitwise}")


# ---------------------------------------------------------------------------
# 8. Stage 4 legacy shim equivalence
# ---------------------------------------------------------------------------

def test_legacy_shim():
    import splr_autograd  # repo-root shim (must re-export the package)

    assert splr_autograd.splr_ot is curegot.autograd.splr_ot, \
        "shim does not re-export the package implementation"
    n, m = 32, 40
    M, a, b = make_problem(n, m, 1)
    W = seeded_randn((n, m), 101)
    g_pkg, g_shim = None, None
    for tag, fn in (("pkg", curegot.torch.splr_ot), ("shim",
                                                    splr_autograd.splr_ot)):
        Ma = M.clone().requires_grad_(True)
        T = fn(Ma, a, b, eta=ETA, tol=TOL, max_iter=MAX_ITER, density=0.10)
        (W * T).sum().backward()
        if tag == "pkg":
            g_pkg = Ma.grad.clone()
        else:
            g_shim = Ma.grad.clone()
    rel = float(torch.linalg.norm(g_pkg - g_shim)
                / torch.linalg.norm(g_shim))
    bitwise = bool(torch.equal(g_pkg, g_shim))
    assert rel <= 1e-12, f"shim vs package rel err {rel:.3e}"
    report("legacy shim equivalence (single implementation)",
           f"identity=True rel={rel:.3e} bitwise={bitwise}")


# ---------------------------------------------------------------------------
# 10. error contract (Stage 4 errors preserved + Stage 5 scalar checks)
# ---------------------------------------------------------------------------

def test_error_contract():
    n, m = 16, 20
    M, a, b = make_problem(n, m, 2)
    ot = curegot.torch.splr_ot

    expect(RuntimeError, lambda: ot(M.cpu(), a.cpu(), b.cpu(), eta=ETA),
           "CPU input")
    expect(RuntimeError,
           lambda: ot(M.float(), a.float(), b.float(), eta=ETA), "float32")
    expect(ValueError, lambda: ot(M, a, b[:-1], eta=ETA), "shape mismatch")
    expect(ValueError, lambda: ot(M, a, b, eta=ETA,
                                  backward_preconditioner="python_diag"),
           "invalid preconditioner")
    expect(NotImplementedError,
           lambda: ot(M, a.clone().requires_grad_(True), b, eta=ETA),
           "a.requires_grad")
    # Stage 5 scalar validation
    expect(ValueError, lambda: ot(M, a, b, eta=0.0), "eta=0")
    expect(ValueError, lambda: ot(M, a, b, eta=-1.0), "eta<0")
    expect(ValueError, lambda: ot(M, a, b, eta=ETA, tol=0.0), "tol=0")
    expect(ValueError, lambda: ot(M, a, b, eta=ETA, max_iter=0),
           "max_iter=0")
    expect(ValueError, lambda: ot(M, a, b, eta=ETA, density=2.0),
           "density>1")
    expect(ValueError, lambda: ot(M, a, b, eta=ETA, density=0.0),
           "density=0")
    expect(ValueError, lambda: ot(M, a, b, eta=ETA, candidate_sinkhorn_iter=-1),
           "candidate_sinkhorn_iter<0")
    expect(ValueError, lambda: ot(M, a, b, eta=ETA, backward_tol=0.0),
           "backward_tol=0")
    expect(ValueError, lambda: ot(M, a, b, eta=ETA, backward_max_iter=0),
           "backward_max_iter=0")
    # batched input is rejected with a clear message (single-problem API)
    expect(ValueError,
           lambda: ot(torch.zeros(2, n, m, dtype=torch.float64,
                                  device=DEVICE),
                      torch.zeros(2, n, dtype=torch.float64, device=DEVICE),
                      torch.zeros(2, m, dtype=torch.float64, device=DEVICE),
                      eta=ETA),
           "batched M rejected (use splr_ot_batch)")
    report("error contract (5 Stage-4 + 8 scalar + batch-reject cases)")


# ---------------------------------------------------------------------------
# 11. fresh-process import from unrelated cwd
# ---------------------------------------------------------------------------

def test_fresh_process_import():
    code = (
        "import curegot; "
        "print(curegot.__file__); "
        "print(curegot.torch.splr_ot); "
        "print(curegot.torch.SPLROT); "
        "print(curegot.torch.sinkhorn_splr_bwd)"
    )
    env = dict(os.environ)  # inherits PYTHONPATH (editable install path)
    proc = subprocess.run([sys.executable, "-c", code], cwd="/tmp",
                          env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"fresh-process import failed (cwd=/tmp):\n{proc.stderr}")
    for line in proc.stdout.strip().splitlines():
        print(f"    /tmp$ {line}")
    report("fresh-process import from /tmp (no cwd dependence)")


# ---------------------------------------------------------------------------
# 12. SPLROT module behavior
# ---------------------------------------------------------------------------

def test_splrot_module():
    layer = curegot.torch.SPLROT(eta=ETA, backward_preconditioner="diag")
    r = repr(layer)
    print(f"    repr: {r}")
    assert "eta=0.05" in r and "backward_preconditioner='diag'" in r
    assert len(layer.state_dict()) == 0, "SPLROT must be parameter-free"
    assert len(list(layer.parameters())) == 0

    n, m = 24, 28
    M, a, b = make_problem(n, m, 3)
    layer.train()
    T_train = layer(M, a, b)
    layer.eval()
    T_eval = layer(M, a, b)
    # Two independent GPU iterative-solver executions: bitwise identity is
    # NOT required (run-to-run scheduling noise), but train()/eval() must
    # not change mathematical behavior -> strict numerical equivalence.
    err = float(torch.linalg.norm(T_train - T_eval)
                / torch.linalg.norm(T_eval))
    assert err <= 1e-12, f"train/eval plan rel err {err:.3e} > 1e-12"
    assert torch.allclose(T_train, T_eval, rtol=1e-12, atol=1e-14)
    print(f"    train/eval rel err = {err:.3e}")
    layer.to(DEVICE)  # must be a safe no-op (no buffers managed)
    T2 = layer(M, a, b)
    err2 = float(torch.linalg.norm(T_train - T2) / torch.linalg.norm(T2))
    assert err2 <= 1e-12, f".to(device) plan rel err {err2:.3e} > 1e-12"
    print(f"    .to(device) rel err = {err2:.3e}")

    Ma = M.clone().requires_grad_(True)
    T = layer(Ma, a, b)
    (T ** 2).sum().backward()
    assert Ma.grad is not None and bool(torch.isfinite(Ma.grad).all())

    torch.save(layer.state_dict(), os.devnull)  # empty state_dict saves fine
    report("SPLROT module (repr/state_dict/train-eval invariance/grad)")


# ---------------------------------------------------------------------------
# 13. splr_ot_batch
# ---------------------------------------------------------------------------

def test_batch_wrapper():
    B, n, m = 3, 16, 20
    M, a, b = make_problem(n, m, 4)
    Mb = M.unsqueeze(0).repeat(B, 1, 1)
    Mb = Mb + 0.05 * seeded_randn((B, n, m), 102)  # per-problem variation
    ab = a.unsqueeze(0).repeat(B, 1)
    bb = b.unsqueeze(0).repeat(B, 1)
    Mb = Mb.requires_grad_(True)
    T = curegot.torch.splr_ot_batch(Mb, ab, bb, eta=ETA, tol=TOL,
                                    max_iter=MAX_ITER, density=0.10)
    assert T.shape == (B, n, m)
    # matches per-slice splr_ot. Two independent GPU iterative solves are
    # NOT required to be bitwise identical -- use strict numerical
    # equivalence instead.
    max_err = 0.0
    for i in range(B):
        Ti = curegot.torch.splr_ot(Mb[i].detach(), ab[i], bb[i], eta=ETA,
                                   tol=TOL, max_iter=MAX_ITER, density=0.10)
        err = float(torch.linalg.norm(T[i].detach() - Ti)
                    / torch.linalg.norm(Ti))
        max_err = max(max_err, err)
        assert err <= 1e-12, f"slice {i} rel err {err:.3e} > 1e-12"
        assert torch.allclose(T[i].detach(), Ti, rtol=1e-12, atol=1e-14)
    print(f"    max slice rel err = {max_err:.3e}")
    W = seeded_randn((B, n, m), 103)
    (W * T).sum().backward()
    assert Mb.grad is not None and bool(torch.isfinite(Mb.grad).all())
    assert bool((Mb.grad != 0).any())
    report("splr_ot_batch (sequential; per-slice numerical equivalence; "
           "grads flow)", f"max_slice_rel_err={max_err:.3e}")


# ---------------------------------------------------------------------------
# 14/15. torch.compile / TorchScript audits (non-fatal)
# ---------------------------------------------------------------------------

def test_compile_audit():
    n, m = 24, 28
    M, a, b = make_problem(n, m, 5)
    W = seeded_randn((n, m), 104)

    def loss_fn(MM):
        T = curegot.torch.splr_ot(MM, a, b, eta=ETA, tol=TOL,
                                  max_iter=MAX_ITER, density=0.10)
        return (W * T).sum()

    ref = float(loss_fn(M))
    try:
        compiled = torch.compile(loss_fn)
        got = float(compiled(M))
        rel = abs(got - ref) / abs(ref)
        status = (f"runs under torch.compile (graph breaks expected around "
                  f"the custom Function), rel={rel:.3e}")
        assert rel <= 1e-10
    except Exception as e:  # noqa: BLE001
        status = f"unsupported: {type(e).__name__}: {str(e)[:80]}"
    print(f"    torch.compile audit: {status}")
    print("    (recorded, non-fatal -- torch.compile is NOT guaranteed)")
    report("torch.compile audit recorded (non-fatal)")


def test_torchscript_audit():
    layer = curegot.torch.SPLROT(eta=ETA)
    try:
        torch.jit.script(layer)
        status = "unexpectedly scriptable"
    except Exception as e:  # noqa: BLE001
        status = f"unsupported as expected: {type(e).__name__}"
    print(f"    TorchScript audit: {status}")
    report("TorchScript audit recorded (unsupported, non-fatal)")


# ---------------------------------------------------------------------------

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 5 API validation requires CUDA")
    print("=" * 72)
    print("STAGE 5 API VALIDATION")
    print("=" * 72)
    test_import_surface()
    test_forward_backward()
    test_legacy_shim()
    test_error_contract()
    test_fresh_process_import()
    test_splrot_module()
    test_batch_wrapper()
    test_compile_audit()
    test_torchscript_audit()
    print("=" * 72)
    print(f"STAGE 5 API VALIDATION: PASS ({len(_PASSED)} test groups)")


if __name__ == "__main__":
    main()
