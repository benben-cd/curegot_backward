#!/usr/bin/env python3
"""Installed public-surface acceptance; never builds CUDA or imports root shims.

The parent runs this file's contents in a fresh isolated Python process in /tmp.
-I removes cwd/PYTHONPATH influence while retaining installed editable finders.
Missing installation, metadata, CUDA, or API is a failure, never a skipped PASS.
"""
import importlib.abc
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    class BlockRootShims(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in {
                'splr_autograd', 'splr_backward', 'backward_core'
            }:
                raise ImportError(f'release API must not import root shim {fullname}')
            return None

    sys.meta_path.insert(0, BlockRootShims())
    import curegot
    import torch

    check(Path.cwd() == Path('/tmp'), 'worker must run in /tmp')
    check(sys.flags.isolated, 'worker must use isolated Python import paths')
    package = Path(curegot.__file__).resolve()
    print(f'package: {package}', flush=True)
    version = importlib.metadata.version('curegot')
    check(version == curegot.__version__, 'installed/package version mismatch')
    print(f'[PASS] fresh-process import, metadata/version: {version}', flush=True)
    for name in ('splr_ot', 'splr_ot_with_info', 'SPLROT', 'splr_ot_batch',
                 'solve_backward', 'sinkhorn_splr_bwd', 'SPLRBackwardContext',
                 'sinkhorn_splr', 'sinkhorn_bcd'):
        check(callable(getattr(curegot.torch, name, None)), f'missing API: {name}')
    check(curegot.torch.__name__ == 'curegot._internal_torch.torch',
          'unexpected native torch namespace')
    print('[PASS] public, advanced and legacy API surface', flush=True)
    check(torch.cuda.is_available(), 'CUDA GPU required for release acceptance')
    print(f'GPU: {torch.cuda.get_device_name(0)}; torch: {torch.__version__}', flush=True)

    n, m, d = 64, 80, 8
    g = torch.Generator(device='cpu').manual_seed(0)
    X = torch.rand(n, d, generator=g, dtype=torch.float64).cuda()
    Y = torch.rand(m, d, generator=g, dtype=torch.float64).cuda()
    M = ((X[:, None, :] - Y[None, :, :]) ** 2).mean(-1).requires_grad_(True)
    a = torch.full((n,), 1.0/n, device='cuda', dtype=torch.float64)
    b = torch.full((m,), 1.0/m, device='cuda', dtype=torch.float64)
    eta = 0.05
    ot = curegot.torch.splr_ot
    T = ot(M, a, b, eta=eta, tol=1e-8)
    check(T.shape == M.shape and T.device == M.device and T.dtype == M.dtype,
          'plan shape/device/dtype mismatch')
    check(bool(torch.isfinite(T).all()) and bool((T >= 0).all()), 'invalid plan')
    check(torch.allclose(T.sum(1), a, atol=1e-6, rtol=0), 'row marginals')
    check(torch.allclose(T.sum(0), b, atol=1e-6, rtol=0), 'column marginals')
    (T ** 2).sum().backward()
    check(M.grad is not None and bool(torch.isfinite(M.grad).all())
          and float(M.grad.norm()) > 0, 'missing/invalid gradient')
    print('[PASS] README forward, nonlinear backward, finite nonzero gradient', flush=True)

    out = curegot.torch.sinkhorn_splr_bwd(
        M.detach(), a, b, eta, 1e-8, 1000, 0,
        candidate_sinkhorn_iter=3, context_mode='adopt')
    check(bool(out['converged']), 'manual forward did not converge')
    res = curegot.torch.solve_backward(out['plan'], 2 * T.detach(), out['context'], eta)
    check(bool(res['converged']), 'manual backward did not converge')
    check(bool(torch.isfinite(res['grad']).all()) and float(res['grad'].norm()) > 0,
          'invalid manual gradient')
    rel = float((M.grad - res['grad']).norm() / res['grad'].norm())
    check(rel <= 1e-10, f'manual gradient relative error {rel:.3e}')
    print(f'[PASS] public/manual gradient relative error {rel:.3e}', flush=True)

    layer = curegot.torch.SPLROT(eta=eta, tol=1e-8, backward_preconditioner='diag')
    Ml = M.detach().clone().requires_grad_(True)
    Tl = layer(Ml, a, b)
    check(torch.allclose(Tl, T, atol=1e-14, rtol=1e-12), 'module mismatch')
    Tl.square().sum().backward()
    check(Ml.grad is not None and bool(torch.isfinite(Ml.grad).all()), 'module backward')
    info = curegot.torch.splr_ot_with_info(M.detach(), a, b, eta=eta)
    check(info['converged'] and not info['plan'].requires_grad, 'debug contract')
    print('[PASS] SPLROT forward/backward and detached info', flush=True)

    def expect(exc, fn):
        try:
            fn()
        except exc:
            return
        raise AssertionError(f'expected {exc.__name__}')

    expect(RuntimeError, lambda: ot(M.cpu(), a.cpu(), b.cpu(), eta=eta))
    expect(RuntimeError, lambda: ot(M.float(), a.float(), b.float(), eta=eta))
    expect(ValueError, lambda: ot(M, a, b[:-1], eta=eta))
    expect(ValueError, lambda: ot(M, a, b, eta=0))
    expect(ValueError, lambda: ot(M, a, b, eta=eta, backward_preconditioner='invalid'))
    expect(NotImplementedError, lambda: ot(M, a.clone().requires_grad_(True), b, eta=eta))
    expect(NotImplementedError, lambda: ot(M, a, b.clone().requires_grad_(True), eta=eta))
    with torch.autocast('cuda'):
        expect(RuntimeError, lambda: ot(M, a, b, eta=eta))
    check(not {'splr_autograd', 'splr_backward', 'backward_core'} & sys.modules.keys(),
          'root shim was imported')
    print('[PASS] error contract and root-shim independence', flush=True)
    print('STAGE 6A RELEASE VALIDATION: PASS', flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--isolated-worker']:
        main()
    else:
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        result = subprocess.run(
            [sys.executable, '-I', '-c', Path(__file__).read_text(), '--isolated-worker'],
            cwd='/tmp', env=env)
        raise SystemExit(result.returncode)
