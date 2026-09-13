#!/usr/bin/env python3
"""Example: learn embeddings through a differentiable SPLR OT layer.

Two sets of points live in R^d. The cost matrix is the mean squared
pairwise distance M_ij = ||X_i - Y_j||^2 / d (the 1/d factor keeps the
cost scale well-conditioned for the solver); the SPLR solver produces the
entropic-OT plan T; a target alignment (point i should match point i)
defines the loss.
Gradients flow THROUGH the OT layer back into the embeddings X:

    loss -> T -> splr_ot backward -> M -> pairwise distance -> X

Run:
    python examples/differentiable_splr_ot.py

Expected: initial/final loss printed, final << initial, PASS.
"""

import torch

import curegot

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    if DEVICE.type != "cuda":
        raise RuntimeError("this example requires CUDA (solver is "
                           "CUDA float64 only)")

    n = m = 48
    d = 8
    eta = 0.05

    g = torch.Generator(device="cpu").manual_seed(0)
    Y = torch.rand((m, d), generator=g, dtype=torch.float64).to(DEVICE)
    a = torch.full((n,), 1.0 / n, dtype=torch.float64, device=DEVICE)
    b = torch.full((m,), 1.0 / m, dtype=torch.float64, device=DEVICE)

    # learnable source embeddings; target: point i matches point i
    X = torch.nn.Parameter(
        torch.rand((n, d), generator=g, dtype=torch.float64).to(DEVICE))
    affinity = torch.eye(n, dtype=torch.float64, device=DEVICE)

    ot = curegot.torch.SPLROT(eta=eta, tol=1e-8, max_iter=1000,
                              density=0.10)
    opt = torch.optim.Adam([X], lr=0.05)

    losses = []
    for step in range(60):
        opt.zero_grad()
        # cost conditioning: average squared distance (per-dimension) keeps
        # the dynamic range of M small enough for tol=1e-8 within
        # max_iter=1000, without cutting the autograd chain to X
        M = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(dim=-1) / d
        T = ot(M, a, b)                    # differentiable OT plan
        loss = -(T * affinity).sum()       # maximize matched mass
        loss.backward()                    # gradients reach X
        opt.step()
        losses.append(float(loss.detach()))

    print(f"initial loss: {losses[0]:.6e}")
    print(f"final loss:   {losses[-1]:.6e}")
    ok = losses[-1] < losses[0] and bool(torch.isfinite(X.grad).all())
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
