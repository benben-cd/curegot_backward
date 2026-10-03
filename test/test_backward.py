"""Small GPU correctness suite; no historical files or compatibility imports.

Run from an installed candidate: python -B test/test_backward.py
"""
import gc
import warnings
from unittest.mock import patch
import torch
import curegot
from curegot.backward import solve_backward
from curegot._backward_core import q_matvec


def problem(n=16,m=19,seed=42):
    g=torch.Generator().manual_seed(seed)
    M=torch.rand(n,m,generator=g,dtype=torch.float64).cuda()
    a=torch.rand(n,generator=g,dtype=torch.float64).cuda()+.5
    b=torch.rand(m,generator=g,dtype=torch.float64).cuda()+.5
    return M,a/a.sum(),b/b.sum()


def relative(x,y):return float(torch.linalg.norm(x-y)/torch.linalg.norm(y).clamp_min(1e-30))


def dense(T,G,eta):
    n,m=T.shape
    Q=torch.cat([torch.cat([torch.diag(T.sum(1)),T[:,:-1]],1),
                 torch.cat([T[:,:-1].T,torch.diag(T.sum(0)[:-1])],1)],0)
    GT=G*T;mu=torch.cat([GT.sum(1),GT.sum(0)[:-1]])
    lam=torch.linalg.solve(Q,mu)
    beta=torch.cat([lam[n:],lam.new_zeros(1)])
    grad=T*(lam[:n,None]+beta[None,:]-G)/eta
    return Q,mu,lam,grad


def run():
    assert torch.cuda.is_available()
    # Forward 1e-8 avoids testing the known near-machine-precision stall.
    # Backward tolerance remains the production default 1e-12.
    M,a,b=problem();eta=.3
    out=curegot.torch.sinkhorn_splr_bwd(M,a,b,eta,1e-8,1000,0,density=.5)
    assert out['converged'] and out['context'].adopted_forward_state
    T=out['plan'];ctx=out['context']
    G=torch.randn(T.shape,generator=torch.Generator().manual_seed(9),dtype=T.dtype).cuda()
    Q,mu,lam,grad=dense(T,G,eta)
    assert relative(q_matvec(T,lam,*T.shape),Q@lam)<1e-13
    for mode in ('none','diag','splr'):
        res=solve_backward(T,G,ctx,eta,preconditioner=mode)
        residual=relative(Q@res['lambda'],mu)
        assert res['converged'] and residual<2e-12
        assert relative(res['lambda'],lam)<1e-8 and relative(res['grad'],grad)<1e-8
        z=solve_backward(T,torch.zeros_like(G),ctx,eta,preconditioner=mode)
        assert z['converged'] and z['iterations']==0 and torch.count_nonzero(z['grad'])==0
        print(f'PASS dense oracle / {mode}: residual={residual:.3e}',flush=True)
    # Internal CSR accessors return copies. Expand the stored symmetric triangle.
    vals=ctx._csr_values();cols=ctx._csr_colind().long();ptr=ctx._csr_rowptr().long()
    A=torch.zeros_like(Q)
    for i in range(Q.shape[0]):A[i,cols[ptr[i]:ptr[i+1]]]=vals[ptr[i]:ptr[i+1]]
    A=A+A.T-torch.diag(A.diag())
    assert relative(ctx.apply_diagonal_preconditioner(torch.ones_like(mu)),1/A.diag())<1e-13
    assert relative(ctx.apply_diagonal_preconditioner(mu),mu/A.diag())<1e-13
    print('PASS native diagonal / adopted context',flush=True)
    fb=curegot.torch.sinkhorn_splr_bwd(M,a,b,eta,1e-7,1000,0,density=.5,x0=out['dual'])
    assert fb['converged'] and fb['niter']==0 and fb['context'].is_fallback and not fb['context'].low_rank_enabled
    rf=solve_backward(fb['plan'],G,fb['context'],eta,preconditioner='splr')
    _,_,lf,gf=dense(fb['plan'],G,eta)
    assert relative(rf['lambda'],lf)<1e-8 and relative(rf['grad'],gf)<1e-8
    print('PASS immediate-convergence fallback',flush=True)
    for mode in ('none','diag','splr'):
        cost=M.clone().requires_grad_(True)
        plan=curegot.torch.splr_ot(cost,a,b,eta=eta,tol=1e-8,density=.5,backward_preconditioner=mode)
        _,_,_,expected=dense(plan.detach(),G,eta)
        loss=(plan*G).sum();loss.backward(retain_graph=True)
        assert relative(cost.grad,expected)<1e-8
        first=cost.grad.clone();cost.grad=None;gc.collect();loss.backward()
        assert relative(cost.grad,first)<1e-8
        # Finite difference of the scalar objective checks the implicit formula.
        eps=1e-3;direction=torch.zeros_like(M);direction[2,3]=1
        with torch.no_grad():
            plus=curegot.torch.splr_ot(M+eps*direction,a,b,eta=eta,tol=1e-8,density=.5)
            minus=curegot.torch.splr_ot(M-eps*direction,a,b,eta=eta,tol=1e-8,density=.5)
        fd=float(((plus-minus)*G).sum()/(2*eps))
        assert abs(fd-float(first[2,3]))<1e-5
        print(f'PASS {mode} autograd / repeated backward / finite difference',flush=True)
    # Forward locals disappear while the graph remains the context owner.
    cost=M.clone().requires_grad_(True)
    def make_loss():return curegot.torch.splr_ot(cost,a,b,eta=eta,tol=1e-8,density=.5,backward_preconditioner='splr').square().sum()
    loss=make_loss();gc.collect();loss.backward();assert torch.isfinite(cost.grad).all()
    try:curegot.torch.splr_ot(M,a.clone().requires_grad_(),b,eta=eta)
    except NotImplementedError:pass
    else:raise AssertionError('a gradient must be rejected')
    print('PASS context lifetime / input contract',flush=True)
    # Unsupported eta gradients must fail before float conversion warns/detaches.
    for device in ('cpu', M.device):
        differentiable_eta=torch.tensor(eta,dtype=torch.float64,device=device,requires_grad=True)
        for call in (
            lambda: curegot.torch.splr_ot(M,a,b,eta=differentiable_eta),
            lambda: curegot.torch.SPLROT(differentiable_eta),
        ):
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                try:call()
                except NotImplementedError as error:assert 'eta' in str(error)
                else:raise AssertionError('eta gradient must be rejected')
    for constant_eta in (eta,torch.tensor(eta,dtype=torch.float64),
                         torch.tensor(eta,dtype=torch.float64,device=M.device)):
        layer=curegot.torch.SPLROT(constant_eta,tol=1e-8,density=.5)
        assert layer.eta==eta and not list(layer.parameters()) and not list(layer.buffers())
        assert not hasattr(layer,'backward_context')
        # This is an API conversion test, independent of repeated forward trajectories.
        with patch.object(curegot.autograd._native_torch, 'sinkhorn_splr_bwd',
                          return_value=out) as native_forward:
            plans=(
                curegot.torch.splr_ot(M,a,b,eta=constant_eta,tol=1e-8,density=.5),
                layer(M,a,b),
            )
            assert native_forward.call_count==2
            for call in native_forward.call_args_list:
                assert isinstance(call.args[3],float) and call.args[3]==eta
            assert all(torch.equal(plan,T) for plan in plans)
    print('PASS constant eta / explicit eta gradient rejection',flush=True)

    # Public autograd must reject real solver iteration-limit failures.
    try:
        curegot.torch.splr_ot(M,a,b,eta=eta,tol=1e-8,max_iter=1,density=.5)
    except RuntimeError as error:
        assert 'forward solver did not converge' in str(error)
    else:
        raise AssertionError('non-converged forward must be rejected')
    cost=M.clone().requires_grad_(True)
    plan=curegot.torch.splr_ot(cost,a,b,eta=eta,tol=1e-8,density=.5,
                             backward_preconditioner='none',backward_max_iter=1)
    try:
        (plan*G).sum().backward()
    except RuntimeError as error:
        assert 'backward PCG did not converge' in str(error)
        assert cost.grad is None
    else:
        raise AssertionError('non-converged PCG must not return a gradient')
    print('PASS explicit forward / backward non-convergence rejection',flush=True)

    # Fixed previously validated rank-2 workload; require actual activation.
    Mr,ar,br=problem(100,120,40)
    ro=curegot.torch.sinkhorn_splr_bwd(Mr,ar,br,.03,1e-8,200,0,density=.1)
    assert ro['converged'] and ro['context'].low_rank_enabled and ro['context'].adopted_forward_state
    Gr=torch.randn(ro['plan'].shape,generator=torch.Generator().manual_seed(13),dtype=torch.float64).cuda()
    _,_,lr,gr=dense(ro['plan'],Gr,.03)
    rr=solve_backward(ro['plan'],Gr,ro['context'],.03,preconditioner='splr')
    assert rr['converged'] and relative(rr['lambda'],lr)<1e-8 and relative(rr['grad'],gr)<1e-8
    rc=ro['context'];dim=rc.dim
    values=rc._csr_values();columns=rc._csr_colind().long();rowptr=rc._csr_rowptr().long()
    Ar=torch.zeros(dim,dim,dtype=torch.float64,device=Mr.device)
    for i in range(dim):Ar[i,columns[rowptr[i]:rowptr[i+1]]]=values[rowptr[i]:rowptr[i+1]]
    Ar=Ar+Ar.T-torch.diag(Ar.diag())
    sec_s=rc._secant_s();sec_y=rc._secant_y();As=Ar@sec_s
    B=Ar+(.03/rc.ys)*torch.outer(sec_y,sec_y)-torch.outer(As,As)/torch.dot(sec_s,As)
    rhs=torch.randn(dim,generator=torch.Generator().manual_seed(5),dtype=torch.float64).cuda()
    assert relative(rc.apply_preconditioner(rhs),torch.linalg.solve(B,rhs))<1e-8
    print('PASS rank-2 active dense oracle / preconditioner B inverse',flush=True)
    print('BACKWARD CORE VALIDATION PASS',flush=True)


if __name__=='__main__':run()
