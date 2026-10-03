"""Capture actual legacy forward outputs, then compare an independently built candidate."""
import argparse
import json
from pathlib import Path
import torch
import curegot


def problems():
    gen=torch.Generator().manual_seed(314)
    for kind in ('random','geometric','nonuniform'):
        n=m=32
        if kind=='random':M=torch.rand(n,m,generator=gen,dtype=torch.float64)
        else:
            x=torch.randn(n,2,generator=gen,dtype=torch.float64)*.3
            y=torch.randn(m,2,generator=gen,dtype=torch.float64)*.3
            M=torch.cdist(x,y).square()
        a=torch.ones(n,dtype=torch.float64);b=torch.ones(m,dtype=torch.float64)
        if kind=='nonuniform':
            a=torch.exp(torch.randn(n,generator=gen,dtype=torch.float64));b=torch.exp(torch.randn(m,generator=gen,dtype=torch.float64))
        yield kind,M.cuda(),(a/a.sum()).cuda(),(b/b.sum()).cuda()


def capture():
    records=[]
    for kind,M,a,b in problems():
        for repeat in range(3):
            out=curegot.torch.sinkhorn_splr(M,a,b,.1,1e-8,1000,0,density=.1,candidate_sinkhorn_iter=3)
            T=out['plan'];dual=out['dual']
            row=T.sum(1)-a;col=T.sum(0)-b
            # Original API has no convergence flag. This is its reduced-gradient criterion.
            gn=float(torch.linalg.norm(torch.cat([row,col[:-1]])))
            rec=dict(kind=kind,repeat=repeat,M=M.cpu(),a=a.cpu(),b=b.cpu(),T=T.cpu(),dual=dual.cpu(),niter=out['niter'],converged=gn<1e-8,reduced_gradient_norm=gn,row_max=float(row.abs().max()),col_max=float(col.abs().max()))
            if hasattr(curegot.torch,'splr_ot'):
                Td=curegot.torch.splr_ot(M,a,b,eta=.1,tol=1e-8,max_iter=1000,density=.1)
                rec['autograd_T_rel']=float(torch.linalg.norm(Td-T)/torch.linalg.norm(T))
                assert rec['autograd_T_rel']<1e-6
            records.append(rec)
    return records


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--save');p.add_argument('--compare');p.add_argument('--report');args=p.parse_args()
    cur=capture()
    if args.save:
        torch.save(cur,args.save);print('Saved upstream baseline',curegot.__file__,flush=True)
    else:
        old=torch.load(args.compare,weights_only=True);report=[]
        for kind in ('random','geometric','nonuniform'):
            a=[r for r in old if r['kind']==kind];b=[r for r in cur if r['kind']==kind]
            rel=lambda x,y:float(torch.linalg.norm(x-y)/torch.linalg.norm(y).clamp_min(1e-30))
            # Compare gauge-normalized duals; last beta is set to zero.
            def gauge(r):
                z=r['dual'].clone();shift=z[-1].clone();z[:32]+=shift;z[32:]-=shift;return z
            tr=max(rel(x['T'],y['T']) for x in b for y in a)
            dr=max(rel(gauge(x),gauge(y)) for x in b for y in a)
            noise=max(rel(x['T'],y['T']) for x in a for y in a)
            rec=dict(kind=kind,T_rel=tr,dual_rel=dr,upstream_T_self_rel=noise,
                     old_niter=[x['niter'] for x in a],new_niter=[x['niter'] for x in b],
                     old_converged=[x['converged'] for x in a],new_converged=[x['converged'] for x in b],
                     old_row_max=max(x['row_max'] for x in a),new_row_max=max(x['row_max'] for x in b),
                     old_col_max=max(x['col_max'] for x in a),new_col_max=max(x['col_max'] for x in b),
                     autograd_T_rel=max(x['autograd_T_rel'] for x in b))
            report.append(rec)
            assert all(x['converged'] for x in a+b),rec
            assert tr<max(1e-6,10*noise) and dr<1e-5,rec
            assert set(rec['old_niter'])==set(rec['new_niter']),rec
        print(json.dumps(report,indent=2))
        if args.report:Path(args.report).write_text(json.dumps(report,indent=2))
        print('FORWARD PARITY PASS')
