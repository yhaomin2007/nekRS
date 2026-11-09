#!/usr/bin/env python3
"""3D GLL wall-masked pressure null modes and stabilized production GCR."""
from pathlib import Path
import subprocess,tempfile
import numpy as np
from numpy.polynomial.legendre import Legendre
root=Path(__file__).resolve().parents[1]
N=3;q=N+1
x=np.r_[-1,Legendre.basis(N).deriv().roots(),1.]
w=2/(N*(N+1)*Legendre.basis(N)(x)**2)
l=np.array([1/np.prod(x[i]-np.delete(x,i)) for i in range(q)])
D=np.zeros((q,q))
for i in range(q):
    for j in range(q):
        if i!=j:D[i,j]=l[j]/(l[i]*(x[i]-x[j]))
    D[i,i]=-D[i].sum()
I=np.eye(q)
G=np.vstack([np.kron(np.kron(I,I),D),np.kron(np.kron(I,D),I),np.kron(np.kron(D,I),I)])
zz,yy,xx=np.indices((q,q,q));freeP=zz.ravel()!=N
freeU=((xx>0)&(xx<N)&(yy>0)&(yy<N)&(zz>0)).ravel()
mass=np.kron(np.kron(w,w),w);weight=np.tile(mass,3)
Gp=G[:,freeP];F=np.tile(freeU,3)
A=Gp.T@(weight[:,None]*F[:,None]*Gp)
K=Gp.T@(weight[:,None]*Gp)
_,sigma,V=np.linalg.svd(A)
rank=np.count_nonzero(sigma>sigma[0]*1e-12)
assert rank==28 and len(A)==48
# Upward predictor, no-slip side walls, prescribed bottom inflow.
u=np.zeros((3,q,q,q));u[2,:,1:-1,1:-1]=.005
b=Gp.T@(weight*u.ravel())
bottom=np.zeros((q,q,q));bottom[0,1:-1,1:-1]=-.005*np.outer(w[1:-1],w[1:-1])
b-=bottom.ravel()[freeP]
null_part=V[rank:]@b
floor=np.linalg.norm(null_part)/np.linalg.norm(b)
# This ideal symmetric inflow is compatible despite the pressure null modes.
assert floor<1e-12
# Full two-phase Helmholtz response on free velocity DOFs. This retains the
# wall-masked null modes while changing the response through viscosity/drag.
free=G[np.tile(freeU,3)][:,freeP]
mass_free=np.diag(weight[F])
stiffness=G.T@(weight[:,None]*G)
stiffness=np.kron(np.eye(3),stiffness)[np.ix_(F,F)]
alpha,rho_g,rho_l,tau=.05,1.2,998.2,.02
lg=25.
ll=alpha*rho_g*lg/((1-alpha)*rho_l)
H=np.block([[(1/tau+lg)*mass_free+.003*stiffness,-lg*mass_free],
            [-ll*mass_free,(1/tau+ll)*mass_free+.006*stiffness]])
load=-np.vstack((mass_free@free/rho_g,mass_free@free/rho_l))
D_m=free.T@mass_free@np.hstack((alpha*np.eye(len(free)),(1-alpha)*np.eye(len(free))))
full_schur=-D_m@np.linalg.solve(H,load)/tau
assert np.linalg.matrix_rank(full_schur,tol=1e-10)==rank
# A nontrivial pressure load gives a compatible, nonzero weak residual.
compatible_rhs=full_schur@np.sin(np.arange(len(A)))
assert np.linalg.norm(compatible_rhs)>1e-3
assert np.linalg.norm(compatible_rhs-full_schur@np.linalg.solve(K,compatible_rhs))>1e-3
# A controlled incompatible component tests failure handling and stabilization;
# it represents a perturbed discrete boundary/RHS, not measured user-run data.
b+=.01*np.linalg.norm(b)*V[rank]
floor=np.linalg.norm(V[rank:]@b)/np.linalg.norm(b)
assert floor>.009
# Native-only mode solves the same K used by the stabilized epsilon=1 limit.
native_pressure=np.linalg.solve(K,b)
assert np.linalg.norm(K@native_pressure-b)/np.linalg.norm(b)<1e-12
native_weak=b-A@native_pressure
assert np.linalg.norm(native_weak-(K-A)@native_pressure)/np.linalg.norm(b)<1e-12
P=np.linalg.inv(K);epsilon=.1;S=(1-epsilon)*A+epsilon*K
assert np.linalg.eigvalsh(S)[0]>0
pressure=np.linalg.solve(S,b)
stab=epsilon*(K-A)@pressure
assert np.linalg.norm((b-A@pressure)-stab)/np.linalg.norm(b)<1e-12
assert np.max(np.abs(((1-1)*A+1*K)-K))==0
n=len(b)
def literal(v):return '{'+','.join(format(a,'.17g') for a in v.flat)+'}'
source=r'''
#include "bubbleColumn3Krylov.hpp"
#include <cassert>
#include <cstdio>
using V=std::vector<double>;
int main(){const int n=SIZE;const V S=SMATRIX,A=AMATRIX,K=KMATRIX,P=PMATRIX,b=BVECTOR,full=FULL_MATRIX,compatible=COMPATIBLE_RHS;
 auto mult=[&](const V& M,const V& x,V& y){y.assign(n,0);for(int i=0;i<n;++i)for(int j=0;j<n;++j)y[i]+=M[n*i+j]*x[j];};
 auto op=[&](const V& x,V& y){mult(S,x,y);};
 bool inexact=false;int calls=0;
 auto dot=[&](const V& x,const V& y){double s=0;for(int i=0;i<n;++i)s+=x[i]*y[i];return s;};
 auto pre=[&](const V& x,V& y){
   mult(P,x,y);
   if(inexact){
     ++calls;V noise(n),kn;for(int i=0;i<n;++i)noise[i]=std::sin(double(i+calls));
     mult(K,noise,kn);const double c=.005*std::sqrt(dot(x,x)/dot(kn,kn));
     for(int i=0;i<n;++i)y[i]+=c*noise[i];
     V residual;mult(K,y,residual);for(int i=0;i<n;++i)residual[i]-=x[i];
     assert(dot(residual,residual)<=.0001*dot(x,x));
   }
 };
 V exact(n,0.0);
 auto fullOp=[&](const V& x,V& y){mult(full,x,y);};
 auto fullResult=method3Projection::solve(compatible,exact,fullOp,pre,dot,100,15,1e-8);
 assert(fullResult.converged);
 printf("PASS: full two-phase Helmholtz Schur on compatible 3D wall-masked space, residual %.3e\n",fullResult.residual);
 V bad;pre(b,bad);
 auto singularOp=[&](const V& x,V& y){mult(A,x,y);};
 auto failed=method3Projection::solve(b,bad,singularOp,pre,dot,200,15,1e-8);
 assert(!failed.converged);
 V p;pre(b,p);auto result=method3Projection::solve(b,p,op,pre,dot,200,15,1e-8);
 assert(result.converged);V r;op(p,r);for(int i=0;i<n;++i)r[i]-=b[i];
 const double relative=std::sqrt(dot(r,r)/dot(b,b));assert(relative<1e-8);
 // Flexible GCR must tolerate changing, deliberately inexact native solves.
 V approximate;mult(P,b,approximate);inexact=true;
 auto flexible=method3Projection::solve(b,approximate,op,pre,dot,200,15,1e-8);
 assert(flexible.converged);op(approximate,r);for(int i=0;i<n;++i)r[i]-=b[i];
 assert(std::sqrt(dot(r,r)/dot(b,b))<1e-8);
 // Warm starting with the converged final pressure should need no inner solve.
 const int previousCalls=calls;
 auto warm=method3Projection::solve(b,approximate,op,pre,dot,200,15,1e-8);
 assert(warm.converged&&warm.iterations==0&&calls==previousCalls);
 printf("PASS: inexact variable preconditioning and zero-iteration final-pressure warm start\n");
 printf("PASS: 3D stabilized production GCR iterations=%d true relative residual=%.3e\n",result.iterations,relative);
}
'''
for key,val in [('SIZE',str(n)),('KMATRIX',literal(K)),('AMATRIX',literal(A)),('SMATRIX',literal(S)),('PMATRIX',literal(P)),('BVECTOR',literal(b)),('FULL_MATRIX',literal(full_schur)),('COMPATIBLE_RHS',literal(compatible_rhs))]:source=source.replace(key,val)
with tempfile.TemporaryDirectory() as tmp:
    f=Path(tmp)/'projection.cpp';f.write_text(source);exe=Path(tmp)/'projection'
    subprocess.run(['c++','-std=c++17','-Wall','-Wextra','-Werror','-I'+str(root),str(f),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
print(f'PASS: 3D pressure nullity={n-rank}, incompatible RHS fraction={floor:.3e}; stabilized continuity balance and native epsilon=1 limit')
