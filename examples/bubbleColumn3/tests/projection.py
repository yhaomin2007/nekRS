#!/usr/bin/env python3
"""Spatial 1D two-element GLL projection, exercising the production Krylov code.

Builds both stiffness and the *assembled/averaged/masked* correction operator.
No NekRS runtime is emulated; MPI/GPU execution is a separate integration check.
"""
from pathlib import Path
import subprocess, tempfile
import numpy as np
root=Path(__file__).resolve().parents[1]
# Four GLL nodes, two unit elements, continuous shared endpoint.
x=np.array([-1.,-1/np.sqrt(5),1/np.sqrt(5),1.])
w=np.array([1/6,5/6,5/6,1/6])/2
lagrange=np.array([1/np.prod(x[i]-np.delete(x,i)) for i in range(4)])
D=np.zeros((4,4))
for i in range(4):
    for j in range(4):
        if i!=j:D[i,j]=lagrange[j]/(lagrange[i]*(x[i]-x[j]))
    D[i,i]=-D[i].sum()
D*=2
mass=np.zeros(7);H=np.zeros((7,7));W=np.zeros((7,7));K=np.zeros((7,7))
beta=np.linspace(.3,1.1,7)
for ids in (np.arange(4),np.arange(3,7)):
    mass[ids]+=w
    H[np.ix_(ids,ids)]+=np.diag(w)@D
    W[np.ix_(ids,ids)]+=D.T@np.diag(w)
    K[np.ix_(ids,ids)]+=D.T@np.diag(w*beta[ids])@D
G=H/mass[:,None]
G[0,:]=0 # prescribed phase inlet velocity; outlet pressure is eliminated
A=(W@np.diag(beta)@G)[:-1,:-1];K=K[:-1,:-1]
expected=np.array([1.2,.9,.8,.55,.3,.2]);b=A@expected
old=np.linalg.solve(K,b)
old_error=np.linalg.norm(b-A@old)/np.linalg.norm(b)
assert old_error>1e-3
invK=np.linalg.inv(K)
def literal(a):return '{'+','.join(format(v,'.17g') for v in a.flat)+'}'
source=r'''
#include "bubbleColumn3Krylov.hpp"
#include <cassert>
#include <cstdio>
using V=std::vector<double>;

void checkRestartHistory()
{
 constexpr int n=30;
 V eigenvalues(n);
 for(int i=0;i<n;++i) eigenvalues[i]=std::pow(10.0,-5.0*i/(n-1));
 auto apply=[&](const V& x,V& y) {
   for(int i=0;i<n;++i) y[i]=eigenvalues[i]*x[i];
 };
 auto identity=[](const V& x,V& y){y=x;};
 auto dot=[](const V& x,const V& y){
   double sum=0;
   for(int i=0;i<n;++i) sum+=x[i]*y[i];
   return sum;
 };
 const V rhs(n,1.0);
 V shortGuess(n,0.0),longGuess(n,0.0);
 const auto shortBasis=method3Projection::solve(rhs,shortGuess,apply,identity,dot,60,15,1e-8);
 const auto longBasis=method3Projection::solve(rhs,longGuess,apply,identity,dot,60,30,1e-8);
 assert(!shortBasis.converged && longBasis.converged);
 assert(shortBasis.residualHistory.size()==5);
 for(int i=0;i<5;++i) assert(shortBasis.residualHistory[i].iteration==15*i);
 const auto& history=longBasis.residualHistory;
 assert(history.front().iteration==0 && history.front().norm==longBasis.initial);
 assert(history.back().iteration==longBasis.iterations && history.back().norm==longBasis.residual);
 printf("PASS: restart residuals with 15 and 30 directions %.3e and %.3e\n",
        shortBasis.residual,longBasis.residual);
}

int main(){
 checkRestartHistory();
 const V A=AMATRIX, P=PMATRIX, b=BVECTOR;
 auto mult=[](const V& M,const V& x,V& y){y.assign(6,0);for(int i=0;i<6;++i)for(int j=0;j<6;++j)y[i]+=M[6*i+j]*x[j];};
 auto op=[&](const V& x,V& y){mult(A,x,y);};
 auto pre=[&](const V& x,V& y){mult(P,x,y);};
 auto dot=[](const V& x,const V& y){double s=0;for(int i=0;i<6;++i)s+=x[i]*y[i];return s;};
 V p=OLDVECTOR;
 auto result=method3Projection::solve(b,p,op,pre,dot,60,6,1e-10);
 assert(result.converged);V residual;op(p,residual);
 for(int i=0;i<6;++i)residual[i]-=b[i];
 assert(std::sqrt(dot(residual,residual)/dot(b,b))<1e-10);
 V fail(6,0);auto limited=method3Projection::solve(b,fail,op,pre,dot,1,1,1e-14);
 assert(!limited.converged);
 auto zero=[](const V&,V& y){y.assign(6,0);};fail.assign(6,0);
 auto singular=method3Projection::solve(b,fail,zero,pre,dot,10,3,1e-10);assert(!singular.converged);
 printf("PASS: spatial composed projection, true residual %.3e, iterations %d; iteration-limit and singular-operator rejection\n",result.residual,result.iterations);
}
'''
for key,a in [('AMATRIX',A),('PMATRIX',invK),('BVECTOR',b),('OLDVECTOR',old)]:source=source.replace(key,literal(a))
with tempfile.TemporaryDirectory() as tmp:
    f=Path(tmp)/'projection.cpp';f.write_text(source)
    exe=Path(tmp)/'projection'
    subprocess.run(['c++','-std=c++17','-Wall','-Wextra','-Werror','-I'+str(root),str(f),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
# Algebraic identity uses the same W and fixed boundary lifting at every stage.
u=np.linspace(.1,.4,7);lift=np.zeros(7);lift[0]=u[0]
correction=beta*(G[:,:-1]@expected)
u1=u-correction;u2=u1.copy();u2[0]=u[0]
r0=(W@u-lift)[:-1];r1=(W@u1-lift)[:-1];r2=(W@u2-lift)[:-1]
assert np.linalg.norm((r0-r1)-A@expected)<1e-14
assert np.linalg.norm(r2-r1)==0
print(f'PASS: native stiffness mismatch {old_error:.3e}; correction identity and zero boundary-restoration change')
