#!/usr/bin/env python3
"""Manufactured two-element SEM audit of the pressure/Helmholtz defect.

This models the actual operator distinction, not the 3D column or its stripes.
The reference force is selected to give a prescribed divergence-free solution.
"""
import numpy as np

x = np.array([-1., -1/np.sqrt(5), 1/np.sqrt(5), 1.])
w = np.array([1/6, 5/6, 5/6, 1/6])/2
l = np.array([1/np.prod(x[i]-np.delete(x, i)) for i in range(4)])
D = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        if i != j:
            D[i, j] = 2*l[j]/(l[i]*(x[i]-x[j]))
    D[i, i] = -D[i].sum()
M = np.zeros((7, 7))
W = np.zeros_like(M)
L = np.zeros_like(M)
for ids in (np.arange(4), np.arange(3, 7)):
    block = np.ix_(ids, ids)
    M[block] += np.diag(w)
    W[block] += D.T@np.diag(w)
    L[block] += D.T@np.diag(w)@D
# Inlet velocity is prescribed, outlet pressure is prescribed.
v = slice(1, None)
q = slice(None, -1)
m, stiffness = M[v, v], L[v, v]
alpha, tau, rho_g, rho_l = .05, .02, 1.2, 998.2
lg = 25.
ll = alpha*rho_g*lg/((1-alpha)*rho_l)
H = np.block([[(1/tau+lg)*m+.003*stiffness, -lg*m],
              [-ll*m, (1/tau+ll)*m+.006*stiffness]])
load = np.vstack((W[v, q]/rho_g, W[v, q]/rho_l))
mixture = np.hstack((alpha*np.eye(6), (1-alpha)*np.eye(6)))
A = -W[q, v]@mixture@np.linalg.solve(H, load)/tau
beta = np.linalg.solve([[1+tau*lg, -tau*lg], [-tau*ll, 1+tau*ll]],
                       [tau/rho_g, tau/rho_l])
K = (alpha*beta[0]+(1-alpha)*beta[1])/tau*L[q, q]
reference_pressure = np.array([2., 1.8, 1.3, 1., .8, .3])
b = A@reference_pressure
old = np.linalg.solve(K, b)
old_defect = np.linalg.norm(b-A@old)
assert old_defect > 1e-3*np.linalg.norm(b)
previous = .98*reference_pressure
predictor_residual = b-A@previous
increment = np.linalg.solve(K, predictor_residual)
final = previous+increment
residual = b-A@final
assert np.linalg.norm(residual-(K-A)@increment) < 1e-12*np.linalg.norm(b)
assert np.linalg.norm(residual) < .03*old_defect
# Same equation with an equilibrated prior pressure has no pressure increment.
assert np.linalg.norm(np.linalg.solve(K, b-A@reference_pressure)) == 0
print(f'PASS: one Poisson increment leaves (K-A)dp, not (K-A)p; '
      f'manufactured defect ratio={np.linalg.norm(residual)/old_defect:.3f}')

# Exercise the production outer iteration with the actual two-phase response.
# Store the velocity and residual from each trial, just as the case does before
# committing the final trial's fields. The inlet velocity has been eliminated.
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
response = np.linalg.solve(H, load)
continuity = W[q, v] @ mixture / tau

def literal(a):
    return '{'+','.join(format(value, '.17g') for value in a.flat)+'}'

source = r'''
#include "bubbleColumn3Krylov.hpp"
#include <cassert>
#include <cstdio>
using V=std::vector<double>;
V multiply(const V& A,const V& x,int rows) {
  V y(rows);
  for(int i=0;i<rows;++i) for(size_t j=0;j<x.size();++j) y[i]+=A[i*x.size()+j]*x[j];
  return y;
}
double dot(const V& a,const V& b) {
  double sum=0;
  for(size_t i=0;i<a.size();++i) sum+=a[i]*b[i];
  return sum;
}
int main() {
  const V response=RESPONSE, continuity=CONTINUITY, invK=INVERSE, b=RHS, reference=REFERENCE;
  for(bool incremental:{false,true}) {
    const V prior=incremental?reference:V(6,0.0);
    auto rhs=multiply(continuity,multiply(response,prior,12),6);
    for(int i=0;i<6;++i) rhs[i]+=b[i];
    V delta(6,0.0);
    auto apply=[&](const V& trial,V& action) {
      double scale=0;
      for(double value:trial) scale=std::max(scale,std::abs(value));
      if(scale==0) {action.assign(6,0);return;}
      V unit=trial;
      for(auto& value:unit)value/=scale;
      action=multiply(continuity,multiply(response,unit,12),6);
      for(auto& value:action)value*=-scale;
    };
    auto precondition=[&](const V& r,V& z){z=multiply(invK,r,6);};
    auto result=method3Projection::solve(rhs,delta,apply,precondition,dot,100,15,0.0,1e-12);
    assert(result.converged);
    V total=prior;
    for(int i=0;i<6;++i)total[i]+=delta[i];
    auto actual=multiply(continuity,multiply(response,total,12),6);
    for(int i=0;i<6;++i)actual[i]+=b[i];
    assert(std::sqrt(dot(actual,actual))<=4*result.target);
    // At tiny pressure, subtracting complete flow residuals can lose the
    // entire response. The normalized homogeneous action retains it.
    V tiny=reference;
    for(auto& value:tiny)value*=1e-19;
    V action;apply(tiny,action);
    auto exact=multiply(continuity,multiply(response,tiny,12),6);
    double error=0,subtracted=0;
    for(int i=0;i<6;++i) {
      error+=std::pow(action[i]+exact[i],2);
      subtracted+=std::pow((rhs[i]-(rhs[i]+exact[i]))-action[i],2);
    }
    assert(std::sqrt(error)<1e-13*std::sqrt(dot(action,action)));
    assert(std::sqrt(subtracted)>.1*std::sqrt(dot(action,action)));
    printf("PASS: full Helmholtz Schur %s solve, measured residual %.3e\n",
        incremental?"increment":"absolute",std::sqrt(dot(actual,actual)));
  }
}
'''
for key, value in [('RESPONSE', response), ('CONTINUITY', continuity), ('INVERSE', np.linalg.inv(K)),
                   ('RHS', b), ('REFERENCE', previous)]:
    source = source.replace(key, literal(value))
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp)/'schur.cpp'
    path.write_text(source)
    executable = Path(tmp)/'schur'
    subprocess.run(['c++','-std=c++17','-Wall','-Wextra','-Werror','-I'+str(root),str(path),
                    '-o',str(executable)],check=True)
    subprocess.run([str(executable)],check=True)
