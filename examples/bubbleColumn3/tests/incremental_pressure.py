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
