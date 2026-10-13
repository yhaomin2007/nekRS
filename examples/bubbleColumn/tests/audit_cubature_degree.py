"""Audit (not a stability test): expose high-degree flux aliasing at default cubature."""
from pathlib import Path
import re
import subprocess
import tempfile
import numpy as np
from numpy.polynomial.legendre import Legendre, leggauss

root = Path(__file__).resolve().parents[1]
p = 7
nq = p + 1
x = np.r_[-1., Legendre.basis(p).deriv().roots(), 1.]
wg = 2 / (p*(p+1)*Legendre.basis(p)(x)**2)
kernel = (root/'bubbleColumnTransport.okl').read_text()
kernel = kernel.replace('@kernel ', '').replace('@ restrict ', '').replace(
    '@shared ', '').replace('@exclusive ', '').replace('@barrier();', '')
kernel = re.sub(r';\s*@outer\(0\)|;\s*@inner\([01]\)|;\s*@tile\(p_blockSize,\s*@outer,\s*@inner\)', '', kernel)

def array(name, value):
    return 'double '+name+'[]={'+','.join(format(v,'.17g') for v in np.asarray(value).ravel())+'};\n'

errors = {}
for cq in (11, 15):
    z, wc = leggauss(cq)
    I = np.ones((cq, nq))
    for i in range(nq):
        for j in range(nq):
            if j != i: I[:,i] *= (z-x[j])/(x[i]-x[j])
    bary = np.array([1/np.prod(z[i]-np.delete(z,i)) for i in range(cq)])
    D = np.zeros((cq,cq))
    for i in range(cq):
        for j in range(cq):
            if j != i: D[i,j] = bary[j]/bary[i]/(z[i]-z[j])
        D[i,i] = -sum(D[i])
    t,s,r = np.meshgrid(x,x,x,indexing='ij')
    tc,sc,rc = np.meshgrid(z,z,z,indexing='ij')
    fields = np.array([.02*(1+.3*Legendre.basis(p)(r)),
                       .3*(1+.3*Legendre.basis(p)(r)), np.zeros_like(r), np.zeros_like(r)])
    # Exact derivative of the product of two independently degree-7 fields.
    exact = .006*.6*(1+.3*Legendre.basis(p)(rc))*Legendre.basis(p).deriv()(rc)
    weights = wc[:,None,None]*wc[None,:,None]*wc[None,None,:]
    mass = wg[:,None,None]*wg[None,:,None]*wg[None,None,:]
    loads = np.einsum('ai,bj,ck,abc->ijk',I,I,I,exact*weights)
    expected = loads/mass
    geo = np.zeros((12,nq**3));cg = np.zeros((12,cq**3))
    for g in (geo,cg): g[0]=g[4]=g[8]=1.
    geo[9]=mass.ravel();geo[10]=1/mass.ravel();cg[9]=weights.ravel()
    prefix = '#include <cmath>\n#include <cstdio>\nusing dfloat=double;using dlong=long;\n'
    for name,value in {'p_Nq':nq,'p_Np':nq**3,'p_cubNq':cq,'p_cubNp':cq**3,
                       'p_Nvgeo':12,'p_RXID':0,'p_RYID':1,'p_RZID':2,
                       'p_SXID':3,'p_SYID':4,'p_SZID':5,'p_TXID':6,'p_TYID':7,
                       'p_TZID':8,'p_JWID':9,'p_IJWID':10}.items():
        prefix += f'#define {name} {value}\n'
    main = 'int main(){\n'+array('I',I.T)+array('D',D)+array('fields',fields)
    main += array('geo',geo)+array('cg',cg)+array('expected',expected)+array('mass',mass)
    main += f'double cube[{4*cq**3}]={{}},products[{4*cq**3}]={{}},out[{4*nq**3}]={{}};\n'
    main += f'gasInterpolate(1,{nq**3},{cq**3},I,fields,cube);alphaConservativeCubature(1,{cq**3},D,cg,cube,products);gasProject(1,{nq**3},{cq**3},I,geo,products,out);\n'
    main += f'double err=0,norm=0,integral=0;for(int i=0;i<{nq**3};++i){{double d=(out[i]-expected[i])*mass[i];err+=d*d;double v=expected[i]*mass[i];norm+=v*v;integral+=out[i]*mass[i];}}\n'
    main += 'printf("%.16e %.16e\\n",sqrt(err/norm),integral);}\n'
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        (path/'check.cpp').write_text(prefix+kernel+main)
        subprocess.run(['g++','-std=c++17','-O2',str(path/'check.cpp'),'-o',str(path/'check')],check=True)
        result = subprocess.run([str(path/'check')],check=True,capture_output=True,text=True)
        error, integral = map(float,result.stdout.split())
    errors[cq] = error
    # Four units of transverse area times the analytical flux difference.
    assert abs(integral-4*.0072)<1e-12
    print(f'Actual conservative kernels: p=7 cubNq={cq}, relative projected-load error={error:.8e}; boundary balance passes.')
assert errors[11] > .1
assert errors[15] < 1e-11
print('Confirmed: boundary balance and low-degree tests do not establish high-degree dealiasing.')
