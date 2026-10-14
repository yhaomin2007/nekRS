"""Check actual case matrix weighting against Nek5000 explicit modal weights."""
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from numpy.polynomial.legendre import Legendre, legvander

case=Path(__file__).resolve().parents[1]
source=(case/'bubbleColumnTerms.hpp').read_text()
helper='inline void weightExplicitFilterMatrix('+source.split(
    'inline void weightExplicitFilterMatrix(',1)[1].split(
    'static deviceMemory<dfloat>',1)[0]
program='''#include <vector>
#include <iostream>
#include <iomanip>
using dfloat=double;
'''+helper+'''
int main(){int nq;double s;std::cin>>nq>>s;std::vector<double>a(nq*nq);
for(auto &v:a)std::cin>>v;weightExplicitFilterMatrix(a,nq,s);
std::cout<<std::setprecision(17);for(auto v:a)std::cout<<v<<" ";}
'''
N=7;nq=N+1
x=np.r_[-1.,Legendre.basis(N).deriv().roots(),1.]
V=legvander(x,N);iV=np.linalg.inv(V)
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory)
    (path/'check.cpp').write_text(program)
    subprocess.run(['g++','-std=c++11',str(path/'check.cpp'),'-o',str(path/'check')],check=True)
    for modes in (1,2,3):
        amp=np.zeros(nq)
        amp[N-modes+1:]=(np.arange(1,modes+1)/modes)**2
        full=(V*(1-amp))@iV
        for s in (0.,.05,.25,1.):
            data=f'{nq} {s} '+' '.join(format(v,'.17g') for v in full.ravel())
            out=subprocess.run([str(path/'check')],input=data,text=True,capture_output=True,check=True).stdout
            F=np.fromstring(out,sep=' ').reshape(nq,nq)
            sigma=1-s*amp
            np.testing.assert_allclose(iV@F@V,np.diag(sigma),atol=2e-14)
            for a,b,c in ((0,0,0),(N,0,0),(N,N,0),(N,N,N),(N-1,N,0)):
                field=np.einsum('i,j,k->ijk',V[:,a],V[:,b],V[:,c])
                filtered=np.einsum('ia,jb,kc,abc->ijk',F,F,F,field)
                np.testing.assert_allclose(filtered,sigma[a]*sigma[b]*sigma[c]*field,atol=3e-14)
    # Documented N=7, two modes, weight .05 transfer coefficients.
    np.testing.assert_allclose(1-.05*np.r_[np.zeros(6),.25,1.],
                               [1,1,1,1,1,1,.9875,.95])
print('Actual case matrix weighting: parabolic 1D weights, constants and mixed 3D modes passed; documented .9875/.95 coefficients reproduced.')
