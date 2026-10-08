"""Run actual case cubature kernels as serial C++, compare to quadrature loads."""
from pathlib import Path
import re,subprocess,tempfile
import numpy as np
from numpy.polynomial.legendre import Legendre,leggauss
root=Path(__file__).resolve().parents[1]
N=3; nq=N+1; cq=6
x=np.r_[-1.,Legendre.basis(N).deriv().roots(),1.]
w=2/(N*(N+1)*Legendre.basis(N)(x)**2)
z,wc=leggauss(cq)
def interp(x,z):
 out=np.ones((len(z),len(x)))
 for i in range(len(x)):
  for j in range(len(x)):
   if i!=j:out[:,i]*=(z-x[j])/(x[i]-x[j])
 return out
def diff(x):
 b=np.array([1/np.prod(x[i]-np.delete(x,i)) for i in range(len(x))]);d=np.zeros((len(x),len(x)))
 for i in range(len(x)):
  for j in range(len(x)):
   if i!=j:d[i,j]=b[j]/b[i]/(x[i]-x[j])
  d[i,i]=-sum(d[i])
 return d
I=interp(x,z);D=diff(z)
# Cartesian affine element: coordinates (r,s,t), identity metrics.
t,s,r=np.meshgrid(x,x,x,indexing='ij');tc,sc,rc=np.meshgrid(z,z,z,indexing='ij')
fields=np.array([.2+.02*r*r+.01*s, .3+.1*r*r, -.05+.02*s*s, .04+.01*t*t])
ac=.2+.02*rc*rc+.01*sc;ux=.3+.1*rc*rc;uy=-.05+.02*sc*sc;uz=.04+.01*tc*tc
expected=np.array([ac*(.2*rc+.04*sc+.02*tc), ux*.2*rc,uy*.04*sc,uz*.02*tc])
# Nonlinear quadrature loads (not GLL pointwise values).
weights=wc[:,None,None]*wc[None,:,None]*wc[None,None,:]
loads=np.einsum('ai,bj,ck,fabc->fijk',I,I,I,expected*weights)
mass=w[:,None,None]*w[None,:,None]*w[None,None,:]
expected_nodal=loads/mass
# The projected load must differ from the old GLL pointwise product.
a0=fields[0];u0,u1,u2=fields[1:]
gll=np.array([a0*(.2*r+.04*s+.02*t),u0*.2*r,u1*.04*s,u2*.02*t])
assert np.max(abs(expected_nodal-gll))>1e-5
geo=np.zeros((12,nq**3));cg=np.zeros((12,cq**3))
for g in (geo,cg):g[0]=g[4]=g[8]=1.
geo[9]=mass.ravel();geo[10]=1/mass.ravel();cg[9]=weights.ravel()
s=(root/'bubbleColumnTransport.okl').read_text()
s=s.replace('@kernel ','').replace('@ restrict ','').replace('@shared ','').replace('@exclusive ','').replace('@barrier();','')
s=re.sub(r';\s*@outer\(0\)','',s);s=re.sub(r';\s*@inner\([01]\)','',s);s=re.sub(r';\s*@tile\(p_blockSize,\s*@outer,\s*@inner\)','',s)
pre='#include <cmath>\n#include <cassert>\nusing dfloat=double;using dlong=long;\n'
for name,value in {'p_Nq':nq,'p_Np':nq**3,'p_cubNq':cq,'p_cubNp':cq**3,'p_Nvgeo':12,'p_RXID':0,'p_RYID':1,'p_RZID':2,'p_SXID':3,'p_SYID':4,'p_SZID':5,'p_TXID':6,'p_TYID':7,'p_TZID':8,'p_JWID':9,'p_IJWID':10}.items():pre+=f'#define {name} {value}\n'
def array(name,a):return 'double '+name+'[]={'+','.join(format(v,'.17g') for v in np.asarray(a).ravel())+'};\n'
main='int main(){\n'+array('I',I.T)+array('D',D)+array('fields',fields)+array('vgeo',geo)+array('cg',cg)+array('expected',expected_nodal)
main+=f'double cube[{4*cq**3}]={{}},products[{4*cq**3}]={{}},out[{4*nq**3}]={{}};\n'
main+=f'gasInterpolate(1,{nq**3},{cq**3},I,fields,cube);gasCubatureProducts(1,{cq**3},D,cg,cube,products);gasProject(1,{nq**3},{cq**3},I,vgeo,products,out);\n'
main+=f'for(int i=0;i<{4*nq**3};++i)assert(std::abs(out[i]-expected[i])<1e-11);\n}}'
with tempfile.TemporaryDirectory() as t:
 p=Path(t);(p/'test.cpp').write_text(pre+s+main)
 subprocess.run(['g++','-std=c++17','-O2',str(p/'test.cpp'),'-o',str(p/'test')],check=True,stderr=subprocess.PIPE)
 subprocess.run([str(p/'test')],check=True)
# Integral of conservative alpha operator equals boundary gas flux on affine box.
volume=np.sum((expected[0]+ux*.04*rc+uy*.01)*weights)
def flux_at(r,s,t):
 r,s,t=np.broadcast_arrays(r,s,t)
 return (.2+.02*r*r+.01*s)*np.array([.3+.1*r*r,-.05+.02*s*s,.04+.01*t*t])
a,b=np.meshgrid(z,z,indexing='ij');ww=wc[:,None]*wc[None,:]
surface=np.sum((flux_at(1,a,b)[0]-flux_at(-1,a,b)[0])*ww)
surface+=np.sum((flux_at(a,1,b)[1]-flux_at(a,-1,b)[1])*ww)
surface+=np.sum((flux_at(a,b,1)[2]-flux_at(a,b,-1)[2])*ww)
assert abs(volume-surface)<1e-12

print('Actual cubature kernels match independent polynomial quadrature; alpha integral matches boundary flux on affine element.')
