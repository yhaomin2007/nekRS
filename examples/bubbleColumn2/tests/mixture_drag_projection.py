"""Explicit UM drag through actual native EXT/BDF, pressure and velocity RHS kernels."""
from pathlib import Path
import re
import subprocess
import tempfile

case = Path(__file__).resolve().parents[1]
repo = case.parents[1]
def serial(s):
    s = s.replace('@kernel ', '').replace('@ restrict ', '')
    return re.sub(r'; @tile\(p_blockSize, @outer, @inner\)', '', s)
eq = (case / 'bubbleColumn2Equations.okl').read_text()
k = serial('@kernel void addExplicitMixtureDrag(' + eq.split('@kernel void addExplicitMixtureDrag(')[1])
cpp = r'''
#include <algorithm>
#include <cassert>
#include <cmath>
using dfloat=double;using dlong=long;
#define p_SUBCYCLING 0
#define p_MovingMesh 0
#define p_ADVECTION 1
#define p_nBDF 2
#define p_nEXT 2
''' + k
for name in ['sumMakef', 'pressureRhsHex3D', 'velocityRhsHex3D']:
    cpp += serial((repo / f'src/solver/fluid/kernels/{name}.okl').read_text())
cpp += r'''
bool close(double x,double y){return std::abs(x-y)<1e-11*std::max(1.,std::max(std::abs(x),std::abs(y)));}
int main(){
 const int N=3,O=5,S=15;
 double mass[N]={2,5,3},rho[N]={2,4,9},rate[N]={0,4,10};
 double um[S]={},ug[S]={},base[2*S]={},force[2*S]={};
 double state[2*S]={},adv[2*S]={},bdf[2]={2,-.5};
 double bf[S]={},gp[S]={},gradDiv[S]={},curl[S]={},mu[N]={.1,.2,.3};
 double velrhs[S]={},prhs[S]={},inverseRho[N]={};
 for(int order:{1,2})for(int vm:{0,1})for(double ramp:{0.,.25,1.}){
  double ext[2]={order==1?1.:2.,order==1?0.:-1.};
  for(int i=0;i<2*S;++i){base[i]=force[i]=12345.;state[i]=adv[i]=0.;}
  for(int n=0;n<N;++n)inverseRho[n]=1/rho[n];
  for(int c=0;c<3;++c)for(int n=0;n<N;++n){
   int id=n+c*O;
   um[id]=.1+.02*id;ug[id]=.3-.01*id;
   state[id]=.05+.01*id;state[id+S]=-.1+.02*id;
   adv[id]=.03*id;adv[id+S]=-.07*id;
   base[id]=1.+vm*(id+1)*(id+1);
   base[id+S]=-.2-vm*(id+2)*(id+2);
   force[id]=base[id];force[id+S]=base[id+S];
   gp[id]=-.2*(id+1);curl[id]=.05*id;
  }
  double ramped[N];for(int n=0;n<N;++n)ramped[n]=ramp*rate[n];
  addExplicitMixtureDrag(N,O,ramped,um,ug,force);
  // Previous explicit force history has its own drag and must not be overwritten.
  double oldUm[S]={},oldUg[S]={},oldRate[N]={2,3,5};
  for(int id=0;id<S;++id){oldUm[id]=-.1+.01*id;oldUg[id]=.2;}
  addExplicitMixtureDrag(N,O,oldRate,oldUm,oldUg,force+S);
  sumMakef(N,mass,100.,ext,bdf,O,S,3,rho,state,adv,force,bf);
  velocityRhsHex3D(N,O,rho,bf,gradDiv,gp,velrhs);
  pressureRhsHex3D(N,O,mu,inverseRho,bf,curl,gradDiv,prhs);
  for(int c=0;c<3;++c)for(int n=0;n<N;++n){
   int id=n+c*O;
   double current=base[id]+ramped[n]*(um[id]-ug[id]);
   double previous=base[id+S]+oldRate[n]*(oldUm[id]-oldUg[id]);
   assert(close(force[id],current));assert(close(force[id+S],previous));
   double expected=mass[n]*(ext[0]*(current-adv[id])+ext[1]*(previous-adv[id+S])
                  +100.*(bdf[0]*state[id]+bdf[1]*state[id+S]));
   assert(close(bf[id],expected));
   assert(close(velrhs[id],rho[n]*expected+gp[id]));
   assert(close(prhs[id],expected-mu[n]/rho[n]*curl[id]));
  }
  for(int c=0;c<3;++c)for(int n=N;n<O;++n){
   assert(force[n+c*O]==12345.);assert(force[n+c*O+S]==12345.);
  }
 }
}
'''
with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp)
    (p / 'check.cpp').write_text(cpp)
    subprocess.run(['g++', '-std=c++17', '-Wall', '-Wextra', '-Werror',
                    '-Wno-unused-parameter', '-Wno-unknown-pragmas',
                    str(p / 'check.cpp'), '-o', str(p / 'check')], check=True)
    subprocess.run([str(p / 'check')], check=True)
t = (case / 'bubbleColumn2Terms.hpp').read_text()
u = (case / 'bubbleColumn2.udf').read_text()
assert 'nrs->preFluid' not in u and 'nrs->fluid->userImplicitLinearTerm' not in u
assert 'nrs->scalar->userImplicitLinearTerm = &bubbleColumn2::implicitGasDrag;' in u
assert 'prepareMixtureDrag' not in t and 'implicitMixtureDrag' not in t
source = t.split('inline void addExplicitSources(double)')[1].split('inline occa::memory implicitGasDrag')[0]
assert source.index('evaluateMixtureForce();') < source.index('addExplicitMixtureDragKernel(') < source.index('fluidTerms.copyFrom(')
assert 'nrs->fluid->o_U, o_ug, o_mixtureForce' in source
print('Explicit UM drag: signed force, ramp/off, VM preservation, padded vectors, EXT1/EXT2 histories and identical known load in native pressure/velocity RHS passed.')
