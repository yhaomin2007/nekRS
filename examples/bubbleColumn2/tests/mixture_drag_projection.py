"""Actual preFluid hook and native BDF/pressure/velocity kernels for UM drag."""
from pathlib import Path
import re,subprocess,tempfile
case=Path(__file__).resolve().parents[1];repo=case.parents[1]
def serial(s):
 s=s.replace('@kernel ','').replace('@ restrict ','')
 return re.sub(r'; @tile\(p_blockSize, @outer, @inner\)','',s)
eq=(case/'bubbleColumn2Equations.okl').read_text()
k=serial('@kernel void normalizeMixtureDrag('+eq.split('@kernel void normalizeMixtureDrag(')[1])
t=(case/'bubbleColumn2Terms.hpp').read_text()
hook='inline void prepareMixtureDrag'+t.split('inline void prepareMixtureDrag')[1].split('inline void addExplicitSources')[0]
cpp=r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>
using dfloat=double;using dlong=long;
#define p_SUBCYCLING 0
#define p_MovingMesh 0
#define p_ADVECTION 1
#define p_nBDF 2
#define p_nEXT 2
'''+k
for name in ['sumMakef','pressureRhsHex3D','velocityRhsHex3D']:
 cpp+=serial((repo/f'src/solver/fluid/kernels/{name}.okl').read_text())
cpp+=r'''
struct Mem {
 std::shared_ptr<std::vector<double>> v;long start=0;
 Mem(long n=0):v(std::make_shared<std::vector<double>>(n)){}
 double *ptr(){return v->data()+start;}
 Mem slice(long off,long){auto m=*this;m.start+=off;return m;}
 void copyFrom(Mem other,long n){std::copy(other.ptr(),other.ptr()+n,ptr());}
};
struct Comm{int mpiComm(){return 0;}};
struct LA{double max(long n,Mem m,int){return *std::max_element(m.ptr(),m.ptr()+n);}} la;
struct Platform{LA *linAlg=&la;Comm comm;} obj;auto platform=&obj;
struct Mesh{long Nlocal=3;Mem o_LMM{3};} mesh;
struct Fluid{long fieldOffsetSum=15;Mem o_JwF{15},o_prop{10};} fluid;
struct Nrs{Mesh *meshV=&mesh;Fluid *fluid=&::fluid;long fieldOffset=5;double g0=1.5,dt[1]={.01};} app;auto nrs=&app;
Mem o_frozenMixtureDragRate(3),o_rhoPressure(3),o_rhoPressureSolve(3),o_ug(15),o_fluidForcingBeforeDrag(15);
bool fluidForcingSnapshotValid=false;
void reconstructGasVelocity(){}
#define nekrsCheck(test,...) do{if(test)throw std::runtime_error("invalid drag diagonal");}while(0)
void normalizeMixtureDragKernel(long n,long o,double a,Mem mass,Mem rho,Mem rate,Mem ug,Mem base,Mem rs,Mem f){
 normalizeMixtureDrag(n,o,a,mass.ptr(),rho.ptr(),rate.ptr(),ug.ptr(),base.ptr(),rs.ptr(),f.ptr());
}
'''+hook+r'''
bool close(double x,double y){return std::abs(x-y)<1e-11*std::max(1.,std::max(std::abs(x),std::abs(y)));}
int main(){
 const int N=3,O=5;
 double mass[3]={2,5,3},rho[3]={2,4,9};
 double state[30]={},adv[30]={},fu[30]={},bf[15]={},bdf[2]={2,-.5};
 double gp[15]={},gradDiv[15]={},curl[15]={},mu[3]={.1,.2,.3};
 double velrhs[15]={},prhs[15]={},inverseRho[3]={};
 for(int extOrder:{1,2}) for(int vm:{0,1}) for(int drag:{0,1}){
  double ext[2]={extOrder==1?1.:2.,extOrder==1?0.:-1.};
  fluidForcingSnapshotValid=false;
  for(int n=0;n<N;++n){
   mesh.o_LMM.ptr()[n]=mass[n];o_rhoPressure.ptr()[n]=rho[n];
   o_frozenMixtureDragRate.ptr()[n]=drag*(n==0?0.:(n==1?4.:10.));
  }
  for(int c=0;c<3;++c)for(int n=0;n<N;++n){
   int id=n+c*O;state[id]=.1+.02*id;state[id+15]=.05+.01*id;
   adv[id]=.03*id;adv[id+15]=-.07*id;
   // Nonlinear signed VM history exposes any accidental reweighting.
   fu[id]=1.+vm*(id+1)*(id+1);fu[id+15]=-.2-vm*(id+2)*(id+2);
   o_ug.ptr()[id]=.3+.01*id;gp[id]=-.2*(id+1);curl[id]=.05*id;
  }
  sumMakef(N,mass,100.,ext,bdf,O,15,3,rho,state,adv,fu,bf);
  std::copy(bf,bf+15,fluid.o_JwF.ptr());
  prepareMixtureDrag(.1,1);
  for(int n=0;n<N;++n){
   double D=o_frozenMixtureDragRate.ptr()[n],chi=1-D/150.;
   assert(close(fluid.o_prop.ptr()[O+n],rho[n]*chi));
   inverseRho[n]=1/fluid.o_prop.ptr()[O+n];
   // Dynamic BDF diagonal is exactly the original diagonal minus rho*D.
   assert(close(150.*fluid.o_prop.ptr()[O+n],rho[n]*(150.-D)));
  }
  velocityRhsHex3D(N,O,fluid.o_prop.ptr()+O,fluid.o_JwF.ptr(),gradDiv,gp,velrhs);
  pressureRhsHex3D(N,O,mu,inverseRho,fluid.o_JwF.ptr(),curl,gradDiv,prhs);
  for(int c=0;c<3;++c)for(int n=0;n<N;++n){
   int id=n+c*O;double D=o_frozenMixtureDragRate.ptr()[n],chi=1-D/150.;
   double rhs=bf[id]-mass[n]*D*o_ug.ptr()[id];
   assert(close(velrhs[id],rho[n]*rhs+gp[id]));
   assert(close(chi*prhs[id],rhs-mu[n]/rho[n]*curl[id]));
   double expected=rhs/chi;
   assert(close(fluid.o_JwF.ptr()[id],expected));
   if(!drag)assert(fluid.o_JwF.ptr()[id]==bf[id]);
  }
  // Repeated preFluid calls must use the unmodified forcing snapshot.
  double first[15];std::copy(fluid.o_JwF.ptr(),fluid.o_JwF.ptr()+15,first);
  prepareMixtureDrag(.1,1);
  for(int i=0;i<15;++i)assert(fluid.o_JwF.ptr()[i]==first[i]);
  o_ug.ptr()[1]+=.1;prepareMixtureDrag(.1,1);
  double D=o_frozenMixtureDragRate.ptr()[1];
  assert(close(fluid.o_JwF.ptr()[1],(bf[1]-mass[1]*D*o_ug.ptr()[1])/(1-D/150.)));
 }
 for(double invalid:{150.,200.}){
  o_frozenMixtureDragRate.ptr()[1]=invalid;
  bool rejected=false;try{prepareMixtureDrag(.1,1);}catch(const std::runtime_error&){rejected=true;}
  assert(rejected);
 }
}
'''
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp);(p/'check.cpp').write_text(cpp)
 subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-Wno-unused-parameter','-Wno-unknown-pragmas',str(p/'check.cpp'),'-o',str(p/'check')],check=True)
 subprocess.run([str(p/'check')],check=True)
# The source stage must exclude UM drag from EXT and freeze its own rate.
forcing=t.split('inline void addExplicitSources(double)')[1].split('inline occa::memory implicitGasDrag')[0]
assert 'o_frozenMixtureDragRate.copyFrom(o_mixtureDragRate' in forcing
assert 'fluidForcingSnapshotValid = false;' in forcing
assert 'freezeDragHistory' not in t
order=(repo/'src/app/nrs/nrs.cpp').read_text().split('bool nrs_t::runInnerStep')[1]
assert order.index('scalar->solve(')<order.index('evaluateProperties(timeNew)')<order.index('preFluid(timeNew, tstep)')<order.index('fluid->solve(')
print('Actual preFluid/native BDF+pressure+velocity: reaction diagonal, full VM/EXT/BDF forcing preserved, variable rho/mass, pressure mobility, repeated stages, updated UG and diagonal guard passed.')
