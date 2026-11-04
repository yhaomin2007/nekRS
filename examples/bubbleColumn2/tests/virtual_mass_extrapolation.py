"""Actual gas VM kernel/helper plus native UG/UM forcing and drag-history isolation."""
from pathlib import Path
import re,subprocess,tempfile
case=Path(__file__).resolve().parents[1];repo=case.parents[1]
terms=(case/'bubbleColumn2Terms.hpp').read_text();eq=(case/'bubbleColumn2Equations.okl').read_text()
def function(s,marker):
 start=s.index(marker);end=s.index('{',start)+1;depth=1
 while depth:
  depth+=(s[end]=='{')-(s[end]=='}');end+=1
 return s[start:end]
def serial(s):
 return re.sub(r';\s*@tile\(p_blockSize,\s*@outer,\s*@inner\)','',s.replace('@kernel ','').replace('@ restrict ',''))
def kernel(name):return serial(function(eq,'@kernel void '+name+'('))
base=kernel('buildEquationTerms');args=base.split(')\n{')[0].split('(',1)[1].split(',')
values=dict(N=2,offset=3,rhoLiquid=998.2,rhoGas=1.2,muLiquid=.001,muGas=1.8e-5,alphaFloor=1e-6,
 gasPressureEnabled=1,dragEnabled=1,mixtureDragEnabled=1,dragAlphaCutoff=0,dragSlipLimitEnabled=0,
 dragSlipMaximum=1,bubbleDiameter=.003,virtualMassEnabled=1,virtualMassCoefficient=.5,
 mixtureVirtualMassEnabled=1,explicitVirtualMassEnabled=0,gravityX=.3,gravityY=-.4,gravityZ=-9.81)
decl=[];names=[]
for a in args:
 name=a.strip().split()[-1].lstrip('*');names.append(name)
 decl.append('double '+name+'[64]={};' if '*' in a else ('long ' if name in ['N','offset'] else 'double ')+name+'='+str(values[name])+';')
call='buildEquationTerms('+','.join(names)+');'
scalar=serial((repo/'src/solver/scalar/kernels/sumMakef.okl').read_text()).replace('sumMakef(','scalarSumMakef(')
fluid=serial((repo/'src/solver/fluid/kernels/sumMakef.okl').read_text()).replace('sumMakef(','fluidSumMakef(')
cpp=r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>
using dfloat=double;using dlong=long;
#define p_nBDF 2
#define p_nEXT 3
#define p_SUBCYCLING 0
#define p_MovingMesh 0
#define p_ADVECTION 0
bool close(double a,double b){return std::abs(a-b)<1e-9*(1+std::abs(b));}
'''+base+kernel('buildMixtureForce')+kernel('extrapolateGasVirtualMass')+scalar+fluid+r'''
struct Mem{
 std::shared_ptr<std::vector<double>> data;long off=0,count=0;
 Mem(long n=0):data(std::make_shared<std::vector<double>>(n,0)),count(n){}
 double& operator[](long i){return data->at(off+i);}double operator[](long i)const{return data->at(off+i);}
 double* ptr(){return data->data()+off;}long size()const{return count;}
 Mem slice(long a,long n)const{auto m=*this;m.off+=a;m.count=n;return m;}
 void copyFrom(const Mem& m){for(long j=0;j<count;j++)(*this)[j]=m[j];}
};
struct LA{
 void fill(long n,double a,Mem& x){for(long j=0;j<n;j++)x[j]=a;}
 void axpby(long n,double a,const Mem& x,double b,Mem& y,long xo=0,long yo=0){for(long j=0;j<n;j++)y[j+yo]=a*x[j+xo]+b*y[j+yo];}
 void axpbyz(long n,double a,const Mem& x,double b,const Mem& y,Mem& z){for(long j=0;j<n;j++)z[j]=a*x[j]+b*y[j];}
}la;
struct Env{LA* linAlg=&la;}env;auto platform=&env;
struct Params{
 int virtualMassExtrapolationEnabled=1,explicitVirtualMassEnabled=0;
 int mixtureForcingEnabled=1,frozenGasVelocityEnabled=0,frozenGasMixtureForcingEnabled=0;
 int virtualMassStartStep=2,virtualMassRampStep=4;
 double virtualMassEnabled=1,virtualMassCoefficient=.5,rhoLiquid=998.2,rhoGas=1.2;
}p;
struct Mesh{long Nlocal=2;}mesh;
struct Fluid{long fieldOffsetSum=9;Mem o_coeffEXT{3},o_EXT{27};}fluid;
struct Nrs{int tstep=0;long fieldOffset=3;Mesh* meshV=&mesh;Fluid* fluid=&::fluid;}app;auto nrs=&app;
Mem o_gasVirtualMassHistory(27),o_liquidAcceleration(9),o_ugSource(9);
Mem o_mixtureDragSource(9),o_previousMixtureDragSource(9),o_mixtureInterphaseAcceleration(9);
Mem o_mixtureVirtualMassSource(9),o_previousMixtureVirtualMassSource(9);
int gasVirtualMassHistoryStep=-1;
'''+function(terms,'inline dfloat virtualMassRampFactor(')+r'''
void extrapolateGasVirtualMassKernel(long N,long O,int stages,int use,int first,int advance,
 double scale,Mem coeff,Mem al,Mem hist,Mem source){
 assert(coeff.data==fluid.o_coeffEXT.data);
 extrapolateGasVirtualMass(N,O,stages,use,first,advance,scale,coeff.ptr(),al.ptr(),hist.ptr(),source.ptr());
}
'''+function(terms,'inline void applyGasVirtualMassExtrapolation(')+function(terms,'inline void freezeExplicitVirtualMassHistory(')+function(terms,'inline void freezeDragHistory(')+r'''
std::array<double,3> weights(std::array<double,3> t,double target){
 std::array<double,3> b{1,1,1};for(int i=0;i<3;i++)for(int j=0;j<3;j++)if(i!=j)b[i]*=(target-t[j])/(t[i]-t[j]);return b;
}
int main(){
'''+''.join(decl)+r'''
 std::fill(o_gasVirtualMassHistory.data->begin(),o_gasVirtualMassHistory.data->end(),999.);
 std::fill(o_ugSource.data->begin(),o_ugSource.data->end(),777.);
 // Startup probes cannot consume history, and the default mode switch is inert.
 auto initial=*o_gasVirtualMassHistory.data;applyGasVirtualMassExtrapolation();
 assert(*o_gasVirtualMassHistory.data==initial && gasVirtualMassHistoryStep==-1);
 p.virtualMassExtrapolationEnabled=0;nrs->tstep=1;applyGasVirtualMassExtrapolation();assert(*o_gasVirtualMassHistory.data==initial);
 p.virtualMassExtrapolationEnabled=1;
 std::vector<std::array<double,9>> gasHistory,mixHistory,otherHistory;
 std::array<double,3> times{};double time=0;
 for(int step=1;step<=9;step++){
  nrs->tstep=step;times[2]=times[1];times[1]=times[0];times[0]=time;
  double h=step<5?.01:(step%2?.015:.008);
  auto beta=step==1?std::array<double,3>{1,0,0}:step==2?std::array<double,3>{2,-1,0}:weights(times,time+h);
  for(int stage=0;stage<3;stage++)fluid.o_coeffEXT[stage]=beta[stage];
  virtualMassEnabled=p.virtualMassEnabled*virtualMassRampFactor(step);
  const double c=virtualMassEnabled*.5*rhoLiquid,eff=rhoGas+c;
  for(int n=0;n<N;n++){
   alpha[n]=.02+.1*n;
   for(int f=0;f<3;f++){
    int id=n+f*offset;
    ug[id]=.3+.03*f;ul[id]=.1-.02*n;uv[id]=alpha[n]*ug[id]+(1-alpha[n])*ul[id];
    gradP[id]=2+f+n;
    liquidAcceleration[id]=(step%2?-1.:1.)*(step*step+f+n);
    virtualMassRelativeAcceleration[id]=.7*step*step-f-.5*n;
    o_liquidAcceleration[id]=liquidAcceleration[id];
   }
  }
'''+call+r'''
  double mixForce[9]={},divDrift[9]={};
  buildMixtureForce(N,offset,alphaFloor,0,gravityX,gravityY,gravityZ,alpha,divDrift,mixtureInterphaseAcceleration,mixForce);
  std::array<double,9> qG{},qM{},other{},base{};
  for(int f=0;f<3;f++)for(int n=0;n<N;n++){
   int id=n+f*offset;
   qG[id]=c/eff*liquidAcceleration[id];
   qM[id]=alpha[n]*(1-rhoGas/rhoLiquid)*c/rhoGas*virtualMassRelativeAcceleration[id];
   assert(close(mixtureInterphaseAcceleration[id]-mixtureDragSource[id],qM[id]));
   double g=f==0?gravityX:(f==1?gravityY:gravityZ);
   double extra=.2*step*(f+1);
   base[id]=(-gradP[id]+rhoGas*g)/eff+dragLambda[n]*ul[id]+extra;
   assert(close(ugSource[id]+extra,base[id]+qG[id]));
   o_ugSource[id]=ugSource[id]+extra;
   o_mixtureDragSource[id]=mixtureDragSource[id];
   o_mixtureInterphaseAcceleration[id]=mixtureInterphaseAcceleration[id];
   other[id]=g+extra;
  }
  gasHistory.insert(gasHistory.begin(),qG);mixHistory.insert(mixHistory.begin(),qM);otherHistory.insert(otherHistory.begin(),other);
  if(step==1)while(gasHistory.size()<3){gasHistory.push_back(qG);mixHistory.push_back(qM);otherHistory.push_back(other);}
  if(gasHistory.size()>3){gasHistory.resize(3);mixHistory.resize(3);otherHistory.resize(3);}
  auto fluidBefore=*fluid.o_EXT.data;
  applyGasVirtualMassExtrapolation();assert(*fluid.o_EXT.data==fluidBefore);
  auto gasAfter=*o_ugSource.data,historyAfter=*o_gasVirtualMassHistory.data;
  // Same-step source callback rebuilds the pointwise load before refreshing VM.
  for(int f=0;f<3;f++)for(int n=0;n<N;n++){int id=n+f*offset;o_ugSource[id]=base[id]+qG[id];}
  applyGasVirtualMassExtrapolation();assert(*o_gasVirtualMassHistory.data==historyAfter);
  for(int f=0;f<3;f++)for(int n=0;n<N;n++){int id=n+f*offset;assert(close(o_ugSource[id],gasAfter[id]));}
  // Fully explicit last-step freezing must be disabled, so UM VM remains EXT.
  freezeExplicitVirtualMassHistory();assert(*fluid.o_EXT.data==fluidBefore);
  freezeDragHistory();
  for(int f=0;f<3;f++)for(int n=0;n<N;n++){
   int id=n+f*offset;fluid.o_EXT[id]=mixForce[id]+.2*step*(f+1);
   double expectedGas=base[id];for(int j=0;j<3;j++)expectedGas+=beta[j]*gasHistory[j][id];
   assert(close(o_ugSource[id],expectedGas));
  }
  double mass[2]={2,5},rho[9],state[18],adv[27]={},sources[27],gasRhs[9],mixRhs[9];
  std::fill_n(rho,9,1.);std::fill_n(state,18,.3);std::fill_n(sources,27,4567.);
  for(int f=0;f<3;f++)for(int n=0;n<N;n++)sources[n+f*offset]=o_ugSource[n+f*offset];
  for(int order=1;order<=2;order++){
   double bdf[2]={order==1?1.:2.,order==1?0.:-.5},g0=order==1?1.:1.5,ext1[3]={1,0,0};
   for(int f=0;f<3;f++)scalarSumMakef(N,mass,1/h,ext1,bdf,f*offset,9,2,rho,state,adv,sources,gasRhs);
   fluidSumMakef(N,mass,1/h,fluid.o_coeffEXT.ptr(),bdf,offset,9,2,rho,state,adv,fluid.o_EXT.ptr(),mixRhs);
   for(int f=0;f<3;f++)for(int n=0;n<N;n++){
    int id=n+f*offset;double temporal=(bdf[0]+bdf[1])*.3/h;
    double extVM=0,expectedMix=o_mixtureDragSource[id];
    for(int j=0;j<3;j++){extVM+=beta[j]*gasHistory[j][id];expectedMix+=beta[j]*(mixHistory[j][id]+otherHistory[j][id]);}
    assert(close(gasRhs[id]/mass[n],temporal+base[id]+extVM));
    assert(close(mixRhs[id]/mass[n],temporal+expectedMix));
    double unew=gasRhs[id]/mass[n]/(g0/h+dragLambda[n]);
    // Added gas inertia remains on the BDF LHS; only its explicit liquid
    // counterpart is extrapolated. Do not divide the UM VM by eff.
    assert(close(eff*((g0*unew-(bdf[0]+bdf[1])*.3)/h+dragLambda[n]*unew),eff*(base[id]+extVM)));
   }
  }
  for(int f=0;f<3;f++){
   assert(o_ugSource[f*offset+2]==777.);
   for(int stage=0;stage<3;stage++)assert(o_gasVirtualMassHistory[stage*9+f*offset+2]==999.);
  }
  for(int stage=2;stage>=1;stage--)for(int j=0;j<9;j++)fluid.o_EXT[stage*9+j]=fluid.o_EXT[(stage-1)*9+j];
  time+=h;
 }
}
'''
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp)/'check.cpp';p.write_text(cpp)
 subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-Wno-unknown-pragmas','-Wno-unused-parameter',str(p),'-o',str(p.with_suffix(''))],check=True)
 subprocess.run([str(p.with_suffix(''))],check=True)
par=(case/'bubbleColumn2.par').read_text()
for key,value in [('virtualMassExtrapolationEnabled','1'),('explicitVirtualMassEnabled','0'),('coupledVirtualMassEnabled','0'),('mixtureVirtualMassEnabled','1'),('polynomialOrder','5'),('cubaturePolynomialOrder','10')]:
 assert re.search(r'^'+key+r'\s*=\s*'+value+r'\s*$',par,re.M)
udf=(case/'bubbleColumn2.udf').read_text();a=udf.index('  nekrsCheck(p.virtualMassExtrapolationEnabled <');b=udf.index('  nekrsCheck(p.explicitVirtualMassEnabled <',a)
guard=udf[a:b]
controls=r'''
#include <cassert>
#include <cstdlib>
#include <stdexcept>
struct P{int virtualMassExtrapolationEnabled=1,explicitVirtualMassEnabled=0,coupledVirtualMassEnabled=0,mixtureVirtualMassEnabled=1,mixtureForcingEnabled=1;double virtualMassEnabled=1;}p;
struct Comm{int mpiComm(){return 0;}};struct Env{Comm comm;}env;auto platform=&env;
#define nekrsCheck(test,...) do{if(test)throw std::runtime_error("mode");}while(0)
void check(){
'''+guard+r'''
}
int main(){check();for(int j=0;j<6;j++){
 p=P{};if(j==0)p.virtualMassExtrapolationEnabled=-1;if(j==1)p.virtualMassExtrapolationEnabled=2;
 if(j==2)p.explicitVirtualMassEnabled=1;if(j==3)p.coupledVirtualMassEnabled=1;
 if(j==4)p.mixtureVirtualMassEnabled=0;if(j==5)p.mixtureForcingEnabled=0;
 bool bad=false;try{check();}catch(const std::runtime_error&){bad=true;}assert(bad);
 p.virtualMassExtrapolationEnabled=0;check();
}p=P{};p.virtualMassEnabled=0;p.mixtureVirtualMassEnabled=0;p.mixtureForcingEnabled=0;check();}
'''
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp)/'guard.cpp';p.write_text(controls)
 subprocess.run(['g++','-std=c++17',str(p),'-o',str(p.with_suffix(''))],check=True)
 subprocess.run([str(p.with_suffix(''))],check=True)
assert 'VM EXTRAPOLATION:' in udf
body=function(terms,'inline void addExplicitSources(')
assert body.index('evaluatePointwiseTerms();')<body.index('applyGasVirtualMassExtrapolation();')<body.index('captureGasPressureSource();')
print('VM EXT passed: actual gas/mixture sources and native BDF1/2 forcing, EXT1/2/3, variable dt, sampled ramp scaling, source isolation, unscaled physical UM VM, startup/repeats/padding, drag-history preservation and mode guards.')
