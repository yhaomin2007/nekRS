"""Compile actual deferred-UG header + native sumMakef; compare direct solves.

This is a C++ host orchestration/operator test, not a native MPI/GPU run.
"""
from pathlib import Path
import re,subprocess,tempfile
case=Path(__file__).resolve().parents[1]
repo=case.parents[1]
header=(case/'bubbleColumn2AfterFluid.hpp').read_text().replace('#pragma once','')
native=(repo/'src/solver/scalar/kernels/sumMakef.okl').read_text()
native=re.sub(r'^#pragma.*$', '', native, flags=re.M)
native=re.sub(r';\s*@tile\(p_blockSize,\s*@outer,\s*@inner\)','',native.replace('@kernel ','').replace('@ restrict ',''))
cpp=r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
using dfloat=double;using dlong=long;
#define p_nBDF 2
#define p_nEXT 3
#define p_SUBCYCLING 0
#define p_MovingMesh 0
#define p_ADVECTION 1
'''+native+r'''
struct Mem{
 std::shared_ptr<std::vector<double>> v;long off=0,count=0;
 Mem(long n=0):v(std::make_shared<std::vector<double>>(n,0)),count(n){}
 void resize(long n){v=std::make_shared<std::vector<double>>(n,0);off=0;count=n;}
 long size()const{return count;}
 double& operator[](long i){return v->at(off+i);}double operator[](long i)const{return v->at(off+i);}
 double* ptr(){return v->data()+off;}const double* ptr()const{return v->data()+off;}
 Mem slice(long o,long n)const{auto m=*this;m.off+=o;m.count=n;return m;}
 void copyFrom(const Mem& x,long n=-1,long dst=0,long src=0){if(n<0)n=count;for(long i=0;i<n;++i)(*this)[i+dst]=x[i+src];}
 void copyFrom(const long* x){for(long i=0;i<count;++i)(*this)[i]=x[i];}
};
namespace occa{using memory=Mem;}
template<class T>using deviceMemory=Mem;
struct LA{
 void fill(long n,double x,Mem m){for(long i=0;i<n;++i)m[i]=x;}
 void axpby(long n,double a,Mem x,double b,Mem y){for(long i=0;i<n;++i)y[i]=a*x[i]+b*y[i];}
 void axmy(long n,double a,Mem x,Mem y){for(long i=0;i<n;++i)y[i]*=a*x[i];}
 void axmyMany(long n,int fields,long O,int,double a,Mem x,Mem y){for(int c=0;c<fields;++c)for(long i=0;i<n;++i)y[i+c*O]*=a*x[i];}
}la;
struct Comm{int mpiComm(){return 0;}int rank=0;int mpiRank(){return rank;}};
struct Options{bool compareArgs(std::string,std::string){return false;}};
struct Par{void extract(std::string,std::string,int&) {}}par;
struct Platform{Comm comm;Options options;Par* par=&::par;LA* linAlg=&la;}env;auto platform=&env;
template<class...Args>void guard(bool test,Args&&...){if(test)throw std::runtime_error("guard");}
#define nekrsCheck guard
struct Mesh{long Nlocal=2,Nelements=1,fieldOffset=4;int dim=3;Mem o_LMM{4},o_invLMM{4},o_vgeo,o_D;int oogs=0;}mesh;
struct Scalar{
 int fieldOffsetSum=20;std::vector<long> fieldOffsetScan{0,5,10,15};
 std::vector<int> compute{1,1,1,1};void* cvode=nullptr;
 std::map<std::string,int> nameToIndex{{"alpha",0},{"ugx",1},{"ugy",2},{"ugz",3}};
 Mem o_coeffEXT{3},o_coeffBDF{2},o_S{60},o_EXT{60},o_JwF{20},o_compute{4},diff{20},rho{20};
 int calls=0,bcCalls=0,explicitCalls=0;double lambda=0;int boundary=0;
 Mem o_transportCoeff(){return rho;}Mem o_diffusionCoeff(std::string s){return diff.slice(fieldOffsetScan[nameToIndex.at(s)],4);}
 void makeExplicit(int is,double,int){++explicitCalls;for(int n=0;n<2;++n)o_EXT[fieldOffsetScan[is]+n]+=.07*is;}
 void applyDirichlet(double){assert(compute==std::vector<int>({0,1,1,1}));++bcCalls;}
 void solve(double,int);
}scalar;
struct Fluid{Mem o_P{12};}fluid;
struct Nrs{
 int tstep=0,Nscalar=4,advectionSubcycingSteps=0;long fieldOffset=4;double dt[2]={.1,.1};bool timeStepConverged=false;
 void* geom=nullptr;void* neknek=nullptr;Mesh* meshV=&mesh;Scalar* scalar=&::scalar;Fluid* fluid=&::fluid;
 void(*userSource)(double)=nullptr;void(*postScalar)(double,int)=nullptr;void(*preFluid)(double,int)=nullptr;
 void(*userProperties)(double)=nullptr;bool(*userConvergenceCheck)(int)=nullptr;
}app;auto nrs=&app;
namespace bubbleColumn2{
struct Params{
 int gasSolveAfterFluid=1,frozenGasVelocityEnabled=0,coupledVirtualMassEnabled=0,gasPressureGradientMethod=0;
 int explicitVirtualMassEnabled=0;double virtualMassEnabled=1,virtualMassCoefficient=.5;
 double rhoGas=1.2,rhoLiquid=998.2,gasPressureEnabled=1,gasPressureGradientFilterEnabled=0;
}p;
Mem o_dragLambda(4),o_gradP(12),o_gradPStrong(12),o_gradPWeak(12),o_ugPredictorPressureGradient(12);
double virtualMassRampFactor(int step){return step==1?.2:.6;}
double scale(){return -1/(p.rhoGas+(p.explicitVirtualMassEnabled?0:p.virtualMassEnabled)*virtualMassRampFactor(nrs->tstep)*p.virtualMassCoefficient*p.rhoLiquid);}
void prepareGasPressureForcing(){o_gradP.copyFrom(p.gasPressureGradientMethod==0?o_gradPStrong:o_gradPWeak);for(int c=0;c<3;++c)for(int n=0;n<2;++n)o_gradP[n+c*4]*=(n==0?.3:.8);}
void postProcessAlpha(double,int){assert(scalar.compute==std::vector<int>({1,0,0,0}));}
void addExplicitSources(double){
 o_gradPStrong.copyFrom(fluid.o_P);o_gradPWeak.copyFrom(fluid.o_P);prepareGasPressureForcing();
 for(int n=0;n<2;++n){
  o_dragLambda[n]=2+n*.3;
  for(int c=0;c<3;++c)scalar.o_EXT[scalar.fieldOffsetScan[c+1]+n]=1.1+c+.2*n+scale()*o_gradP[n+c*4];
 }
}
void updateProperties(double){
 for(int c=0;c<3;++c)for(int n=0;n<2;++n)scalar.diff[scalar.fieldOffsetScan[c+1]+n]=.4+.2*c+.1*n;
 for(int n=0;n<2;++n)o_dragLambda[n]=999; // source-time drag must be restored
}
}
namespace opSEM{void strongGrad(Mesh*,long,Mem in,Mem out){out.copyFrom(in);}}
const int ogsDfloat=0,ogsAdd=0;
namespace oogs{void startFinish(Mem,int,long,int,int,int){}}
void launchKernel(const char*,long N,Mem mass,double idt,Mem ext,Mem bdf,long scan,long sum,long jw,Mem rho,Mem U,Mem adv,Mem FU,Mem BF){
 sumMakef(N,mass.ptr(),idt,ext.ptr(),bdf.ptr(),scan,sum,jw,rho.ptr(),U.ptr(),adv.ptr(),FU.ptr(),BF.ptr());
}
void launchKernel(const char*,long,Mem,Mem,long,Mem P,Mem out){
 for(int c=0;c<3;++c)for(int n=0;n<2;++n)out[n+c*4]=-mesh.o_LMM[n]*P[n+c*4];
}
bool close(double a,double b){return std::abs(a-b)<1e-11*(1+std::abs(b));}
std::array<double,2> solveDirect(double r0,double r1,double lambda0,double lambda1,double nu,int boundary,int c){
 const double d0=2*lambda0+nu,d1=5*lambda1+nu;
 if(boundary){double x=.12*(c+1);return {x,(r1+nu*x)/d1};}
 const double det=d0*d1-nu*nu;return {(r0*d1+nu*r1)/det,(r1*d0+nu*r0)/det};
}
void Scalar::solve(double,int stage){
 assert(stage==1 && nrs->timeStepConverged && compute==std::vector<int>({0,1,1,1}));++calls;
 for(int c=0;c<3;++c){
  int scan=fieldOffsetScan[c+1];double nu=diff[scan];
  assert(close(nu,.4+.2*c));assert(close(bubbleColumn2::o_dragLambda[0],2));
  auto x=solveDirect(o_JwF[scan],o_JwF[scan+1],lambda+bubbleColumn2::o_dragLambda[0],lambda+bubbleColumn2::o_dragLambda[1],nu,boundary,c);
  o_S[scan]=x[0];o_S[scan+1]=x[1];
 }
}
'''+header+r'''
int main(){using namespace bubbleColumn2;
 for(int boundary:{0,1})for(int weak:{0,1})for(int explicitVM:{0,1}){
  nrs->preFluid=nullptr;nrs->userConvergenceCheck=nullptr;scalar.compute={1,1,1,1};
  afterFluid::preparedStep=afterFluid::solvedStep=afterFluid::propertiesStep=-1;
  scalar.calls=scalar.bcCalls=scalar.explicitCalls=0;scalar.boundary=boundary;
  p.gasPressureGradientMethod=weak;p.explicitVirtualMassEnabled=explicitVM;
  mesh.o_LMM[0]=2;mesh.o_LMM[1]=5;mesh.o_invLMM[0]=.5;mesh.o_invLMM[1]=.2;
  for(int i=0;i<20;++i)scalar.rho[i]=1;
  for(int i=0;i<60;++i)scalar.o_S[i]=.1+.01*i;
  for(int i=0;i<12;++i)fluid.o_P[i]=.2+.03*i;
  afterFluid::logInterval=1;
  afterFluid::configure();assert(scalar.compute==std::vector<int>({1,1,1,1}));
  for(int i=0;i<4;++i)assert(scalar.o_compute[i]==(i==0));
  nrs->tstep=0;afterFluid::beginStep(0);assert(afterFluid::preparedStep==-1);
  for(int step=1;step<=2;++step){
   nrs->tstep=step;nrs->timeStepConverged=false;nrs->dt[0]=step==1?.1:.06;
   double r=.6,g0=step==1?1:(1+2*r)/(1+r);
   scalar.lambda=g0/nrs->dt[0];scalar.o_coeffBDF[0]=step==1?1:1+r;scalar.o_coeffBDF[1]=step==1?0:-r*r/(1+r);
   scalar.o_coeffEXT[0]=1;scalar.o_coeffEXT[1]=scalar.o_coeffEXT[2]=0;
   auto old=*scalar.o_S.v;auto oldP=*fluid.o_P.v;
   afterFluid::beginStep(0);assert(scalar.compute==std::vector<int>({1,0,0,0}));
   assert(scalar.explicitCalls==3*step);
   bool rejected=false;try{afterFluid::beginStep(0);}catch(std::runtime_error&){rejected=true;}assert(rejected);
   // Simulate native alpha solve and the one global solution/source lag.
   for(int j=59;j>=20;--j)scalar.o_S[j]=scalar.o_S[j-20];
   for(int j=59;j>=20;--j)scalar.o_EXT[j]=scalar.o_EXT[j-20];
   scalar.o_S[0]=.09;scalar.o_S[1]=.11;nrs->postScalar(.1,step);
   nrs->userProperties(.1);nrs->preFluid(.1,step);
   for(int i=0;i<12;++i)fluid.o_P[i]=-.7+.02*i+.3*step;
   la.fill(20,888,scalar.diff); // pressure/UM scratch updates cannot alter UG operator
   auto lagged=*scalar.o_S.v;auto history=*scalar.o_EXT.v;
   nrs->timeStepConverged=true;afterFluid::solveGas(.1,step);
   assert(scalar.calls==step && scalar.bcCalls==step && scalar.compute==std::vector<int>({1,1,1,1}));
   assert(*scalar.o_EXT.v==history);assert(close(scalar.o_S[0],.09)&&close(scalar.o_S[1],.11));
   for(int c=0;c<3;++c){
    int scan=scalar.fieldOffsetScan[c+1];double rhs[2];
    for(int n=0;n<2;++n){
     double derivative=(scalar.o_coeffBDF[0]*old[scan+n]+scalar.o_coeffBDF[1]*old[20+scan+n])/nrs->dt[0];
     rhs[n]=mesh.o_LMM[n]*(derivative+1.1+c+.2*n+.07*(c+1)+scale()*fluid.o_P[n+c*4]*(n==0?.3:.8));
     assert(close(o_ugPredictorPressureGradient[n+c*4],fluid.o_P[n+c*4]*(n==0?.3:.8)));
    }
    auto expected=solveDirect(rhs[0],rhs[1],scalar.lambda+2,scalar.lambda+2.3,.4+.2*c,boundary,c);
    assert(close(scalar.o_S[scan],expected[0]) && close(scalar.o_S[scan+1],expected[1]));
   }
   for(int j=20;j<60;++j)assert(scalar.o_S[j]==lagged[j]);
   rejected=false;try{afterFluid::solveGas(.1,step);}catch(std::runtime_error&){rejected=true;}assert(rejected);
  }
 }
 // Configuration guards reject modes whose pressure/operator cannot be frozen this way.
 for(int mode=0;mode<5;++mode){
  nrs->preFluid=nullptr;nrs->userConvergenceCheck=nullptr;
  p.coupledVirtualMassEnabled=mode==0;p.gasPressureGradientMethod=mode==1?2:0;
  p.gasPressureGradientFilterEnabled=mode==2?1:0;p.frozenGasVelocityEnabled=mode==3;
  nrs->advectionSubcycingSteps=mode==4?1:0;
  bool rejected=false;try{afterFluid::configure();}catch(std::runtime_error&){rejected=true;}assert(rejected);
 }
 p.gasSolveAfterFluid=0;nrs->userSource=nullptr;afterFluid::configure();assert(nrs->userSource==nullptr);
 platform->comm.rank=1;afterFluid::stageLog("SHOULD_NOT_PRINT");
 assert(afterFluid::singlePass(1));bool rejected=false;try{afterFluid::singlePass(2);}catch(std::runtime_error&){rejected=true;}assert(rejected);
}
'''
with tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp)/'check.cpp';path.write_text(cpp)
    subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-Wno-unused-parameter',str(path),'-o',str(path.with_suffix(''))],check=True)
    output=subprocess.run([str(path.with_suffix(''))],check=True,capture_output=True,text=True).stdout
    rows=re.findall(r'method2Order step=(\d+) stage=(\S+) hostSolve\(alpha,ugx,ugy,ugz\)=\(([^)]+)\)',output)
    expected=[('ALPHA_PENDING','1,0,0,0'),('ALPHA_DONE','1,0,0,0'),('P_UM_BEGIN','1,0,0,0'),
              ('P_UM_DONE','1,0,0,0'),('UG_BEGIN_CURRENT_PRESSURE','0,1,1,1'),
              ('UG_DONE','0,1,1,1'),('STEP_FIELDS_READY','1,1,1,1')]
    assert len(rows)==8*2*7,len(rows)
    for start in range(0,len(rows),7):
        group=rows[start:start+7]
        assert [(stage,mask) for _,stage,mask in group]==expected
    assert 'SHOULD_NOT_PRINT' not in output
terms=(case/'bubbleColumn2Terms.hpp').read_text();udf=(case/'bubbleColumn2.udf').read_text()
assert '(p.frozenGasVelocityEnabled || p.gasSolveAfterFluid) && i > 0' in terms
assert 'if (p.gasSolveAfterFluid) p.gasPressureCorrectorEnabled = 0;' in udf
execute=udf.split('void UDF_ExecuteStep')[1]
assert execute.index('afterFluid::solveGas(time, tstep)')<execute.index('updateVirtualMassHistory();')
print('Actual C++ after-fluid orchestration + native BDF forcing: two steps, variable dt, strong/weak pressure, ramped semiimplicit/explicit VM mobility, drag/diffusion snapshots, Dirichlet/periodic solves, host/device masks, guards and history preservation passed.')
