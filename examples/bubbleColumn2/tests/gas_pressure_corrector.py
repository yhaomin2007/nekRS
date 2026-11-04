"""Compile actual corrector/capture helpers against a small diffusion operator.

Compare predictor+correction with a direct new-pressure solve, including drag,
nonuniform mass/diffusion, fixed boundary values and post-predictor mutations.
This exercises host algebra/routing; it is not a native MPI/GPU test.
"""
from pathlib import Path
import subprocess
import tempfile

case=Path(__file__).resolve().parents[1]
s=(case/'bubbleColumn2Terms.hpp').read_text()
def function(marker):
    start=s.index(marker);pos=s.index('{',start);end=pos+1;depth=1
    while depth:
        depth+=(s[end]=='{')-(s[end]=='}');end+=1
    return s[start:end]

cpp=r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "bubbleColumn2Elliptic.hpp"
using bubbleColumn2::solveGasPressureCorrection;
using dfloat=double;using dlong=long;
struct Mem {
 std::shared_ptr<std::vector<double>> v;long off=0,count=0;
 Mem(long n=0):v(std::make_shared<std::vector<double>>(n,0)),count(n){}
 Mem slice(long o,long n)const{auto m=*this;m.off+=o;m.count=n;return m;}
 double& operator[](long n){return v->at(off+n);}
 double operator[](long n)const{return v->at(off+n);}
 void copyFrom(const Mem& x,long n){for(long j=0;j<n;j++)(*this)[j]=x[j];}
};
namespace occa {using memory=Mem;}
Mem o_NULL;
struct LA {
 void fill(long n,double a,Mem& y){for(long j=0;j<n;j++)y[j]=a;}
 void axpby(long n,double a,const Mem& x,double b,Mem& y){for(long j=0;j<n;j++)y[j]=a*x[j]+b*y[j];}
 void axpbyz(long n,double a,const Mem& x,double b,const Mem& y,Mem& z){for(long j=0;j<n;j++)z[j]=a*x[j]+b*y[j];}
 void axmy(long n,double a,const Mem& x,Mem& y){for(long j=0;j<n;j++)y[j]*=a*x[j];}
 void axmyz(long n,double a,const Mem& x,const Mem& y,Mem& z){for(long j=0;j<n;j++)z[j]=a*x[j]*y[j];}
 void axmyMany(long n,int nf,long o,int,double a,const Mem& x,Mem& y){for(int c=0;c<nf;c++)for(long j=0;j<n;j++)y[c*o+j]*=a*x[j];}
 double amax(long n,const Mem& x,int){double v=0;for(long j=0;j<n;j++)v=std::max(v,std::abs(x[j]));return v;}
}la;
struct Comm{int mpiComm(){return 0;}int mpiRank(){return 1;}};
struct Platform{LA* linAlg=&la;Comm comm;bool verbose(){return true;}}env;
auto platform=&env;
struct Mesh{long Nlocal=4,Nelements=1;int dim=3;Mem o_LMM{4},o_invLMM{4},o_vgeo,o_D;int oogs=0;}mesh;
using Vec=std::array<double,4>;using Matrix=std::array<Vec,4>;
Matrix matrix(const Mem& diff,const Mem& lambda){
 Matrix A{};
 for(int j=0;j<4;j++)A[j][j]=mesh.o_LMM[j]*lambda[j];
 for(int j=0;j<3;j++){
  double k=.5*(diff[j]+diff[j+1]);
  A[j][j]+=k;A[j+1][j+1]+=k;A[j][j+1]-=k;A[j+1][j]-=k;
 }
 return A;
}
Vec solve(Matrix A,Vec b,double boundary){
 for(int j=1;j<4;j++){b[j]-=A[j][0]*boundary;A[j][0]=0;}
 A[0]={1,0,0,0};b[0]=boundary;
 for(int k=0;k<4;k++){
  for(int j=k+1;j<4;j++){
   double w=A[j][k]/A[k][k];for(int m=k;m<4;m++)A[j][m]-=w*A[k][m];b[j]-=w*b[k];
  }
 }
 Vec x{};for(int j=3;j>=0;j--){x[j]=b[j];for(int k=j+1;k<4;k++)x[j]-=A[j][k]*x[k];x[j]/=A[j][j];}
 return x;
}
struct Solver {
 int it=7,calls=0;double r0=9,rg=8,r=1e-9;
 int Niter(){return it;}void Niter(int a){it=a;}
 double initialResidual(){return r0;}void initialResidual(double a){r0=a;}
 double initialGuessResidual(){return rg;}void initialGuessResidual(double a){rg=a;}
 double finalResidual(){return r;}void finalResidual(double a){r=a;}
 void solve(Mem diff,Mem lambda,const Mem& rhs,Mem delta){
  assert(delta[0]==0);Vec b{};for(int j=0;j<4;j++)b[j]=rhs[j];
  Vec x=::solve(matrix(diff,lambda),b,0);
  for(int j=0;j<4;j++)delta[j]=x[j];
  calls++;it=3;r=1e-13;r0=4;rg=5;
 }
};
double g0=1.5,dt[1]={.02};
struct Scalar {
 std::map<std::string,Mem> u,diff,rho;
 std::vector<Solver*> ellipticSolver;
 double* g0=&::g0;double* dt=::dt;
 Mem o_solution(std::string n){return u.at(n);}
 Mem o_diffusionCoeff(std::string n){return diff.at(n);}
 Mem o_transportCoeff(std::string n){return rho.at(n);}
}scalar;
struct Fluid{Mem o_P{4},o_U{18};}fluid;
struct Nrs{int tstep=1;long fieldOffset=6;Mesh* meshV=&mesh;Scalar* scalar=&::scalar;Fluid* fluid=&::fluid;}app;
auto nrs=&app;
struct Params {
 int gasPressureCorrectorEnabled=1,gasPressureCorrectorGradientMethod=0;
 int coupledVirtualMassEnabled=0,explicitVirtualMassEnabled=0;
 int frozenGasVelocityEnabled=0,gasPressureInletMaskEnabled=0,gasPressureInletRegionEnabled=0;
 int stabilityMonitorInterval=1;
 double gasPressureEnabled=1,rhoGas=1.2,rhoLiquid=998.2,virtualMassEnabled=0,virtualMassCoefficient=.5;
}p;
double virtualMassRampFactor(int){return .5;}
Mem o_ugPredictorPressureGradient(18);
Mem o_gradP(18),o_ugPressureUsed(18),o_ugCorrectorDiff(18),o_ugCorrectorLambda(18);
Mem o_ugCorrectorRhs(4),o_ugCorrectorDelta(4),o_dragLambda(6);
Mem o_coupledVMCoefficients(24),o_ugCorrectorPressureResponse(6);
Mem o_gasPressureInletMask(6),o_gasPressureRegionWeight(6),newGrad(18);
int ugPressureSourceStep=-1,ugPressureCorrectedStep=-1,ugOperatorStep[3]={-1,-1,-1};
double ugPressureSourceScale=0;
int refreshes=0;
void reconstructLiquidVelocity(){refreshes++;}
namespace opSEM {
 void strongGrad(Mesh*,long,const Mem&,Mem& g){g.copyFrom(newGrad,18);}
}
constexpr int ogsDfloat=0,ogsAdd=0;
namespace oogs {void startFinish(Mem&,int,long,int,int,int){}}
void launchKernel(const char*,long,const Mem&,const Mem&,long offset,const Mem&,Mem& g){
 for(int c=0;c<3;c++)for(int j=0;j<4;j++)g[c*offset+j]=-mesh.o_LMM[j]*newGrad[c*offset+j];
}
#define EXIT_FAILURE 1
#define nekrsCheck(test,...) do{if(test)throw std::runtime_error("snapshot guard");}while(0)
''' + function('inline void captureGasPressureSource(')+'\n'+function('inline occa::memory implicitGasDrag(')+'\n'+function('inline void correctGasPressure(')+r'''
int main(){
 Solver solvers[4];scalar.ellipticSolver={&solvers[0],&solvers[1],&solvers[2],&solvers[3]};
 const char* names[3]={"ugx","ugy","ugz"};
 for(auto n:names){scalar.u.emplace(n,Mem(6));scalar.diff.emplace(n,Mem(6));scalar.rho.emplace(n,Mem(6));}
 for(int j=0;j<4;j++){mesh.o_LMM[j]=j+2;mesh.o_invLMM[j]=1./mesh.o_LMM[j];fluid.o_P[j]=100+j;}
 const auto originalP=*fluid.o_P.v,originalUM=*fluid.o_U.v;
 for(int coupled:{0,1})for(int method:{0,1})for(int mask:{0,1})for(int vm:{0,1})for(int pressure:{0,1})for(int explicitVM:{0,1}){
  p.coupledVirtualMassEnabled=coupled; p.explicitVirtualMassEnabled=explicitVM;
  nrs->tstep++;p.gasPressureCorrectorGradientMethod=method;
  p.gasPressureInletMaskEnabled=mask;p.gasPressureInletRegionEnabled=mask;
  p.virtualMassEnabled=vm;p.gasPressureEnabled=pressure;
  Vec expected[3];Matrix A[3];Mem savedDiff[3],savedLambda[3];
  for(int j=0;j<6;j++){o_gasPressureInletMask[j]=j==1?0:1;o_gasPressureRegionWeight[j]=j==2?.4:1;}
  for(int j=0;j<6;j++)o_coupledVMCoefficients[12+j]=.002+.0003*j;
  for(int c=0;c<3;c++)for(int j=0;j<6;j++){
   o_gradP[c*6+j]=2+c+.2*j;
   newGrad[c*6+j]=5-c+.3*j;
   scalar.diff.at(names[c])[j]=.2+c+.1*j;
   scalar.rho.at(names[c])[j]=1;
   o_dragLambda[j]=2+.5*j;
  }
  captureGasPressureSource();
  for(int id=0;id<18;id++)assert(o_ugPredictorPressureGradient[id]==o_gradP[id]);
  if(!coupled){
   double eff=p.rhoGas+(explicitVM?0.:vm*.5*p.virtualMassCoefficient*p.rhoLiquid);
   assert(std::abs(ugPressureSourceScale+pressure/eff)<1e-14);
  }
  for(int c=0;c<3;c++){
   implicitGasDrag(0,c+1);
   savedDiff[c]=o_ugCorrectorDiff.slice(c*6,6);
   savedLambda[c]=o_ugCorrectorLambda.slice(c*6,6);
   A[c]=matrix(savedDiff[c],savedLambda[c]);
   Vec predictorRhs{},targetRhs{};
   for(int j=0;j<4;j++){
    double base=3+c+j;
    predictorRhs[j]=base+mesh.o_LMM[j]*o_ugPressureUsed[c*6+j];
    double w=mask?o_gasPressureInletMask[j]*o_gasPressureRegionWeight[j]:1;
    double pressureScale=coupled?-pressure*o_ugCorrectorPressureResponse[j]:ugPressureSourceScale;
    targetRhs[j]=base+mesh.o_LMM[j]*pressureScale*newGrad[c*6+j]*w;
   }
   Vec pred=solve(A[c],predictorRhs,.3+c);
   expected[c]=solve(A[c],targetRhs,.3+c);
   for(int j=0;j<4;j++)scalar.u.at(names[c])[j]=pred[j];
   scalar.u.at(names[c])[4]=456.;
  }
  // Mimic post-scalar properties and monitor callbacks changing live buffers.
  for(auto n:names)for(int j=0;j<6;j++)scalar.diff.at(n)[j]=900+j;
  for(int j=0;j<6;j++)o_dragLambda[j]=9000;
  for(int j=0;j<18;j++)o_gradP[j]=-999;
  for(int j=0;j<6;j++)o_coupledVMCoefficients[12+j]=999; // Preserve predictor mobility snapshot.
  correctGasPressure();
  for(int c=0;c<3;c++){
   for(int j=0;j<4;j++)assert(std::abs(scalar.u.at(names[c])[j]-expected[c][j])<1e-12);
   assert(scalar.u.at(names[c])[4]==456.);
   assert(solvers[c+1].Niter()==7 && solvers[c+1].finalResidual()==1e-9);
  }
  int before=solvers[1].calls;correctGasPressure();assert(solvers[1].calls==before);
  assert(*fluid.o_P.v==originalP && *fluid.o_U.v==originalUM);
 }
 p.gasPressureCorrectorEnabled=0;nrs->tstep++;int before=solvers[1].calls;
 correctGasPressure();assert(solvers[1].calls==before);
 p.gasPressureCorrectorEnabled=1;bool rejected=false;
 try{correctGasPressure();}catch(const std::runtime_error&){rejected=true;}assert(rejected);
 assert(refreshes==64);
}
'''
with tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp);(path/'check.cpp').write_text(cpp)
    subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-I',str(case),str(path/'check.cpp'),'-o',str(path/'check')],check=True)
    subprocess.run([str(path/'check')],check=True)
    # Repeat the complete algebra/routing test with only the NekRS-LS API.
    modern=cpp.replace(' void solve(Mem diff,Mem lambda,const Mem& rhs,Mem delta){',
        ' Mem diff,lambda;\n'
        ' void coeff0HLM(const Mem& x){diff=x;}\n'
        ' void coeff1HLM(const Mem& x){lambda=x;}\n'
        ' void solve(const Mem& rhs,Mem delta){')
    assert modern != cpp
    (path/'modern.cpp').write_text(modern)
    subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-I',str(case),str(path/'modern.cpp'),'-o',str(path/'modern')],check=True)
    subprocess.run([str(path/'modern')],check=True)
udf=(case/'bubbleColumn2.udf').read_text()
execute=udf.split('void UDF_ExecuteStep')[1]
assert execute.index('correctGasPressure();')<execute.index('updateVirtualMassHistory();')
assert execute.index('correctGasPressure();')<execute.index('if (nrs->checkpointStep)')
source=function('inline void addExplicitSources(')
assert source.index('evaluatePointwiseTerms();')<source.index('captureGasPressureSource();')<source.index('subtractScalarDiffusion();')
assert 'gaspressurecorrectorenabled' in udf
assert 'hasRobin' in udf
print('Actual UG corrector/captures: direct re-solve equivalence, diffusion/drag/VM, strong/weak target, masks, mass, Dirichlet/Neumann, property mutations, padding, disable/idempotence/stale-step guards and output/history order passed.')
