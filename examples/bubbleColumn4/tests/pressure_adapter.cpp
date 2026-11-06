// Numerical adapter for the PRODUCTION crossStress + solvePhase functions.
// Two-node Galerkin element: M=diag(.5,.5), K=[[1,-1],[-1,1]].
// Variation is in x, so the longitudinal full-stress stiffness is 2*nu*K;
// transverse stiffness is nu*K. This distinguishes full stress from Laplacian.
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
using dfloat=double;using dlong=int;
#define p_blockSize 1
#define p_Nfaces 6
#define p_Nfp 4
#define p_Nsgeo 4
#define p_NXID 0
#define p_NYID 1
#define p_NZID 2
#define p_WSJID 3
#define nekrsCheck(condition,comm,code,fmt,...) do {if(condition) throw std::runtime_error(fmt);}while(0)
#include "equations_serial.hpp"
struct Mem{
 std::shared_ptr<std::vector<double>> data;int offset=0,n=0;
 Mem()=default;explicit Mem(int count):data(std::make_shared<std::vector<double>>(count)),n(count){}
 void resize(int count){*this=Mem(count);}int size()const{return n;}
 Mem slice(int off,int count=-1)const{Mem m=*this;m.offset+=off;m.n=count<0?n-off:count;assert(off+m.n<=n);return m;}
 double&operator[](int i)const{assert(i>=0&&i<n);return (*data)[offset+i];}
 void copyFrom(Mem src,int count=-1,int dest=0,int from=0){if(count<0)count=n;std::vector<double>tmp(count);for(int i=0;i<count;++i)tmp[i]=src[from+i];for(int i=0;i<count;++i)(*this)[dest+i]=tmp[i];}
 operator double*()const{return data->data()+offset;}
};
namespace occa{using memory=Mem;}template<class T>using deviceMemory=Mem;
const Mem o_NULL;
struct LA{
 void fill(int n,double v,Mem out){for(int i=0;i<n;++i)out[i]=v;}
 void axpby(int n,double a,Mem x,double b,Mem y){for(int i=0;i<n;++i)y[i]=a*x[i]+b*y[i];}
 void axpbyz(int n,double a,Mem x,double b,Mem y,Mem z){for(int i=0;i<n;++i)z[i]=a*x[i]+b*y[i];}
 void axmyz(int n,double a,Mem x,Mem y,Mem z){for(int i=0;i<n;++i)z[i]=a*x[i]*y[i];}
 void axmyMany(int n,int fields,int O,int,double a,Mem x,Mem y){for(int c=0;c<fields;++c)for(int i=0;i<n;++i)y[i+c*O]*=a*x[i];}
}la;
struct Options{void getArgs(const std::string& key,double& out){out=key.find("01")!=std::string::npos||key.find("02")!=std::string::npos||key.find("03")!=std::string::npos?.1:.2;}};
struct Comm{int mpiComm(){return 0;}int mpiRank(){return 1;}};
struct Platform{LA*linAlg=&la;Options options;Comm comm;}platformObject;auto platform=&platformObject;
namespace ellipticBcType {const int NO_OP=0,DIRICHLET=1,NEUMANN=2;}
struct Mesh{int Nlocal=2,Nelements=1;double volume=1;Mem o_LMM{2},o_invLMM{2},o_sgeo{96},o_vmapM{24},o_EToB{6};
 template<class F>std::vector<int> createEToB(F f){return {f(1),f(2),f(3)};}
}mesh;
struct elliptic{
 elliptic()=default;
 elliptic(const char*,Mesh*,int,std::vector<int> map,Mem,Mem){assert(map[0]==2&&map[1]==1&&map[2]==2);}
 double initialResidual(){return 1;}double finalResidual(){return 0;}int Niter(){return 1;}
 void solve(Mem beta,Mem,Mem rhs,Mem out){out[0]=rhs[0]/(.5*(beta[0]+beta[1]));out[1]=0;}
 void Ax(Mem nu,Mem,Mem u,Mem out){out[0]=nu[0]*(u[0]-u[1]);out[1]=-out[0];}
};
std::array<std::string,7> names={"alpha","ugx","ugy","ugz","ulx","uly","ulz"};
struct Scalar{
 int fieldOffsetSum=14;std::vector<int>fieldOffsetScan{0,2,4,6,8,10,12},compute=std::vector<int>(7,1);
 Mem o_JwF{14},solution{14},nu{14};std::vector<elliptic*>ellipticSolver;
 int index(std::string key){return std::find(names.begin(),names.end(),key)-names.begin();}
 Mem o_solution(std::string key){return solution.slice(2*index(key),2);}
 Mem o_diffusionCoeff(std::string key){return nu.slice(2*index(key),2);}
 void applyDirichlet(double){}
 void solve(double,int);
}scalar;
struct Fluid{Mem o_U{6},pressure{2};Mem o_solution(const char*){return pressure;}}fluid;
struct Nrs{Fluid*fluid=&::fluid;Scalar*scalar=&::scalar;Mesh*meshV=&mesh;int fieldOffset=2,tstep=1;double g0=1.5,dt[1]={.3};}nrsObject;auto nrs=&nrsObject;
struct Params{double rhoGas=1.2,rhoLiquid=998.2;}p;
Mem gradientTensor(18),flux(6),fixedLoad(6),momentumBase(14),pressureGrad(6),extrapG(6),extrapL(6),previous(6),defect(6),lap(2),zero(2);
auto transposeKernel=transposeFlux;auto momentumLoadKernel=momentumLoad;
std::string scalarDigitStr(int i){return i<10?"0"+std::to_string(i):std::to_string(i);}
namespace opSEM{
void strongGrad(Mesh*,int O,Mem q,Mem out){for(int c=0;c<3;++c)for(int i=0;i<2;++i)out[i+c*O]=c==0?q[1]-q[0]:0;}
void strongGradVec(Mesh*,int O,Mem u,Mem grad,bool avg){assert(!avg);
 for(int c=0;c<3;++c)for(int j=0;j<3;++j)for(int i=0;i<2;++i)grad[i+(3*c+j)*O]=j==0?.5*(u[c*O+1]-u[c*O]):0;
}
void divergence(Mesh*,int,Mem v,Mem out){out[0]=-.5*(v[0]+v[1]);out[1]=-out[0];}
}
void assembleVector(Mem&){} // one element, no duplicate nodes
void zeroRows(elliptic*,Mem&){} // no Dirichlet rows in this periodic-mode adapter
void validateLinearSolve(elliptic*,const std::string&){}
double velocityNorm(Mem v){double s=0;for(int i=0;i<6;++i)s+=.5*v[i]*v[i];return std::sqrt(s);}
void Scalar::solve(double,int){
 const double cm=.5*nrs->g0/nrs->dt[0];
 for(int is=0;is<7;++is)if(compute[is]){
  const double k=nu[2*is],d=cm+k,det=d*d-k*k;auto r=o_JwF.slice(2*is,2);auto u=solution.slice(2*is,2);
  u[0]=(d*r[0]+k*r[1])/det;u[1]=(k*r[0]+d*r[1])/det;
 }
}

std::unique_ptr<elliptic> pressureSolver;
Mem beta(2),pressureRhs(2),boundaryFlux(6),crossG(6),crossL(6),viscG(6),viscL(6),lambdaG(2),lambdaL(2),bdfG(6),bdfL(6),bdfMixture(6);
int preparedStep=1,completedStep=-1,alphaStepDone=1,pressureStepDone=-1;
auto pressureFluxKernel=pressureFlux;
void allocateFluidStorage(){}
void reconstructMixture(){auto a=scalar.o_solution("alpha"),g=scalar.solution.slice(2,6),l=scalar.solution.slice(8,6);reconstruct(2,2,a,g,l,fluid.o_U);}
void viscousAcceleration(const Mem&,int,Mem&,Mem& out){for(int i=0;i<6;++i)out[i]=0;}
void surfaceKernel(int,int,Mem,Mem,Mem,Mem u,Mem r){r[0]+=u[0];}
// PRODUCTION_FUNCTIONS
int main(){
 mesh.o_LMM[0]=mesh.o_LMM[1]=.5;
 scalar.solution[0]=.2;scalar.solution[1]=.6; // new alpha
 for(int is=1;is<7;++is)for(int i=0;i<2;++i){
  scalar.solution[2*is+i]=.1*is;momentumBase[2*is+i]=.5*(is+i*.2);
 }
 for(int i=0;i<6;++i){extrapG[i]=.3;extrapL[i]=.1;bdfG[i]=.1*(i+1);bdfL[i]=.2*(i+1);bdfMixture[i]=.7+.05*i;}
 lambdaG[0]=lambdaG[1]=.4;lambdaL[0]=lambdaL[1]=.02;
 const double c=nrs->g0/nrs->dt[0];
 const double umLeft=.2*.1+.8*.4;
 auto F=[&](int i){const double a=scalar.solution[i];return bdfMixture[i]
     +a*(1+.2*i-bdfG[i]-.4*.2)+(1-a)*(4+.2*i-bdfL[i]+.02*.2);};
 const double b0=.2/1.2+.8/998.2,b1=.6/1.2+.4/998.2;
 const double expected=(-.5*(F(0)+F(1))+c*umLeft)/(.5*(b0+b1));
 pressureStep(.3,1);
 assert(std::abs(fluid.pressure[0]-expected)<1e-12);assert(fluid.pressure[1]==0);
 assert(pressureStepDone==1);
 for(int i=0;i<2;++i)assert(std::abs(pressureGrad[i]+expected)<1e-12);
 bool blocked=false;try{pressureStep(.3,1);}catch(...){blocked=true;}assert(blocked);
 pressureStepDone=-1;alphaStepDone=-1;blocked=false;
 try{pressureStep(.3,1);}catch(...){blocked=true;}assert(blocked);
 puts("PASS: actual pressure host assembly/solve matches (99)/(104): historical-mixture BDF, new-alpha explicit terms, variable beta, future prescribed flux and latest gradient");
}
