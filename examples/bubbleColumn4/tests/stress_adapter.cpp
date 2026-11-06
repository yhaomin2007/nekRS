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
struct Mesh{int Nlocal=2;double volume=1;Mem o_LMM{2},o_invLMM{2};}mesh;
struct elliptic{
 void Ax(Mem nu,Mem mass,Mem u,Mem out){assert(mass.size()>=2&&mass[0]==0&&mass[1]==0);out[0]=nu[0]*(u[0]-u[1]);out[1]=-out[0];}
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
struct Nrs{Scalar*scalar=&::scalar;Mesh*meshV=&mesh;int fieldOffset=2,tstep=1;double g0=1.5,dt[1]={.3};}nrsObject;auto nrs=&nrsObject;
struct Params{int stressMaxIterations=200;double stressTolerance=1e-11;}p;
Mem gradientTensor(18),flux(6),fixedLoad(6),momentumBase(14),pressureGrad(6),extrapG(6),extrapL(6),previous(6),defect(6),lap(2),zero(2);
auto transposeKernel=transposeFlux;auto momentumLoadKernel=momentumLoad;
std::string scalarDigitStr(int i){return i<10?"0"+std::to_string(i):std::to_string(i);}
namespace opSEM{
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
// PRODUCTION_FUNCTIONS
int main(){
 mesh.o_LMM[0]=mesh.o_LMM[1]=.5;mesh.o_invLMM[0]=mesh.o_invLMM[1]=2;
 std::array<elliptic,7>solvers;for(auto& e:solvers)scalar.ellipticSolver.push_back(&e);
 for(int is=1;is<7;++is){for(int i=0;i<2;++i){scalar.nu[2*is+i]=is<4?.1:.2;
  momentumBase[2*is+i]=.4+is*.2+i*.3;scalar.solution[2*is+i]=.1*is;}}
 for(int k=0;k<6;++k){pressureGrad[k]=.03*(k+1);extrapG[k]=.2;extrapL[k]=.1;}
 Mem gas(6),liquid(6),cross(6),lg(2),ll(2);lg[0]=lg[1]=.4;ll[0]=ll[1]=.02;
 Mem manufactured(6),viscous(6),crossTest(6);
 for(int i=0;i<6;++i)manufactured[i]=.2*i;
 viscousAcceleration(manufactured,1,crossTest,viscous);
 for(int c=0;c<3;++c){double expected=-2*.1*(c==0?2:1)*(manufactured[2*c]-manufactured[2*c+1]);
  assert(std::abs(viscous[2*c]-expected)<1e-12);assert(std::abs(viscous[2*c+1]+expected)<1e-12);}
 auto pressureBefore=*pressureGrad.data;
 auto check=[&](int first,Mem vel,double rho,Mem lambda,double sign){
  const double cm=.5*nrs->g0/nrs->dt[0];
  for(int c=0;c<3;++c){double rhs[2];for(int i=0;i<2;++i){int k=i+c*2;
   rhs[i]=momentumBase[2*(first+c)+i]+.5*(-pressureGrad[k]/rho+sign*lambda[i]*(extrapL[k]-extrapG[k]));}
   const double k=(first==1?.1:.2)*(c==0?2:1),d=cm+k,det=d*d-k*k;
   assert(std::abs(vel[2*c]-(d*rhs[0]+k*rhs[1])/det)<2e-10);
   assert(std::abs(vel[2*c+1]-(k*rhs[0]+d*rhs[1])/det)<2e-10);
  }
 };
 auto liquidBefore=std::vector<double>{scalar.solution[8],scalar.solution[9]};
 solvePhase(0,1,gas,cross,1.2,lg,1);check(1,gas,1.2,lg,1);
 assert(scalar.solution[8]==liquidBefore[0]&&scalar.solution[9]==liquidBefore[1]);
 auto gasBefore=*gas.data;
 solvePhase(0,4,liquid,cross,998.2,ll,-1);check(4,liquid,998.2,ll,-1);
 assert(*gas.data==gasBefore);assert(*pressureGrad.data==pressureBefore);
 p.stressMaxIterations=1;bool failed=false;
 for(int i=0;i<6;++i)gas[i]=0;
 try{solvePhase(0,1,gas,cross,1.2,lg,1);}catch(const std::runtime_error&){failed=true;}assert(failed);
 puts("PASS: actual phase host solver matches direct full-stress block solve; gas then liquid, frozen shared pressure/drag, phase isolation and nonconvergence guard");
}
