#!/usr/bin/env python3
"""Exercise production host callbacks with native-style BDF/EXT history shifts.

The adapter solves uniform scalar modes algebraically. It checks scheduling and
coupling, not the distributed SEM elliptic implementation.
"""
from pathlib import Path
import re, subprocess, tempfile
root=Path(__file__).resolve().parents[1]
udf=(root/'bubbleColumn3.udf').read_text()
def extract(marker):
    start=udf.index(marker);brace=udf.index('{',start);depth=1;i=brace+1
    while depth:
        depth+=(udf[i]=='{')-(udf[i]=='}');i+=1
    return udf[start:i]
equations=(root/'bubbleColumn3Equations.okl').read_text().replace('@kernel','').replace('@restrict','').replace('@atomic','')
equations=re.sub(r';\s*@tile\([^;\n]*?\)\)',')',equations)
equations=re.sub(r';\s*@(?:outer|inner)\(\d+\)', '', equations)
prefix=r'''
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
#define EXIT_FAILURE 1
#define nekrsCheck(cond,comm,code,fmt,...) do{if(cond)throw std::runtime_error(fmt);}while(0)
std::string upperCase(std::string s){for(auto&c:s)c=std::toupper(c);return s;}
#include "equations_serial.hpp"
struct Mem{
 std::shared_ptr<std::vector<double>> data;int offset=0,n=0;
 Mem()=default;explicit Mem(int count):data(std::make_shared<std::vector<double>>(count)),n(count){}
 int size()const{return n;}void resize(int count){*this=Mem(count);}
 Mem slice(int off,int count=-1)const{Mem m=*this;m.offset+=off;m.n=count<0?n-off:count;assert(off+m.n<=n);return m;}
 double&operator[](int i)const{assert(i>=0&&i<n);return (*data)[offset+i];}
 void copyFrom(Mem src,int count=-1,int dest=0,int from=0){if(count<0)count=n;assert(dest+count<=n&&from+count<=src.n);std::vector<double>tmp(count);for(int i=0;i<count;++i)tmp[i]=src[from+i];for(int i=0;i<count;++i)(*this)[dest+i]=tmp[i];}
 operator double*()const{return data->data()+offset;}
};
template<class T>using deviceMemory=Mem;
struct Comm{int mpiComm(){return 0;}int mpiRank(){return 0;}};
struct LA{
 void fill(int n,double a,Mem x){for(int i=0;i<n;++i)x[i]=a;}
 void axpby(int n,double a,Mem x,double b,Mem y){for(int i=0;i<n;++i)y[i]=a*x[i]+b*y[i];}
 double weightedNorm2Many(int n,int fields,int stride,Mem w,Mem x,int){double sum=0;for(int c=0;c<fields;++c)for(int i=0;i<n;++i)sum+=w[i]*x[i+c*stride]*x[i+c*stride];return std::sqrt(sum);}
}la;
struct Options{std::string getArgs(std::string){return "1e-8";}};
struct elliptic{Options cfg;double residual=0;double initialResidual(){return 1;}double finalResidual(){return residual;}Options&options(){return cfg;}int Niter(){return 1;}};
struct Mesh{int Nelements=1,Nlocal=2,fieldOffset=4;Mem o_vgeo{1},o_D{1};double volume=2;Mem o_LMM{4};}mesh;
struct Fluid{Mem o_U{12};std::vector<elliptic*>ellipticSolver;elliptic*ellipticSolverP=nullptr;}fluid;
struct Scalar{
 int fieldOffsetSum=28;std::vector<int>compute=std::vector<int>(7,1);std::vector<int>fieldOffsetScan{0,4,8,12,16,20,24};
 Mem o_S{56},o_JwF{28},o_EXT{56},o_coeffEXT{2},o_coeffBDF{2},rho{28};
 std::vector<elliptic*>ellipticSolver;int explicitCalls=0,historyShifts=0,solutionShifts=0,solveCalls=0,dirichletCalls=0,lastStage=0;
 int index(const std::string&name){static std::vector<std::string>v={"alpha","ugx","ugy","ugz","ulx","uly","ulz"};return std::find(v.begin(),v.end(),name)-v.begin();}
 Mem o_solution(const std::string&name){return o_S.slice(4*index(name),4);}
 Mem o_explicitTerms(const std::string&name){return o_EXT.slice(4*index(name),4);}
 Mem o_transportCoeff(){return rho;}
 void makeExplicit(int,double,int){++explicitCalls;}
 void applyDirichlet(double){++dirichletCalls;for(int is=0;is<7;++is)if(compute[is])assert(ellipticSolver.at(is));}
 void solve(double,int stage);
}scalar;
struct Nrs{Scalar*scalar=::std::addressof(::scalar);Fluid*fluid=::std::addressof(::fluid);Mesh*meshV=&mesh;int fieldOffset=4,tstep=1;double g0=1.5,dt[1]={1e-5};}nrsObject;
auto nrs=&nrsObject;
struct Platform{Comm comm;LA*linAlg=&la;}platformObject;
auto platform=&platformObject;
struct Parameters{double rhoGas=1.2,rhoLiquid=998.2,muLiquid=1e-3,bubbleDiameter=3e-3,dragEnabled=1,alphaMinimum=0,alphaMaximum=.99,gravity[3]={0,0,-9.81},dragCouplingTolerance=1e-12;int dragCouplingMaxIterations=30,virtualMassEnabled=1;double virtualMassCoefficient=.5;}p;
std::array<std::string,7>names={"alpha","ugx","ugy","ugz","ulx","uly","ulz"};
Mem ug(12),ul(12),lambdaG(4),lambdaL(4),transportG(16),transportL(16),zeroAdvection(56),predictorBase(28),phasePrevious(24),phaseDelta(24);
Mem materialBase(28),materialHistory(56);
int preparedStep=-1,completedStep=-1,dragIterations=0;double dragChange=0;
auto reconstructKernel=reconstruct;auto dragKernel=drag;auto sourceKernel=phaseSource;auto clipKernel=clipAlpha;auto dragLoadKernel=addDragLoad;auto virtualMassKernel=virtualMass;
void convection(Mem,Mem result){for(int i=0;i<result.size();++i)result[i]=0;}
void launchKernel(const char*,int n,Mem mass,double idt,Mem ext,Mem bdf,int scan,int sum,int,Mem rho,Mem u,Mem,Mem source,Mem rhs){
 for(int i=0;i<n;++i){double f=0,h=0;for(int k=0;k<ext.size();++k)f+=ext[k]*source[i+scan+k*sum];for(int k=0;k<bdf.size();++k)h+=bdf[k]*u[i+scan+k*sum];rhs[i+scan]=mass[i]*(f+rho[i+scan]*idt*h);}
}
'''
functions='\n'.join(extract(m) for m in [
 'inline void packVelocities()', 'inline void unpackVelocities()', 'inline void reconstructMixture()',
 'inline occa::memory implicitDrag(', 'inline void beginStep(', 'inline void validateLinearSolve(elliptic *solver',
 'inline void postAlpha(', 'inline void solvePhasePredictors('])
# Compile the production pressure-load and final-phase helpers too.
pressure_header=(root/'bubbleColumn3Pressure.hpp').read_text()
for marker in ['inline void addNativePhasePressureLoad(', 'inline void solveNativePressureVelocities(']:
    saved_udf=udf;udf=pressure_header;functions+='\n'+extract(marker);udf=saved_udf
prefix+=r"""
Mem pressureGrad(12);
void launchKernel(const char* name,int elements,Mem,Mem,int offset,const Mem& pressure,Mem grad){
 assert(std::string(name)=="core-wGradientVolumeHex3D"&&elements==1&&offset==4);
 grad.copyFrom(pressure);
}
"""
functions=functions.replace('occa::memory','Mem').replace('return o_NULL;','return Mem();')
suffix=r'''
void Scalar::solve(double,int stage){++solveCalls;lastStage=stage;for(int is=1;is<7;++is)if(compute[is]){Mem l=implicitDrag(0,is);for(int i=0;i<2;++i)o_S[4*is+i]=o_JwF[4*is+i]/(mesh.o_LMM[i]*(nrs->g0/nrs->dt[0]+l[i]));}}
int main(){
 elliptic e;scalar.ellipticSolver.assign(7,&e);for(int i=0;i<2;++i)mesh.o_LMM[i]=1;
 for(int i=0;i<28;++i)scalar.rho[i]=1;
 scalar.o_coeffEXT[0]=2;scalar.o_coeffEXT[1]=-1;scalar.o_coeffBDF[0]=2;scalar.o_coeffBDF[1]=-.5;
 for(int is=0;is<7;++is)for(int i=0;i<2;++i){double value=is==0?.05:((is==3||is==6)?.1:0);scalar.o_S[is*4+i]=value;scalar.o_S[28+is*4+i]=value;scalar.o_EXT[28+is*4+i]=(is==3||is==6)?-9.81:0;}
 beginStep(0);assert(preparedStep==1);assert(scalar.explicitCalls==6);assert(scalar.compute[0]==1);
 for(int is=1;is<7;++is)assert(scalar.compute[is]==0);
 // Native alpha makeForcing then ONE all-field EXT shift and solution lag.
 for(int k=scalar.o_coeffEXT.size()-1;k>0;--k)for(int i=0;i<28;++i)scalar.o_EXT[28*k+i]=scalar.o_EXT[28*(k-1)+i];++scalar.historyShifts;
 for(int k=scalar.o_coeffEXT.size()-1;k>0;--k)for(int i=0;i<28;++i)scalar.o_S[28*k+i]=scalar.o_S[28*(k-1)+i];++scalar.solutionShifts;
 postAlpha(1e-5,1);assert(scalar.compute[0]==1);
 solvePhasePredictors(1e-5);
 const double expected=.1-9.81*nrs->dt[0]/nrs->g0;
 for(int i=0;i<2;++i){assert(std::abs(scalar.o_solution("ugz")[i]-expected)<1e-12);assert(std::abs(scalar.o_solution("ulz")[i]-expected)<1e-12);}
 assert(scalar.historyShifts==1&&scalar.solutionShifts==1&&scalar.explicitCalls==6);
 assert(dragIterations>1&&dragIterations<=30&&scalar.solveCalls==dragIterations);
 assert(dragChange<=p.dragCouplingTolerance);
 // Native-style weak pressure enters BOTH RHSs before frozen-coefficient
 // Helmholtz/drag solves; no histories/explicit sources are regenerated.
 Mem fixedPressure(12),savedBase(28);savedBase.copyFrom(predictorBase);
 for(int i=0;i<2;++i)fixedPressure[8+i]=p.rhoLiquid*9.81;
 const double a=nrs->g0/nrs->dt[0],lg=lambdaG[0],ll=lambdaL[0];
 const double bg=a*expected+fixedPressure[8]/p.rhoGas;
 const double bl=a*expected+fixedPressure[8]/p.rhoLiquid;
 const double determinant=a*(a+lg+ll);
 const double expectedGas=((a+ll)*bg+lg*bl)/determinant;
 const double expectedLiquid=(ll*bg+(a+lg)*bl)/determinant;
 for(int repeat=0;repeat<2;++repeat){
  solveNativePressureVelocities(1e-5,fixedPressure);
  for(int i=0;i<2;++i){
   assert(std::abs(scalar.o_solution("ugz")[i]-expectedGas)<1e-11);
   assert(std::abs(scalar.o_solution("ulz")[i]-expectedLiquid)<1e-11);
  }
  for(int is=1;is<7;++is)for(int i=0;i<2;++i)
   assert(predictorBase[4*is+i]==savedBase[4*is+i]);
  assert(scalar.historyShifts==1&&scalar.solutionShifts==1&&scalar.explicitCalls==6);
  assert(lambdaG[0]==lg&&lambdaL[0]==ll);
 }
 puts("PASS: production native weak-pressure RHS, density scaling, coupled phase response, restored base, no duplicate pressure/history");
 // The homogeneous pressure response excludes the BDF/EXT base and inlet
 // lifting. Its small load must scale linearly, with the same coupled drag.
 const int boundaryCalls=scalar.dirichletCalls;
 for(double scale:{1.0,1e-8}) {
  Mem load(12);
  for(int i=0;i<2;++i)load[8+i]=scale*fixedPressure[8+i];
  solveNativePressureVelocities(1e-5,load,true);
  for(int i=0;i<2;++i) {
   assert(std::abs(scalar.o_solution("ugz")[i]/scale-(expectedGas-expected))<1e-11);
   assert(std::abs(scalar.o_solution("ulz")[i]/scale-(expectedLiquid-expected))<1e-11);
  }
  for(int is=1;is<7;++is)for(int i=0;i<2;++i)
   assert(predictorBase[4*is+i]==savedBase[4*is+i]);
  assert(scalar.dirichletCalls==boundaryCalls&&scalar.lastStage>1);
  assert(scalar.historyShifts==1&&scalar.solutionShifts==1&&scalar.explicitCalls==6);
 }
 puts("PASS: pressure-only homogeneous phase response scales to small directions without history or inlet loads");

 // Enable VM without drag. Its native BDF/EXT source includes material
 // convection differences and must conserve the mixture-volume reaction.
 p.dragEnabled=0;p.virtualMassEnabled=1;
 for(int c=0;c<3;++c)for(int i=0;i<2;++i) {
   predictorBase[4*(1+c)+i]=2.1+c;
   predictorBase[4*(4+c)+i]=-.7-c;
 }
 const Mem originalBase=[&](){Mem v(28);v.copyFrom(predictorBase);return v;}();
 materialBase.copyFrom(originalBase);
 // Distinguish material acceleration from a numerical regularization load.
 for(int c=0;c<3;++c)for(int i=0;i<2;++i) {
   materialBase[4*(1+c)+i]+=100;
   materialBase[4*(4+c)+i]-=40;
 }
 postAlpha(1e-5,1);
 const double vg=p.virtualMassCoefficient*p.rhoLiquid/p.rhoGas;
 const double vl=p.virtualMassCoefficient*.05/.95;
 for(int c=0;c<3;++c)for(int i=0;i<2;++i) {
   const double dg=predictorBase[4*(1+c)+i]-originalBase[4*(1+c)+i];
   const double dl=predictorBase[4*(4+c)+i]-originalBase[4*(4+c)+i];
   assert(std::abs(dg-vg*(materialBase[4*(1+c)+i]-materialBase[4*(4+c)+i]))<1e-10);
   assert(std::abs(.05*p.rhoGas*dg+.95*p.rhoLiquid*dl)<1e-10);
 }
 solveNativePressureVelocities(1e-5,fixedPressure);
 for(int c=0;c<3;++c)for(int i=0;i<2;++i) {
   const double rhsG=predictorBase[4*(1+c)+i]+fixedPressure[4*c+i]/p.rhoGas;
   const double rhsL=predictorBase[4*(4+c)+i]+fixedPressure[4*c+i]/p.rhoLiquid;
   const double determinant=a*a*(1+vg+vl);
   const double gas=((a+a*vl)*rhsG+a*vg*rhsL)/determinant;
   const double liquid=(a*vl*rhsG+(a+a*vg)*rhsL)/determinant;
   assert(std::abs(scalar.o_solution(names[1+c])[i]-gas)<1e-10);
   assert(std::abs(scalar.o_solution(names[4+c])[i]-liquid)<1e-10);
 }
 puts("PASS: symmetric VM native BDF/EXT source and full coupled pressure response");
 // A new timestep is forbidden until the deferred step is marked completed.
 nrs->tstep=2;bool blocked=false;try{beginStep(1e-5);}catch(...){blocked=true;}assert(blocked);
 e.residual=1;blocked=false;try{validateLinearSolve(&e,"test");}catch(...){blocked=true;}assert(blocked);e.residual=0;
 p.dragCouplingMaxIterations=1;for(int i=0;i<2;++i){scalar.o_solution("ugz")[i]=.5;scalar.o_solution("ulz")[i]=0;}
 blocked=false;try{solvePhasePredictors(1e-5);}catch(...){blocked=true;}assert(blocked);
 puts("PASS: production callbacks with native BDF/EXT, one history shift, deferred six-phase solves, coupled common gravity, drag convergence, missing-step and failed-solver guards");
}
'''
for ext_order in (2,3):
    source=prefix+functions+suffix
    if ext_order==3:
        source=source.replace('o_S{56},o_JwF{28},o_EXT{56},o_coeffEXT{2}',
                              'o_S{84},o_JwF{28},o_EXT{84},o_coeffEXT{3}')
        source=source.replace('materialHistory(56)','materialHistory(84)')
        source=source.replace('scalar.o_coeffEXT[0]=2;scalar.o_coeffEXT[1]=-1;',
                              'scalar.o_coeffEXT[0]=3;scalar.o_coeffEXT[1]=-3;scalar.o_coeffEXT[2]=1;')
        source=source.replace('beginStep(0);assert',
                              'for(int i=0;i<28;++i){scalar.o_EXT[56+i]=scalar.o_EXT[28+i];scalar.o_S[56+i]=scalar.o_S[28+i];}beginStep(0);assert')
    with tempfile.TemporaryDirectory() as tmp:
        out=Path(tmp);(out/'equations_serial.hpp').write_text(equations)
        (out/'lifecycle.cpp').write_text(source)
        subprocess.run(['c++','-std=c++17','-I'+tmp,str(out/'lifecycle.cpp'),'-o',str(out/'test')],check=True)
        subprocess.run([str(out/'test')],check=True)
        print(f'PASS: BDF2/EXT{ext_order} callback adapter')
