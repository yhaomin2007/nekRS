#!/usr/bin/env python3
"""Execute the production bounded outer loop on a manufactured coupled response.

The native Poisson inverse and phase response use two DOFs. This checks pressure
accumulation, true residual acceptance, budget exhaustion and strict failure.
"""
from pathlib import Path
import re
import subprocess
import tempfile
root = Path(__file__).resolve().parents[1]
header = (root/'bubbleColumn3Pressure.hpp').read_text()
start = header.index('inline void solveOuterPhasePressure(')
end = header.index('\ninline void solvePressureAndCorrect(', start)
function = header[start:end]
source = r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
using dfloat=double;
#define EXIT_FAILURE 1
#define MPI_Wtime() 0.0
#define nekrsCheck(c,comm,code,...) do {if(c) throw std::runtime_error("check");} while(0)
struct Mem {
 std::shared_ptr<std::vector<double>> data;
 int offset=0,count=0;
 Mem()=default;
 explicit Mem(int n):data(std::make_shared<std::vector<double>>(n)),count(n){}
 void resize(int n){if(n>count)*this=Mem(n);}
 double& operator[](int i)const{return (*data)[offset+i];}
 Mem slice(int i,int n)const{Mem v=*this;v.offset+=i;v.count=n;return v;}
 void copyFrom(Mem m,int n){for(int i=0;i<n;++i)(*this)[i]=m[i];}
};
template<class T>using deviceMemory=Mem;
struct Comm {int mpiComm(){return 0;}int mpiRank(){return 0;}};
struct LA {
 void fill(int n,double a,Mem m){for(int i=0;i<n;++i)m[i]=a;}
 void axpby(int n,double a,Mem x,double b,Mem y){for(int i=0;i<n;++i)y[i]=a*x[i]+b*y[i];}
 template<class T>void mask(int,Mem,Mem){}
 double weightedNorm2(int n,Mem,Mem m,int){double d=0;for(int i=0;i<n;++i)d+=m[i]*m[i];return std::sqrt(d);}
 double max(int n,Mem m,int){double v=m[0];for(int i=1;i<n;++i)v=std::max(v,m[i]);return v;}
}la;
struct Platform {LA*linAlg=&la;Comm comm;}platformObject;
auto platform=&platformObject;
struct Ogs {Mem o_invDegree{2};}ogs;
struct Mesh {int Nlocal=2;Ogs*ogs=&::ogs;int oogs=0;Mem o_LMM{2};}mesh;
namespace oogs {void startFinish(Mem,int,int,int,int,int){}}
constexpr int ogsDfloat=0,ogsAdd=0;
Mem pressure(2),pressureRhs(2),pressureDiffusion(2),o_NULL;
Mem lambdaG(2),lambdaL(2),ug(6),ul(6),predictorBase(14),materialBase(14);
struct Fluid {Mem o_solution(const char*){return pressure;}}fluid;
struct Scalar {
 std::vector<int>fieldOffsetScan{0,2,4,6,8,10,12};int boundaryCalls=0;
 Mem o_solution(const char*){return Mem(2);}
 void applyDirichlet(double){++boundaryCalls;}
}scalar;
struct Nrs {Mesh*meshV=&mesh;int fieldOffset=2;Fluid*fluid=&::fluid;Scalar*scalar=&::scalar;}nrsObject;
auto nrs=&nrsObject;
struct Parameters {
 double pressureSchurAbsoluteTolerance=1e-12,pressureSchurTolerance=1e-8;
 int pressureOuterCorrections=3,pressureOuterRequireConvergence=0,virtualMassEnabled=0;
 double pressureOuterRelaxation=1,dragCouplingTolerance=1e-12;
 double virtualMassCoefficient=.5,rhoGas=1.2,rhoLiquid=998.2;
}p;
int phaseCalls=0,dragIterations=1;double dragChange=0;
double response[2]={1,1.5};
struct Solver {
 int calls=0;
 void solve(Mem,Mem,Mem rhs,Mem x){++calls;for(int i=0;i<2;++i)x[i]=rhs[i]/(2+i);}
 int Niter(){return 1;}int Nmasked(){return 0;}Mem o_maskIds(){return Mem();}
 double initialResidual(){return 1;}double finalResidual(){return 0;}
};
auto pressureSolver=std::make_unique<Solver>();
auto& solver=*pressureSolver;
void validateLinearSolve(Solver*,const char*){}
void solveNativePressureVelocities(double,Mem,bool,double){++phaseCalls;}
void reconstructMixture(){}
double divergenceNorm(){return 0;}double phaseCFL(Mem){return 0;}
void formPressureRhs(double){for(int i=0;i<2;++i)pressureRhs[i]=(i+1)-response[i]*pressure[i];}
double phaseMomentumResidual(double,bool){return 0;}
void couplingBalanceKernel(int n,int,double,double,double,Mem,Mem,Mem,Mem,Mem,Mem,Mem,Mem,Mem balance){la.fill(n,0,balance);}
'''
source += function
source += r'''
void reset(){solver.calls=phaseCalls=scalar.boundaryCalls=0;la.fill(2,0,pressure);formPressureRhs(1);}
int main(){
 reset();solveOuterPhasePressure(0,1,true,1);
 assert(solver.calls==3&&phaseCalls==3&&scalar.boundaryCalls==3);
 for(int i=0;i<2;++i)assert(std::abs(pressureRhs[i]-(i+1)/8.0)<1e-14);
 reset();p.pressureOuterRequireConvergence=1;
 bool rejected=false;try{solveOuterPhasePressure(0,1,false,1);}catch(...){rejected=true;}
 assert(rejected&&solver.calls==3&&phaseCalls==3);
 response[0]=2;response[1]=3;reset();
 solveOuterPhasePressure(0,1,false,1);
 assert(solver.calls==1&&phaseCalls==1&&la.weightedNorm2(2,Mem(),pressureRhs,0)<1e-12);
 solver.calls=phaseCalls=0;solveOuterPhasePressure(0,2,false,1);
 assert(solver.calls==0&&phaseCalls==0);
 puts("PASS: production outer loop pressure accumulation, true continuity acceptance, fixed budget and strict failure");
}
'''
with tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp)
    (path/'outer.cpp').write_text(source)
    subprocess.run(['c++','-std=c++17',str(path/'outer.cpp'),'-o',str(path/'outer')],check=True)
    subprocess.run([str(path/'outer')],check=True)
