#!/usr/bin/env python3
"""Production source preparation/post-alpha callbacks with BDF2/EXT1,2,3.
The adapter checks history ownership and freezing of new-alpha drag loads.
"""
from pathlib import Path
import re,subprocess,tempfile
root=Path(__file__).resolve().parents[1]
udf=(root/'bubbleColumn4.udf').read_text()
def extract(marker):
 start=udf.index(marker);b=udf.index('{',start);end=b+1;depth=1
 while depth:
  depth+=(udf[end]=='{')-(udf[end]=='}');end+=1
 return udf[start:end]
equations=(root/'bubbleColumn4Equations.okl').read_text().replace('@kernel','').replace('@restrict','').replace('@atomic','')
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
 double innerProd(int n,Mem a,Mem b,int){double r=0;for(int i=0;i<n;++i)r+=a[i]*b[i];return r;}
 std::vector<std::pair<double,double>> minMax(int n,std::vector<Mem> fields,int){std::vector<std::pair<double,double>> r;for(auto f:fields){double lo=f[0],hi=f[0];for(int i=1;i<n;++i){lo=std::min(lo,f[i]);hi=std::max(hi,f[i]);}r.push_back({lo,hi});}return r;}
 void axpby(int n,double a,Mem x,double b,Mem y){for(int i=0;i<n;++i)y[i]=a*x[i]+b*y[i];}
 double weightedNorm2Many(int n,int fields,int stride,Mem w,Mem x,int){double sum=0;for(int c=0;c<fields;++c)for(int i=0;i<n;++i)sum+=w[i]*x[i+c*stride]*x[i+c*stride];return std::sqrt(sum);}
}la;
struct Options{std::string getArgs(std::string){return "1e-8";}};
struct elliptic{Options cfg;double residual=0;double initialResidual(){return 1;}double finalResidual(){return residual;}Options&options(){return cfg;}int Niter(){return 1;}};
struct Mesh{int Nlocal=2,fieldOffset=4;double volume=2;Mem o_LMM{4};}mesh;
struct Fluid{Mem o_U{12};std::vector<elliptic*>ellipticSolver;elliptic*ellipticSolverP=nullptr;}fluid;
struct Scalar{
 int fieldOffsetSum=28;std::vector<int>compute=std::vector<int>(7,1);std::vector<int>fieldOffsetScan{0,4,8,12,16,20,24};
 Mem o_Se{28},o_S{84},o_JwF{28},o_EXT{84},o_coeffEXT{3},o_coeffBDF{2},rho{28};
 std::vector<elliptic*>ellipticSolver;int explicitCalls=0,historyShifts=0,solutionShifts=0,solveCalls=0;
 int index(const std::string&name){static std::vector<std::string>v={"alpha","ugx","ugy","ugz","ulx","uly","ulz"};return std::find(v.begin(),v.end(),name)-v.begin();}
 Mem o_solution(const std::string&name){return o_S.slice(4*index(name),4);}
 Mem o_explicitTerms(const std::string&name){return o_EXT.slice(4*index(name),4);}
 Mem o_transportCoeff(){return rho;}
 void makeExplicit(int,double,int){++explicitCalls;}
 void applyDirichlet(double){for(int is=0;is<7;++is)if(compute[is])assert(ellipticSolver.at(is));}
 void solve(double,int stage);
}scalar;
struct Nrs{Scalar*scalar=::std::addressof(::scalar);Fluid*fluid=::std::addressof(::fluid);Mesh*meshV=&mesh;int fieldOffset=4,tstep=1;double g0=1.5,dt[1]={1e-5};}nrsObject;
auto nrs=&nrsObject;
struct Platform{Comm comm;LA*linAlg=&la;}platformObject;
auto platform=&platformObject;
struct Parameters{double rhoGas=1.2,rhoLiquid=998.2,muLiquid=1e-3,bubbleDiameter=3e-3,dragEnabled=1,alphaMinimum=0,alphaMaximum=.99,gravity[3]={0,0,-9.81},dragCouplingTolerance=1e-12;int alphaClippingEnabled=0,dragCouplingMaxIterations=30;}p;
std::array<std::string,7>names={"alpha","ugx","ugy","ugz","ulx","uly","ulz"};
Mem ug(12),ul(12),lambdaG(4),lambdaL(4),transportG(16),transportL(16),zeroAdvection(56),momentumBase(28),extrapG(12),extrapL(12),bdfG(12),bdfL(12),bdfMixture(12),phasePrevious(24),phaseDelta(24);
int preparedStep=-1,completedStep=-1,alphaStepDone=-1,dragIterations=0;double dragChange=0,alphaClipMass=0;
auto reconstructKernel=reconstruct;auto dragKernel=drag;auto sourceKernel=phaseSource;auto clipKernel=clipAlpha;auto historyKernel=historyBDF;
void convection(Mem,Mem result){for(int i=0;i<result.size();++i)result[i]=0;}
void launchKernel(const char*,int n,Mem mass,double idt,Mem ext,Mem bdf,int scan,int sum,int,Mem rho,Mem u,Mem,Mem source,Mem rhs){
 for(int i=0;i<n;++i){double f=0,h=0;for(int k=0;k<ext.size();++k)f+=ext[k]*source[i+scan+k*sum];for(int k=0;k<bdf.size();++k)h+=bdf[k]*u[i+scan+k*sum];rhs[i+scan]=mass[i]*(f+rho[i+scan]*idt*h);}
}
'''
suffix=r'''
int main(){
 elliptic e;scalar.ellipticSolver.assign(7,&e);for(int i=0;i<2;++i)mesh.o_LMM[i]=1;
 for(int i=0;i<28;++i)scalar.rho[i]=1;
 scalar.o_coeffEXT[0]=COEF0;scalar.o_coeffEXT[1]=COEF1;scalar.o_coeffEXT[2]=COEF2;
 scalar.o_coeffBDF[0]=2;scalar.o_coeffBDF[1]=-.5;
 for(int is=0;is<7;++is)for(int i=0;i<2;++i){
  double value=is==0?.05:.1*is;
  scalar.o_S[is*4+i]=value;scalar.o_S[28+is*4+i]=.7*value;
  scalar.o_Se[is*4+i]=1.3*value;
  for(int k=1;k<3;++k)scalar.o_EXT[k*28+is*4+i]=(is==3||is==6)?-9.81:0;
 }
 beginStep(0);assert(preparedStep==1);assert(scalar.explicitCalls==6);
 assert(scalar.compute[0]==1);for(int is=1;is<7;++is)assert(scalar.compute[is]==0);
 for(int is=1;is<7;++is)for(int i=0;i<2;++i){
  double expected=(2*.1*is-.5*.7*.1*is)/nrs->dt[0]+((is==3||is==6)?-9.81:0);
  assert(std::abs(momentumBase[4*is+i]-expected)<1e-8);
 }
 for(int c=0;c<3;++c)for(int i=0;i<2;++i){
  const double ag=.05,agOld=.035;
  const double ug=.1*(1+c),ul=.1*(4+c),ugOld=.7*ug,ulOld=.7*ul;
  assert(std::abs(bdfG[4*c+i]-(2*ug-.5*ugOld)/nrs->dt[0])<1e-8);
  assert(std::abs(bdfL[4*c+i]-(2*ul-.5*ulOld)/nrs->dt[0])<1e-8);
  const double expected=(2*(ag*ug+(1-ag)*ul)-.5*(agOld*ugOld+(1-agOld)*ulOld))/nrs->dt[0];
  assert(std::abs(bdfMixture[4*c+i]-expected)<1e-8);
 }
 // Emulate native makeForcing's one EXT shift and runInnerStep's one BDF lag.
 for(int k=2;k>0;--k)for(int i=0;i<28;++i)scalar.o_EXT[28*k+i]=scalar.o_EXT[28*(k-1)+i];++scalar.historyShifts;
 for(int i=0;i<28;++i)scalar.o_S[28+i]=scalar.o_S[i];++scalar.solutionShifts;
 // Native alpha solve has completed. New alpha differs from the old .05.
 scalar.o_solution("alpha")[0]=.2;scalar.o_solution("alpha")[1]=.8;
 postAlpha(1e-5,1);assert(alphaStepDone==1);
 assert(alphaClipMass==0);
 assert(scalar.compute[0]==1);for(int is=1;is<7;++is)assert(scalar.compute[is]==0);
 for(int i=0;i<2;++i){double a=scalar.o_solution("alpha")[i];
  assert(std::abs(lambdaL[i]/lambdaG[i]-a*p.rhoGas/((1-a)*p.rhoLiquid))<1e-12);
  assert(std::abs(extrapG[8+i]-.39)<1e-12);assert(std::abs(extrapL[8+i]-.78)<1e-12);
 }
 assert(scalar.historyShifts==1&&scalar.solutionShifts==1&&scalar.explicitCalls==6);
 nrs->tstep=2;bool blocked=false;try{beginStep(1e-5);}catch(...){blocked=true;}assert(blocked);
 e.residual=1;blocked=false;try{validateLinearSolve(&e,"test");}catch(...){blocked=true;}assert(blocked);
 puts("PASS: actual BDF/EXT callbacks, historical-mixture BDF, one history shift, alpha-first scheduling, new-alpha explicit drag freezing, strict no-clipping baseline and lifecycle guards");
}
'''
functions='\n'.join(extract(m) for m in [
 'inline void packVelocities()', 'inline void reconstructMixture()',
 'inline void beginStep(', 'inline void validateLinearSolve(elliptic *solver', 'inline void postAlpha('])
for coefs in [(1,0,0),(2,-1,0),(3,-3,1)]:
 source=prefix+functions+suffix
 for i,v in enumerate(coefs):source=source.replace('COEF'+str(i),str(v))
 with tempfile.TemporaryDirectory() as tmp:
  p=Path(tmp);(p/'equations_serial.hpp').write_text(equations);(p/'test.cpp').write_text(source)
  subprocess.run(['c++','-std=c++17','-I'+tmp,str(p/'test.cpp'),'-o',str(p/'test')],check=True)
  subprocess.run([str(p/'test')],check=True)
 print('PASS: BDF2 / EXT coefficients',coefs)
