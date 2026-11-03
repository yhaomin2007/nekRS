"""Compile actual mixture-force routing; frozen transport must omit feedback."""
from pathlib import Path
import subprocess
import tempfile

case=Path(__file__).resolve().parents[1]
s=(case/'bubbleColumn2Terms.hpp').read_text()
start=s.index('inline void evaluateMixtureForce()')
end=s.index('inline void subtractScalarDiffusion()',start)
helper=s[start:end]
cpp=r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <string>
#include <vector>
using dfloat=double;using dlong=long;
struct Mem{
 std::vector<double> values=std::vector<double>(18,9.);
 Mem slice(long,long)const{return *this;}
};
struct Params{
 int frozenGasVelocityEnabled=1,frozenGasMixtureForcingEnabled=0;
 double driftStressEnabled=1,alphaFloor=1e-6,gravity[3]={0,0,-9.81};
 double rhoLiquid=998.2,rhoGas=1.2,muLiquid=.001,muGas=1.8e-5;
 double mixtureViscousCorrectionEnabled=1;
}p;
struct Mesh{long Nlocal=4;}mesh;
struct Scalar{Mem o_solution(const char*){return Mem{};}}scalar;
struct Nrs{long fieldOffset=6;Mesh* meshV=&mesh;Scalar* scalar=&::scalar;}solver;
auto nrs=&solver;
struct LA{void fill(long n,double value,Mem& m){std::fill_n(m.values.begin(),n,value);}}la;
struct Platform{LA* linAlg=&la;}env;auto platform=&env;
Mem o_mixtureForce,o_mixtureDragSource,o_driftStress,o_divDriftStress;
Mem o_mixtureInterphaseAcceleration,o_rhoPressure,o_gradUl,o_gradUg,o_gradUv;
Mem o_exactMixtureStress,o_baseNativeStress,o_divExactMixtureStress,o_divBaseNativeStress;
int calls=0;
namespace opSEM{template<class... T>void strongDivergence(T&&...){calls++;}}
template<class... T>void buildMixtureForceKernel(T&&...){calls++;}
template<class... T>void buildMixtureStressCorrectionKernel(T&&...){calls++;}
template<class... T>void addMixtureStressCorrectionKernel(T&&...){calls++;}
''' + helper + r'''
int main(){
 for(int frozen:{0,1})for(int forcing:{0,1}){
  p.frozenGasVelocityEnabled=frozen;p.frozenGasMixtureForcingEnabled=forcing;
  calls=0;std::fill(o_mixtureForce.values.begin(),o_mixtureForce.values.end(),9.);
  std::fill(o_mixtureDragSource.values.begin(),o_mixtureDragSource.values.end(),7.);
  evaluateMixtureForce();
  if(frozen&&!forcing){
   assert(calls==0);
   for(double x:o_mixtureForce.values)assert(x==0);
   for(double x:o_mixtureDragSource.values)assert(x==0);
  }else{
   assert(calls>0);
   for(double x:o_mixtureForce.values)assert(x==9.);
   for(double x:o_mixtureDragSource.values)assert(x==7.);
  }
 }
 // Fixed Ki linearization: reconstructed UL makes frozen-UG mixture drag
 // increase with UM. This positive slope is not drag damping of UM.
 double a=.016667,Ki=33.,rhoG=1.2,rhoL=998.2,ug=.3;
 auto force=[&](double um){double ul=(um-a*ug)/(1-a);
  return a*(1-rhoG/rhoL)*(Ki/rhoG)*(ul-ug);};
 double h=1e-6;
 double slope=(force(.005+h)-force(.005-h))/(2*h);
 assert(slope>0);
 assert(std::abs(slope-a*(1-rhoG/rhoL)*Ki/(rhoG*(1-a)))<1e-9);
}
'''
with tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp);(path/'check.cpp').write_text(cpp)
    subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror',str(path/'check.cpp'),'-o',str(path/'check')],check=True)
    subprocess.run([str(path/'check')],check=True)
body=s.split('inline void addExplicitSources(')[1].split('inline occa::memory implicitGasDrag')[0]
assert body.index('evaluateMixtureForce();')<body.index('rampMixtureDrag(')<body.index('freezeDragHistory();')
assert 'fluidTerms.copyFrom(o_mixtureForce' in body
print('Frozen transport: zero user mixture force and drag history input; evolving/coupled modes retained; positive frozen-UG drag feedback confirmed.')
