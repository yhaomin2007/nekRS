"""Compile the actual properties callback: AVM follows resets and runs once per call."""
from pathlib import Path
import subprocess
import tempfile

case = Path(__file__).resolve().parents[1]
terms = (case / 'bubbleColumn2Terms.hpp').read_text()
hook = 'inline void updateProperties(double)' + terms.split('inline void updateProperties(double)')[1].split('inline void enforceZeroDivergence')[0]
cpp = r'''
#include <cassert>
using dlong=long;
struct Mem {
 double *p;
 Mem slice(long off,long){return Mem{p+off};}
 void copyFrom(Mem other){*p=*other.p;}
};
int stage=0, avmCalls=0;
double diffusion[4]={},baseline[4]={1,2,3,4};
bool enabled[4]={};
struct Scalar {
 void applyAVM(){
  assert(stage==3);++avmCalls;
  for(int i=0;i<4;++i)if(enabled[i])diffusion[i]+=.5;
  stage=4;
 }
} scalar;
double props[2]={},mu=7,rho=9;
Mem o_muM{&mu},o_rhoPressure{&rho};
struct Fluid{Mem o_prop{props};} fluid;
struct Nrs{Scalar* scalar=&::scalar;Fluid* fluid=&::fluid;long fieldOffset=1;} app;
auto nrs=&app;
void postProcessGasVelocity(){assert(stage==0);stage=1;}
void prepareGasAdvection(){assert(stage==1);stage=2;}
void evaluatePointwiseTerms(){
 assert(stage==2);
 for(int i=0;i<4;++i)diffusion[i]=baseline[i];
 stage=3;
}
''' + hook + r'''
int main(){
 for(int mask=0;mask<16;++mask){
  for(int i=0;i<4;++i)enabled[i]=(mask>>i)&1;
  for(int repeat=0;repeat<3;++repeat){
   stage=0;int before=avmCalls;updateProperties(.1);
   assert(avmCalls==before+1 && stage==4);
   assert(props[0]==mu && props[1]==rho);
   for(int i=0;i<4;++i)assert(diffusion[i]==baseline[i]+(enabled[i]?.5:0));
  }
 }
}
'''
with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp)
    (p/'check.cpp').write_text(cpp)
    subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror',str(p/'check.cpp'),'-o',str(p/'check')],check=True)
    subprocess.run([str(p/'check')],check=True)
repo=case.parents[1]
core=(repo/'src/app/nrs/nrs.cpp').read_text()
assert 'if (userProperties) {\n    userProperties(timeNew);\n  } else {' in core
udf=(case/'bubbleColumn2.udf').read_text()
assert 'nrs->userProperties = &bubbleColumn2::updateProperties;' in udf
print('Properties callback AVM: actual callback compiles; one invocation after diffusion reset, all scalar enable combinations, repeated updates and fluid properties passed. AVM numerics are not mocked as validated.')
