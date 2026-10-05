"""Compile the actual step-ramp helper against a serial memory/axpby harness."""
from pathlib import Path
import subprocess
import tempfile
s=(Path(__file__).resolve().parents[1]/'bubbleColumn2Terms.hpp').read_text()
helper='inline dfloat mixtureDragRampFactor'+s.split('inline dfloat mixtureDragRampFactor')[1].split('// All EXT history')[0]
cpp=r'''
#include <vector>
#include <memory>
#include <cassert>
#include <cmath>
using dlong=long; using dfloat=double;
struct Memory {
 std::shared_ptr<std::vector<double>> v; long start=0;
 Memory(long n=0):v(std::make_shared<std::vector<double>>(n)){}
 Memory slice(long i,long){auto m=*this;m.start+=i;return m;}
 double& operator[](long i){return v->at(start+i);}
 void copyFrom(Memory x,long n){for(long i=0;i<n;i++)(*this)[i]=x[i];}
};
struct Algebra{void axpby(long n,double a,Memory x,double b,Memory y){for(long i=0;i<n;i++)y[i]=a*x[i]+b*y[i];}} algebra;
struct Platform{Algebra* linAlg=&algebra;} storage;auto platform=&storage;
struct Mesh{long Nlocal=1;} mesh;
struct Nrs{Mesh* meshV=&mesh;long fieldOffset=2;int tstep=1;} solver;auto nrs=&solver;
struct Params{double mixtureDragRampEnabled=0; int mixtureDragRampStartStep=2, mixtureDragRampSteps=4;}p;
Memory o_mixtureDragSource(6),o_mixtureForce(6);
void set(double raw){for(int i=0;i<3;i++){o_mixtureDragSource[i*2]=raw*(i+1);o_mixtureForce[i*2]=10+raw*(i+1);}}
void check(double used){for(int i=0;i<3;i++){assert(std::abs(o_mixtureDragSource[i*2]-used*(i+1))<1e-12);assert(std::abs(o_mixtureForce[i*2]-(10+used*(i+1)))<1e-12);assert(o_mixtureForce[i*2+1]==0);}}
'''+helper+r'''
int main(){
set(8);rampMixtureDrag(0);check(8);
p.mixtureDragRampEnabled=1;
for(int step: {0,2,3,4,6,10}) {
 set(8);rampMixtureDrag(step);
 check(8*std::max(0.,std::min(1.,double(step-2)/4)));
}
// No force/filter memory: the current raw force determines each evaluation.
set(100);rampMixtureDrag(4);check(50);
set(10);rampMixtureDrag(4);check(5);
p.mixtureDragRampStartStep=0;p.mixtureDragRampSteps=10000;
assert(mixtureDragRampFactor(1)==.0001);
assert(mixtureDragRampFactor(10000)==1);
}

'''
with tempfile.TemporaryDirectory() as t:
    src=Path(t)/'check.cpp';src.write_text(cpp)
    subprocess.run(['g++','-std=c++17',str(src),'-o',str(Path(t)/'check')],check=True)
    subprocess.run([str(Path(t)/'check')],check=True)
print('Actual step ramp helper: disabled/start/midpoint/end, current raw force, components, padding and other-force checks passed.')
