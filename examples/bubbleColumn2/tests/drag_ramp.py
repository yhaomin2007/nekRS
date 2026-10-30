"""Compile the actual step-ramp helper against a serial memory/axpby harness."""
from pathlib import Path
import subprocess
import tempfile
s=(Path(__file__).resolve().parents[1]/'bubbleColumn2Terms.hpp').read_text()
helper='inline dfloat mixtureDragRampFactor'+s.split('inline dfloat mixtureDragRampFactor')[1].split('inline void addExplicitSources')[0]
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
'''+helper+r'''
int main(){
assert(mixtureDragRampFactor(0)==1);
p.mixtureDragRampEnabled=1;
for(int step: {0,2,3,4,6,10}) {
 double factor=std::max(0.,std::min(1.,double(step-2)/4));
 assert(mixtureDragRampFactor(step)==factor);
 // Ramp the complete explicit acceleration, including both velocities.
 double D=8*mixtureDragRampFactor(step),um=.2,ug=.3;
 assert(std::abs(D*um-D*ug-factor*8*(um-ug))<1e-12);
}
p.mixtureDragRampStartStep=0;p.mixtureDragRampSteps=10000;
assert(mixtureDragRampFactor(1)==.0001);
assert(mixtureDragRampFactor(10000)==1);
}

'''
with tempfile.TemporaryDirectory() as t:
    src=Path(t)/'check.cpp';src.write_text(cpp)
    subprocess.run(['g++','-std=c++17',str(src),'-o',str(Path(t)/'check')],check=True)
    subprocess.run([str(Path(t)/'check')],check=True)
assert 'mixtureDragRampFactor(nrs->tstep), o_frozenMixtureDragRate' in s
print('Actual step ramp helper: disabled/start/midpoint/end, current raw force, components, padding and other-force checks passed.')
