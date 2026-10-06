"""Check actual surface-node mask construction and native vector multiplication."""
from pathlib import Path
import re, subprocess, tempfile
root=Path(__file__).resolve().parents[1]
repo=root.parents[1]
terms=(root/'bubbleColumn2Terms.hpp').read_text()
block=terms[terms.index('  auto mesh = nrs->meshV;',terms.index('// Suppress QG pressure')):]
block=block[:block.index('  o_rhoM.resize')]
kernel=(repo/'src/platform/linAlg/kernels/axmyVector.okl').read_text()
kernel=kernel.replace('@kernel ', '').replace('@ restrict ', '')
kernel=re.sub(r'; @tile\(p_blockSize, @outer, @inner\)', '',kernel)
cpp=r'''
#include <vector>
#include <cassert>
using dlong=long;using dfloat=double;
#define p_NVec 3
struct Memory {std::vector<double> v;void copyFrom(std::vector<double> x){v=x;}} o_qgPressureWeight;
struct Mesh {long Nelements=2,Nlocal=10;int Nfaces=3,Nfp=2;
 int EToB[6]={1,0,3,0,2,3};
 long vmapM[12]={0,1,2,3,4,5,6,7,8,9,4,5};int oogs=0;};
struct Nrs {Mesh* meshV;};Nrs *nrs;
int ogsDfloat=0,ogsMin=1;
namespace oogs {void startFinish(Memory &m,int,int,int,int op,int){
 assert(op==ogsMin);
 // Nodes 0/6 and 8/2 represent shared CG copies across elements.
 for(auto pair:std::vector<std::vector<int>>{{0,6},{8,2}}){
  double value=std::min(m.v[pair[0]],m.v[pair[1]]);
  m.v[pair[0]]=m.v[pair[1]]=value;
 }
}}
void makeMask(){long offset=12;
'''+block+r'''
}
'''+kernel+r'''
int main(){Mesh mesh;Nrs app{&mesh};nrs=&app;makeMask();
 double raw[36],masked[36];for(int i=0;i<36;i++)raw[i]=masked[i]=i+1;
 axmyVector(10,12,0,1.,o_qgPressureWeight.v.data(),masked);
 for(int c=0;c<3;c++)for(int n=0;n<12;n++){
  bool blocked=n==0||n==1||n==2||n==6||n==8||n==9;
  assert(masked[c*12+n]==(blocked?0:raw[c*12+n]));
  assert(raw[c*12+n]==c*12+n+1);
 }
}
'''
with tempfile.TemporaryDirectory() as t:
 f=Path(t)/'mask.cpp';f.write_text(cpp)
 subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-Wno-unknown-pragmas',str(f),'-o',str(Path(t)/'mask')],check=True)
 subprocess.run([str(Path(t)/'mask')],check=True)
assert 'o_virtualMassRelativeAcceleration,\n                           o_gradPForQG,' in terms
assert 'o_gradP,\n' not in terms[terms.index('  buildEquationTermsKernel('):terms.index('  buildEquationTermsKernel(')+2500]
for path in ['src/solver/fluid/fluidSolver.hpp','src/solver/scalar/scalarSolver.hpp','src/solver/scalar/scalarSolver.cpp']:
 text=(repo/path).read_text()
 for removed in ['postPressure','solveDeferred','deferSolve','userRhs']:
  assert removed not in text
print('Actual boundary mask and native vector kernel: inlet/outlet, shared copies, wall/interior, components, padding and original core API passed.')
