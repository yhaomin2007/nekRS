"""Check actual native forcing kernel with independent scalar EXT1 coefficients."""
from pathlib import Path
import re
import subprocess
import tempfile
root = Path(__file__).resolve().parents[3]
source = (root / 'src/solver/scalar/kernels/sumMakef.okl').read_text()
source = source.replace('@kernel ', '').replace('@ restrict ', '')
source = re.sub(r';\s*@tile\(p_blockSize,\s*@outer,\s*@inner\)', '', source)
prefix = '''#include <cassert>
#include <cmath>
using dfloat=double; using dlong=long;
#define p_nBDF 2
#define p_nEXT 2
#define p_SUBCYCLING 0
#define p_MovingMesh 0
#define p_ADVECTION 1
'''
main = '''
int main() {
  double mass[1]={2}, rho[4]={1,1,1,1}, bdf[2]={2,-.5};
  double state[8]={.2,.3,.4,.5,.1,.2,.3,.4};
  double adv[8]={.6,.7,.8,.9,2,3,4,5};
  double rhs[8]={-.1,1,2,3,-4,-5,-6,-7}, bf[4]={};
  double scalarExt[2]={1,0}, fluidExt[2]={2,-1};
  for(int f=0;f<4;++f) {
    sumMakef(1,mass,10,scalarExt,bdf,f,4,1,rho,state,adv,rhs,bf);
    double time=20*(2*state[f]-.5*state[f+4]);
    assert(std::abs(bf[f]-(time+2*(rhs[f]-adv[f])))<1e-12);
    sumMakef(1,mass,10,fluidExt,bdf,f,4,1,rho,state,adv,rhs,bf);
    assert(std::abs(bf[f]-(time+2*(2*(rhs[f]-adv[f])-(rhs[f+4]-adv[f+4]))))<1e-12);
  }
  assert(fluidExt[0]==2 && fluidExt[1]==-1);
}
'''
with tempfile.TemporaryDirectory() as t:
    p=Path(t)
    (p/'test.cpp').write_text(prefix+source+main)
    subprocess.run(['g++','-std=c++17',str(p/'test.cpp'),'-o',str(p/'test')],check=True)
    subprocess.run([str(p/'test')],check=True)
case=(root/'examples/bubbleColumn/bubbleColumnTerms.hpp').read_text()
prepare=case.split('inline void prepareGasAdvection()')[1].split('inline void initializeHistory()')[0]
assert 'o_Se' not in prepare
assert 'o_gasAdvector.copyFrom(o_ug' in prepare
assert 'nrs->scalar->o_coeffEXT = o_scalarCoeffEXT;' in case
assert 'o_scalarCoeffEXT.resize(nEXT)' in case
assert 'nrs->o_coeffEXT.copyFrom' not in case
print('Native forcing: EXT1 applies latest convection/RHS to all four scalars; BDF and fluid EXT preserved. Advector is latest UG, not predicted UG.')
