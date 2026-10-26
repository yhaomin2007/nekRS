"""Actual flux product and direct gas RHS, including shared-node assembly."""
from pathlib import Path
import re
import subprocess
import tempfile

root = Path(__file__).resolve().parents[3]
def serial(path):
    source = (root / path).read_text().replace('@kernel ', '').replace('@ restrict ', '')
    return re.sub(r';\s*@tile\(p_blockSize,\s*@outer,\s*@inner\)', '', source)

cpp = '''#include <cmath>
#include <cassert>
#include <initializer_list>
using dfloat=double; using dlong=long;
#define p_nBDF 2
#define p_nEXT 2
#define p_SUBCYCLING 0
#define p_MovingMesh 0
#define p_ADVECTION 1
''' + serial('src/platform/linAlg/kernels/axmyMany.okl') + serial(
    'src/solver/scalar/kernels/sumMakef.okl') + '''
int main() {
  // Form the shared-alpha three-component flux with the actual linAlg kernel.
  double alpha[5]={.1,.2,.3,99,99};
  double flux[15]={1,2,3,99,99, 4,5,6,99,99, -1,-2,-3,99,99};
  axmyMany(3,3,5,0,1.,alpha,flux);
  assert(std::abs(flux[0]-.1)<1e-14 && std::abs(flux[6]-1.)<1e-14);
  assert(std::abs(flux[12]+.9)<1e-14 && flux[13]==99 && flux[3]==99);

  // Two element copies of one shared node have unequal masses.
  // No native advection or cancellation enters the direct gas RHS.
  double mass[2]={2.,5.},rho[4]={1,1,1,1},state[4]={.2,.2,.1,.1};
  double bdf[2]={2.,-.5},adv[4]={},rhs[4],bf[2];
  double divFlux[2]={.9,-.2},extra[2]={.07,.03};
  for(int h=0;h<2;++h) {
    for(int i=0;i<2;++i) rhs[2*h+i]=-divFlux[h]+extra[h];
  }
  for(int extrapolate: {0,1}) {
    double ext[2]={extrapolate?2.:1.,extrapolate?-1.:0.};
    sumMakef(2,mass,10.,ext,bdf,0,2,2,rho,state,adv,rhs,bf);
    double time=7.*10.*(2.*.2-.5*.1);
    double expected=time+7.*(ext[0]*(-divFlux[0]+extra[0])
                            +ext[1]*(-divFlux[1]+extra[1]));
    assert(std::abs(bf[0]+bf[1]-expected)<1e-12);
  }
}
'''
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory)
    (path/'check.cpp').write_text(cpp)
    subprocess.run(['g++','-std=c++17',str(path/'check.cpp'),'-o',str(path/'check')],check=True)
    subprocess.run([str(path/'check')],check=True)

case = (root/'examples/bubbleColumn/bubbleColumnTerms.hpp').read_text()
evaluation = case.split('inline void evaluatePointwiseTerms()')[1].split(
    'inline void evaluateNativeAlphaAdvection')[0]
assert 'evaluateAlphaTransport();' in evaluation
assert 'evaluateGasCubatureTransport(true);' in case.split('inline void evaluateAlphaTransport()')[1].split('inline void evaluatePointwiseTerms()')[0]
forcing = case.split('inline void addExplicitSources(double)')[1].split(
    'inline void buildDivergenceFromAlphaRhs')[0]
assert 'evaluateNativeAlphaAdvection' not in forcing
assert 'o_alphaGasAdvection' not in case
reconstruction = case.split('inline void buildDivergenceFromAlphaRhs')[1].split(
    'inline void buildDivergenceFromAlphaBdf')[0]
assert 'recoverAlphaRhsKernel' in reconstruction
assert 'core-weakLaplacianHex3D' in reconstruction
assert 'o_alphaBeforePostprocess' in reconstruction
print('Actual nodal flux product and direct gas RHS with EXT1/EXT2 and zero native advection '
      'pass unequal shared-node mass assembly. Direct RHS reconstruction routing checked.')
