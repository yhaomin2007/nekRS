"""Compile actual diagnostic kernels and check stride, geometry and CFL scaling."""
from pathlib import Path
import re
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root/'bubbleColumnDiagnostics.okl').read_text()
source = source.replace('@kernel ', '').replace('@ restrict ', '')
source = re.sub(r'; @tile\(p_blockSize, @outer, @inner\)', '', source)
prefix = '''#include <cmath>
#include <cassert>
#include <algorithm>
#include <initializer_list>
using dfloat=double; using dlong=long;
#define p_Nq 3
#define p_Np 27
#define p_Nvgeo 12
#define p_RXID 0
#define p_RYID 1
#define p_RZID 2
#define p_SXID 3
#define p_SYID 4
#define p_SZID 5
#define p_TXID 6
#define p_TYID 7
#define p_TZID 8
'''
main = '''
int main() {
  double fields[9*8]={},magnitude[8];
  fields[8]=.3; // Pure y velocity must not report zero.
  fields[1]=3.;fields[9]=4.;
  fields[2]=2.;fields[8*8+2]=5.;
  std::fill(magnitude,magnitude+8,123.);
  fieldMagnitude(3,3,8,fields,magnitude);
  assert(std::abs(magnitude[0]-.3)<1e-14 && magnitude[1]==5.);
  assert(magnitude[2]==2. && magnitude[3]==123.);
  fieldMagnitude(3,9,8,fields,magnitude);
  assert(std::abs(magnitude[2]-std::sqrt(29.))<1e-14);

  const int n=54,offset=60;
  double geo[2*p_Np*p_Nvgeo]={},v[3*offset]={},cfl[offset],spacing[3]={2.,1.,3.};
  for(int id=0;id<n;++id) {
    int e=id/p_Np,node=id%p_Np,g=e*p_Np*p_Nvgeo+node;
    geo[g+p_RXID*p_Np]=2.;geo[g+p_SYID*p_Np]=3.;geo[g+p_TZID*p_Np]=4.;
    // Skew second element checks off-diagonal metric terms and element stride.
    if(e) {geo[g+p_RYID*p_Np]=1.;geo[g+p_SXID*p_Np]=-.5;}
    v[id]=1.;v[id+offset]=2.;v[id+2*offset]=3.;
  }
  v[n-1]=v[n-1+offset]=0.;v[n-1+2*offset]=10.;
  for(double dt: {.002,.001}) {
    std::fill(cfl,cfl+offset,123.);
    gasPointwiseCfl(n,dt,offset,geo,spacing,v,cfl);
    assert(std::abs(cfl[0]-dt*(2.*2.+6.*2.+12.*2.))<1e-14);
    assert(std::abs(cfl[27]-dt*(4.*2.+5.5*2.+12.*2.))<1e-14);
    assert(std::abs(*std::max_element(cfl,cfl+n)-dt*120.)<1e-14);
    assert(cfl[n]==123.);
  }
  for(int id=0;id<n;++id) v[id]=v[id+offset]=v[id+2*offset]=0.;
  gasPointwiseCfl(n,.002,offset,geo,spacing,v,cfl);
  assert(*std::max_element(cfl,cfl+n)==0.);
}
'''
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory)
    (path/'check.cpp').write_text(prefix+source+main)
    subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror',
                    str(path/'check.cpp'),'-o',str(path/'check')],check=True)
    subprocess.run([str(path/'check')],check=True)
host=(root/'bubbleColumnTerms.hpp').read_text()
assert 'platform->linAlg->entrywiseMag' not in host
assert 'o_gasCfl.resize(nrs->meshV->Nlocal)' in host
print('Actual monitor kernels: vector/tensor magnitude with padded strides, affine/skew '
      'metrics, two elements, late-node CFL maximum, dt scaling and zero velocity passed.')
