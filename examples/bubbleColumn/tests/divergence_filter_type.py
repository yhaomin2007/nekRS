"""Exercise actual filter callback: off, relaxed and full low-pass selection."""
from pathlib import Path
import subprocess
import tempfile

case = Path(__file__).resolve().parents[1]
source = (case / 'bubbleColumnTerms.hpp').read_text()
body = source.split('inline void filterDivergence(')[1].split('inline void assembleDivergence')[0]
prefix = r'''
#include <cassert>
#include <algorithm>
#include <cmath>
using dfloat=double; using dlong=long;
struct Mem {
 double *a;
 void copyFrom(Mem b,long n,long dst=0,long src=0) {
  std::copy(b.a+src,b.a+src+n,a+dst);
 }
};
template<class T> using deviceMemory=Mem;
struct LA {void fill(long n,double x,Mem m){std::fill(m.a,m.a+n,x);}} la;
struct Platform {LA *linAlg;} obj{&la},*platform=&obj;
struct Mesh {long Nlocal=2,Nelements=1;} mesh;
struct Nrs {Mesh *meshV;long fieldOffset;} nr{&mesh,4},*nrs=&nr;
struct Parameters {double divergenceFilterEnabled,divergenceFilterStrength;int divergenceFilterType;} p;
double scratch[12],matrix[1];
Mem o_validationMassFlux{scratch},o_divFilterMatrix{matrix};
int calls;double usedStrength;
void launchKernel(const char*,long,Mem,double s,long offset,Mem input,Mem output){
 ++calls;usedStrength=s;
 // A constant-preserving low-pass removes the alternating high mode.
 for(int field=0;field<3;++field){
  long o=field*offset;double mean=.5*(input.a[o]+input.a[o+1]);
  double a=input.a[o],b=input.a[o+1];
  output.a[o]+=-s*(a-mean);output.a[o+1]+=-s*(b-mean);
 }
}
'''
main = r'''
int main(){
 for(int enabled=0;enabled<=1;++enabled)
 for(int type=0;type<=1;++type)
 for(double s:{0.,.25,1.}){
  double values[]={1.,3.,99.,99.};Mem q{values};calls=0;
  p.divergenceFilterEnabled=enabled;p.divergenceFilterType=type;p.divergenceFilterStrength=s;
  filterDivergence(q);
  double effective=enabled?(type==1?1.:s):0.;
  assert(std::abs(values[0]-(1.+effective))<1e-14);
  assert(std::abs(values[1]-(3.-effective))<1e-14);
  assert(values[2]==99.&&values[3]==99.);
  assert(calls==enabled);
  if(enabled)assert(usedStrength==(type==1?1.:s));
 }
}
'''
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory)
    (path/'check.cpp').write_text(prefix+'inline void filterDivergence('+body+main)
    subprocess.run(['g++','-std=c++11',str(path/'check.cpp'),'-o',str(path/'check')],check=True)
    subprocess.run([str(path/'check')],check=True)
print('Actual filter callback: disabled unchanged, relaxed strength honored, full explicit ignores strength; padding preserved.')
