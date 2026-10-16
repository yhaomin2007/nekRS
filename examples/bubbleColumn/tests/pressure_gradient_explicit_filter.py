"""Actual UG gradient spatial-filter routing, raw preservation and repeat calls."""
from pathlib import Path
import subprocess
import tempfile

case=Path(__file__).resolve().parents[1]
source=(case/'bubbleColumnTerms.hpp').read_text()
body=source.split('inline void filterGasPressureGradient()')[1].split('inline void evaluatePointwiseTerms()')[0]
prefix=r'''
#include <cassert>
#include <cmath>
#include <algorithm>
using dfloat=double;using dlong=long;
struct Mem {double *a;Mem slice(long i,long)const{return {a+i};}
void copyFrom(Mem b,long n){std::copy(b.a,b.a+n,a);}};
double strong[12]={1,3,91,92,-2,4,93,94,5,1,95,96},weak[12],filtered[12];
Mem o_gradPStrong{strong},o_gradPWeak{weak},o_gradPSpatial{filtered};
struct Mesh {} mesh;struct Nrs {Mesh *meshV;long fieldOffset;} nr{&mesh,4},*nrs=&nr;
struct Parameters {int gasPressureGradientMethod,gasPressureGradientExplicitFilterModes;double gasPressureGradientExplicitFilterStrength;} p;
int calls,assemblies;
namespace userExplicitFilter {
void explicitFilterApply(const char*,Mesh*,int modes,double strength,Mem field){
 assert(modes==2);++calls;double a=field.a[0],b=field.a[1];
 field.a[0]=(1-strength)*a+strength*.5*(a+b);
 field.a[1]=(1-strength)*b+strength*.5*(a+b);
}}
void assembleDivergence(Mem){++assemblies;}
'''
main=r'''
int main(){for(int i=0;i<12;++i)weak[i]=2.*strong[i];
for(int method=0;method<2;++method)for(double s:{0.,.05,1.}){
 p.gasPressureGradientMethod=method;p.gasPressureGradientExplicitFilterModes=2;
 p.gasPressureGradientExplicitFilterStrength=s;
 for(int repeat=0;repeat<2;++repeat){calls=assemblies=0;filterGasPressureGradient();
  assert(calls==(s>0?3:0)&&assemblies==calls);
  for(int c=0;c<3;++c){double factor=method?2.:1.;double a=factor*strong[4*c],b=factor*strong[4*c+1];
   assert(std::abs(filtered[4*c]-((1-s)*a+s*.5*(a+b)))<1e-14);
   assert(std::abs(filtered[4*c+1]-((1-s)*b+s*.5*(a+b)))<1e-14);
   assert(filtered[4*c+2]==factor*strong[4*c+2]&&filtered[4*c+3]==factor*strong[4*c+3]);
  }
 }
}
assert(strong[0]==1.&&strong[5]==4.&&weak[0]==2.&&weak[5]==8.);
}
'''
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory);(path/'test.cpp').write_text(prefix+'inline void filterGasPressureGradient()'+body+main)
    subprocess.run(['g++','-std=c++11',str(path/'test.cpp'),'-o',str(path/'test')],check=True)
    subprocess.run([str(path/'test')],check=True)
evaluation=source.split('inline void evaluatePointwiseTerms()')[1].split('inline void evaluateNativeAlphaAdvection')[0]
assert evaluation.index('filterGasPressureGradient();') < evaluation.index('if (p.gasPressureGradientFilterEnabled') < evaluation.index('if (p.gasPressureInletMaskEnabled)') < evaluation.index('buildEquationTermsKernel')
assert 'const auto &o_selectedGradP = o_gradPSpatial;' in evaluation
print('Actual pressure-gradient routing: all components, strong/weak selection, zero/full strength, raw preservation, repeated calls and spatial-before-temporal/mask ordering passed.')
