"""Compile actual target callback and test enable/disable and cosine ramp."""
from pathlib import Path
import subprocess
import tempfile
root=Path(__file__).resolve().parents[1]
source=(root/'bubbleColumnTerms.hpp').read_text()
body='inline void updateDivergence(double)'+source.split('inline void updateDivergence(double)')[1].split('inline void updateDivergenceHistory')[0]
cpp='''#include <cmath>
#include <cassert>
#include <algorithm>
using dfloat=double;using dlong=long;
struct Mem {double *a; void copyFrom(Mem b,long n)const {std::copy(b.a,b.a+n,a);}};
struct LA {
 void fill(long n,double v,Mem m){std::fill(m.a,m.a+n,v);}
 void scale(long n,double v,Mem m){for(long i=0;i<n;++i)m.a[i]*=v;}
 void axpby(long n,double x,Mem a,double y,Mem b){for(long i=0;i<n;++i)b.a[i]=x*a.a[i]+y*b.a[i];}
}la;
struct App {int tstep; } app;
struct Platform {LA *linAlg;App *app;} object{&la,&app},*platform=&object;
double values[2]={2.,-3.},previous[2]={1.,-2.},older[2]={.5,-1.},scratch[2],target[2];
Mem o_divSource{values},o_divPrevious{previous},o_divOlder{older},o_divExtrapolated{scratch};
struct Mesh {long Nlocal=2;}mesh;
struct Fluid {Mem o_div{target};}fluid;
struct Nrs {Mesh *meshV;Fluid *fluid;} nrso{&mesh,&fluid},*nrs=&nrso;
struct Parameters {int mixtureDivergenceEnabled,divergenceExtrapolationEnabled,divergenceRampEnabled,divergenceRampSteps;}p;
int divergenceHistoryCount=0;
'''+body+'''
int main(){
 p.divergenceRampSteps=100;p.divergenceRampEnabled=1;
 for(int enabled=0;enabled<2;++enabled) for(int step: {0,50,100,200}) {
   p.mixtureDivergenceEnabled=enabled;app.tstep=step;
   updateDivergence(0.);
   double ramp=step<100?.5*(1-std::cos(3.14159265358979323846*step/100.)):1.;
   for(int i=0;i<2;++i)assert(std::abs(target[i]-(enabled?ramp*values[i]:0.))<1e-14);
 }
 p.mixtureDivergenceEnabled=1;p.divergenceRampEnabled=0;app.tstep=1;
 updateDivergence(0.);assert(target[0]==values[0] && target[1]==values[1]);
}
'''
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory);(path/'check.cpp').write_text(cpp)
    subprocess.run(['g++','-std=c++17',str(path/'check.cpp'),'-o',str(path/'check')],check=True)
    subprocess.run([str(path/'check')],check=True)
print('Actual divergence callback: disabled zero target, enabled RHS target, cosine ramp endpoints/midpoint passed.')
