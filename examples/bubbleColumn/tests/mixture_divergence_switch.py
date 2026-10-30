"""Compile actual target callback and test enable/disable and cosine ramp."""
from pathlib import Path
import re
import subprocess
import tempfile
root=Path(__file__).resolve().parents[1]
source=(root/'bubbleColumnTerms.hpp').read_text()
body='inline void updateDivergence(double)'+source.split('inline void updateDivergence(double)')[1].split('inline void updateDivergenceHistory')[0]
kernel=(root/'bubbleColumnEquations.okl').read_text().split('@kernel void maskMixtureDivergenceInlet',1)[1].split('@kernel void buildRelativeMixtureDivergence',1)[0]
kernel='void maskMixtureDivergenceInletKernel'+re.sub(r';\s*@tile\(p_blockSize,\s*@outer,\s*@inner\)','',kernel.replace('@ restrict ',''))
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
double z[2]={0.0,0.01};
double values[2]={2.,-3.},previous[2]={1.,-2.},older[2]={.5,-1.},scratch[2],target[2];
Mem o_divSource{values},o_divPrevious{previous},o_divOlder{older},o_divExtrapolated{scratch};
struct Mesh {long Nlocal=2;Mem o_z{z};}mesh;
struct Fluid {Mem o_div{target};}fluid;
struct Nrs {Mesh *meshV;Fluid *fluid;} nrso{&mesh,&fluid},*nrs=&nrso;
struct Parameters {int mixtureDivergenceEnabled,divergenceExtrapolationEnabled,divergenceRampEnabled,divergenceRampSteps; double divergenceInletThickness=0.;}p;
int divergenceHistoryCount=0;
''' + kernel.replace('const dfloat *z','Mem z').replace('dfloat *divergence','Mem divergence').replace('z[n]','z.a[n]').replace('divergence[n]','divergence.a[n]') + body+'''
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
 // Test final spatial weighting across EXT startup/history and temporal ramp paths.
 p.divergenceInletThickness=.01;p.divergenceRampEnabled=1;app.tstep=50;
 for(int ext=0;ext<2;++ext)for(int history=0;history<3;++history){
   p.divergenceExtrapolationEnabled=ext;divergenceHistoryCount=history;
   updateDivergence(0.);
   assert(target[0]==0.); // z=0, inside layer
   double expected=(!ext||history==0)?values[1]:history==1?previous[1]:2*previous[1]-older[1];
   assert(std::abs(target[1]-.5*expected)<1e-14); // z=delta is NOT masked
 }
 // Zero plateau, ramp midpoint, duplicate shared nodes, full recovery and padding.
 double heights[8]={0.,.004,.005,.0075,.0075,.01,.02,-1.};
 double field[8]={1,2,3,4,4,5,6,99};
 maskMixtureDivergenceInletKernel(7,.01,Mem{heights},Mem{field});
 assert(field[0]==0&&field[1]==0&&field[2]==0);
 assert(std::abs(field[3]-2.)<1e-14&&field[3]==field[4]);
 assert(field[5]==5&&field[6]==6&&field[7]==99);
 field[0]=7;maskMixtureDivergenceInletKernel(7,0.,Mem{heights},Mem{field});assert(field[0]==7);
 // Dense sweep checks the intended shape and endpoint smoothness.
 double last=-1.;
 for(int j=0;j<=1000;++j){
   double h=.01*j/1000.,v=-3.;
   maskMixtureDivergenceInletKernel(1,.01,Mem{&h},Mem{&v});
   double weight=v/(-3.);
   double expected=j<=500?0.:.5*(1-std::cos(3.14159265358979323846*(2*h/.01-1)));
   assert(std::abs(weight-expected)<1e-14&&weight>=last&&weight>=0&&weight<=1);
   last=weight;
 }
 // Callback composes temporal ramp (0.5) and spatial ramp midpoint (0.5).
 z[1]=.0075;p.divergenceExtrapolationEnabled=0;updateDivergence(0.);
 assert(std::abs(target[1]-.25*values[1])<1e-14);
 assert(values[0]==2&&previous[0]==1&&older[0]==.5); // raw/history unaffected
}
'''
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory);(path/'check.cpp').write_text(cpp)
    subprocess.run(['g++','-std=c++17',str(path/'check.cpp'),'-o',str(path/'check')],check=True)
    subprocess.run([str(path/'check')],check=True)
print('Actual divergence callback: disabled zero target, enabled RHS target, cosine ramp, zero plateau and spatial cosine ramp after EXT/temporal ramp, midpoint, monotonicity, shared nodes and padding passed.')
