"""Compile actual properties routing; filter either method without changing raw diagnostics."""
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / 'bubbleColumnTerms.hpp').read_text()
body = source.split('inline void updateProperties(double)')[1].split(
    'inline void printDivergenceComparison')[0]
prefix = '''#include <cassert>
#include <cmath>
#include <algorithm>
using dfloat=double; using dlong=long;
struct Mem {
 double *a;
 Mem slice(long start,long) const { return {a+start}; }
 void copyFrom(Mem b,long n=2,long dst=0,long src=0) const {
   std::copy(b.a+src,b.a+src+n,a+dst);
 }
};
struct LA {
 void axpby(long n,double x,Mem a,double y,Mem b) {
   for(long i=0;i<n;++i)b.a[i]=x*a.a[i]+y*b.a[i];
 }
} la;
struct Platform { LA *linAlg; } platformObject{&la},*platform=&platformObject;
double props[4],mu[2]={4,5},rho[2]={6,7},target[2],comparison[6],scratch[6];
Mem o_divSource{target},o_divComparison{comparison},o_validationMassFlux{scratch};
Mem o_muM{mu},o_rhoM{rho};
struct Fluid { Mem o_prop{props}; } fluid;
struct Mesh { long Nlocal=2; } mesh;
struct Nrs { Mesh *meshV; Fluid *fluid; long fieldOffset=2; } nrsObject{&mesh,&fluid},*nrs=&nrsObject;
struct Parameters { int mixtureDivergenceMethod; double divergenceFilterStrength; } p;
int filterCalls,assemblyCalls;
void postProcessGasFlux() {}
void evaluatePointwiseTerms() {}
void buildDivergenceFromAlphaBdf(Mem out) {out.a[0]=2.;out.a[1]=6.;}
void buildDivergenceFromAlphaRhs(Mem out) {out.a[0]=1.;out.a[1]=3.;}
void assembleDivergence(Mem) {++assemblyCalls;}
void filterDivergence(Mem out) {
 if(!p.divergenceFilterStrength)return;
 ++filterCalls;
 // Emulate modal smoothing and the filter's destructive shared scratch use.
 std::fill(scratch,scratch+6,-999.);
 double mean=(out.a[0]+out.a[1])/2.;
 for(int i=0;i<2;++i)out.a[i]=.75*out.a[i]+.25*mean;
}
'''
main = '''
int main() {
 for(int method=0;method<=1;++method) for(int enabled=0;enabled<=1;++enabled) {
   p.mixtureDivergenceMethod=method;p.divergenceFilterStrength=enabled;
   filterCalls=assemblyCalls=0;
   std::fill(comparison,comparison+6,123.);
   updateProperties(0.);
   assert(comparison[0]==1. && comparison[1]==3.);
   assert(comparison[2]==2. && comparison[3]==6.);
   assert(comparison[4]==-1. && comparison[5]==-3.);
   double factor=method?2.:1.;
   assert(std::abs(target[0]-factor*(enabled?1.25:1.))<1e-14);
   assert(std::abs(target[1]-factor*(enabled?2.75:3.))<1e-14);
   assert(filterCalls==enabled && assemblyCalls==3);
   assert(props[0]==4. && props[1]==5. && props[2]==6. && props[3]==7.);
 }
}
'''
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory)
    (path/'check.cpp').write_text(prefix+'inline void updateProperties(double)'+body+main)
    subprocess.run(['g++','-std=c++17',str(path/'check.cpp'),'-o',str(path/'check')],check=True)
    subprocess.run([str(path/'check')],check=True)
print('Actual updateProperties: both selected methods filtered once, disabled targets '
      'unchanged, post-filter assembly and raw diagnostics preserved despite scratch reuse.')
