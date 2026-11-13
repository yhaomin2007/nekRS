"""Verify removed extra filter and retained clipping/volume diagnostics."""
from pathlib import Path
import subprocess
import tempfile

case=Path(__file__).resolve().parents[1]
source=(case/'bubbleColumn2Terms.hpp').read_text()
body=source.split('inline void postProcessAlpha(double, int)')[1].split('inline void clipVolumeVelocity')[0]
prefix=r'''
#include <cassert>
#include <cmath>
#include <algorithm>
using dfloat=double;using dlong=long;using MPI_Comm=int;
struct Mem {double *a;void copyFrom(Mem b,long n){std::copy(b.a,b.a+n,a);}};
double values[3],before[3],rate[3],mass[3]={2,3,5};
Mem o_alphaBeforePostprocess{before},o_alphaPostprocessRate{rate};
struct Mesh {long Nlocal=3;Mem o_LMM{mass};} mesh;
int order,filterCalls,bcCalls,clipCalls;
struct Scalar {Mem o_solution(const char*){return {values};}
 void applyDirichlet(double){assert(order==2);order=3;++bcCalls;values[1]=.02;}} scalar;
struct Nrs {Mesh *meshV;Scalar *scalar;double g0,dt[1];} nr{&mesh,&scalar,1.5,{.1}},*nrs=&nr;
struct Comm {int mpiComm(){return 0;}};
struct LA {void fill(long n,double x,Mem a){std::fill(a.a,a.a+n,x);}
 double innerProd(long n,Mem a,Mem b,int){double sum=0;for(int i=0;i<n;++i)sum+=a.a[i]*b.a[i];return sum;}
 void axpby(long n,double x,Mem a,double y,Mem b){for(int i=0;i<n;++i)b.a[i]=x*a.a[i]+y*b.a[i];}} la;
struct Platform {Comm comm;LA *linAlg;} obj{{},&la},*platform=&obj;
struct Parameters {int alphaExplicitFilterModes;double alphaExplicitFilterStrength,alphaClippingEnabled,alphaMinimum,alphaMaximum;} p;
double alphaFilterDeltaVolume,cumulativeAlphaFilterDeltaVolume,alphaClipDeltaVolume,cumulativeAlphaClipDeltaVolume;
namespace userExplicitFilter {void explicitFilterApply(const char*,Mesh*,int,double,Mem a){
 assert(order==0);order=1;++filterCalls;a.a[0]=-.03;a.a[1]=.03;a.a[2]=.06;}}
void assembleNodalField(Mem){assert(order==1);order=2;}
void clipAlphaKernel(long n,double lo,double hi,Mem a){
 assert(order==0||order==3);order=4;++clipCalls;for(int i=0;i<n;++i)a.a[i]=std::max(lo,std::min(hi,a.a[i]));}
'''
main=r'''
int main(){for(int clipping=0;clipping<=1;++clipping){
 const double original[3]={-.1,.02,.08};std::copy(original,original+3,values);
 order=filterCalls=bcCalls=clipCalls=0;
 cumulativeAlphaFilterDeltaVolume=cumulativeAlphaClipDeltaVolume=0.;
 p.alphaClippingEnabled=clipping;p.alphaMinimum=0.;p.alphaMaximum=.05;
 postProcessAlpha(.1,1);
 double dvClip=0;
 for(int i=0;i<3;++i){double expected=clipping?std::max(0.,std::min(.05,original[i])):original[i];
 assert(std::abs(values[i]-expected)<1e-14);
 dvClip+=mass[i]*(expected-original[i]);}
 assert(alphaFilterDeltaVolume==0 && cumulativeAlphaFilterDeltaVolume==0);
 assert(std::abs(alphaClipDeltaVolume-dvClip)<1e-14);
 assert(std::abs(cumulativeAlphaClipDeltaVolume-dvClip)<1e-14);
 assert(filterCalls==0 && bcCalls==0 && clipCalls==clipping);
}}
'''
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory);(path/'check.cpp').write_text(prefix+'inline void postProcessAlpha(double, int)'+body+main)
    subprocess.run(['g++','-std=c++11',str(path/'check.cpp'),'-o',str(path/'check')],check=True)
    subprocess.run([str(path/'check')],check=True)
assert 'nrs->postScalar = &bubbleColumn2::postProcessAlpha' in (case/'bubbleColumn2.udf').read_text()
assert 'userExplicitFilter::' not in source
assert 'alphaExplicitFilterStrength' not in source
assert 'alphaExplicitFilterStrength' not in (case/'bubbleColumn2.par').read_text()
print('Actual alpha postScalar: no extra filtering/BC resets; clipping toggle and volume diagnostics passed.')
