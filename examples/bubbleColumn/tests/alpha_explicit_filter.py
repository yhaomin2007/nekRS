"""Exercise actual postScalar filter/BC/clip ordering and BDF correction."""
from pathlib import Path
import subprocess
import tempfile

case=Path(__file__).resolve().parents[1]
source=(case/'bubbleColumnTerms.hpp').read_text()
body=source.split('inline void postProcessAlpha(double time, int)')[1].split('inline void clipMixtureVelocity')[0]
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
void assembleDivergence(Mem){assert(order==1);order=2;}
void clipAlphaKernel(long n,double lo,double hi,Mem a){
 assert(order==0||order==3);order=4;++clipCalls;for(int i=0;i<n;++i)a.a[i]=std::max(lo,std::min(hi,a.a[i]));}
'''
main=r'''
int main(){for(int filtering=0;filtering<=1;++filtering)for(int clipping=0;clipping<=1;++clipping){
 const double original[3]={-.1,.02,.08};std::copy(original,original+3,values);
 order=filterCalls=bcCalls=clipCalls=0;cumulativeAlphaFilterDeltaVolume=cumulativeAlphaClipDeltaVolume=0.;
 p.alphaExplicitFilterModes=1;p.alphaExplicitFilterStrength=filtering?.05:0.;
 p.alphaClippingEnabled=clipping;p.alphaMinimum=0.;p.alphaMaximum=.05;
 postProcessAlpha(.1,1);
 double afterFilter[3]={filtering?-.03:-.1,.02,filtering?.06:.08};
 double dvFilter=0,dvClip=0;
 for(int i=0;i<3;++i){double expected=clipping?std::max(0.,std::min(.05,afterFilter[i])):afterFilter[i];
 assert(std::abs(values[i]-expected)<1e-14);assert(before[i]==original[i]);
 assert(std::abs(rate[i]-15.*(expected-original[i]))<1e-14);
 dvFilter+=mass[i]*(afterFilter[i]-original[i]);dvClip+=mass[i]*(expected-afterFilter[i]);}
 assert(std::abs(alphaFilterDeltaVolume-dvFilter)<1e-14);
 assert(std::abs(alphaClipDeltaVolume-dvClip)<1e-14);
 assert(std::abs(cumulativeAlphaFilterDeltaVolume-dvFilter)<1e-14);
 assert(std::abs(cumulativeAlphaClipDeltaVolume-dvClip)<1e-14);
 assert(filterCalls==filtering&&bcCalls==filtering&&clipCalls==clipping);
}}
'''
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory);(path/'check.cpp').write_text(prefix+'inline void postProcessAlpha(double time, int)'+body+main)
    subprocess.run(['g++','-std=c++11',str(path/'check.cpp'),'-o',str(path/'check')],check=True)
    subprocess.run([str(path/'check')],check=True)
assert 'nrs->postScalar = &bubbleColumn::postProcessAlpha' in (case/'bubbleColumn.udf').read_text()
assert 'o_alphaBeforePostprocess' in source.split('inline void buildDivergenceFromAlphaRhs')[1].split('inline void buildDivergenceFromAlphaBdf')[0]
print('Actual alpha postScalar: filter/assembly/BC/clip order, independent switches, total BDF correction and separate volume changes passed.')
