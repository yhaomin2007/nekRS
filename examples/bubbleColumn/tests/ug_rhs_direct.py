"""Compile the actual source callback: direct UG convection, no UM cancellation."""
from pathlib import Path
import subprocess,tempfile
case=Path(__file__).resolve().parents[1]
text=(case/'bubbleColumnTerms.hpp').read_text()
body='inline void addExplicitSources(double)'+text.split('inline void addExplicitSources(double)',1)[1].split('inline void buildDivergenceFromAlphaRhs',1)[0]
pre=r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <string>
#include <vector>
using dfloat=double;using dlong=long;
struct Mem {double *a;
 Mem slice(long off,long){return {a+off};}
 void copyFrom(Mem b,long n,long dst=0,long src=0){std::copy(b.a+src,b.a+src+n,a+dst);}
};
struct LA {void axpby(long n,double a,Mem x,double b,Mem y){for(long i=0;i<n;++i)y.a[i]=a*x.a[i]+b*y.a[i];}}la;
struct Platform {LA *linAlg=&la;}object,*platform=&object;
struct Mesh {long Nlocal=3;}mesh;
double fields[20],velocity[15],fluidRhs[15];
struct Scalar {Mem o_explicitTerms(std::string s){int f=s=="alpha"?0:s=="ugx"?1:s=="ugy"?2:3;return {fields+f*5};}}scalar;
struct Fluid {Mem o_U{velocity};Mem o_explicitTerms(){return {fluidRhs};}}fluid;
struct Nrs {Mesh *meshV=&mesh;Scalar *scalar;Fluid *fluid;long fieldOffset=5;} nr{&mesh,&scalar,&fluid},*nrs=&nr;
double ugSource[15],transport[20],alphaSource[5],mixtureForce[15],snapshot[15];
Mem o_ugSource{ugSource},o_gasTransport{transport},o_alphaSource{alphaSource},o_mixtureForce{mixtureForce},o_mixtureAdvectorUsed{snapshot};
void prepareGasAdvection(){}
void evaluatePointwiseTerms(){
 for(int c=0;c<3;++c)for(int n=0;n<3;++n){ugSource[c*5+n]=10*(c+1)+n;mixtureForce[c*5+n]=.5*c+n;}
 for(int n=0;n<3;++n)alphaSource[n]=-.1*(n+1);
}
void subtractScalarDiffusion(){}void storeAlphaTransportRhs(){}void evaluateMixtureForce(){}
void evaluateGasCubatureTransport(bool conservative){assert(!conservative);
 for(int c=0;c<3;++c)for(int n=0;n<3;++n)transport[(c+1)*5+n]=(c+1)+.25*n;
}
'''
main=r'''
int main(){
 std::fill(fields,fields+20,123.);std::fill(fluidRhs,fluidRhs+15,123.);
 for(double um:{-.7,0.,1.3}){
 std::fill(velocity,velocity+15,um);addExplicitSources(0.);
 for(int n=0;n<3;++n)assert(std::abs(fields[n]+.1*(n+1))<1e-14);
 for(int c=0;c<3;++c){for(int n=0;n<3;++n){
 assert(fields[(c+1)*5+n]==10*(c+1)+n-((c+1)+.25*n));
 assert(fluidRhs[c*5+n]==.5*c+n);assert(snapshot[c*5+n]==um);}
 assert(fields[(c+1)*5+3]==123.&&fluidRhs[c*5+3]==123.);
 }
 }
}
'''
with tempfile.TemporaryDirectory() as d:
 p=Path(d);(p/'check.cpp').write_text(pre+body+main)
 subprocess.run(['g++','-std=c++17',str(p/'check.cpp'),'-o',str(p/'check')],check=True)
 subprocess.run([str(p/'check')],check=True)
print('Actual source callback: all three UG RHS components are physical source minus gas convection; no UM cancellation, correct sign/padding, alpha and fluid forcing preserved.')
