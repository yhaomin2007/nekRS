"""Actual alpha transport callback: frozen-UG nodal flux or cubature selection."""
from pathlib import Path
import subprocess
import tempfile

case=Path(__file__).resolve().parents[1]
source=(case/'bubbleColumnTerms.hpp').read_text()
body=source.split('inline void evaluateAlphaTransport()')[1].split('inline void evaluatePointwiseTerms()')[0]
cpp=r'''
#include <cassert>
#include <algorithm>
#include <cmath>
using dfloat=double;using dlong=long;
struct Mem {double *a;Mem slice(long i,long)const{return {a+i};}
void copyFrom(Mem b,long n,long dst=0,long src=0){std::copy(b.a+src,b.a+src+n,a+dst);}};
double alpha[6]={.1,.2,.4,.4,.25,.15},ug[24],flux[24],out[8],cub[8];
Mem o_gasAdvector{ug},o_alphaConservativeFlux{flux},o_alphaSource{out},o_gasTransport{cub};
struct Mesh {long Nlocal=6;} mesh;
struct Scalar {Mem o_solution(const char*){return {alpha};}} scalar;
struct Nrs {Mesh *meshV;Scalar *scalar;long fieldOffset;} nr{&mesh,&scalar,8},*nrs=&nr;
struct LA {void axmy(long n,double s,Mem a,Mem b){for(int i=0;i<n;++i)b.a[i]*=s*a.a[i];}
void scale(long n,double s,Mem a){for(int i=0;i<n;++i)a.a[i]*=s;}
void axpby(long n,double s,Mem a,double t,Mem b){for(int i=0;i<n;++i)b.a[i]=s*a.a[i]+t*b.a[i];}} la;
struct Platform {LA *linAlg;} obj{&la},*platform=&obj;
struct Parameters {int alphaTransportForm;} p;
int nodalCalls,cubCalls;
const double D[9]={-1.5,2,-.5,-.5,0,.5,.5,-2,1.5};
namespace opSEM {void strongDivergence(Mesh*,long offset,Mem q,Mem result,bool avg){
assert(avg&&offset==8);++nodalCalls;
// Two 1D affine elements with unequal masses at the shared CG node.
for(int e=0;e<2;++e)for(int i=0;i<3;++i){double v=0;for(int j=0;j<3;++j)v+=D[3*i+j]*q.a[e*3+j];result.a[e*3+i]=v;}
double shared=(result.a[2]+2.*result.a[3])/3.;result.a[2]=result.a[3]=shared;
}}
void evaluateGasCubatureTransport(bool conservative){assert(conservative);++cubCalls;for(int i=0;i<6;++i)cub[i]=9.+i;}
'''
main=r'''
int main(){
for(int i=0;i<24;++i){ug[i]=99.;flux[i]=123.;}
for(int i=0;i<6;++i){ug[i]=.3+.01*i;ug[8+i]=-.1;ug[16+i]=.2;}
// Advector is single-valued at the shared node.
ug[3]=ug[2];out[6]=out[7]=88.;
p.alphaTransportForm=1;evaluateAlphaTransport();assert(nodalCalls==1&&cubCalls==0);
double expected[6];for(int e=0;e<2;++e)for(int i=0;i<3;++i){expected[e*3+i]=0;for(int j=0;j<3;++j)expected[e*3+i]-=D[3*i+j]*alpha[e*3+j]*ug[e*3+j];}
double shared=(expected[2]+2.*expected[3])/3.;expected[2]=expected[3]=shared;
for(int i=0;i<6;++i){assert(std::abs(out[i]-expected[i])<1e-14);for(int k=0;k<3;++k)assert(flux[k*8+i]==alpha[i]*ug[k*8+i]);}
assert(out[6]==88.&&out[7]==88.&&flux[6]==123.);
p.alphaTransportForm=2;evaluateAlphaTransport();assert(nodalCalls==1&&cubCalls==1);
for(int i=0;i<6;++i)assert(out[i]==-9.-i);assert(out[6]==88.&&out[7]==88.);
}
'''
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory);(path/'test.cpp').write_text(cpp+'inline void evaluateAlphaTransport()'+body+main)
    subprocess.run(['g++','-std=c++11',str(path/'test.cpp'),'-o',str(path/'test')],check=True)
    subprocess.run([str(path/'test')],check=True)
assert '"alphatransportform", p.alphaTransportForm' in (case/'bubbleColumn.udf').read_text()
print('Actual selection: nodal alpha*frozenUG, assembled SEM divergence, negative RHS sign, cubature fallback and padded storage passed.')
