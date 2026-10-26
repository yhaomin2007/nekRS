"""Actual relative-flux preparation and prefactor kernel, both spatial routes."""
from pathlib import Path
import re,subprocess,tempfile
case=Path(__file__).resolve().parents[1]
s=(case/'bubbleColumnTerms.hpp').read_text()
helper='inline void prepareRelativeDivergenceRhs()'+s.split('inline void prepareRelativeDivergenceRhs()',1)[1].split('inline void storeAlphaTransportRhs()',1)[0]
k=(case/'bubbleColumnEquations.okl').read_text().split('@kernel void buildRelativeMixtureDivergence',1)[1]
k='void buildRelativeMixtureDivergence'+re.sub(r';\s*@tile\(p_blockSize,\s*@outer,\s*@inner\)','',k.replace('@ restrict ',''))
cpp=r'''
#include <cassert>
#include <cmath>
#include <algorithm>
using dfloat=double;using dlong=long;
struct Mem{double *a;operator double*()const{return a;}Mem slice(long o,long){return {a+o};}
void copyFrom(Mem b,long n,long dst=0,long src=0){std::copy(b.a+src,b.a+src+n,a+dst);}};
struct Mesh{long Nlocal=3,Nelements=1,cubNp=3;Mem o_cubInterpT{nullptr},o_cubD{nullptr},o_cubvgeo{nullptr},o_vgeo{nullptr};}mesh;
double alpha[5]={.1,.2,.3,99,99},um[15],ug[15],flux[15],rhs[5],gas[5]={-.03,-.03,-.03,99,99},saved[5];
double fields[20],cub[20],products[20],transport[20];
Mem o_mixtureAdvectorUsed{um},o_gasAdvector{ug},o_alphaRelativeFlux{flux},o_alphaRelativeRhs{rhs},o_alphaSource{gas},o_alphaTransportOnly{saved};
Mem o_gasTransportFields{fields},o_gasCubatureFields{cub},o_gasCubatureProducts{products},o_gasTransport{transport};
struct Scalar{Mem o_solution(const char*){return {alpha};}}scalar;
struct Nrs{Mesh *meshV=&mesh;Scalar *scalar;long fieldOffset=5;}nr{&mesh,&scalar},*nrs=&nr;
struct Parameters{int mixtureDivergenceForm=2,alphaTransportForm=1;}p;
struct LA{void axpby(long n,double a,Mem x,double b,Mem y){for(int i=0;i<n;++i)y.a[i]=a*x.a[i]+b*y.a[i];}
void axmy(long n,double a,Mem x,Mem y){for(int i=0;i<n;++i)y.a[i]*=a*x.a[i];}}la;
struct Platform{LA *linAlg=&la;}object,*platform=&object;
void deriv(const double *f,double *r){r[0]=-1.5*f[0]+2*f[1]-.5*f[2];r[1]=-.5*f[0]+.5*f[2];r[2]=.5*f[0]-2*f[1]+1.5*f[2];}
int semCalls=0,cubCalls=0,assemblies=0;
namespace opSEM{void strongDivergence(Mesh*,long,Mem f,Mem r,bool avg){assert(avg);++semCalls;deriv(f.a,r.a);}}
void assembleDivergence(Mem){++assemblies;}
// Controlled cubature stages check interpolation inputs and signed product;
// true conservative cubature kernels have their own polynomial regression.
void gasInterpolateKernel(long,long offset,long,Mem,Mem in,Mem out){assert(offset==5);out.copyFrom(in,20);}
void alphaConservativeCubatureKernel(long,long,Mem,Mem,Mem in,Mem out){
++cubCalls;double f[3];for(int i=0;i<3;++i){assert(std::abs(in.a[5+i]+.2)<1e-14);f[i]=in.a[i]*in.a[5+i];}deriv(f,out.a);}
void gasProjectKernel(long,long,long,Mem,Mem,Mem in,Mem out){out.copyFrom(in,3);}
'''+helper+k+r'''
int main(){
for(int form:{1,2}){
p.alphaTransportForm=form;std::fill(um,um+15,.1);std::fill(ug,ug+15,.3);std::fill(rhs,rhs+5,123.);
prepareRelativeDivergenceRhs();
for(int i=0;i<3;++i){assert(std::abs(rhs[i]+.02)<1e-14);assert(saved[i]==gas[i]);}
assert(rhs[3]==123.);double diffusion[5]={.01,.01,.01},out[5]={99,99,99,99,99};
buildRelativeMixtureDivergence(3,10,2,rhs,diffusion,out);
for(int i=0;i<3;++i)assert(std::abs(out[i]+.008)<1e-14);assert(out[3]==99.);
buildRelativeMixtureDivergence(3,10,10,rhs,diffusion,out);for(int i=0;i<3;++i)assert(out[i]==0.);
}
assert(semCalls==1&&cubCalls==1&&assemblies==1);
p.alphaTransportForm=1;std::copy(um,um+15,ug);prepareRelativeDivergenceRhs();for(int i=0;i<3;++i)assert(rhs[i]==0.);
p.mixtureDivergenceForm=1;rhs[0]=123.;prepareRelativeDivergenceRhs();assert(rhs[0]==123.);
}
'''
with tempfile.TemporaryDirectory() as d:
 pth=Path(d);(pth/'check.cpp').write_text(cpp)
 subprocess.run(['g++','-std=c++17',str(pth/'check.cpp'),'-o',str(pth/'check')],check=True)
 subprocess.run([str(pth/'check')],check=True)
print('Actual relative-flux helper/prefactor: SEM and cubature routing, UM-UG sign, alpha product, constant rhoL denominator, equal-density/velocity limits and padding passed.')
