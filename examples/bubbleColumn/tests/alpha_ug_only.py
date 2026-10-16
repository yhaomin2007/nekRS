"""Actual masks and native kernels: alpha advection histories remain zero."""
from pathlib import Path
import re
import subprocess
import tempfile

root=Path(__file__).resolve().parents[3]
case=(root/'examples/bubbleColumn/bubbleColumnTerms.hpp').read_text()
helper='inline void disableNativeAlphaAdvection()'+case.split(
    'inline void disableNativeAlphaAdvection()',1)[1].split('inline void initializeHistory()',1)[0]
def serial(s):
    s=s.replace('@kernel ','').replace('@ restrict ','').replace('@shared ','').replace('@exclusive ','').replace('@barrier();','')
    return re.sub(r';\s*@(?:outer|inner)\([01]\)|;\s*@tile\(p_blockSize,\s*@outer,\s*@inner\)','',s)
prefix=r'''
#include <cassert>
#include <vector>
#include <map>
#include <string>
#include <cmath>
using dfloat=double;using dlong=long;
#define p_Nq 2
#define p_Np 8
#define p_cubNq 3
#define p_cubNp 27
#define p_Nvgeo 12
#define p_IJWID 10
#define p_JWID 9
#define p_RXID 0
#define p_RYID 1
#define p_RZID 2
#define p_SXID 3
#define p_SYID 4
#define p_SZID 5
#define p_TXID 6
#define p_TYID 7
#define p_TZID 8
#define p_nEXT 2
#define p_nBDF 2
#define p_SUBCYCLING 0
#define p_MovingMesh 0
#define p_ADVECTION 1
const double c_I[3][2]={{}},c_D[3][3]={{}};
struct Mem {std::vector<long>a;void resize(size_t n){a.resize(n);}
void copyFrom(const std::vector<long>&b){a=b;}void copyFrom(const long*b){std::copy(b,b+a.size(),a.begin());}};
Mem o_userAdvectionCompute;
struct Scalar {std::vector<int>compute{1,1,1,1};Mem o_compute;
std::map<std::string,int>nameToIndex{{"alpha",0}};} scalar;
struct Nrs {Scalar *scalar;} nr{&scalar},*nrs=&nr;
'''
native=(root/'src/core/kernels/strongAdvectionVolumeScalarHex3D.okl').read_text()
cub=(root/'src/core/kernels/strongAdvectionCubatureVolumeScalarHex3D.okl').read_text()
forcing=(root/'src/solver/scalar/kernels/sumMakef.okl').read_text()
main=r'''
int main(){scalar.o_compute.resize(4);disableNativeAlphaAdvection();
assert(scalar.compute[0]==1);assert(scalar.o_compute.a[0]==0);
for(int i=0;i<4;++i){assert(o_userAdvectionCompute.a[i]==1);if(i)assert(scalar.o_compute.a[i]==1);}
double geo[96]={},D[4]={},I[6]={},DI[6]={},P[6]={},rho[8],S[24]={},Urst[81]={},adv[16]={};
for(int i=0;i<8;++i){geo[80+i]=1.;rho[i]=1.;S[8+i]=.1;}
long offsets[1]={0};double ext[2]={1.,0.},bdf[2]={1.,0.},mass[8],source[16],out[8];
for(int i=0;i<8;++i)mass[i]=1.;for(int i=0;i<16;++i)source[i]=-.03;
for(int step=0;step<4;++step){
// Arbitrary changing mixture advector cannot enter masked alpha transport.
for(int i=0;i<81;++i)Urst[i]=(step+1)*(.3+i);
strongAdvectionVolumeScalarHex3D(1,1,0,geo,D,scalar.o_compute.a.data(),offsets,8,S,Urst,rho,adv);
strongAdvectionCubatureVolumeScalarHex3D(1,1,0,0,geo,DI,I,P,scalar.o_compute.a.data(),offsets,8,27,S,Urst,rho,adv);
for(int i=0;i<16;++i)assert(adv[i]==0.);
sumMakef(8,mass,10.,ext,bdf,0,8,8,rho,S,adv,source,out);
for(int i=0;i<8;++i)assert(std::abs(out[i]-(10.*S[i]-.03))<1e-14);
for(int i=0;i<8;++i)adv[i+8]=adv[i];
}
// The independent user mask still evaluates a nonzero UM.grad(alpha) for
// divergence reconstruction, using the actual native GLL advection kernel.
D[0]=-.5;D[1]=.5;D[2]=-.5;D[3]=.5;
for(int i=0;i<81;++i)Urst[i]=0.;
for(int i=0;i<8;++i){S[i]=.2+.01*(i%2?1.:-1.);Urst[i]=.7;}
strongAdvectionVolumeScalarHex3D(1,1,0,geo,D,o_userAdvectionCompute.a.data(),offsets,8,S,Urst,rho,adv);
for(int i=0;i<8;++i)assert(std::abs(adv[i]-.007)<1e-14);
}
'''
with tempfile.TemporaryDirectory() as directory:
    p=Path(directory);(p/'test.cpp').write_text(prefix+helper+serial(native)+serial(cub)+serial(forcing)+main)
    subprocess.run(['g++','-std=c++17',str(p/'test.cpp'),'-o',str(p/'test')],check=True)
    subprocess.run([str(p/'test')],check=True)
# Confirm v26 initializes private ADV with zeros and device flags are only
# used by advection, not the scalar solve or HPF.
allocation=(root/'src/platform/device.tpp').read_text()
assert 'std::calloc(entries, sizeof(T))' in allocation
scalarSource=(root/'src/solver/scalar/scalarSolver.cpp').read_text()
assert 'o_ADV = platform->device.malloc<dfloat>(nFieldsAlloc * fieldOffsetSum)' in scalarSource
advection=scalarSource.split('void scalar_t::makeAdvection')[1].split('void scalar_t::makeExplicit')[0]
assert scalarSource.count('o_compute + is')==advection.count('o_compute + is')==2
forcing=case.split('inline void addExplicitSources(double)')[1].split('inline void buildDivergenceFromAlphaRhs')[0]
assert 'for (int i = 1; i < 4; ++i)' in forcing
userOp=case.split('inline void evaluateNativeAlphaAdvection')[1].split('inline void prepareGasAdvection')[0]
assert 'nrs->scalar->o_compute' not in userOp and 'o_userAdvectionCompute' in userOp
print('Actual masks/GLL/cubature/sumMakef: alpha solve stays enabled, native advection and histories remain zero despite changing UM; user divergence mask remains enabled, UG flags unchanged.')
