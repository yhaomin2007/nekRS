"""Execute actual UDF transport-history storage; no shift on repeated calls."""
from pathlib import Path
import subprocess,tempfile
case=Path(__file__).resolve().parents[1]
s=(case/'bubbleColumnTerms.hpp').read_text()
helper='inline void storeAlphaTransportRhs()'+s.split('inline void storeAlphaTransportRhs()',1)[1].split('inline void addExplicitSources',1)[0]
cpp=r'''
#include <algorithm>
#include <cassert>
#include <cmath>
using dlong=long;
struct Mem{double *a;void copyFrom(Mem b,long n,long dst=0,long src=0){std::copy(b.a+src,b.a+src+n,a+dst);}};
struct Parameters{int mixtureDivergenceForm=1;double subtractAlphaDiffusion=0;}p;
struct LA{void axpby(long n,double a,Mem x,double b,Mem y){for(long i=0;i<n;++i)y.a[i]=a*x.a[i]+b*y.a[i];}}la;
struct Platform{LA *linAlg=&la;}object,*platform=&object;
Mem o_alphaRelativeRhs{nullptr},o_alphaTransportOnly{nullptr};
struct Coeff{int size(){return 3;}};
struct Scalar{Coeff o_coeffEXT;}scalar;
struct Mesh{long Nlocal=3;}mesh;
struct Nrs{int tstep=0;long fieldOffset=5;Scalar *scalar;Mesh *meshV;}nr{0,5,&scalar,&mesh},*nrs=&nr;
double history[15]={},source[5]={1,2,3,99,99};
Mem o_alphaTransportRhsHistory{history},o_alphaSource{source};int alphaTransportHistoryStep=-1;
'''+helper+r'''
int main(){
 nrs->tstep=1;storeAlphaTransportRhs();
 for(int i=0;i<3;++i){assert(history[i]==i+1);assert(history[i+5]==0&&history[i+10]==0);}
 // Same-step refresh updates only current source; old history cannot advance.
 source[0]=4;storeAlphaTransportRhs();assert(history[0]==4&&history[5]==0);
 nrs->tstep=2;source[0]=5;storeAlphaTransportRhs();assert(history[0]==5&&history[5]==4&&history[10]==0);
 nrs->tstep=3;source[0]=6;storeAlphaTransportRhs();assert(history[0]==6&&history[5]==5&&history[10]==4);
 source[0]=123;assert(history[0]==6); // No alias to refreshed pointwise sources.
 for(int h=0;h<3;++h)assert(history[5*h+3]==0&&history[5*h+4]==0);
 double relative[5]={.1,.2,.3},pure[5]={-1,-2,-3};
 o_alphaRelativeRhs={relative};o_alphaTransportOnly={pure};p.mixtureDivergenceForm=2;
 nrs->tstep=4;storeAlphaTransportRhs();assert(history[0]==.1&&history[5]==6);
 p.subtractAlphaDiffusion=1;source[0]=-.9;relative[0]=.1;
 storeAlphaTransportRhs();assert(std::abs(history[0]-.2)<1e-14&&history[5]==6);

}
'''
with tempfile.TemporaryDirectory() as d:
 p=Path(d);(p/'check.cpp').write_text(cpp)
 subprocess.run(['g++','-std=c++17',str(p/'check.cpp'),'-o',str(p/'check')],check=True)
 subprocess.run([str(p/'check')],check=True)
print('Actual source-history helper: three time levels, one shift per timestep, independent source copy and padded strides passed.')
