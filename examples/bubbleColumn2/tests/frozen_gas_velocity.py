"""Exercise actual profile/reconstruction helpers and verify freeze routing."""
from pathlib import Path
import subprocess
import tempfile
import re

case = Path(__file__).resolve().parents[1]
s = (case / 'bubbleColumn2Terms.hpp').read_text()
udf = (case / 'bubbleColumn2.udf').read_text()

def function(marker):
    start = s.index(marker)
    pos = s.index('{', start)
    end, depth = pos + 1, 1
    while depth:
        depth += (s[end] == '{') - (s[end] == '}')
        end += 1
    return s[start:end]

cpp = r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <map>
#include <string>
#include <vector>
using dfloat=double; using dlong=long;
struct Mem {
 std::vector<double> v;
 Mem(int n=18):v(n,123.){}
 void copyFrom(const Mem& src,long n,long dst=0,long off=0){
  std::copy_n(src.v.begin()+off,n,v.begin()+dst);
 }
};
struct Scalar {
 std::map<std::string,Mem> fields;
 Mem& o_solution(std::string n){return fields[n];}
} scalar;
struct Mesh{long Nlocal=4;} mesh;
struct Nrs{Mesh* meshV=&mesh;Scalar* scalar=&::scalar;long fieldOffset=6;} solver;
auto nrs=&solver;
struct Params{int frozenGasVelocityEnabled=1,gasVelocityLowAlphaCutoffEnabled=1;
 double alphaFloor=1e-6;}p;
Mem o_ug,o_qg,o_frozenUg;
void reconstructGasVelocityKernel(long n,long offset,int cutoff,double floor,
 const Mem& a,const Mem& x,const Mem& y,const Mem& z,Mem& ug){
 for(long j=0;j<n;j++)for(int c=0;c<3;c++){
  double v=(c==0?x.v[j]:(c==1?y.v[j]:z.v[j]));
  ug.v[j+c*offset]=(cutoff&&a.v[j]<=floor)?0:v;
 }
}
void buildGasFluxKernel(long n,long offset,const Mem& a,const Mem& ug,Mem& q){
 for(long j=0;j<n;j++)for(int c=0;c<3;c++)
  q.v[j+c*offset]=std::max(0.,std::min(1.,a.v[j]))*ug.v[j+c*offset];
}
''' + function('inline dfloat frozenGasWallWeight(') + '\n' + function('inline void reconstructGasVelocity(') + r'''
int main(){
 const double R=.075,d=.005;
 assert(frozenGasWallWeight(0,R,d)==1);
 assert(frozenGasWallWeight(R-d,R,d)==1);
 assert(std::abs(frozenGasWallWeight(R-.5*d,R,d)-.5)<1e-14);
 assert(frozenGasWallWeight(R,R,d)==0);
 assert(frozenGasWallWeight(R+.01,R,d)==0);
 double prev=1;
 for(int i=0;i<=1000;i++){
  double w=frozenGasWallWeight(R*i/1000,R,d);
  assert(w>=0&&w<=prev+1e-14);prev=w;
 }
 double h=1e-9;
 assert(frozenGasWallWeight(R-h,R,d)/h<.001);
 assert((1-frozenGasWallWeight(R-d+h,R,d))/h<.001);
 for(int c=0;c<3;c++)for(int j=0;j<6;j++)
  o_frozenUg.v[c*6+j]=(c==2 && j<4)? .3*frozenGasWallWeight(R*j/3,R,d):0;
 auto& a=scalar.o_solution("alpha");a.v={0,1e-10,.01,.02,123,123};
 for(int step=0;step<5;step++){
  for(auto name:{"ugx","ugy","ugz"})
   std::fill(scalar.o_solution(name).v.begin(),scalar.o_solution(name).v.end(),999.);
  reconstructGasVelocity();
  for(int j=0;j<4;j++)for(int c=0;c<3;c++){
   assert(o_ug.v[c*6+j]==o_frozenUg.v[c*6+j]);
   assert(o_qg.v[c*6+j]==a.v[j]*o_ug.v[c*6+j]);
  }
  assert(o_ug.v[12]==.3); // Retain UG even where alpha=0.
  assert(o_ug.v[15]==0); // No slip at wall.
  assert(scalar.o_solution("ugz").v[4]==999.); // Do not overwrite padding.
 }
 p.frozenGasVelocityEnabled=0;
 scalar.o_solution("ugz").v[2]=.7;
 reconstructGasVelocity();
 assert(o_ug.v[14]==.7 && o_ug.v[12]==0);
}
'''
with tempfile.TemporaryDirectory() as temp:
    path = Path(temp)
    (path/'check.cpp').write_text(cpp)
    subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror',str(path/'check.cpp'),'-o',str(path/'check')],check=True)
    subprocess.run([str(path/'check')],check=True)
assert 'nrs->scalar->compute[i] = 0;' in udf
assert 'nrs->scalar->o_compute.copyFrom(compute.data(), compute.size());' in udf
assert udf.index('bubbleColumn2::reconstructGasVelocity();') < udf.index('bubbleColumn2::initializeHistory();')
assert 'p.frozenGasVelocityEnabled && p.virtualMassEnabled != 0.0' in udf
assert '(p.frozenGasVelocityEnabled || p.gasSolveAfterFluid) && i > 0' in function('inline void addExplicitSources(')
assert 'if (p.frozenGasVelocityEnabled) return o_NULL;' in function('inline occa::memory implicitGasDrag(')
assert 'p.frozenGasVelocityEnabled && std::string(name) != "alpha"' in function('inline void applyGasVelocityAVM(')
assert 'reconstructGasVelocity();' in function('inline void prepareGasAdvection(')
assert 'o_gasAdvector.copyFrom(o_ug, 3 * offset);' in function('inline void prepareGasAdvection(')
assert 'if (p.frozenGasVelocityEnabled)' in function('inline void postProcessGasVelocity(')
assert 'MPI_Allreduce(&localRadius, &radius' in s
assert 'ug[2*offset+n] = p.ugInlet*weight[n];' in s
assert 'if (p.frozenGasVelocityEnabled || p.smoothGasVelocityProfileEnabled)' in function('inline void allocate(')
initial=udf.split('if (bubbleColumn2::p.smoothGasVelocityProfileEnabled')[1].split('if (bubbleColumn2::p.frozenGasVelocityEnabled)')[0]
assert 'RESTART FILE NAME' in initial
assert 'nrs->scalar->o_solution(names[i]).copyFrom(bubbleColumn2::o_frozenUg' in initial
assert 'compute' not in initial
assert 'nrs->fluid->o_U.copyFrom(bubbleColumn2::o_qg' in initial
assert 'p.smoothGasVelocityProfileEnabled' not in function('inline void reconstructGasVelocity(')
print('Frozen UG: smooth wall ramp, endpoints, padding, repeated restoration, dilute gas, flux consistency and solver/transport routing passed.')

# Compile the actual allocation download calls against the real wrapper's
# vector-only copyTo overload. The earlier profile mock did not cover allocate.
header = (case.parents[1]/'src/platform/deviceMemory.hpp').read_text()
start = header.index('  void copyTo(std::vector<T> &dest, size_type count,')
end = header.index('\n  void copyTo(std::vector<T> &dest) const', start)
overload = header[start:end]
calls = re.findall(r'o_inletVelocityWeight\.copyTo\([^;]+;', function('inline void allocate('))
assert len(calls) == 2
transfer_cpp = r'''
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>
struct Backend {
 std::vector<double> data{0., .25, .5, 1.};
 void copyTo(double* out, std::size_t count, std::size_t offset) const {
  std::copy_n(data.begin()+offset, count, out);
 }
};
template<class T> struct Wrapper {
 using size_type=std::size_t;
 Backend occa_memory_;
''' + overload + r'''
};
int main(){
 Wrapper<double> o_inletVelocityWeight;
 const long offset=4;
 std::vector<double> weight(offset, -1.);
''' + '\n'.join(calls) + r'''
 assert(weight==o_inletVelocityWeight.occa_memory_.data);
}
'''
with tempfile.TemporaryDirectory() as temp:
    path = Path(temp)
    (path/'transfers.cpp').write_text(transfer_cpp)
    subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror',str(path/'transfers.cpp'),'-o',str(path/'transfers')],check=True)
    subprocess.run([str(path/'transfers')],check=True)
print('Frozen UG allocation: both host downloads compile against the actual deviceMemory vector overload and preserve values.')
