"""Run the actual UDF prescribed-UG initialization against CPU memory stubs."""
from pathlib import Path
import subprocess
import tempfile

case = Path(__file__).resolve().parents[1]
udf = (case / 'bubbleColumn2.udf').read_text()
terms = (case / 'bubbleColumn2Terms.hpp').read_text()
start = udf.index('  if (bubbleColumn2::p.prescribedGasVelocityEnabled) {', udf.index('void UDF_Setup()'))
end = udf.index('  bubbleColumn2::initializeHistory();', start)
body = udf[start:end]
assert 'std::string("SCALAR") + sid + " SOLVER", "NONE"' in udf
assert 'for (const auto *sid : {"01", "02", "03"})' in udf
assert 'p.prescribedGasVelocityEnabled ? 0.0 : p.gasVelocityLowAlphaCutoffEnabled' in terms
assert 'if (p.prescribedGasVelocityEnabled\n      ||' in terms
assert 'i < (p.prescribedGasVelocityEnabled ? 1 : 4)' in terms
assert 'i < (p.prescribedGasVelocityEnabled ? 0 : 3)' in terms
assert 'if (!p.prescribedGasVelocityEnabled) evaluateGasCubatureTransport(false)' in terms
# Native disabled scalars skip both solves and Dirichlet updates.
native = (case.parents[1] / 'src/solver/scalar/scalarSolver.cpp').read_text()
for function in ('void scalar_t::applyDirichlet', 'void scalar_t::solve'):
    pos = native.index(function)
    assert 'if (!compute[is]' in native[pos:pos+2500]

cpp = r'''
#include <cassert>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>
using dlong=long; using dfloat=double;
struct Mem {
  std::vector<double>* v; long off=0;
  Mem slice(long o,long) { return {v,off+o}; }
};
struct LA {
 void fill(long n,double a,Mem m) { for(long i=0;i<n;++i)(*m.v)[m.off+i]=a; }
 void axpby(long n,double a,Mem x,double b,Mem y) {
   for(long i=0;i<n;++i)(*y.v)[y.off+i]=a*(*x.v)[x.off+i]+b*(*y.v)[y.off+i];
 }
};
struct Scalar { std::map<std::string,std::vector<double>> s;
 Mem o_solution(std::string n) {return {&s.at(n)};}
};
struct Fluid { Mem o_U; };
struct Mesh { long Nlocal=3; };
struct Nrs { Mesh* meshV; Scalar* scalar; Fluid* fluid; long fieldOffset=5; } *nrs;
struct Options { std::string restart; std::string getArgs(std::string){return restart;} };
struct Comm {int mpiRank(){return 1;}};
struct Platform { LA* linAlg; Options options; Comm comm; } *platform;
namespace bubbleColumn2 { struct {int prescribedGasVelocityEnabled; double ugInitial;} p; }
void initialize() {
''' + body + r'''
}
int main() {
 LA la; Platform plat{&la}; platform=&plat;
 Mesh mesh; Scalar scalar; std::vector<double> um(15,9.);
 Fluid fluid{{&um}}; Nrs obj{&mesh,&scalar,&fluid}; nrs=&obj;
 for(auto name:{"alpha","ugx","ugy","ugz"})scalar.s[name]=std::vector<double>(5,7.);
 scalar.s["alpha"]={0.,.01,.4,0.,0.};
 for(bool restart:{false,true}) for(double speed:{.3,-.2,0.}) {
  plat.options.restart=restart?"restart.fld":"";
  std::fill(um.begin(),um.end(),9.);
  bubbleColumn2::p={1,speed}; initialize();
  for(int i=0;i<3;++i) {
   assert(scalar.s["ugx"][i]==0. && scalar.s["ugy"][i]==0.);
   assert(scalar.s["ugz"][i]==speed);
   assert(um[i]==(restart?9.:0.));
   assert(um[5+i]==(restart?9.:0.));
   assert(std::abs(um[10+i]-(restart?9.:scalar.s["alpha"][i]*speed))<1e-14);
  }
 }
 bubbleColumn2::p.prescribedGasVelocityEnabled=0;
 scalar.s["ugz"][0]=123.; initialize(); assert(scalar.s["ugz"][0]==123.);
}
'''
with tempfile.TemporaryDirectory() as tmp:
    src = Path(tmp) / 'test.cpp'
    src.write_text(cpp)
    exe = Path(tmp) / 'test'
    subprocess.run(['c++', '-std=c++17', str(src), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
print('PASS: prescribed UG fresh/restart/disabled paths, native solve/BC skip and active alpha forcing')
