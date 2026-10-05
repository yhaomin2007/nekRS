"""Compile actual scalar solve bodies and fluid orchestration against serial stubs."""
from pathlib import Path
import re,subprocess,tempfile
repo=Path(__file__).resolve().parents[3]
scalar=(repo/'src/solver/scalar/scalarSolver.cpp').read_text()
bodies=scalar[scalar.index('void scalar_t::solve(double'):scalar.index('void scalar_t::lagSolution')]
fluid=(repo/'src/solver/fluid/fluidSolver.hpp').read_text()
solve=re.search(r'  void solve\(double time, int stage\) override\n  \{.*?\n  \};',fluid,re.S).group(0).replace(' override','')
cpp=r'''
#include <cassert>
#include <functional>
#include <vector>
#include <string>
#include <memory>
using dfloat=double;using dlong=long;
std::vector<int> events;
namespace occa {struct memory {
 std::shared_ptr<std::vector<double>> v;int start=0;
 memory(int n=16):v(std::make_shared<std::vector<double>>(n)){}
 memory slice(int off,int)const {auto m=*this;m.start+=off;return m;}
 void copyFrom(memory m,int n,int dst=0,int src=0){for(int i=0;i<n;i++)v->at(start+dst+i)=m.v->at(m.start+src+i);}
 void copyTo(memory m,int n,int dst=0){m.copyFrom(*this,n,dst);}
 bool isInitialized(){return true;}
 int size(){return 1;}
};}
struct Options{bool compareArgs(std::string,std::string){return false;}};
struct BC{occa::memory o_usrwrk;bool hasRobin(std::string){return false;}};
struct Timer{void tic(std::string){} void toc(std::string){}};
struct Pool{template<class T>occa::memory reserve(long n){return occa::memory(n);}};
struct Algebra {template<class... T>void axpby(T...){};};
struct App{BC* bc;};
struct Platform{Timer timer;Pool deviceMemoryPool;Options options;Algebra* linAlg;App* app;} storage;
auto platform=&storage;
template<class... T>void launchKernel(T...){}
std::string scalarDigitStr(int n){return std::to_string(n);}
struct Mesh{int Nlocal=1,Nelements=1;occa::memory o_sgeo,o_vmapM,o_EToB,o_x,o_y,o_z;};
struct Elliptic{int index;void solve(occa::memory,occa::memory,occa::memory rhs,occa::memory){
 events.push_back(index);if(index>=1 && index<=3)assert(rhs.v->at(rhs.start)==42);}};
struct scalar_t {
 int NSfields=4,vFieldOffset=4,_fieldOffset=4,EToBOffset=6;
 std::vector<int> compute{1,1,1,1},cvodeSolve{0,0,0,0},fieldOffsetScan{0,1,2,3};
 std::vector<Mesh*> _mesh;std::vector<Elliptic*> ellipticSolver;
 std::vector<occa::memory> o_name=std::vector<occa::memory>(4);
 occa::memory o_JwF,o_Ue,o_S,o_EToB,o_diff,o_rho,o_Se;
 double g=1,delta[1]{.1};double* g0=&g;double* dt=delta;
 std::function<bool(int)> deferSolve=nullptr;
 std::function<void(double,int,occa::memory)> userRhs=nullptr;
 std::function<occa::memory(double,int)> userImplicitLinearTerm=nullptr;
 void solve(double,int);void solveDeferred(double,int);void solveFields(double,int,bool);
};
'''+bodies+r'''
struct Fluid {
 std::function<void(double,int)> postPressure=nullptr;
 void solvePressure(double,int){events.push_back(10);}
 void solveVelocity(double,int){events.push_back(20);}
'''+solve+r'''
};
int main(){BC bc;App app{&bc};Algebra a;storage.linAlg=&a;storage.app=&app;
Mesh m;Elliptic e0{0},e1{1},e2{2},e3{3};scalar_t s;s._mesh={&m,&m,&m,&m};s.ellipticSolver={&e0,&e1,&e2,&e3};
Fluid f;
// Default: all scalar fields then pressure/velocity, no callbacks required.
s.userRhs=[](double,int,occa::memory rhs){rhs.v->at(rhs.start)=42;};
s.solve(.1,1);f.solve(.1,1);assert((events==std::vector<int>{0,1,2,3,10,20}));events.clear();
s.deferSolve=[](int i){return i>=1;};
f.postPressure=[&](double t,int stage){assert(events.back()==10);s.solveDeferred(t,stage);};
s.solve(.1,1);f.solve(.1,1);assert((events==std::vector<int>{0,10,1,2,3,20}));
// Actual solve bodies do not mutate time/history or rebuild forcing.
}
'''
with tempfile.TemporaryDirectory() as t:
 f=Path(t)/'check.cpp';f.write_text(cpp)
 subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror',str(f),'-o',str(Path(t)/'check')],check=True)
 subprocess.run([str(Path(t)/'check')],check=True)
for bad in ['lagSolution(', 'makeForcing(', 'setTimeIntegrationCoeffs(']:assert bad not in bodies
print('Actual scalar solve bodies and fluid hook: default order, alpha-pressure-QG-velocity order, RHS hook and fixed histories passed.')
