"""Execute the production restarted linear GMRES with a nonsymmetric operator.

This isolates the host Givens/orthogonalization logic from the real SEM-kernel
tests in alpha_residual.py, including preservation of a Dirichlet component.
"""
from pathlib import Path
import subprocess,tempfile

case=Path(__file__).resolve().parents[1]
source=(case/'bubbleColumnAlphaResidual.hpp').read_text()
direction=source[source.index('inline void direction('):source.index('inline void solve(')]
code=r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
using dlong=int;using dfloat=double;
template<class T> struct deviceMemory {
 std::shared_ptr<std::vector<T>> a=std::make_shared<std::vector<T>>(8);
 void copyFrom(const deviceMemory& b,int n){std::copy_n(b.a->begin(),n,a->begin());}
};
struct LA {
 void scale(int n,double c,deviceMemory<double> &v){for(int i=0;i<n;++i)(*v.a)[i]*=c;}
 void fill(int n,double c,deviceMemory<double> &v){std::fill_n(v.a->begin(),n,c);}
 void axpby(int n,double c,const deviceMemory<double>& x,double b,deviceMemory<double>& y){
  for(int i=0;i<n;++i)(*y.a)[i]=c*(*x.a)[i]+b*(*y.a)[i];}
};
struct Comm{int mpiComm(){return 0;}};
struct Platform{LA la;LA*linAlg=&la;Comm comm;} platformObject,*platform=&platformObject;
struct Mesh{int Nlocal=8;};
struct Scalar{deviceMemory<double> alpha;deviceMemory<double> o_solution(const char*){return alpha;}};
struct NRS{Mesh mesh;Mesh*meshV=&mesh;Scalar scalarObject;Scalar*scalar=&scalarObject;double g0=1.5,dt[1]={1e-5};}nrsObject,*nrs=&nrsObject;
deviceMemory<double> baseResidual,operatorVector,linearStep;
std::vector<deviceMemory<double>> krylov(21);
int lastIterations=0;double projectedNorm=0;
#define nekrsCheck(test,...) do{if(test)throw std::runtime_error("GMRES breakdown");}while(0)
double dot(const deviceMemory<double>& a,const deviceMemory<double>& b){
 double r=0;for(int i=0;i<8;++i)r+=(*a.a)[i]*(*b.a)[i]/8;return r;}
double length(const deviceMemory<double>& a){return std::sqrt(dot(a,a));}
double h=150000;
void apply(const deviceMemory<double>& v,deviceMemory<double>& out){
 auto&a=*v.a;auto&r=*out.a;r[0]=0;
 for(int i=1;i<8;++i)r[i]=h*(a[i]+.3*a[(i+1)%8]-.2*a[(i+7)%8]);
}
DIRECTION
int main(){
 deviceMemory<double> correction,product,original;
 for(int i=0;i<8;++i)(*nrs->scalar->alpha.a)[i]=.01*(i+1);
 original.copyFrom(nrs->scalar->alpha,8);
 for(int i=1;i<8;++i)(*baseResidual.a)[i]=.1*i;
 deviceMemory<double> initial;initial.copyFrom(baseResidual,8);
 const double tolerance=1e-9;
 int cycles=0;
 while(length(baseResidual)>tolerance&&lastIterations<200){
  // A budget of two forces multiple real production restarts.
  direction(2,tolerance*.5);++cycles;
  assert((*linearStep.a)[0]==0);
  platform->linAlg->axpby(8,1.,linearStep,1.,correction);
  apply(correction,product);baseResidual.copyFrom(initial,8);
  platform->linAlg->axpby(8,1.,product,1.,baseResidual);
 }
 assert(cycles>1&&lastIterations<200);
 assert(length(baseResidual)<=tolerance);
 for(int i=0;i<8;++i)assert((*nrs->scalar->alpha.a)[i]==(*original.a)[i]);
 assert((*correction.a)[0]==0);
}

'''.replace('DIRECTION',direction)
with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp);(p/'check.cpp').write_text(code)
    subprocess.run(['g++','-std=c++17','-O2',str(p/'check.cpp'),'-o',str(p/'check')],check=True)
    subprocess.run([str(p/'check')],check=True)
print('Production restarted linear GMRES: nonsymmetric operator, true tolerance, unchanged alpha during operator actions and Dirichlet preservation passed.')
