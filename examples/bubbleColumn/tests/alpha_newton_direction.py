"""Execute the production Newton/GMRES direction with a nonsymmetric Jacobian.

This isolates the host Givens/orthogonalization logic from the real SEM-kernel
tests in alpha_residual.py, including preservation of a Dirichlet component.
"""
from pathlib import Path
import subprocess,tempfile

case=Path(__file__).resolve().parents[1]
source=(case/'bubbleColumnAlphaResidual.hpp').read_text()
direction=source[source.index('inline void direction()'):source.index('inline void solve(')]
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
deviceMemory<double> previous,baseResidual,residualStrong,jacobianVector,newtonStep;
std::vector<deviceMemory<double>> krylov(21);
int lastLinearIterations=0;
#define nekrsCheck(test,...) do{if(test)throw std::runtime_error("GMRES breakdown");}while(0)
double dot(const deviceMemory<double>& a,const deviceMemory<double>& b){
 double r=0;for(int i=0;i<8;++i)r+=(*a.a)[i]*(*b.a)[i]/8;return r;}
double length(const deviceMemory<double>& a){return std::sqrt(dot(a,a));}
double h=150000;
void evaluate(){
 auto&a=*nrs->scalar->alpha.a;auto&r=*residualStrong.a;r[0]=0;
 for(int i=1;i<8;++i)r[i]=h*(a[i]+.3*a[(i+1)%8]-.2*a[(i+7)%8]+.1*a[i]*a[i]*a[i]-.001*i);
}
double norm(){return length(residualStrong);}
DIRECTION
int main(){
 for(int i=0;i<8;++i)(*previous.a)[i]=.01*(i+1);
 nrs->scalar->alpha.copyFrom(previous,8);evaluate();baseResidual.copyFrom(residualStrong,8);
 direction();assert(lastLinearIterations>0&&lastLinearIterations<=20);
 assert((*newtonStep.a)[0]==0); // constrained inlet component
 for(int i=0;i<8;++i)assert((*nrs->scalar->alpha.a)[i]==(*previous.a)[i]);
 double sum=0,base=0;
 for(int i=1;i<8;++i){
  auto&d=*newtonStep.a;double ai=(*previous.a)[i];
  double jd=h*((1+.3*ai*ai)*d[i]+.3*d[(i+1)%8]-.2*d[(i+7)%8]);
  sum+=std::pow(jd+(*baseResidual.a)[i],2);base+=std::pow((*baseResidual.a)[i],2);
 }
 assert(std::sqrt(sum/base)<.05001);
}
'''.replace('DIRECTION',direction)
with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp);(p/'check.cpp').write_text(code)
    subprocess.run(['g++','-std=c++17','-O2',str(p/'check.cpp'),'-o',str(p/'check')],check=True)
    subprocess.run([str(p/'check')],check=True)
print('Production host GMRES direction: nonsymmetric nonlinear Jacobian, linear tolerance, alpha restoration and Dirichlet preservation passed.')
