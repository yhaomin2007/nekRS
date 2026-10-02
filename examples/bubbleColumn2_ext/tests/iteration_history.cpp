#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <vector>
#include <cassert>
using dfloat=double;using dlong=long long;
namespace occa {
struct memory {
 std::shared_ptr<std::vector<double>> v; long start=0,length=0;
 memory()=default; memory(long n):v(std::make_shared<std::vector<double>>(n)),length(n){}
 long size()const{return length;} double& at(long n)const{return v->at(start+n);}
 memory slice(long off,long n)const{memory m=*this;m.start+=off;m.length=n;return m;}
 void copyFrom(const memory& src,long n=-1){if(n<0)n=std::min(size(),src.size());std::vector<double> a(n);for(long i=0;i<n;++i)a[i]=src.at(i);for(long i=0;i<n;++i)at(i)=a[i];}
 void copyFrom(const double* src){for(long i=0;i<size();++i)at(i)=src[i];}
};}
template<class T>struct deviceMemory:occa::memory{void resize(long n){if(this->size()!=n)static_cast<occa::memory&>(*this)=occa::memory(n);}};
#define nekrsCheck(cond,comm,code,fmt,...) do{if(cond){char b[1024];std::snprintf(b,sizeof(b),fmt,__VA_ARGS__);throw std::runtime_error(b);}}while(0)
struct Comm{int mpiRank(){return 0;}int mpiComm(){return 0;}};
struct Par{template<class T>void extract(const char*,const char*,T&) {}};
struct Algebra {
 void fill(long n,double a,occa::memory m){for(long i=0;i<n;++i)m.at(i)=a;}
 void axpbyz(long n,double a,const occa::memory& x,double b,const occa::memory& y,occa::memory z){for(long i=0;i<n;++i)z.at(i)=a*x.at(i)+b*y.at(i);}
 double amax(long n,occa::memory x,int){double m=0;for(long i=0;i<n;++i)m=std::max(m,std::abs(x.at(i)));return m;}
 double weightedNorm2(long n,occa::memory w,occa::memory x,int){double v=0;for(long i=0;i<n;++i)v+=w.at(i)*x.at(i)*x.at(i);return std::sqrt(v);}
};
struct Platform{Comm comm;Par p;Par* par=&p;Algebra alg;Algebra* linAlg=&alg;}storage;auto platform=&storage;
struct Mesh{long Nlocal=1;occa::memory o_LMM{1};double volume=1;}mesh;
struct Fluid{long fieldOffsetSum=3;occa::memory o_U{6},o_Ue{3},o_P{2},o_EXT{6},o_coeffEXT{2},rhs{3};
 void makeAdvection(double,int){}void makeExplicit(double,int){}
 void makeForcing(){for(int i=0;i<3;++i)rhs.at(i)=2*o_U.at(i)-.5*o_U.at(i+3)+o_EXT.at(i);}
};
struct Scalar{long fieldOffsetSum=4;int Nsubsteps=0;long fieldOffsetScan[4]={0,1,2,3};occa::memory o_S{8},o_Se{4},o_EXT{8},o_coeffEXT{2},rhs{4};
 occa::memory o_solution(const char* key){int i=key[0]=='a'?0:key[2]=='x'?1:key[2]=='y'?2:3;return o_S.slice(i,1);}
 void makeAdvection(int,double,int){}void makeExplicit(int,double,int){}
 void makeForcing(){for(int i=0;i<4;++i)rhs.at(i)=2*o_S.at(i)-.5*o_S.at(i+4)+o_EXT.at(i);}
};
struct Nrs{Mesh* meshV=&mesh;Fluid* fluid;Scalar* scalar;long fieldOffset=1;int tstep=1,advectionSubcycingSteps=0;void* geom=nullptr;void* neknek=nullptr;double dt[1]={.1};occa::memory o_coeffEXT{2};void computeUrst(){}} nrsStorage;auto nrs=&nrsStorage;
namespace opSEM{void strongDivergence(Mesh*,long,occa::memory,occa::memory out){out.at(0)=0;}}
namespace bubbleColumn2_ext{
double cumulativeAlphaClipDeltaVolume=0,cumulativeQgMaskDeltaMagnitudeIntegral=0,cumulativeQgMaskDeltaIntegral[3]={};
void addExplicitSources(double){for(int i=0;i<3;++i)nrs->fluid->o_EXT.at(i)=.1*nrs->fluid->o_U.at(i);
for(int i=0;i<4;++i)nrs->scalar->o_EXT.at(i)=.1*nrs->scalar->o_S.at(i);}
void clipVolumeVelocity(){}
struct Parameters{double virtualMassEnabled=0;}p;
deviceMemory<double> o_ugPrevious,o_ulPrevious;
void updateVirtualMassHistory(){}
}
#include "../bubbleColumn2_extIteration.hpp"
int main(){
bubbleColumn2_ext::o_ugPrevious.resize(3);bubbleColumn2_ext::o_ulPrevious.resize(3);
Fluid fluid;Scalar scalar;nrsStorage.fluid=&fluid;nrsStorage.scalar=&scalar;mesh.o_LMM.at(0)=1;
for(int i=0;i<6;++i)fluid.o_U.at(i)=i+1;
for(int i=0;i<8;++i)scalar.o_S.at(i)=i+1;
bubbleColumn2_ext::beginCoupledStep(0);
for(int i=0;i<6;++i)fluid.o_U.at(i)=100+i;
for(int i=0;i<8;++i)scalar.o_S.at(i)=200+i;
for(int repeat=0;repeat<3;++repeat){bubbleColumn2_ext::refreshCoupledRhs(.1);
for(int i=0;i<6;++i)assert(fluid.o_U.at(i)==100+i);
for(int i=0;i<8;++i)assert(scalar.o_S.at(i)==200+i);
assert(std::abs(fluid.rhs.at(0)-(2*1-.5*4+.1*100))<1e-12);
assert(std::abs(scalar.rhs.at(0)-(2*1-.5*5+.1*200))<1e-12);
assert(fluid.o_coeffEXT.at(0)==1 && fluid.o_coeffEXT.at(1)==0);
}
bubbleColumn2_ext::couplingRelaxation=1;
bubbleColumn2_ext::couplingRequireConvergence=0;
bubbleColumn2_ext::couplingMinIterations=2;
bubbleColumn2_ext::o_iterU.copyFrom(fluid.o_U,3);
bubbleColumn2_ext::o_iterS.copyFrom(scalar.o_S,4);
bubbleColumn2_ext::o_iterP.copyFrom(fluid.o_P,1);
assert(!bubbleColumn2_ext::couplingConverged(1));
assert(bubbleColumn2_ext::couplingConverged(2));
bubbleColumn2_ext::couplingMaxIterations=2;
bubbleColumn2_ext::couplingRequireConvergence=1;
scalar.o_S.at(0)+=1;bool failed=false;
try{bubbleColumn2_ext::couplingConverged(2);}catch(const std::runtime_error&){failed=true;}assert(failed);
puts("Fixed BDF history, iterate restoration, minimum iterations and failure checks passed.");
}
