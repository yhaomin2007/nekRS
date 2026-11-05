"""Compile actual C++ history, source and gradient routing; controlled CPU arrays.
No full NekRS elliptic/MPI/GPU integration is claimed.
"""
from pathlib import Path
import re, subprocess, tempfile
case=Path(__file__).resolve().parents[1]
terms=(case/'bubbleColumnTerms.hpp').read_text()
def function(marker):
 a=terms.index(marker);b=terms.index('{',a);d=1;e=b+1
 while d:d+=(terms[e]=='{')-(terms[e]=='}');e+=1
 return terms[a:e]
prefix=r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>
using dfloat=double;using dlong=long;
struct Mem {
 std::shared_ptr<std::vector<double>> v;long off=0,len=0;
 Mem(){} explicit Mem(long n){resize(n);}
 void resize(long n){v=std::make_shared<std::vector<double>>(n,0);off=0;len=n;}
 bool isInitialized()const{return bool(v);}
 long size()const{return len;}
 Mem slice(long a,long n)const{Mem r=*this;r.off+=a;r.len=n;return r;}
 double &operator[](long i)const{assert(i>=0&&i<len);return (*v)[off+i];}
 void copyFrom(const Mem &b,long n,long dst=0,long src=0){
  std::vector<double> t(n);for(long i=0;i<n;++i)t[i]=b[src+i];
  for(long i=0;i<n;++i)(*this)[dst+i]=t[i];}
 void copyTo(double *p)const{for(long i=0;i<len;++i)p[i]=(*this)[i];}
};
template<class T>using deviceMemory=Mem;
namespace occa{using memory=Mem;}
struct LA{
 void fill(long n,double x,Mem a){for(long i=0;i<n;++i)a[i]=x;}
 void axpby(long n,double a,Mem x,double b,Mem y){for(long i=0;i<n;++i)y[i]=a*x[i]+b*y[i];}
 void axmyMany(long n,int c,long offset,long,double a,Mem x,Mem y){
  for(int k=0;k<c;++k)for(long i=0;i<n;++i)y[k*offset+i]*=a*x[i];}
}la;
struct Comm{int mpiComm(){return 0;}int mpiRank(){return 0;}};
struct Options{bool compareArgs(const char*,const char*){return false;}}options;
struct Par{void extract(const char*,const char*,int&){}};
struct Platform{LA *linAlg=&la;Comm comm;Options options;std::shared_ptr<Par> par=std::make_shared<Par>();}platformObject,*platform=&platformObject;
struct Mesh{long Nlocal=2,Nelements=1;int dim=3;Mem o_vgeo{1},o_D{1},o_invLMM{4};int oogs=0;}mesh;
struct Fluid{Mem o_U{12},o_P{4},o_coeffEXT{3},o_EXT{12};Mem o_explicitTerms(){return o_EXT;}}fluid;
struct Scalar{
 std::map<std::string,int> nameToIndex{{"alpha",0},{"ugx",1},{"ugy",2},{"ugz",3}};
 std::vector<int>compute{1,1,1,1};int cvode=0;Mem o_EXT{16};
 Mem o_explicitTerms(const char*name){return o_EXT.slice(4*nameToIndex.at(name),4);}
}scalar;
struct Nrs{
 Mesh *meshV=&mesh;Fluid *fluid;Scalar *scalar;long fieldOffset=4;int tstep=0,Nscalar=4;
 int geom=0,neknek=0,advectionSubcycingSteps=0;
 std::function<void(double,int)> postScalar=nullptr,preFluid=nullptr;
 std::function<bool(int)>userConvergenceCheck=nullptr;
}nr{&mesh,&fluid,&scalar},*nrs=&nr;
void nekrsCheck(bool bad,int,int,const char*,...){if(bad)throw std::runtime_error("guard");}
const int ogsDfloat=0,ogsAdd=0;
namespace oogs{void startFinish(Mem,int,long,int,int,int){}}
double rawGrad=0,velocity=0,rawAlpha=0,relativeFlux=0,explicitDiff=0;
namespace opSEM{void strongGrad(Mesh*,long offset,Mem,Mem result){
 for(int c=0;c<3;++c)for(int i=0;i<2;++i)result[c*offset+i]=rawGrad*(c+1)+i;}}
template<class...T>void launchKernel(const char*,T...){}
'''
physics=r'''
namespace bubbleColumn {
namespace ugTerms {void captureSource(double){} void captureSolved(double){} void capturePostprocessed(double){}}
struct Parameters{int gasPressureGradientExtrapolation=0,mixtureDivergenceForm=1;}p;
bool assemblingExplicitSources=false;
ExplicitHistory convectionHistory,pressureHistory,divergenceTransportHistory;
Mem o_gradPStrong{12},o_gradPWeak{12},o_gradP{12};
Mem o_alphaSource{4},o_alphaRelativeRhs{4},o_divergenceTransportEXT{4};
Mem o_convectionSample{16},o_convectionEXT{16},o_alphaExplicitDiffusion{4};
Mem o_gasTransport{16},o_ugSource{12},o_mixtureForce{12},o_mixtureAdvectorUsed{12};
std::vector<std::string> events;
void prepareGasAdvection(){events.push_back("prepare");}
void evaluatePointwiseTerms();
void prepareRelativeDivergenceRhs(){for(int i=0;i<2;++i)o_alphaRelativeRhs[i]=relativeFlux+i;}
void subtractScalarDiffusion(){
 for(int i=0;i<2;++i)o_alphaSource[i]+=explicitDiff*(i+1);
 for(int c=0;c<3;++c)for(int i=0;i<2;++i)o_ugSource[c*4+i]+=.7;
}
void evaluateMixtureForce(){for(int i=0;i<12;++i)o_mixtureForce[i]=123;}
void evaluateGasCubatureTransport(bool conservative){
 assert(!conservative);for(int c=0;c<3;++c)for(int i=0;i<2;++i)
 o_gasTransport[(c+1)*4+i]=(c+1)*velocity*velocity+i;
}
void postProcessAlpha(double,int){events.push_back("alphaPost");}
void postProcessGasFlux(){events.push_back("ugPost");}
'''
main=r'''
void evaluatePointwiseTerms(){
 evaluateGasPressureGradient();
 for(int i=0;i<2;++i)o_alphaSource[i]=-rawAlpha*velocity+i;
 for(int c=0;c<3;++c)for(int i=0;i<2;++i)
  o_ugSource[c*4+i]=5+c-o_gradP[c*4+i]/4.;
}
} // namespace
#include "bubbleColumnOneThrough.hpp"
int main(){
 using namespace bubbleColumn;
 for(int mode:{0,1})for(int form:{1,2}){
  convectionHistory=ExplicitHistory();pressureHistory=ExplicitHistory();divergenceTransportHistory=ExplicitHistory();
  p.gasPressureGradientExtrapolation=mode;p.mixtureDivergenceForm=form;
  std::vector<std::vector<double>> oldConv,oldPressure,oldRelative;
  // Nonuniform-step Lagrange weights (target .9, source times .6,.2,0):
  // 2.625,-3.375,1.75. Constant-step EXT1/2/3 are also exercised.
  const std::vector<std::vector<double>> W{{1,0,0},{2,-1,0},{3,-3,1},{2.625,-3.375,1.75}};
  for(int step=1;step<=4;++step){
   nrs->tstep=step;for(int h=0;h<3;++h)fluid.o_coeffEXT[h]=W[step-1][h];
   auto beforeW=W[step-1];rawGrad=step*step+3;velocity=.1*step+.2;
   rawAlpha=.03*step;relativeFlux=-.06*step;explicitDiff=.002*step;
   std::vector<double> conv(16,0),grad(12,0),rel(4,0);
   for(int i=0;i<2;++i){conv[i]=-rawAlpha*velocity+i;rel[i]=(form==2?relativeFlux+i:conv[i]);
    for(int c=0;c<3;++c){conv[(c+1)*4+i]=(c+1)*velocity*velocity+i;grad[c*4+i]=rawGrad*(c+1)+i;}}
   oldConv.insert(oldConv.begin(),conv);oldPressure.insert(oldPressure.begin(),grad);oldRelative.insert(oldRelative.begin(),rel);
   // Native EXT1 is the carrier; addExplicitSources must supply the final selected values.
   for(int i=0;i<16;++i)scalar.o_EXT[i]=999;
   oneThrough::beginStep(0);
   for(int i=0;i<2;++i){
    double expectedAlpha=0,expectedDiv=0;
    for(int h=0;h<3;++h){int j=std::min(h,step-1);expectedAlpha+=W[step-1][h]*oldConv[j][i];expectedDiv+=W[step-1][h]*oldRelative[j][i];}
    assert(std::abs(scalar.o_EXT[i]-expectedAlpha-explicitDiff*(i+1))<1e-12);
    assert(std::abs(o_divergenceTransportEXT[i]-expectedDiv)<1e-12);
    assert(std::abs(o_alphaExplicitDiffusion[i]-explicitDiff*(i+1))<1e-12);
    for(int c=0;c<3;++c){
     double expectedConvection=0,expectedGradient=0;
     for(int h=0;h<3;++h){int j=std::min(h,step-1);
      expectedConvection+=W[step-1][h]*oldConv[j][(c+1)*4+i];
      expectedGradient+=(mode?W[step-1][h]:double(h==0))*oldPressure[j][c*4+i];}
     assert(std::abs(o_gradP[c*4+i]-expectedGradient)<1e-12);
     assert(std::abs(scalar.o_EXT[(c+1)*4+i]-(5+c+.7-expectedGradient/4.-expectedConvection))<1e-12);
    }
   }
   assert(scalar.o_EXT[2]==999&&scalar.o_EXT[6]==999);
   // Repeat source evaluation at the same timestep: history must not shift twice.
   auto saved=convectionHistory.samples;std::vector<double> snapshot(saved.size());saved.copyTo(snapshot.data());
   oneThrough::beginStep(0);
   for(long i=0;i<saved.size();++i)assert(convectionHistory.samples[i]==snapshot[i]);
   // Property evaluations see newly solved scalars, but may not touch source-time histories/selected G.
   auto selected=o_gradP;std::vector<double> g(12);selected.copyTo(g.data());
   rawGrad=999;evaluateGasPressureGradient();
   for(int c=0;c<3;++c)for(int i=0;i<2;++i)assert(o_gradP[c*4+i]==g[c*4+i]);
   for(int h=0;h<3;++h)assert(fluid.o_coeffEXT[h]==beforeW[h]);
   oneThrough::postScalar(0,step);oneThrough::preFluid(0,step);oneThrough::finishStep();
   assert(scalar.compute==std::vector<int>({1,1,1,1}));
  }
 }
 // Confirm variable-step weights independently from quadratic interpolation.
 const double t[3]={.6,.2,0},target=.9,w[3]={2.625,-3.375,1.75};
 for(int degree=0;degree<=2;++degree){double val=0;for(int i=0;i<3;++i)val+=w[i]*std::pow(t[i],degree);assert(std::abs(val-std::pow(target,degree))<1e-12);}
 // Setup evaluations do not create a physical-time history.
 ExplicitHistory setup;Mem setupSample(4),setupOut(4);setupSample[0]=4;
 nrs->tstep=0;setup.apply(setupSample,setupOut,1,true);
 assert(!setup.initialized()&&setupOut[0]==4);
 // Native startup and restart constant continuation.
 ExplicitHistory restart;Mem sample(4),out(4);sample[0]=7;sample[1]=-3;
 nrs->tstep=100;fluid.o_coeffEXT[0]=3;fluid.o_coeffEXT[1]=-3;fluid.o_coeffEXT[2]=1;
 restart.apply(sample,out,1,true);assert(out[0]==7&&out[1]==-3);
 bool stopped=false;try{oneThrough::singlePass(2);}catch(...){stopped=true;}assert(stopped);
 puts("Actual C++: full nonlinear convection EXT1/2/3 and variable-step weights; independent strong pressure latest/EXT; relative/material divergence histories; explicit diffusion once; repeat-source/setup/restart handling; no mask changes passed.");
}
'''
# The production helper needs standard headers from the main UDF environment.
source=prefix+'\n#include <stdexcept>\n#include "bubbleColumnTimeExtrapolation.hpp"\n'+physics
source+=function('inline void evaluateGasPressureGradient()')+'\n'
source+=function('inline void storeAlphaTransportRhs()')+'\n'
source+=function('inline void extrapolateGasConvection()')+'\n'
source+=function('inline void addExplicitSources(double)')+'\n'+main
with tempfile.TemporaryDirectory() as d:
 p=Path(d);(p/'check.cpp').write_text(source)
 subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-Wno-unused-parameter','-I',str(case),str(p/'check.cpp'),'-o',str(p/'check')],check=True)
 subprocess.run([str(p/'check')],check=True)
order=(case/'bubbleColumnOneThrough.hpp').read_text()
udf=(case/'bubbleColumn.udf').read_text()
assert '->solve(' not in order and 'compute =' not in order
assert 'oneThrough::finishStep()' in udf and 'solveGas' not in udf
assert 'scalar->o_coeffEXT = o_scalarCoeffEXT' in terms
assert 'coefficients[0] = 1.0' in terms
assert 'gasPressureGradientFilter' not in terms
assert 'gasPressureInletMask' not in terms
print('Native all-scalar schedule and EXT1 carrier/no legacy pressure processing source checks passed.')

