"""Run the production deferred-alpha host code with controlled device/solver doubles.

Checks pre-lag BDF retention, latest UG, native regularization, mass weighting,
relative-flux target, and exactly one alpha-only solve without a second lag.
The real native Helmholtz/SEM operators still require an integration run.
"""
from pathlib import Path
import subprocess
import tempfile

case = Path(__file__).resolve().parents[1]
host = (case / 'bubbleColumnPhaseMomentum.hpp').read_text()
snapshot = '  // Capture alpha BDF load' + host.split('  // Capture alpha\'s BDF load')[1].split('  sourceTime=time;')[0]
post = 'inline void postScalar(double time,int tstep)' + host.split('inline void postScalar(double time,int tstep)')[1].split('inline void preFluid')[0]
source = r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>
using dfloat=double; using dlong=int;
struct Memory {
 std::shared_ptr<std::vector<double>> data;
 int offset=0,count=0;
 Memory(int n=0):data(std::make_shared<std::vector<double>>(n)),count(n){}
 int size()const{return count;}
 double &operator[](int i)const{return data->at(offset+i);}
 Memory slice(int start,int n)const{Memory m=*this;m.offset+=start;m.count=n;return m;}
 void copyFrom(Memory m,int n,int dst=0,int src=0){for(int i=0;i<n;++i)(*this)[dst+i]=m[src+i];}
 void copyTo(double *v)const{for(int i=0;i<count;++i)v[i]=(*this)[i];}
};
struct Algebra {
 void fill(int n,double a,Memory x){for(int i=0;i<n;++i)x[i]=a;}
 void axpby(int n,double a,Memory x,double b,Memory y){for(int i=0;i<n;++i)y[i]=a*x[i]+b*y[i];}
 void axmy(int n,double a,Memory x,Memory y){for(int i=0;i<n;++i)y[i]*=a*x[i];}
};
struct Platform {Algebra *linAlg;}; Algebra alg; Platform plat{&alg}; Platform *platform=&plat;
constexpr int N=2,O=2;
struct Mesh {int Nlocal=N;Memory o_LMM{N};};
std::vector<std::string> stages;
struct Scalar {
 std::vector<int> compute=std::vector<int>(7,1);
 std::map<std::string,int> nameToIndex{{"alpha",0}};
 std::vector<int> fieldOffsetScan{0,2,4,6,8,10,12};
 int fieldOffsetSum=7*O, solves=0,explicitCalls=0;
 Memory o_coeffBDF{2},o_S{2*7*O},o_JwF{7*O},ext{2*7*O};
 Memory o_explicitTerms(std::string){return ext.slice(0,O);}
 void makeExplicit(int is,double,int){assert(is==0);++explicitCalls;for(int i=0;i<N;++i)ext[i]+=-.3*o_S[i];}
 void applyDirichlet(double){assert(compute[0]==1);for(int k=1;k<7;++k)assert(compute[k]==0);stages.push_back("alphaBC");}
 void solve(double,int stage){assert(stage==1);++solves;assert(compute[0]==1);for(int k=1;k<7;++k)assert(compute[k]==0);stages.push_back("alphaSolve");}
};
struct Nrs {std::unique_ptr<Scalar> scalar=std::make_unique<Scalar>();Mesh *meshV;double dt[1]={.1};int tstep=2;};
Mesh mesh;Nrs state;Nrs *nrs=&state;
struct Params {int mixtureDivergenceForm=2,alphaCompressionEnabled=1;}p;
Memory alphaHistory(N),alphaRegularization(N),o_alphaSource(N),o_alphaRelativeRhs(N),
       o_divergenceTransportEXT(N),o_alphaExplicitDiffusion(N),gas(N);
std::vector<int> activeScalars(7,1);
void trace(const char *s){stages.push_back(s);}
void prepareGasAdvection(){for(int i=0;i<N;++i)gas[i]=nrs->scalar->o_S[O+i];stages.push_back("newUG");}
void evaluateAlphaTransport(){for(int i=0;i<N;++i)o_alphaSource[i]=-(p.alphaCompressionEnabled?2.:1.)*gas[i]*nrs->scalar->o_S[i];}
void prepareRelativeDivergenceRhs(){for(int i=0;i<N;++i)o_alphaRelativeRhs[i]=o_alphaSource[i]+.7;}
void postProcessAlpha(double,int){stages.push_back("alphaPostprocess");}
void snapshot(double time){const int N=mesh.Nlocal;
'''
source += snapshot + '\n}\n' + post
source += r'''
int main(){
 state.meshV=&mesh;
 auto &s=*state.scalar;
 for(int form:{1,2})for(int compression:{0,1})for(int startup:{0,1}){
  p.mixtureDivergenceForm=form;p.alphaCompressionEnabled=compression;
  s.compute=activeScalars;s.solves=0;s.explicitCalls=0;stages.clear();
  s.o_coeffBDF[0]=startup?1:2;s.o_coeffBDF[1]=startup?0:-.5;
  double old[N]={.02,.04},older[N]={.01,.03},expected[N];
  for(int i=0;i<N;++i){
   s.o_S[i]=old[i];s.o_S[s.fieldOffsetSum+i]=older[i];
   s.o_S[O+i]=.1;mesh.o_LMM[i]=i+1.5;o_alphaExplicitDiffusion[i]=.01*(i+1);
  }
  snapshot(0.1);
  assert(s.explicitCalls==1 && s.compute[0]==0);
  // Native history lag occurs exactly once, before the phase sweep.
  for(int i=0;i<N;++i){s.o_S[s.fieldOffsetSum+i]=old[i];s.o_S[O+i]=.6+.1*i;}
  for(int i=0;i<N;++i){
   double transport=-(compression?2.:1.)*s.o_S[O+i]*old[i];
   expected[i]=mesh.o_LMM[i]*((s.o_coeffBDF[0]*old[i]+s.o_coeffBDF[1]*older[i])/.1+
       transport+o_alphaExplicitDiffusion[i]-.3*old[i]);
  }
  postScalar(.2,2);
  assert(s.solves==1 && s.explicitCalls==1 && s.compute==activeScalars);
  for(int i=0;i<N;++i){
   assert(std::abs(s.o_JwF[i]-expected[i])<1e-13);
   double transport=-(compression?2.:1.)*(.6+.1*i)*old[i];
   assert(std::abs(o_divergenceTransportEXT[i]-(transport+(form==2?.7:0)))<1e-13);
   assert(s.o_S[s.fieldOffsetSum+i]==old[i]);
   assert(s.o_S[O+i]==.6+.1*i); // alpha solve never overwrites new UG
  }
  assert((stages==std::vector<std::string>{"UG_UL_DONE","alphaBC","newUG",
      "ALPHA_BEGIN_NEW_UG","alphaSolve","alphaPostprocess","ALPHA_DONE"}));
 }
}
'''
with tempfile.TemporaryDirectory() as t:
    d = Path(t)
    (d / 'check.cpp').write_text(source)
    subprocess.run(['g++', '-std=c++17', str(d/'check.cpp'), '-o', str(d/'check')], check=True)
    subprocess.run([str(d/'check')], check=True)
print('Production deferred-alpha host path: new UG, pre-lag BDF1/BDF2, HPF once, local mass, matching pressure flux, one alpha solve, restored phase mask and unchanged histories passed.')
