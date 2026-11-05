"""Actual UG diagnostic kernels and C++ host callbacks; controlled CPU fields."""
from pathlib import Path
import re,subprocess,tempfile,ast
case=Path(__file__).resolve().parents[1]
def extract(s,marker):
 a=s.index(marker);b=s.index('{',a);d=1;e=b+1
 while d:d+=(s[e]=='{')-(s[e]=='}');e+=1
 return s[a:e]
def serial(s):
 return re.sub(r';\s*@tile\(p_blockSize,\s*@outer,\s*@inner\)','',s.replace('@kernel ','').replace('@ restrict ',''))
kernels=(case/'bubbleColumnUGTerms.okl').read_text()
physics=(case/'bubbleColumnEquations.okl').read_text()
cpp=r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <cstdio>
using std::isfinite;using dfloat=double;using dlong=long;
'''+serial(kernels)+serial(extract(physics,'@kernel void buildEquationTerms('))+serial(extract(physics,'@kernel void addGasStress('))+r'''
int main(){
 const long N=3,o=5,stride=3*o;const int count=20;
 double alpha[o]={0,.001,.2},ug[15]={},um[15]={},ul[15]={},gradA[o]={},divQ[o]={};
 double gradU[45]={},acc[15]={},gp[15]={},rho[o]={},mu[o]={},drift[45]={},stress[45]={};
 double alphaSource[o]={},source[15]={},lambda[o]={},divStress[15]={},diffComp[15]={},cv[20]={},raw[20]={};
 double terms[count*stride],native[15]={},weak[15]={},solved[15]={},mask[15],flag[o]={};
 const double floor=1e-6;
 for(int vm:{0,1})for(int drag:{0,1})for(int pressure:{0,1}){
  const double eff=1.2+vm*.5*998.2;
  std::fill(terms,terms+count*stride,999.);std::fill(mask,mask+stride,1.);
  for(int c=0;c<3;++c)for(int i=0;i<N;++i){int v=c*o+i;
   ug[v]=.2+.01*c+.02*i;um[v]=.02;ul[v]=.01-.003*c;
   acc[v]=.4+.2*c-.1*i;gp[v]=-5+2*c-i;divStress[v]=1e-7*(c+1)*(i+1);
   cv[(c+1)*o+i]=.03+.01*c+.02*i;raw[(c+1)*o+i]=cv[(c+1)*o+i]/2;
   diffComp[v]=c==0?std::numeric_limits<double>::quiet_NaN():.01*(i+1);
  }
  buildEquationTerms(N,o,998.2,1.2,.001,1.8e-5,floor,pressure,drag,.003,vm,.5,
      .2,-.4,-9.81,alpha,um,ug,ul,gradA,divQ,gradU,acc,gp,rho,mu,drift,stress,
      alphaSource,source,lambda);
  for(int c=0;c<3;++c){addGasStress(N,floor,eff,alpha,divStress+c*o,source+c*o);
   for(int i=0;i<N;++i)source[c*o+i]+=(c==0?0:(c==1?.5:1.)*diffComp[c*o+i])-cv[(c+1)*o+i];}
  ugCaptureTerms(N,o,1.2,eff,vm*.5*998.2,pressure,floor,.2,-.4,-9.81,0,.5,1,
      alpha,gp,lambda,ul,acc,divStress,diffComp,cv,raw,source,terms);
  for(int c=0;c<3;++c)for(int i=0;i<N;++i){int v=c*o+i;
   assert(std::abs(terms[v+14*stride])<1e-10);
   assert(std::abs(terms[v]+pressure*gp[v]/eff)<1e-12);
   assert(terms[v+1*stride]==-cv[(c+1)*o+i]);
   assert(terms[v+16*stride]==-raw[(c+1)*o+i]);
   assert(std::abs(terms[v+3*stride]-lambda[i]*ul[v])<1e-12);
   assert(std::abs(terms[v+4*stride]-vm*.5*998.2/eff*acc[v])<1e-12);
   if(c==0)assert(terms[v+6*stride]==0.);
   solved[v]=.1+.01*c+.02*i;native[v]=source[v]-.007*(c+1);
   // Construct an exactly balanced weak free-node equation independently.
   terms[v+19*stride]=9.5+.1*i;
   weak[v]=15.*solved[v]-terms[v+19*stride]-native[v]+lambda[i]*solved[v];
  }
  long masked[1]={0};ugTermFreeMask(1,masked,mask);weak[0]+=20; // Only x/node0 masked.
  ugFinishTerms(N,o,15.,solved,lambda,native,weak,mask,terms);
  for(int c=0;c<3;++c)for(int i=0;i<N;++i){int v=c*o+i;
   assert(std::abs(terms[v+8*stride]+.007*(c+1))<1e-12);
   assert(terms[v+9*stride]==-lambda[i]*solved[v]);
   assert(terms[v+10*stride]==weak[v]);
   assert(std::abs(terms[v+11*stride]-(15.*solved[v]-terms[v+19*stride]))<1e-12);
   assert(std::abs(terms[v+13*stride])<1e-10);
   assert(terms[v+18*stride]==15.*solved[v]);
   assert(std::abs(terms[v+17*stride]-(lambda[i]*ul[v]-lambda[i]*solved[v]))<1e-12);
  }
  double x[o],y[o],z[o];for(int i=0;i<N;++i){x[i]=solved[i];y[i]=solved[o+i];z[i]=solved[2*o+i]-.1;}
  ugPostprocessTerm(N,o,15.,solved,x,y,z,terms);
  for(int i=0;i<N;++i){assert(terms[i+15*stride]==0);assert(std::abs(terms[i+2*o+15*stride]+1.5)<1e-12);}
  for(int t=0;t<count;++t)for(int c=0;c<3;++c)
   for(int i=N;i<o;++i)assert(terms[t*stride+c*o+i]==999); // No padded-node writes.
  ugTermNonfinite(N,o,terms,flag);for(int i=0;i<N;++i)assert(flag[i]==0);
  terms[o+1]=std::numeric_limits<double>::infinity();ugTermNonfinite(N,o,terms,flag);assert(flag[1]==1);
  terms[o+1]=std::numeric_limits<double>::quiet_NaN();ugTermNonfinite(N,o,terms,flag);assert(flag[1]==1);
 }
 puts("Actual diagnostic + physics kernels: signs/units, VM/drag/pressure toggles, zero-alpha stress/floor, disabled stale compensation, exact weak balance, per-component BC masks, postprocessing, padding and nonfinite detection passed.");
}
'''
with tempfile.TemporaryDirectory() as d:
 p=Path(d);(p/'kernels.cpp').write_text(cpp)
 subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-Wno-unused-parameter',str(p/'kernels.cpp'),'-o',str(p/'kernels')],check=True)
 subprocess.run([str(p/'kernels')],check=True)

# Compile the entire production host module against CPU device-memory/IO doubles.
# Reuse the independently tested field/memory fixture, extending only required APIs.
tree=ast.parse((case/'tests/time_extrapolation.py').read_text())
prefix=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='prefix' for t in n.targets))
prefix=prefix.replace('#include <algorithm>','#include <algorithm>\n#include <fstream>\n#include <iomanip>\n#include <tuple>')
prefix=prefix.replace('namespace occa{using memory=Mem;}','''namespace occa{using memory=Mem;struct kernel{template<class...T>void operator()(T...)const{}};}
struct deviceKernelProperties{};
struct Requests{void add(const std::string&,const char*,deviceKernelProperties&){}occa::kernel load(const std::string&,const char*){return {};}};''')
prefix=prefix.replace('struct LA{','''struct LA{
 double min(long n,Mem a,int){double x=a[0];for(int i=1;i<n;++i)x=std::min(x,a[i]);return x;}
 double max(long n,Mem a,int){double x=a[0];for(int i=1;i<n;++i)x=std::max(x,a[i]);return x;}
 void axmy(long n,double a,Mem x,Mem y){for(int i=0;i<n;++i)y[i]*=a*x[i];}''')
prefix=prefix.replace('struct Platform{LA *linAlg=&la;', 'struct Platform{Requests kernelRequests;LA *linAlg=&la;')
prefix=prefix.replace('Mem o_vgeo{1},o_D{1},o_invLMM{4}', 'Mem o_vgeo{1},o_D{1},o_ggeo{1},o_LMM{4},o_invLMM{4}')
prefix=prefix.replace('struct Scalar{','''struct Elliptic{int Nmasked(){return 0;}Mem o_maskIds(){return Mem(0);}};
struct Scalar{
 std::vector<Elliptic*>ellipticSolver;
 std::vector<long>fieldOffsetScan{0,4,8,12};long fieldOffsetSum=16;
 Mem o_coeffBDF{2},o_S{48},diff{4};
 Mem o_solution(const char*n){return o_S.slice(4*nameToIndex.at(n),4);}
 Mem o_diffusionCoeff(const char*){return diff;}''')
prefix=prefix.replace('int geom=0,neknek=0,advectionSubcycingSteps=0;', '''int geom=0,neknek=0,advectionSubcycingSteps=0;bool checkpointStep=true;
 double g0=1.5,dt[1]={.1};
 std::vector<std::string>exports={};
 void writeToFile(const char*n,double,Mesh*,const std::vector<std::tuple<std::string,std::vector<Mem>>>&){exports.push_back(n);}''')
parent=r'''
namespace bubbleColumn {
struct P{double rhoGas=1.2,rhoLiquid=998.2,virtualMassEnabled=1,virtualMassCoefficient=.5;
 double gasPressureEnabled=1,alphaFloor=1e-6,gravity[3]={0,0,-9.81},subtractUgDiffusion[3]={0,0,0};}p;
Mem o_dragLambda{4},o_gradP{12},o_ul{12},o_liquidAcceleration{12},o_divGasStress{12};
Mem o_ugDiffusionDivergence{12},o_convectionEXT{16},o_convectionSample{16},o_scalarFieldOffsetScan{1};
}
#include "bubbleColumnUGTerms.hpp"
int main(){
 using namespace bubbleColumn::ugTerms;
 Elliptic e;scalar.ellipticSolver={&e,&e,&e,&e};
 for(int i=0;i<2;++i){mesh.o_LMM[i]=1;mesh.o_invLMM[i]=1;}
 scalar.o_coeffBDF[0]=2;scalar.o_coeffBDF[1]=-.5;
 for(int i=0;i<48;++i)scalar.o_S[i]=.01*(i+1);
 for(int i=0;i<16;++i)scalar.o_EXT[i]=-.03*i;
 configure();nrs->tstep=1;
 std::vector<double>old(48);scalar.o_S.copyTo(old.data());
 captureSource(0.);
 auto hist=field(TimeHistory);
 for(int c=0;c<3;++c)for(int i=0;i<2;++i)
  assert(std::abs(hist[c*4+i]-(2*old[(c+1)*4+i]-.5*old[16+(c+1)*4+i])/.1)<1e-12);
 // Simulate native solution-history shift/update; captured BDF history must survive.
 for(int i=0;i<48;++i)scalar.o_S[i]=99+i;
 std::vector<double>physical(48);scalar.o_S.copyTo(physical.data());
 captureSolved(.1);capturePostprocessed(.1);
 for(int i=0;i<48;++i)assert(scalar.o_S[i]==physical[i]);
 auto pressure=field(Pressure);pressure[0]=-7;pressure[1]=2;
 writeCheckpoint(.1);assert(nrs->exports.size()==22);
 for(int t=0;t<Count;++t)assert(nrs->exports[t]==prefixes[t]);
 assert(nrs->exports[20]=="gmk"&&nrs->exports[21]=="gcf");
 csv.close();
 std::ifstream report("bubbleColumn_ug_terms.csv");std::string line;int rows=0;bool pressureRow=false;
 while(std::getline(report,line)){++rows;if(line.find("checkpoint,pressure,x,-7.")!=std::string::npos)pressureRow=true;}
 assert(rows==1+3*(Count+Count)&&pressureRow); // 20 source/solved rows + 20 checkpoint rows, xyz.
 enabled=0;nrs->exports.clear();captureSource(.1);captureSolved(.2);capturePostprocessed(.2);writeCheckpoint(.2);
 assert(nrs->exports.empty());
 puts("Whole production C++ diagnostics header: allocation, pre-lag BDF snapshot, read-only source/solved callbacks, global range/CSV rows, all 20 checkpoint vectors + mask/source coefficients, and disabled path passed.");
}
'''
with tempfile.TemporaryDirectory() as d:
 p=Path(d);(p/'host.cpp').write_text(prefix+parent)
 subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-Wno-unused-parameter','-I',str(case),str(p/'host.cpp'),'-o',str(p/'host')],check=True)
 r=subprocess.run([str(p/'host')],cwd=p,capture_output=True,text=True)
 if r.returncode:print(r.stdout+r.stderr)
 assert r.returncode==0
 print(r.stdout.splitlines()[-1])
# Verify observations are placed around real scalar solving/postprocessing.
order=(case/'bubbleColumnOneThrough.hpp').read_text()
post=extract(order,'inline void postScalar(')
assert post.index('captureSolved')<post.index('postProcessAlpha')<post.index('postProcessGasFlux')<post.index('capturePostprocessed')
begin=extract(order,'inline void beginStep(')
assert begin.index('addExplicitSources')<begin.index('captureSource')
udf=(case/'bubbleColumn.udf').read_text()
assert 'ugTerms::registerKernels(kernelInfo)' in udf and 'ugTerms::writeCheckpoint(time)' in udf
print('Source/solved/postprocessing capture timing and checkpoint/kernel registration routing passed.')


# Build all new kernels with the real OCCA Serial backend when available.
import os
candidates=[Path(os.environ.get('OCCA_DIR','/nonexistent')),case.parents[2]/'occa-cub-parser',case.parents[1]/'3rd_party/occa']
occa=next((p for p in candidates if (p/'include/occa.hpp').exists() and (p/'lib/libocca.so').exists()),None)
if occa:
 real=r'''
#include <occa.hpp>
#include <cassert>
#include <cmath>
#include <limits>
int main(int argc,char**argv){
 occa::device device;device.setup(std::string("{mode:'Serial'}"));
 occa::properties p;p["defines/dfloat"]="double";p["defines/dlong"]="long";p["defines/p_blockSize"]=256;
 for(const char*name:{"ugCaptureTerms","ugFinishTerms","ugPostprocessTerm","ugTermFreeMask","ugTermNonfinite"})
  device.buildKernel(argv[1],name,p);
 auto finite=device.buildKernel(argv[1],"ugTermNonfinite",p);
 auto mask=device.buildKernel(argv[1],"ugTermFreeMask",p);
 double x[15]={},flag[5]={},m[5]={1,1,1,1,1};long ids[1]={1};
 x[11]=std::numeric_limits<double>::infinity();
 auto ox=device.malloc<double>(15,x),of=device.malloc<double>(5,flag),om=device.malloc<double>(5,m);
 auto oi=device.malloc<long>(1,ids);
 finite(long(3),long(5),ox,of);of.copyTo(flag);assert(flag[0]==0&&flag[1]==1&&flag[2]==0);
 mask(long(1),oi,om);om.copyTo(m);assert(m[0]==1&&m[1]==0&&m[2]==1&&m[3]==1);
}
'''
 with tempfile.TemporaryDirectory() as d:
  p=Path(d);(p/'real.cpp').write_text(real)
  subprocess.run(['g++','-std=c++17',str(p/'real.cpp'),'-I'+str(occa/'include'),'-L'+str(occa/'lib'),'-Wl,-rpath,'+str(occa/'lib'),'-locca','-o',str(p/'real')],check=True)
  subprocess.run([str(p/'real'),str(case/'bubbleColumnUGTerms.okl')],check=True,env=dict(os.environ,OCCA_CACHE_DIR=str(p/'cache')))
  with (p/'terms.cu').open('w') as output:
   subprocess.run([str(occa/'bin/occa'),'translate','-m','CUDA','-D','p_blockSize=256','-D','dfloat=double','-D','dlong=long',str(case/'bubbleColumnUGTerms.okl')],stdout=output,check=True)
 print('All five diagnostic kernels built with OCCA Serial; real nonfinite/mask execution and CUDA translation passed.')

