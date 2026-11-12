"""Active simplified Method2: actual new kernels and native BDF/EXT forcing."""
from pathlib import Path
import re,subprocess,tempfile
case=Path(__file__).resolve().parents[1]
repo=case.parents[1]
native=(repo/'src/solver/scalar/kernels/sumMakef.okl').read_text()
new=(case/'bubbleColumn2Native.okl').read_text()
fluid=(repo/'src/solver/fluid/kernels/sumMakef.okl').read_text().replace('sumMakef(', 'fluidSumMakef(')
def serial(s):
 s=s.replace('@kernel ','').replace('@ restrict ','')
 s=re.sub(r'^#pragma.*$','',s,flags=re.M)
 return re.sub(r';\s*@tile\(p_blockSize,\s*@outer,\s*@inner\)','',s)
header=(case/'bubbleColumn2Native.hpp').read_text()
a=header.index('inline void pinGasDragHistory(');b=header.index('{',a)+1;depth=1
while depth:depth+=(header[b]=='{')-(header[b]=='}');b+=1
pin=header[a:b]
cpp=r'''
#include <cassert>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
using dlong=long;using dfloat=double;
#define p_nEXT 3
#define p_nBDF 2
#define p_SUBCYCLING 0
#define p_MovingMesh 0
#define p_ADVECTION 1
bool close(double a,double b){return std::abs(a-b)<1e-10*(1+std::abs(b));}
'''+serial(new)+serial(native)+serial(fluid)+r'''
struct Mem{
 double* p=nullptr;long count=0;
 long size()const{return count;}Mem slice(long off,long n)const{return {p+off,n};}
 void copyFrom(Mem x,long n){std::copy_n(x.p,n,p);}
};
struct Scalar{Mem o_coeffEXT,o_EXT;long fieldOffsetSum=12;std::array<long,4>fieldOffsetScan{0,3,6,9};}scalar;
struct Mesh{long Nlocal=2;}mesh;
struct Nrs{int tstep=0;long fieldOffset=3;Scalar* scalar=&::scalar;Mesh* meshV=&mesh;}app;auto nrs=&app;
struct LA{void axpby(long n,double a,Mem x,double b,Mem y){for(long i=0;i<n;++i)y.p[i]=a*x.p[i]+b*y.p[i];}}la;
struct Env{LA* linAlg=&la;}env;auto platform=&env;
Mem o_dragLoad,o_previousDragLoad;
'''+pin+r'''
int main(){
 constexpr long N=2,O=3,S=12,V=9;
 double hist[27],ul[V]={},pred[V],lambda[O],load[V],source[V],previous[V]={};
 double FU[36]={},ADV[36]={},U[24]={},rho[S],mass[3]={2,5,999},BF[S]={};
 double alpha[3]={.03,.12,888},grad[27]={},comp[3]={0,0,777};
 std::fill_n(hist,27,999);std::fill_n(pred,V,666);std::fill_n(load,V,555);std::fill_n(rho,S,1);
 std::vector<std::array<double,V>> samples;
 std::vector<std::array<double,S>> bases,advs;
 for(int step=1;step<=6;++step){
  std::array<double,3>b=step==1?std::array<double,3>{1,0,0}:step==2?std::array<double,3>{2,-1,0}:
      step%2?std::array<double,3>{3,-3,1}:std::array<double,3>{2.8,-2.4,.6};
  double cb[2]={step==1?1.:1.6,step==1?0.:-.225};
  for(int c=0;c<3;++c)for(int n=0;n<N;++n)ul[n+c*O]=.03*step*step+.02*c+.01*n;
  std::array<double,V> sample{};std::copy_n(ul,V,sample.begin());samples.insert(samples.begin(),sample);
  while(samples.size()<3)samples.push_back(sample);
  liquidEXT(N,O,3,step==1,step!=1,b.data(),ul,hist,pred);
  for(int c=0;c<3;++c)for(int n=0;n<N;++n){
    double expect=0;for(int j=0;j<3;++j)expect+=b[j]*samples[j][n+c*O];assert(close(pred[n+c*O],expect));
  }
  // Same-step refresh must not shift history again.
  auto snapshot=std::vector<double>(hist,hist+27);
  liquidEXT(N,O,3,0,0,b.data(),ul,hist,pred);
  for(int i=0;i<27;++i)assert(hist[i]==snapshot[i]);
  for(int n=0;n<N;++n){lambda[n]=.7+.3*step+.1*n;grad[n]=.1*step;grad[n+4*O]=-.02;grad[n+8*O]=.03;}
  alphaCompression(N,O,alpha,grad,comp);
  std::array<double,S> base{},adv{};
  for(int n=0;n<N;++n){base[n]=comp[n];assert(close(comp[n],-alpha[n]*(.1*step+.01)));}
  for(int c=0;c<3;++c)for(int n=0;n<N;++n){
    int id=n+c*O;base[n+(c+1)*O]=.4+.01*step*step+.2*c;
    source[id]=base[n+(c+1)*O]+lambda[n]*ul[id];
  }
  replaceGasDragLoad(N,O,lambda,ul,pred,source,load);
  bases.insert(bases.begin(),base);while(bases.size()<3)bases.push_back(base);
  for(int f=0;f<4;++f)for(int n=0;n<N;++n){
    int id=n+f*O;adv[id]=.2*step*step+.1*f+.03*n;
    ADV[id]=adv[id];U[id]=.1+step*.01+id*.02;U[id+S]=.08+id*.01;
    FU[id]=f==0?comp[n]:source[n+(f-1)*O];
  }
  advs.insert(advs.begin(),adv);while(advs.size()<3)advs.push_back(adv);
  // Execute the actual header helper, then compare to an independent RHS.
  scalar.o_coeffEXT={b.data(),3};scalar.o_EXT={FU,36};
  o_dragLoad={load,V};o_previousDragLoad={previous,V};nrs->tstep=step;
  pinGasDragHistory();
  for(int f=0;f<4;++f){
    sumMakef(N,mass,10,b.data(),cb,f*O,S,O,rho,U,ADV,FU,BF);
    for(int n=0;n<N;++n){
      double expected=0;
      for(int j=0;j<3;++j){
        const auto &bj=step>j?bases[j]:bases[0];
        const auto &aj=step>j?advs[j]:advs[0];
        // Startup unused coefficients are zero; later all slots are real samples.
        expected+=b[j]*(bj[n+f*O]-aj[n+f*O]);
      }
      if(f)expected+=load[n+(f-1)*O];
      expected=mass[n]*(expected+10*(cb[0]*U[n+f*O]+cb[1]*U[n+f*O+S]));
      assert(close(BF[n+f*O],expected));
    }
  }
  // Full UM drag must keep its nonlinear force history, not the latest load.
  double um[3]={double(step*step),double((step-1)*(step-1)),double((step-2)*(step-2))};
  double ext=b[0]*um[0]+b[1]*um[1]+b[2]*um[2];
  if(step>=3)assert(!close(ext,um[0]));
  double umFU[27]={},umADV[27]={},umU[27]={},umBF[9]={},zeroBDF[2]={};
  for(int j=0;j<3;++j)for(int c=0;c<3;++c)for(int n=0;n<N;++n)umFU[j*V+c*O+n]=um[j]*(c+1);
  fluidSumMakef(N,mass,10,b.data(),zeroBDF,O,V,O,rho,umU,umADV,umFU,umBF);
  for(int c=0;c<3;++c)for(int n=0;n<N;++n)assert(close(umBF[c*O+n],mass[n]*ext*(c+1)));
  // Native history shifts once, after RHS formation.
  for(int j=2;j>0;--j)for(int i=0;i<S;++i){FU[j*S+i]=FU[(j-1)*S+i];ADV[j*S+i]=ADV[(j-1)*S+i];}
 }
 assert(comp[2]==777);
 for(int c=0;c<3;++c){assert(pred[c*O+2]==666);assert(load[c*O+2]==555);for(int j=0;j<3;++j)assert(hist[j*V+c*O+2]==999);}
}
'''
with tempfile.TemporaryDirectory() as t:
 p=Path(t)/'native.cpp';p.write_text(cpp)
 subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-Wno-unused-parameter',str(p),'-o',str(p.with_suffix(''))],check=True)
 subprocess.run([str(p.with_suffix(''))],check=True)
h=(case/'bubbleColumn2Native.hpp').read_text();u=(case/'bubbleColumn2.udf').read_text();par=(case/'bubbleColumn2.par').read_text()
assert 'nrs->scalar->o_U=o_ug' in h and 'nrs->scalar->o_relUrst=o_gasUrst' in h
assert 'computeGasUrst();' in h and 'o_scalarCoeffEXT' not in h
for forbidden in ('correctGasPressure','freezeDragHistory','explicitFilterApply','buildOutletRamp','gasInterpolate','o_scalarConvectionHistory'):
 assert forbidden not in h+u,forbidden
assert 'cubaturePolynomialOrder =' not in par
assert 'virtualMassEnabled = 1' in par
assert 'std::round(1.5*(N+1)-1)-1' in u
print('Active native Method2: actual compression/UL-EXT/drag kernels, native BDF/EXT, no double UL prediction, changing coefficients, repeats, padding and transport routing passed.')
# Check every explicit GPU launch against the actual OKL signature. OCCA's
# variadic host API alone cannot diagnose a wrong launch argument count.
physics=(case/'bubbleColumn2Equations.okl').read_text()
def function(text,marker):
 a=text.index(marker);b=text.index('{',a)+1;depth=1
 while depth:depth+=(text[b]=='{')-(text[b]=='}');b+=1
 return text[a:b]
def argcount(text):
 depth=0;count=1
 for ch in text:
  if ch in '([{':depth+=1
  elif ch in ')]}':depth-=1
  elif ch==',' and depth==0:count+=1
 return count
mapping={'reconstructGasVelocityKernel':'reconstructGasVelocity','buildGasFluxKernel':'buildGasFlux',
 'buildLiquidVelocityKernel':'buildLiquidVelocity','buildEquationTermsKernel':'buildEquationTerms',
 'addGasStressKernel':'addGasStress','buildMixtureForceKernel':'buildMixtureForce',
 'buildMixtureStressCorrectionKernel':'buildMixtureStressCorrection','addMixtureStressCorrectionKernel':'addMixtureStressCorrection',
 'updateVirtualMassHistoryKernel':'updateVirtualMassHistory','initializePlumeKernel':'initializePlume',
 'clipAlphaKernel':'clipAlpha','compressionKernel':'alphaCompression','liquidEXTKernel':'liquidEXT','replaceDragKernel':'replaceGasDragLoad'}
for host,kernel in mapping.items():
 definition=function(physics+new,'@kernel void '+kernel+'(')
 expected=argcount(definition.split('(',1)[1].split(')\n{',1)[0])
 for call in re.finditer(r'\b'+host+r'\(',h+u):
  start=call.end();end=start;depth=1
  while depth:depth+=((h+u)[end]=='(')-((h+u)[end]==')');end+=1
  assert argcount((h+u)[start:end-1])==expected,(host,expected)
print('All active host-to-OKL launch argument counts match actual kernel signatures.')
