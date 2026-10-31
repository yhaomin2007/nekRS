"""Compile actual inlet weight and UG forcing helper with native vector multiplication."""
from pathlib import Path
import re
import subprocess
import tempfile

case=Path(__file__).resolve().parents[1];repo=case.parents[1]
s=(case/'bubbleColumn2Terms.hpp').read_text()
def function(marker):
 start=s.index(marker);pos=s.index('{',start);depth=1;end=pos+1
 while depth:
  if s[end]=='{':depth+=1
  elif s[end]=='}':depth-=1
  end+=1
 return s[start:end]
weight=function('inline dfloat gasPressureInletRampWeight(')
forcing=function('inline void prepareGasPressureForcing(')
k=(repo/'src/platform/linAlg/kernels/axmyVector.okl').read_text()
k=k.replace('@kernel ','').replace('@ restrict ','')
k=re.sub(r'; @tile\(p_blockSize, @outer, @inner\)','',k)
validation=s[s.index('  nekrsCheck(p.gasPressureInletRegionEnabled'):s.index('  nekrsCheck(!std::isfinite(p.gasPressureGradientFilterWeight)')]
cpp=r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
using dfloat=double;using dlong=long;
#define p_NVec 3
'''+k+r'''
struct Mem{
 std::shared_ptr<std::vector<double>> data;long off=0;
 Mem(long n=0):data(std::make_shared<std::vector<double>>(n,0)){}
 double* ptr()const{return data->data()+off;}
 Mem slice(long start,long)const{auto out=*this;out.off+=start;return out;}
 void copyFrom(Mem x,long n){std::copy(x.ptr(),x.ptr()+n,ptr());}
};
struct LA{
 void axpby(long n,double a,const Mem& x,double b,Mem& y){for(long i=0;i<n;i++)y.ptr()[i]=a*x.ptr()[i]+b*y.ptr()[i];}
 void axmyMany(long n,int fields,long offset,int,double a,Mem x,Mem y){assert(fields==3);axmyVector(n,offset,0,a,x.ptr(),y.ptr());}
} la;
struct Comm{int mpiComm(){return 0;}};
struct Platform{LA* linAlg=&la;Comm comm;} env;auto platform=&env;
struct Mesh{long Nlocal=7;} mesh;
struct Nrs{Mesh* meshV=&mesh;long fieldOffset=9;int tstep=1;} solver;auto nrs=&solver;
struct Params{
 int gasPressureInletRegionEnabled=0,gasPressureInletMaskEnabled=0,gasPressureGradientMethod=0;
 double gasPressureInletZeroEndZ=.01,gasPressureInletRampEndZ=.02;
 double gasPressureGradientFilterEnabled=0,gasPressureGradientFilterWeight=.2;
}p;
Mem o_gradP(27),o_gradPStrong(27),o_gradPWeak(27),o_gradPFilterHistory(27);
Mem o_gasPressureRegionWeight(9),o_gasPressureInletMask(9);
bool pressureGradientFilterInitialized=false;int pressureGradientFilterStep=-1;
#define nekrsCheck(test,...) do{if(test)throw std::runtime_error("invalid region");}while(0)
void validate(){
'''+validation+'}\n'+weight+'\n'+forcing+r'''
bool close(double x,double y){return std::abs(x-y)<1e-12;}
int main(){
 for(double z:{-.1,0.,.01,.0125,.015,.0175,.02,.1})assert(gasPressureInletRampWeight(z)==1);
 p.gasPressureInletRegionEnabled=1;validate();
 assert(gasPressureInletRampWeight(-.1)==0 && gasPressureInletRampWeight(.01)==0);
 assert(close(gasPressureInletRampWeight(.0125),.15625));
 assert(close(gasPressureInletRampWeight(.015),.5));
 assert(close(gasPressureInletRampWeight(.0175),.84375));
 assert(gasPressureInletRampWeight(.02)==1 && gasPressureInletRampWeight(.1)==1);
 double prev=0;for(int j=0;j<=100;j++){double w=gasPressureInletRampWeight(.01+.0001*j);assert(w>=prev-1e-14 && w<=1);prev=w;}
 double dz=1e-8;
 assert(gasPressureInletRampWeight(.01+dz)/dz<.001);
 assert((1-gasPressureInletRampWeight(.02-dz))/dz<.001);
 for(double end:{.01,0.,std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity()}){
  p.gasPressureInletRampEndZ=end;bool rejected=false;try{validate();}catch(const std::runtime_error&){rejected=true;}assert(rejected);
 }
 p.gasPressureInletRampEndZ=.02;p.gasPressureInletRegionEnabled=2;
 bool rejected=false;try{validate();}catch(const std::runtime_error&){rejected=true;}assert(rejected);
 p.gasPressureInletRegionEnabled=1;
 double z[7]={0,.005,.01,.0125,.015,.02,.1};
 for(int i=0;i<9;i++){
  o_gasPressureRegionWeight.ptr()[i]=i<7?gasPressureInletRampWeight(z[i]):1;
  o_gasPressureInletMask.ptr()[i]=i==0?0:1;
 }
 for(int i=0;i<27;i++){o_gradPStrong.ptr()[i]=i+1;o_gradPWeak.ptr()[i]=3*(i+1);}
 for(int method:{0,1})for(int region:{0,1})for(int mask:{0,1}){
  p.gasPressureGradientMethod=method;p.gasPressureInletRegionEnabled=region;p.gasPressureInletMaskEnabled=mask;
  prepareGasPressureForcing();
  for(int c=0;c<3;c++)for(int n=0;n<9;n++){
   int id=c*9+n;double raw=(method==1?3:1)*(id+1);
   double w=(region&&n<7?o_gasPressureRegionWeight.ptr()[n]:1)*(mask&&n<7?o_gasPressureInletMask.ptr()[n]:1);
   assert(close(o_gradP.ptr()[id],raw*w));
  }
  prepareGasPressureForcing(); // A repeated callback must not square the ramp.
  assert(close(o_gradP.ptr()[4],(method==1?15:5)*(region?.5:1)));
 }
 // Preserve unattenuated temporal history for subsequent steps.
 p.gasPressureGradientMethod=0;p.gasPressureInletRegionEnabled=1;p.gasPressureInletMaskEnabled=1;
 p.gasPressureGradientFilterEnabled=1;prepareGasPressureForcing();
 assert(close(o_gradP.ptr()[4],2.5));assert(o_gradPFilterHistory.ptr()[4]==5);
 for(int i=0;i<27;i++)o_gradPStrong.ptr()[i]=2*(i+1);
 prepareGasPressureForcing(); // Same step: no temporal re-filtering.
 assert(close(o_gradP.ptr()[4],2.5));
 nrs->tstep++;prepareGasPressureForcing();
 assert(close(o_gradPFilterHistory.ptr()[4],6));assert(close(o_gradP.ptr()[4],3));
 assert(o_gradP.ptr()[0]==0 && close(o_gradPFilterHistory.ptr()[0],1.2));
 assert(o_gradPStrong.ptr()[4]==10 && o_gradPWeak.ptr()[4]==15);
 prepareGasPressureForcing();assert(close(o_gradP.ptr()[4],3));
}
'''
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp);(p/'check.cpp').write_text(cpp)
 subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-Wno-unknown-pragmas',str(p/'check.cpp'),'-o',str(p/'check')],check=True)
 subprocess.run([str(p/'check')],check=True)
body=s.split('inline void evaluatePointwiseTerms()')[1].split('inline void ')[0]
assert body.index('opSEM::strongGrad(')<body.index('prepareGasPressureForcing();')<body.index('buildEquationTermsKernel(')
assert 'o_liquidAcceleration,\n                           o_gradP,' in body
assert 'o_gradPFilterHistory.resize(3 * offset)' in s
udf=(case/'bubbleColumn2.udf').read_text()
for key in ['gaspressureinletregionenabled','gaspressureinletzeroendz','gaspressureinletrampendz']:assert key in udf
print('UG inlet pressure region: plateau/smoothstep/full recovery, endpoint slopes, validation, strong/weak gradients, legacy mask composition, components/padding, temporal-history isolation and repeated callbacks passed.')
