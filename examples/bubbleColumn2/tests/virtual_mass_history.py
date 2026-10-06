"""Compile actual VM difference helper/kernel; verify variable-dt history and gating."""
from pathlib import Path
import re, subprocess, tempfile
root=Path(__file__).resolve().parents[1]
terms=(root/'bubbleColumn2Terms.hpp').read_text()
eq=(root/'bubbleColumn2Equations.okl').read_text()
helper=terms[terms.index('inline void virtualMassTimeCoefficients'):terms.index('inline void updateVirtualMassHistory()')]
kernel='void updateVirtualMassHistory'+eq.split('@kernel void updateVirtualMassHistory')[1].split('@kernel')[0]
kernel=kernel.replace('@ restrict ', '')
kernel=re.sub(r'; @tile\(p_blockSize, @outer, @inner\)', '',kernel)
cpp=r'''
#include <cassert>
#include <cmath>
#include <initializer_list>
using dfloat=double;using dlong=long;
'''+helper+kernel+r'''
void check(double h,double k,bool second){
 double c0,c1,c2;virtualMassTimeCoefficients(h,k,second,c0,c1,c2);
 assert(std::abs(c0+c1+c2)<1e-12);
 const int off=3,N=2;
 double ug[9],ul[9],up[9],lp[9],up2[9],lp2[9],gu[27]={},gl[27]={},a[9];
 double t=1.3,tp=t-h,tpp=tp-k;
 for(int i=0;i<9;i++){
  ug[i]=t*t;up[i]=tp*tp;up2[i]=tpp*tpp;
  ul[i]=3*t*t;lp[i]=3*tp*tp;lp2[i]=3*tpp*tpp;a[i]=99;
 }
 // Add prescribed nonzero gradient for x component; convection stays current.
 for(int n=0;n<N;n++){gl[n]=.4;gu[n]=.2;}
 updateVirtualMassHistory(N,off,c0,c1,c2,ug,ul,lp,up,lp2,up2,gl,gu,a);
 for(int i=0;i<3;i++)for(int n=0;n<N;n++){
  int id=n+i*off;
  double derivative=second?4*t:4*t-2*h;
  double convection=i==0?.4*3*t*t-.2*t*t:0;
  assert(std::abs(a[id]-derivative-convection)<1e-12);
  assert(lp2[id]==3*tp*tp && up2[id]==tp*tp);
  assert(lp[id]==3*t*t && up[id]==t*t);
 }
 for(int i=0;i<3;i++)assert(a[i*off+2]==99);
}
int main(){
 for(double h:{.01,.1,.2})for(double k:{.02,.1,.3}){
  check(h,k,true);check(h,k,false);
 }
 // First completed step must be first order even with requested order 2.
 double c0,c1,c2;virtualMassTimeCoefficients(.1,0,false,c0,c1,c2);
 assert(c0==10 && c1==-10 && c2==0);
 virtualMassTimeCoefficients(.1,.1,true,c0,c1,c2);
 assert(std::abs(c0-15)<1e-12 && std::abs(c1+20)<1e-12 && std::abs(c2-5)<1e-12);
}
'''
with tempfile.TemporaryDirectory() as t:
 f=Path(t)/'vm.cpp';f.write_text(cpp)
 subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror',str(f),'-o',str(Path(t)/'vm')],check=True)
 subprocess.run([str(Path(t)/'vm')],check=True)
assert 'nrs->tstep > p.virtualMassStartStep ? p.virtualMassEnabled : 0.0,' in terms
assert 'p.virtualMassTimeDerivativeOrder == 2 && virtualMassHistorySamples >= 1' in terms
udf=(root/'bubbleColumn2.udf').read_text()
execute=udf[udf.index('void UDF_ExecuteStep'):]
assert 'virtualMassStartStep' not in execute  # delayed sources must still collect history
assert 'updateVirtualMassHistory();' in execute
# Both VM sources use the single enabled argument from the gated call.
assert 'a * virtualMassEnabled' in eq and 'gasVirtualMassScale = virtualMassEnabled' in eq
print('Actual VM helper/kernel: first/second order, unequal timesteps, quadratic acceleration, convection, history shifts, padding and delayed-source wiring passed.')
