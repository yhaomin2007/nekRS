"""Execute production LD/lift kernel, verify closure signs, VM scaling and limits."""
from pathlib import Path
import re
import subprocess
import tempfile

case=Path(__file__).resolve().parents[1]
s=(case/'bubbleColumnEquations.okl').read_text().split('@kernel void addDispersionLift')[1]
s='void addDispersionLift'+s.replace('@ restrict ','')
s=re.sub(r';\s*@tile\(p_blockSize,\s*@outer,\s*@inner\)','',s)
prefix='''#include <cassert>
#include <cmath>
#include <algorithm>
#include <initializer_list>
using dlong=int;using dfloat=double;
bool near(double a,double b){return std::abs(a-b)<1e-11*(1+std::abs(a)+std::abs(b));}
'''
main=r'''
int main(){
 constexpr int N=3,O=5;
 const double rl=998.2,rg=1.2,mu=.001,diam=.003,cdis=4.545,cl=.5;
 double a[O]={0,.016667,1},g[3*O]={},l[3*O]={},gradA[3*O]={},gradL[9*O]={},out[3*O];
 for(int c=0;c<3;++c)for(int i=0;i<N;++i){g[c*O+i]=.1*(c+1);l[c*O+i]=-.02*c;gradA[c*O+i]=.3*(c-1);}
 // UL=(z,2x,3y) has curl=(3,1,2); ordering must follow opSEM's vector gradient.
 for(int i=0;i<N;++i){gradL[2*O+i]=1;gradL[3*O+i]=2;gradL[7*O+i]=3;}
 for(double scale:{.1,1.})for(double vm:{0.,.5,2.})for(int ld:{0,1})for(int lift:{0,1}){
  for(int c=0;c<3;++c)for(int i=0;i<N;++i)g[c*O+i]=scale*.1*(c+1);
  std::fill(out,out+3*O,.123);
  addDispersionLift(N,O,rl,rg,mu,diam,vm,ld,cdis,lift,cl,a,g,l,gradA,gradL,out);
  for(int i=0;i<N;++i){
   double ur[3],force[3],cross[3],curl[3]={3,1,2};
   for(int c=0;c<3;++c)ur[c]=g[c*O+i]-l[c*O+i];
   double speed=std::sqrt(ur[0]*ur[0]+ur[1]*ur[1]+ur[2]*ur[2]);
   double re=rl*speed*diam/mu;
   double cd=re<1000?24/re*(1+.15*std::pow(re,.687)):.44;
   double delta=.75*cd*cdis*rl*a[i]*(1-1.166*a[i]+.5*a[i]*a[i])*speed*speed;
   double liftPower=0;
   for(int c=0;c<3;++c){
    int j=(c+1)%3,k=(c+2)%3;
    cross[c]=ur[j]*curl[k]-ur[k]*curl[j];
    double flift=-cl*rl*a[i]*cross[c];
    force[c]=-ld*delta*gradA[c*O+i]+lift*flift;
    double accel=out[c*O+i]-.123;
    assert(std::isfinite(accel));
    if(a[i]>0)assert(near(a[i]*(rg+vm*rl)*accel,force[c]));
    liftPower+=ur[c]*flift;
    if(!ld && !lift)assert(out[c*O+i]==.123);
   }
   assert(near(liftPower,0)); // classic lift is perpendicular to slip
  }
  for(int c=0;c<3;++c)for(int i=N;i<O;++i)assert(out[c*O+i]==.123);
 }
 // Pure positive x alpha gradient: LD gas acceleration must be negative x.
 std::fill(gradA,gradA+3*O,0);for(int i=0;i<N;++i)gradA[i]=2;
 std::fill(out,out+3*O,0);
 addDispersionLift(N,O,rl,rg,mu,diam,.5,1,cdis,0,cl,a,g,l,gradA,gradL,out);
 for(int i=0;i<N;++i){assert(out[i]<0);assert(out[O+i]==0 && out[2*O+i]==0);}
 // Zero slip is exactly finite zero, even with nonzero alpha gradients/vorticity.
 std::copy(l,l+3*O,g);std::fill(out,out+3*O,0);
 addDispersionLift(N,O,rl,rg,mu,diam,.5,1,cdis,1,cl,a,g,l,gradA,gradL,out);
 for(double v:out)assert(v==0);
 // Pure shear UL_z=s*x, vertical slip => negative x gas lift for positive Cl.
 std::fill(l,l+3*O,0);std::fill(g,g+3*O,0);std::fill(gradL,gradL+9*O,0);
 for(int i=0;i<N;++i){g[2*O+i]=.3;gradL[6*O+i]=4;}
 addDispersionLift(N,O,rl,rg,mu,diam,.5,0,cdis,1,cl,a,g,l,gradA,gradL,out);
 for(int i=0;i<N;++i){assert(near(out[i],-cl*rl/(rg+.5*rl)*1.2));assert(out[O+i]==0 && out[2*O+i]==0);}
}
'''
with tempfile.TemporaryDirectory() as t:
 p=Path(t);(p/'test.cpp').write_text(prefix+s+main)
 subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror',str(p/'test.cpp'),'-o',str(p/'test')],check=True)
 subprocess.run([str(p/'test')],check=True)
terms=(case/'bubbleColumnTerms.hpp').read_text()
pointwise=terms.split('inline void evaluatePointwiseTerms()')[1].split('inline void evaluateNativeAlphaAdvection')[0]
assert pointwise.count('addDispersionLiftKernel(')==1
assert pointwise.index('buildEquationTermsKernel(')<pointwise.index('addDispersionLiftKernel(')
assert pointwise.index('strongGradVec(mesh, offset, o_ul, o_gradUl)')<pointwise.index('addDispersionLiftKernel(')
assert 'addDispersionLift' not in terms.split('inline void evaluateMixtureForce()')[1].split('inline void subtractScalarDiffusion')[0]
print('Production P-LD/classic-lift: force signs, SN high/low Re, VM normalization, alpha=0/1, zero slip, perpendicular lift, shear direction, toggles, padded strides and single-add UG-only routing passed.')

import os
import signal
occa=Path(os.environ.get('OCCA_DIR',case.parents[1]/'3rd_party/occa'))
if (occa/'lib/libocca.so').exists():
 probe=subprocess.run([str(occa/'bin/occa'),'version'],capture_output=True)
 if probe.returncode==-signal.SIGILL:
  print('OCCA checks unavailable: installed runtime fails its version probe with SIGILL.')
  raise SystemExit(0)
 probe.check_returncode()
 code=r'''
#include <occa.hpp>
#include <cmath>
#include <cassert>
int main(int argc,char **argv){
 occa::device d;d.setup(std::string("{mode:'Serial'}"));occa::properties p;
 p["defines/dfloat"]="double";p["defines/dlong"]="int";p["defines/p_blockSize"]=256;
 auto k=d.buildKernel(argv[1],"addDispersionLift",p);
 double a[1]={.02},g[3]={0,0,.3},l[3]={},ga[3]={1,2,3},gl[9]={},out[3]={};gl[6]=4;
 auto A=d.malloc<double>(1,a),G=d.malloc<double>(3,g),L=d.malloc<double>(3,l),
      GA=d.malloc<double>(3,ga),GL=d.malloc<double>(9,gl),S=d.malloc<double>(3,out);
 k(1,1,998.2,1.2,.001,.003,.5,1,4.545,1,.5,A,G,L,GA,GL,S);S.copyTo(out);
 double re=998.2*.003*.3/.001,cd=24/re*(1+.15*std::pow(re,.687)),eff=1.2+.5*998.2;
 double ld=.75*cd*4.545*998.2*(1-1.166*.02+.5*.02*.02)*.3*.3/eff;
 assert(std::abs(out[0]-(-ld-.5*998.2/eff*1.2))<1e-12);
 assert(std::abs(out[1]+2*ld)<1e-12 && std::abs(out[2]+3*ld)<1e-12);
}
'''
 with tempfile.TemporaryDirectory() as t:
  p=Path(t);(p/'occa.cpp').write_text(code)
  subprocess.run(['g++','-std=c++17',str(p/'occa.cpp'),'-I'+str(occa/'include'),'-L'+str(occa/'lib'),'-Wl,-rpath,'+str(occa/'lib'),'-locca','-o',str(p/'occa')],check=True)
  subprocess.run([str(p/'occa'),str(case/'bubbleColumnEquations.okl')],check=True,env=dict(os.environ,OCCA_CACHE_DIR=str(p/'cache')))
  with (p/'equations.cu').open('w') as output:
   subprocess.run([str(occa/'bin/occa'),'translate','-m','CUDA','-D','p_blockSize=256','-D','dfloat=double','-D','dlong=int',str(case/'bubbleColumnEquations.okl')],stdout=output,check=True)
 print('Production LD/lift compiled and executed with OCCA Serial; full equations OKL translated for CUDA.')
else:
 print('OCCA checks skipped: set OCCA_DIR to a built OCCA checkout.')
