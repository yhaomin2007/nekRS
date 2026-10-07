"""Compile actual uniform initialization kernel as serial C++."""
from pathlib import Path
import re
import subprocess
import tempfile
root=Path(__file__).resolve().parents[1]
s=(root/'bubbleColumn2Equations.okl').read_text()
k='void initializeUniform'+s.split('@kernel void initializeUniform')[1].split('@kernel')[0]
k=k.replace('@ restrict ', '')
k=re.sub(r'; @tile\(p_blockSize, @outer, @inner\)', '', k)
cpp='''#include <cassert>
#include <cmath>
using dfloat=double;using dlong=long;
'''+k+'''
int main(){
for(double a: {0.,.02,.05}) for(double ug: {0.,.1,-.1}) {
 double alpha[6],x[6],y[6],z[6],u[18];
 for(int i=0;i<6;i++)alpha[i]=x[i]=y[i]=z[i]=99;
 for(int i=0;i<18;i++)u[i]=99;
 initializeUniform(4,6,a,ug,alpha,x,y,z,u);
 for(int i=0;i<4;i++) {
 assert(alpha[i]==a && x[i]==0 && y[i]==0 && z[i]==ug);
 assert(u[i]==0 && u[6+i]==0 && u[12+i]==a*z[i]);
 assert((u[12+i]-a*z[i])/(1-a)==0);
 }
 assert(alpha[4]==99 && z[5]==99 && u[16]==99);
}
}
'''
cpp=cpp.replace('#include <cassert>', '#include <cassert>\n#include <initializer_list>')
with tempfile.TemporaryDirectory() as t:
 f=Path(t)/'check.cpp';f.write_text(cpp)
 subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror',str(f),'-o',str(Path(t)/'check')],check=True)
 subprocess.run([str(Path(t)/'check')],check=True)
udf=(root/'bubbleColumn2.udf').read_text()
assert 'if (platform->options.getArgs("RESTART FILE NAME").empty())' in udf
assert 'if (p.uniformInitialCondition == 1)' in udf
print('Uniform alpha/UG/mixture initialization, stationary liquid, signed velocity and padding checks passed.')

k='void initializePlume'+s.split('@kernel void initializePlume')[1].split('@kernel')[0]
k=k.replace('@ restrict ', '')
k=re.sub(r'; @tile\(p_blockSize, @outer, @inner\)', '', k)
cpp='''#include <cassert>
#include <cmath>
using dfloat=double;using dlong=long;
'''+k+'''
int main(){
 double height[6]={-.1,0,.05,.1,.2,99};
 double alpha[6],qx[6],qy[6],qz[6],u[18];
 for(int i=0;i<6;i++)alpha[i]=qx[i]=qy[i]=qz[i]=99;
 for(double &v:u)v=99;
 initializePlume(5,6,.05,.3,.05,.01,height,alpha,qx,qy,qz,u);
 for(int i=0;i<5;i++){
  double profile=.5*(1-std::tanh((height[i]-.05)/.01));
  assert(alpha[i]==.05*profile && qx[i]==0 && qy[i]==0);
  assert(std::abs(qz[i]-.3*profile)<1e-16);
  assert(u[i]==0 && u[6+i]==0 && u[12+i]==alpha[i]*qz[i]);
  assert((u[12+i]-alpha[i]*qz[i])/(1-alpha[i])==0);
  if(i)assert(alpha[i]<=alpha[i-1]);
 }
 assert(alpha[5]==99 && qz[5]==99 && u[17]==99);
 assert(std::abs(alpha[2]-.025)<1e-15);
}
'''
with tempfile.TemporaryDirectory() as t:
 f=Path(t)/'plume.cpp';f.write_text(cpp)
 subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror',str(f),'-o',str(Path(t)/'plume')],check=True)
 subprocess.run([str(Path(t)/'plume')],check=True)
print('Plume profile, consistent UG/mixture, stationary liquid and padding checks passed.')
