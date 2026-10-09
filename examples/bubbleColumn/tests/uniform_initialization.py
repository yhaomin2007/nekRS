"""Compile actual uniform initialization kernel as serial C++."""
from pathlib import Path
import re
import subprocess
import tempfile
root=Path(__file__).resolve().parents[1]
s=(root/'bubbleColumnEquations.okl').read_text()
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
 const double rl=998.2,rg=1.2,rho=(1-a)*rl+a*rg;
 initializeUniform(4,6,rl,rg,a,ug,alpha,x,y,z,u);
 for(int i=0;i<4;i++) {
 assert(alpha[i]==a && x[i]==0 && y[i]==0 && z[i]==ug);
 assert(u[i]==0 && u[6+i]==0 && u[12+i]==rg*a*z[i]/rho);
 assert(std::abs((rho*u[12+i]-rg*a*z[i])/((1-a)*rl))<1e-14);
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
udf=(root/'bubbleColumn.udf').read_text()
assert 'if (platform->options.getArgs("RESTART FILE NAME").empty())' in udf
assert 'if (p.initialConditionMode == 1)' in udf
assert 'initializePlumeKernel' in udf
assert "nrs->userDivergence = &bubbleColumn::updateDivergence;" in udf
print('Uniform alpha/UG/mixture initialization, stationary liquid, signed velocity and padding checks passed.')
