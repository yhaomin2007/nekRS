"""Check the active four-scalar Method1 path and reconstructed liquid velocity."""
from pathlib import Path
import configparser
import re
import subprocess
import tempfile
case = Path(__file__).resolve().parents[1]
par = configparser.ConfigParser(inline_comment_prefixes=('#',))
par.read_string('[ROOT]\n' + (case/'bubbleColumn.par').read_text())
assert [s.strip() for s in par['GENERAL']['scalars'].split(',')] == ['ALPHA','UGX','UGY','UGZ']
assert not any(s.startswith('SCALAR UL') for s in par.sections())
udf = (case/'bubbleColumn.udf').read_text()
host = (case/'bubbleColumnPhaseMomentum.hpp').read_text()
terms = (case/'bubbleColumnTerms.hpp').read_text()
assert 'phaseMomentum::correct()' not in udf
assert 'phaseMomentum::registerKernels' not in udf
assert 'nrs->Nscalar!=4' in host
assert 'predictorKernel' not in host and 'liquidNames' not in host
assert 'scalar->solve(time,1)' in host and 'fluid->solve(' not in udf+host
assert 'updateVirtualMassHistory();' in udf
reconstruct = terms.split('inline void reconstructLiquidVelocity()')[1].split('inline void evaluateGasCubatureTransport')[0]
assert 'buildLiquidVelocityKernel' in reconstruct and 'o_solution(names' not in reconstruct
source = (case/'bubbleColumnEquations.okl').read_text()
kernel = source.split('@kernel void buildLiquidVelocity')[1].split('@kernel void updateVirtualMassHistory')[0]
kernel = 'void buildLiquidVelocity' + kernel.replace('@ restrict ', '')
kernel = re.sub(r';\s*@tile\(p_blockSize,\s*@outer,\s*@inner\)', '', kernel)
cpp = '#include <cassert>\n#include <cmath>\nusing dlong=int; using dfloat=double;\n' + kernel + r'''
int main(){
 constexpr int N=4,O=6;
 double a[O]={0,.016667,.5,1},ug[3*O]={},um[3*O]={},q[3*O]={},ul[3*O];
 for(int i=0;i<3*O;++i){ug[i]=.3+.01*i;um[i]=.01-.002*i;ul[i]=123.;}
 for(int c=0;c<3;++c)for(int i=0;i<N;++i)q[c*O+i]=a[i]*ug[c*O+i];
 buildLiquidVelocity(N,O,998.2,1.2,1e-6,a,um,q,ul);
 for(int c=0;c<3;++c)for(int i=0;i<N;++i){
   int k=i+c*O;double rho=(1-a[i])*998.2+a[i]*1.2;
   assert(std::isfinite(ul[k]));
   if(a[i]<1)assert(std::abs((1-a[i])*998.2*ul[k]+a[i]*1.2*ug[k]-rho*um[k])<1e-12);
   if(i==0)assert(ul[k]==um[k]);
 }
 for(int c=0;c<3;++c)assert(ul[c*O+N]==123.);
 // Fresh-start plume helper initializes UM for UL=0.
 for(int c=0;c<3;++c)for(int i=0;i<N-1;++i){
   int k=i+c*O;double rho=(1-a[i])*998.2+a[i]*1.2;um[k]=a[i]*1.2*ug[k]/rho;
 }
 buildLiquidVelocity(N,O,998.2,1.2,1e-6,a,um,q,ul);
 for(int c=0;c<3;++c)for(int i=0;i<N-1;++i)assert(std::abs(ul[c*O+i])<1e-15);
}
'''
with tempfile.TemporaryDirectory() as t:
    p=Path(t);(p/'test.cpp').write_text(cpp)
    subprocess.run(['g++','-std=c++17',str(p/'test.cpp'),'-o',str(p/'test')],check=True)
    subprocess.run([str(p/'test')],check=True)
print('Four-scalar Method1: no UL solve/correction, active liquid-history update, actual density-weighted reconstruction, dilute/full-phase safety, stationary-liquid initialization and padding passed.')
