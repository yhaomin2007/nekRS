"""Exercise actual startup guards for the coupled-VM model assumptions."""
from pathlib import Path
import subprocess
import tempfile

case=Path(__file__).resolve().parents[1]
terms=(case/'bubbleColumn2Terms.hpp').read_text()
udf=(case/'bubbleColumn2Legacy.udf').read_text()
params=terms[terms.index('struct Parameters {'):terms.index('\nstatic Parameters p;')]
start=udf.index('  nekrsCheck(p.coupledVirtualMassEnabled\n')
end=udf.index('  platform->par->extract("casedata", "stabilitymonitorenabled"',start)
guard=udf[start:end]
cpp=r'''
#include <cassert>
#include <stdexcept>
using dfloat=double;
''' + params + r'''
Parameters p{};
struct Comm{int mpiComm(){return 0;}};
struct Env{Comm comm;}env;auto platform=&env;
#define EXIT_FAILURE 1
#define nekrsCheck(test,...) do{if(test)throw std::runtime_error("incompatible controls");}while(0)
void validate(){
''' + guard + r'''
}
void defaults(){
 p=Parameters{};
 p.coupledVirtualMassEnabled=1;p.mixtureVirtualMassEnabled=1;
 p.mixtureForcingEnabled=1;p.driftStressEnabled=1;p.mixtureDragEnabled=1;
 p.gasPressureEnabled=1;
}
int main(){
 defaults();validate();
 // Reject combinations which break the common pressure/paired-force model.
 for(int j=0;j<12;j++){
  defaults();
  if(j==0)p.frozenGasVelocityEnabled=1;
  if(j==1)p.mixtureVirtualMassEnabled=0;
  if(j==2)p.mixtureForcingEnabled=0;
  if(j==3)p.driftStressEnabled=0;
  if(j==4)p.mixtureDragEnabled=0;
  if(j==5)p.mixtureDragRampEnabled=1;
  if(j==6)p.gasPressureEnabled=0;
  if(j==7)p.gasPressureGradientMethod=2;
  if(j==8)p.gasPressureGradientFilterEnabled=1;
  if(j==9)p.gasPressureInletMaskEnabled=1;
  if(j==10)p.gasVelocityClipEnabled=1;
  if(j==11)p.volumeVelocityClipEnabled=1;
  bool rejected=false;try{validate();}catch(const std::runtime_error&){rejected=true;}
  assert(rejected);
  p.coupledVirtualMassEnabled=0;validate(); // Legacy diagnostics remain usable.
 }
}
'''
with tempfile.TemporaryDirectory() as temp:
    path=Path(temp);(path/'check.cpp').write_text(cpp)
    subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror',str(path/'check.cpp'),'-o',str(path/'check')],check=True)
    subprocess.run([str(path/'check')],check=True)
# Inspect host scheduling around the tested physical kernels.
pointwise=terms.split('inline void evaluatePointwiseTerms()')[1].split('inline void evaluateNativeAlphaAdvection')[0]
assert pointwise.index('buildEquationTermsKernel(')<pointwise.index('configureCoupledVirtualMassKernel(')
assert pointwise.index('configureCoupledVirtualMassKernel(')<pointwise.index('addCoupledVMStressKernel(')
execute=udf.split('void UDF_ExecuteStep')[1]
assert '&& !bubbleColumn2::p.coupledVirtualMassEnabled' in execute
print('Coupled VM controls: actual startup validation accepts paired mode, rejects incompatible diagnostics, preserves legacy mode and bypasses lagged VM history.')
