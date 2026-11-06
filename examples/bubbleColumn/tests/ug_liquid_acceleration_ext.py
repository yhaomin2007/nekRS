"""Run production liquid-acceleration EXT helper and history on CPU arrays."""
from pathlib import Path
import ast, subprocess, tempfile
case=Path(__file__).resolve().parents[1]
a=ast.parse((case/'tests/time_extrapolation.py').read_text())
prefix=next(ast.literal_eval(n.value) for n in a.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='prefix' for t in n.targets))
terms=(case/'bubbleColumnTerms.hpp').read_text()
a=terms.index('inline void prepareVirtualMassAcceleration()'); b=terms.index('inline void addExplicitSources',a)
source=prefix+'\n#include "bubbleColumnTimeExtrapolation.hpp"\nnamespace bubbleColumn {\nstruct {double virtualMassEnabled=1;}p;\nExplicitHistory liquidAccelerationHistory;\nMem o_liquidAcceleration{12},o_liquidAccelerationEXT{12};\n'+terms[a:b]+r'''
}
int main(){using namespace bubbleColumn;
 const double W[4][3]={{1,0,0},{2,-1,0},{3,-3,1},{2.625,-3.375,1.75}};
 const double raw[4]={0,2,5,9};
 for(int s=0;s<4;++s){nrs->tstep=s+1;
  for(int h=0;h<3;++h)fluid.o_coeffEXT[h]=W[s][h];
  for(int c=0;c<3;++c)for(int i=0;i<2;++i)o_liquidAcceleration[c*4+i]=(c+1)*raw[s]+i;
  prepareVirtualMassAcceleration();
  for(int c=0;c<3;++c)for(int i=0;i<2;++i){double expected=0;
   for(int h=0;h<3;++h)expected+=W[s][h]*((c+1)*raw[std::max(0,s-h)]+i);
   assert(std::abs(o_liquidAccelerationEXT[c*4+i]-expected)<1e-12);
  }
  auto samples=liquidAccelerationHistory.samples;std::vector<double> old(samples.size());samples.copyTo(old.data());
  prepareVirtualMassAcceleration();for(int i=0;i<samples.size();++i)assert(samples[i]==old[i]);
 }
 p.virtualMassEnabled=0;prepareVirtualMassAcceleration();
 for(int i=0;i<12;++i)assert(o_liquidAccelerationEXT[i]==0);
}
'''
with tempfile.TemporaryDirectory() as d:
 p=Path(d);(p/'test.cpp').write_text(source)
 subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-I',str(case),str(p/'test.cpp'),'-o',str(p/'test')],check=True)
 subprocess.run([str(p/'test')],check=True)
assert 'assemblingExplicitSources ? o_liquidAccelerationEXT : o_liquidAcceleration' in terms
assembly=terms[terms.index('inline void addExplicitSources'):]
assert assembly.index('prepareVirtualMassAcceleration();')<assembly.index('evaluatePointwiseTerms();')
assert 'implicitGasDrag' not in terms
assert 'userImplicitLinearTerm' not in (case/'bubbleColumn.udf').read_text()
print('Production VM acceleration EXT1/2/3, variable-step weights, startup zero, repeated source, disabled VM, and explicit-drag registration checks passed.')
