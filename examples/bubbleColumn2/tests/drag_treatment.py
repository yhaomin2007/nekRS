"""Semi-implicit gas and mixture drag checks; requires g++."""
from pathlib import Path
import re
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
terms = (root / 'bubbleColumn2Terms.hpp').read_text()
udf = (root / 'bubbleColumn2.udf').read_text()
for obsolete in ('frozenDragEnabled', 'mixtureImplicitDragEnabled',
'addMixtureStressAndSplitDrag',
                 'mixtureDragRelaxation', 'relaxMixtureDrag', 'o_relaxedMixtureDragSource',
                 'mixtureDragRampStartTime', 'mixtureDragRampDuration'):
    assert obsolete not in terms + udf
assert 'userImplicitLinearTerm = &bubbleColumn2::implicitGasDrag' in udf
assert 'freezeDragHistory' not in terms
assert 'nrs->preFluid = &bubbleColumn2::prepareMixtureDrag;' in udf
okl = (root / 'bubbleColumn2Equations.okl').read_text()
kernel = okl.split('@kernel void buildEquationTerms')[1].split('@kernel')[0]
kernel = 'void buildEquationTerms' + kernel
kernel = kernel.replace('@ restrict ', '')
kernel = re.sub(r'; @tile\(p_blockSize, @outer, @inner\)', '', kernel)
args = kernel.split(')\n{')[0].split('(', 1)[1].split(',')
values = dict(N=1, offset=2, rhoLiquid=1000, rhoGas=1, muLiquid=.001,
              muGas=.00001, alphaFloor=1e-6, gasPressureEnabled=0,
              dragEnabled=1, mixtureDragEnabled=1,
              dragAlphaCutoff=0, dragSlipLimitEnabled=0, dragSlipMaximum=1,
              bubbleDiameter=.003, virtualMassEnabled=0, virtualMassCoefficient=0,
              gravityX=0, gravityY=0, gravityZ=0)
arrays = []
call = []
for arg in args:
    name = arg.strip().split()[-1].lstrip('*')
    if '*' in arg:
        arrays.append(name)
        call.append(name)
    else:
        call.append(str(values[name]))
setup = '\n'.join(f'double {name}[32]={{}};' for name in arrays)
call = 'buildEquationTerms(' + ','.join(call) + ');'
cpp = r'''
#include <cassert>
#include <cmath>
#include <cstdio>
using dfloat=double;using dlong=long;
'''+kernel
cpp += '\nint main(){\n' + setup + r'''
 alpha[0]=.05; uv[0]=.2; ug[0]=1; ul[0]=(.2-.05)/.95;
'''
cpp += call + r'''
 const double gas=dragLambda[0]*(uv[0]-ug[0]);
 assert(gas<0);
 assert(std::abs(ugSource[0]-dragLambda[0]*uv[0])<1e-12);
 assert(std::abs(ugSource[0]-dragLambda[0]*ug[0]-gas)<1e-12);
 assert(mixtureInterphaseAcceleration[0]==0);
 assert(std::abs(mixtureDragRate[0]*(uv[0]-ug[0])-.05*.999*gas)<1e-12);
 assert(dragLambda[0]>0);
'''
for parameter, value, assertion in [
    ('mixtureDragEnabled', '0', 'assert(dragLambda[0]>0 && mixtureDragRate[0]==0 && mixtureInterphaseAcceleration[0]==0);'),
    ('dragEnabled', '0', 'assert(ugSource[0]==0 && mixtureDragRate[0]==0 && dragLambda[0]==0);'),
    ('dragAlphaCutoff', '.1', 'assert(ugSource[0]==0 && mixtureDragRate[0]==0 && dragLambda[0]==0);')]:
    variant = list(call.removesuffix(';').removeprefix('buildEquationTerms(').removesuffix(')').split(','))
    names = [arg.strip().split()[-1].lstrip('*') for arg in args]
    variant[names.index(parameter)] = value
    cpp += 'buildEquationTerms(' + ','.join(variant) + ');\n' + assertion + '\n'
cpp += r'''
 puts("UG/UM drag rates and known-velocity source, no explicit UM drag, switches/cutoff passed.");
}
'''
with tempfile.TemporaryDirectory() as folder:
    source=Path(folder)/'test.cpp';source.write_text(cpp)
    binary=Path(folder)/'test'
    subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror',str(source),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)
