"""Serial checks of actual frozen-drag kernel and EXT-history helper; requires g++."""
from pathlib import Path
import re
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
terms = (root / 'bubbleColumn2Terms.hpp').read_text()
udf = (root / 'bubbleColumn2.udf').read_text()
for obsolete in ('frozenDragEnabled', 'mixtureImplicitDragEnabled',
                 'implicitGasDrag', 'implicitMixtureDrag', 'o_mixtureDragDiagonal',
                 'o_mixtureDragRate', 'addMixtureStressAndSplitDrag'):
    assert obsolete not in terms + udf
assert 'userImplicitLinearTerm' not in udf
helper = terms.split('inline void freezeDragHistory()')[1].split('inline void addExplicitSources')[0]
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
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
using dfloat=double; using dlong=long long;
struct Memory {
 std::vector<double> v;
 Memory(long n=0):v(n){}
 long size()const{return v.size();}
 void copyFrom(const Memory& m){v=m.v;}
};
struct Algebra {
 void axpby(long n,double a,const Memory& x,double b,Memory& y,long xo,long yo){
  for(long i=0;i<n;++i)y.v.at(yo+i)=a*x.v.at(xo+i)+b*y.v.at(yo+i);
 }
};
struct Platform {Algebra a; Algebra* linAlg=&a;} platformStore; auto platform=&platformStore;
struct Mesh{long Nlocal=1;} mesh;
struct Scalar{long fieldOffsetSum=8;std::vector<long> fieldOffsetScan{0,2,4,6};Memory o_coeffEXT{3},o_EXT{24};} scalar;
struct Fluid{long fieldOffsetSum=6;Memory o_coeffEXT{3},o_EXT{18};} fluid;
struct Nrs{int tstep=0;long fieldOffset=2;Mesh* meshV=&mesh;Scalar* scalar=&::scalar;Fluid* fluid=&::fluid;} nrsStore;auto nrs=&nrsStore;
Memory o_gasDragSource{6},o_mixtureDragSource{6},o_previousGasDragSource{6},o_previousMixtureDragSource{6};
'''
cpp += 'inline void freezeDragHistory()' + helper + kernel
cpp += '\nint main(){\n' + setup + r'''
 alpha[0]=.05; uv[0]=.2; ug[0]=1; ul[0]=(.2-.05)/.95;
'''
cpp += call + r'''
 const double gas=gasDragSource[0];
 assert(gas<0);
 assert(std::abs(qgSource[0]-gas)<1e-12);
 assert(std::abs(mixtureInterphaseAcceleration[0]-.999*gas)<1e-12);
 assert(std::abs(mixtureDragSource[0]-.999*gas)<1e-12);
 assert(dragLambda[0]>0);
'''
for parameter, value, assertion in [
    ('mixtureDragEnabled', '0', 'assert(gasDragSource[0]<0 && mixtureDragSource[0]==0 && mixtureInterphaseAcceleration[0]==0);'),
    ('dragEnabled', '0', 'assert(gasDragSource[0]==0 && mixtureDragSource[0]==0 && dragLambda[0]==0);'),
    ('dragAlphaCutoff', '.1', 'assert(gasDragSource[0]==0 && mixtureDragSource[0]==0 && dragLambda[0]==0);')]:
    variant = list(call.removesuffix(';').removeprefix('buildEquationTerms(').removesuffix(')').split(','))
    names = [arg.strip().split()[-1].lstrip('*') for arg in args]
    variant[names.index(parameter)] = value
    cpp += 'buildEquationTerms(' + ','.join(variant) + ');\n' + assertion + '\n'
cpp += r'''
 // Startup CFL source probe must not alter stored history.
 o_gasDragSource.v[0]=4; freezeDragHistory();
 assert(o_previousGasDragSource.v[0]==0 && scalar.o_EXT.v[10]==0);
 // Distinct signed/nonlinear forces expose accidental EXT extrapolation.
 for(int step=1;step<=8;++step){
  nrs->tstep=step;
  double d=(step%2 ? -1.0 : 1.0)*step*step;
  for(int i=0;i<3;++i){o_gasDragSource.v[2*i]=d*(i+1);o_mixtureDragSource.v[2*i]=.999*d*(i+1);}
  // Native clears current source only. Alpha and padded entries untouched.
  for(int i=0;i<8;++i)scalar.o_EXT.v[i]=0;
  for(int i=0;i<6;++i)fluid.o_EXT.v[i]=0;
  scalar.o_EXT.v[0]=100+step;
  for(int i=0;i<3;++i){scalar.o_EXT.v[2*(i+1)]=o_gasDragSource.v[2*i]+10*step;fluid.o_EXT.v[2*i]=o_mixtureDragSource.v[2*i]+20*step;}
  freezeDragHistory();
  std::vector<double> weights=step==1?std::vector<double>{1,0,0}:step==2?std::vector<double>{2,-1,0}:std::vector<double>{3,-3,1};
  for(int i=0;i<3;++i){
   double sg=0,su=0,ng=0,nu=0;
   for(int j=0;j<3;++j){sg+=weights[j]*scalar.o_EXT.v[8*j+2*(i+1)];su+=weights[j]*fluid.o_EXT.v[6*j+2*i];ng+=weights[j]*10*(step-j);nu+=weights[j]*20*(step-j);}
   assert(std::abs(sg-(d*(i+1)+ng))<1e-10);
   assert(std::abs(su-(.999*d*(i+1)+nu))<1e-10);
  }
  assert(scalar.o_EXT.v[0]==100+step);
  for(int j=2;j>=1;--j){for(int i=0;i<8;++i)scalar.o_EXT.v[8*j+i]=scalar.o_EXT.v[8*(j-1)+i];for(int i=0;i<6;++i)fluid.o_EXT.v[6*j+i]=fluid.o_EXT.v[6*(j-1)+i];}
 }
 puts("Explicit paired drag, switches/cutoff, startup, EXT1/2/3 and other-source preservation passed.");
}
'''
with tempfile.TemporaryDirectory() as folder:
    source=Path(folder)/'test.cpp';source.write_text(cpp)
    binary=Path(folder)/'test'
    subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror',str(source),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)
