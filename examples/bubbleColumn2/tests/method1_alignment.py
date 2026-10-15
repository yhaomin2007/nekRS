"""Compare actual UG force/drag kernels and transport modules between methods."""
from pathlib import Path
import re
import subprocess
import tempfile

case=Path(__file__).resolve().parents[1]
one=case.parent/'bubbleColumn'
def function(s, marker):
 start=s.index(marker);at=s.index('{',start);depth=1;end=at+1
 while depth:
  if s[end]=='{':depth+=1
  elif s[end]=='}':depth-=1
  end+=1
 return s[start:end]
for name in ['Transport.okl','Cubature.hpp']:
 a=(one/('bubbleColumn'+name)).read_text()
 b=(case/('bubbleColumn2'+name)).read_text().replace('bubbleColumn2','bubbleColumn')
 assert a==b, name
for name in ['explicitFilter.hpp','explicitFilter.okl']:
 assert (one/name).read_bytes()==(case/name).read_bytes(),name
m=(one/'bubbleColumnTerms.hpp').read_text();t=(case/'bubbleColumn2Terms.hpp').read_text()
for name in ['evaluateGasCubatureTransport','prepareGasAdvection','evaluateNativeAlphaAdvection']:
 assert function(m,'inline void '+name+'(')==function(t,'inline void '+name+'('),name
# Compile both actual physical UG kernels with identical prescribed alpha/UL/UG.
prefix='#include <cmath>\n#include <cassert>\n#include <initializer_list>\nusing dfloat=double;using dlong=long;\n'
values=dict(N=2,offset=4,rhoLiquid=998.2,rhoGas=1.2,muLiquid=.001,muGas=.000018,alphaFloor=1e-6,
 gasPressureEnabled=1,dragEnabled=1,mixtureDragEnabled=1,dragAlphaCutoff=0,dragSlipLimitEnabled=0,
 dragSlipMaximum=1,bubbleDiameter=.003,virtualMassEnabled=1,virtualMassCoefficient=.5,
 gravityX=.1,gravityY=-.2,gravityZ=-9.81)
outputs={'rhoMField','rhoPressureField','muMField','driftStress','gasStress','alphaSource','ugSource',
         'dragLambda','mixtureInterphaseAcceleration','mixtureDragRate'}
common=set();decl=[];calls=[];kernels=[]
for idx,directory in enumerate([one,case]):
 name=directory.name
 k=function((directory/(name+'Equations.okl')).read_text(),'@kernel void buildEquationTerms(')
 args=k.split(')\n{')[0].split('(',1)[1].split(',')
 k=k.replace('@kernel ','').replace('@ restrict ','').replace('buildEquationTerms','equation'+str(idx))
 k=re.sub(r'; @tile\(p_blockSize, @outer, @inner\)','',k);kernels.append(k)
 args_call=[]
 for arg in args:
  key=arg.strip().split()[-1].lstrip('*')
  if '*' in arg:
   mapped='um' if key=='uv' else key
   if key in outputs:mapped+=str(idx)
   if mapped not in common:decl.append('double '+mapped+'[64]={};');common.add(mapped)
   args_call.append(mapped)
  else:args_call.append(key)
 calls.append('equation'+str(idx)+'('+','.join(args_call)+');')
cpp=prefix+'\n'.join(kernels)+'\nint main(){'+''.join('double '+k+'='+str(v)+';' for k,v in values.items())+''.join(decl)+r'''
 for(double fraction:{0.,1e-8,.02,.2}) for(double vm:{0.,.5,1.}) {
  alpha[0]=alpha[1]=fraction;virtualMassEnabled=vm;
  for(int i=0;i<3;i++)for(int n=0;n<2;n++){
   int id=n+4*i;ug[id]=.3+.05*i;ul[id]=.1-.02*i;
   gradP[id]=2-i;liquidAcceleration[id]=.2*i;
   um[id]=(1-fraction)*ul[id]+fraction*ug[id];
  }
'''+''.join(calls)+r'''
  for(int i=0;i<3;i++)for(int n=0;n<2;n++){
   int id=n+4*i;assert(std::abs((ugSource0[id]-dragLambda0[n]*ug[id])-(ugSource1[id]-dragLambda1[n]*ug[id]))<1e-9);
  }
  for(int n=0;n<2;n++)assert(std::abs(dragLambda0[n]-(1-fraction)*dragLambda1[n])<1e-12);
 }
}
'''
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp);(p/'check.cpp').write_text(cpp)
 subprocess.run(['g++','-std=c++17',str(p/'check.cpp'),'-o',str(p/'check')],check=True)
 subprocess.run([str(p/'check')],check=True)
# Host/kernel call arity check catches wiring errors beyond serial kernel math.
eq=(case/'bubbleColumn2Equations.okl').read_text()
for name in ['buildEquationTerms','initializePlume','updateVirtualMassHistory','postProcessGasVelocity','reconstructGasVelocity','splitMixtureDrag']:
 kernel=function(eq,'@kernel void '+name+'(')
 count=len(kernel.split(')\n{')[0].split('(',1)[1].split(','))
 source=t+(case/'bubbleColumn2.udf').read_text()
 for match in re.finditer(r'\b'+name+r'Kernel\(',source):
  start=match.end();pos=start;depth=1
  while depth:
   if source[pos]=='(':depth+=1
   elif source[pos]==')':depth-=1
   pos+=1
  args=source[start:pos-1];level=0;nargs=1
  for ch in args:
   if ch in '({[':level+=1
   elif ch in ')}]':level-=1
   elif ch==',' and level==0:nargs+=1
  assert nargs==count,(name,nargs,count)
assert 'nrs->userDivergence = &bubbleColumn2::enforceZeroDivergence;' in source
assert 'o_rhoPressure' in t and 'volumeWeight' in eq
print('Method1 alignment: identical transport/filter modules, matching unsplit UG force and reconstructed-UG diagonal at zero/dilute/finite alpha with VM on/off; host/kernel arities and method2 projection retained.')
