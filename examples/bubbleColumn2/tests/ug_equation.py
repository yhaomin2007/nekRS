"""Actual UG force kernel and native BDF forcing: effective VM inertia and flux."""
from pathlib import Path
import re,subprocess,tempfile
root=Path(__file__).resolve().parents[1];repo=root.parents[1]
eq=(root/'bubbleColumn2Equations.okl').read_text()
def translate(s):
 s=s.replace('@kernel ', '').replace('@ restrict ', '')
 return re.sub(r'; @tile\(p_blockSize, @outer, @inner\)', '',s)
def kernel(name):
 return translate('@kernel void '+name+'('+eq.split('@kernel void '+name+'(')[1].split('@kernel')[0])+'\n'
k=kernel('buildEquationTerms')
args=k.split(')\n{')[0].split('(',1)[1].split(',')
defaults=dict(N=1,offset=2,rhoLiquid=998.2,rhoGas=1.2,muLiquid=.001,muGas=.000018,
 alphaFloor=1e-6,gasPressureEnabled=1,dragEnabled=0,mixtureDragEnabled=1,
 dragAlphaCutoff=0,dragSlipLimitEnabled=0,dragSlipMaximum=1,bubbleDiameter=.003,
 virtualMassEnabled=1,virtualMassCoefficient=.5,gravityX=0,gravityY=0,gravityZ=0)
declarations=[];names=[]
for a in args:
 name=a.strip().split()[-1].lstrip('*');names.append(name)
 if '*' in a:declarations.append('double '+name+'[32]={};')
 else:declarations.append(('long ' if name in ['N','offset'] else 'double ')+name+'='+str(defaults[name])+';')
call='buildEquationTerms('+','.join(names)+');'
native=translate((repo/'src/solver/scalar/kernels/sumMakef.okl').read_text())
cpp=r'''
#include <cassert>
#include <cmath>
#include <algorithm>
#define p_SUBCYCLING 0
#define p_MovingMesh 0
#define p_ADVECTION 0
#define p_nBDF 2
#define p_nEXT 1
using dfloat=double;using dlong=long;
'''+k+kernel('buildGasFlux')+kernel('buildLiquidVelocity')+kernel('addGasStress')+native+r'''
int main(){
'''+''.join(declarations)+r'''
 alpha[0]=.05;ug[0]=.3;ul[0]=.1;
 gradP[0]=-12.;liquidAcceleration[0]=2;
'''+call+r'''
 double c=.5*rhoLiquid,eff=rhoGas+c;
 assert(std::abs(ugSource[0]-(12+c*2)/eff)<1e-12);
 // Changing alpha does not multiply the UG pressure/VM acceleration.
 double source=ugSource[0];alpha[0]=.01;
'''+call+r'''
 assert(std::abs(ugSource[0]-source)<1e-12);
 // Explicit gas acceleration no longer feeds back into the UG RHS.
 virtualMassRelativeAcceleration[0]=1e12;
'''+call+r'''
 assert(std::abs(ugSource[0]-source)<1e-12);
 // Native BDF histories give the same system as added inertia on the LHS.
 double h=.001,old=.3,older=.1;
 double U[4]={old,0,older,0},RHO[2]={1,1},M[2]={1,1};
 double BDF[2]={2,-.5},EXT[1]={1},ADV[2]={},F[2]={ugSource[0],0},rhs[2]={};
 sumMakef(1,M,1/h,EXT,BDF,0,2,2,RHO,U,ADV,F,rhs);
 double unew=rhs[0]/(1.5/h);
 assert(std::abs(eff*(1.5*unew-2*old+.5*older)/h-(12+c*2))<1e-8);
 // Gas force is recovered when VM is disabled.
 virtualMassEnabled=0;
'''+call+r'''
 assert(std::abs(ugSource[0]-10)<1e-12);
 // Native mixture convection cancellation leaves gas advective convection.
 uv[0]=.2;ug[0]=.3;gradUg[0]=.4;gradP[0]=0;
'''+call+r'''
 assert(std::abs(ugSource[0]-(.2-.3)*.4)<1e-12);
 // Derived flux and liquid reconstruction preserve volume average.
 double a[2]={.02,0},gas[6]={.4,0,0,0,.3,0},q[6]={},mix[6]={.008,0,0,0,.006,0},liquid[6]={};
 buildGasFlux(1,2,a,gas,q);buildLiquidVelocity(1,2,1e-6,a,mix,q,liquid);
 assert(q[0]==.008 && q[4]==.006 && liquid[0]==0 && liquid[4]==0);
 // Physical stress is divided by alpha and effective inertia once.
 double stress[2]={3,0},out[2]={};addGasStress(1,1e-6,eff,a,stress,out);
 assert(std::abs(out[0]-3/(.02*eff))<1e-12);
}
'''
with tempfile.TemporaryDirectory() as t:
 f=Path(t)/'ug.cpp';f.write_text(cpp)
 subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-Wno-unknown-pragmas','-Wno-unused-parameter',str(f),'-o',str(Path(t)/'ug')],check=True)
 subprocess.run([str(Path(t)/'ug')],check=True)
terms=(root/'bubbleColumn2Terms.hpp').read_text()
par=(root/'bubbleColumn2.par').read_text();bc=(root/'bubbleColumn2Boundary.oudf').read_text()
assert 'scalars = ALPHA, UGX, UGY, UGZ' in par
assert 'scalar ugz' in bc and '? p_UG_INLET : 0.0' in bc
assert 'o_solution("qgx")' not in terms and 'o_solution("ugx")' in terms
assert 'o_ugSource' in terms and 'o_gradUg.slice' in terms
print('UG actual-kernel/native-BDF checks: added inertia/history, pressure, convection, flux/liquid reconstruction, stress and BC/storage wiring passed.')
