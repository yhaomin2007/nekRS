"""Mesh topology/geometry plus compiled actual periodic and method2 force kernels.
Run with Python + NumPy and g++; does not claim native MPI/GPU verification.
"""
from pathlib import Path
import sys,struct,math,subprocess,tempfile,re
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import prepare

def check_mesh(case):
    path=ROOT/f'case{case}/case{case}.re2'
    data=path.read_bytes();header=data[:80].decode();assert header.startswith('#v004')
    n=int(header[5:21]);assert n==(64 if case<3 else 512)
    assert abs(struct.unpack_from('<f',data,80)[0]-6.54321)<1e-6
    pos=84;elements=[]
    for _ in range(n):
        vals=struct.unpack_from('<25d',data,pos);pos+=200
        elements.append(np.array(vals[1:]).reshape(3,8).T)
    nc=int(struct.unpack_from('<d',data,pos)[0]);pos+=8;curves={}
    for _ in range(nc):
        vals=struct.unpack_from('<7d',data,pos);code=data[pos+56:pos+64].decode().strip();pos+=64
        assert code=='C';curves[(int(vals[0]),int(vals[1]))]=vals[2]
    nb=int(struct.unpack_from('<d',data,pos)[0]);pos+=8;b={}
    for _ in range(nb):
        v=struct.unpack_from('<7d',data,pos);code=data[pos+56:pos+64].decode().strip();pos+=64
        b[(int(v[0]),int(v[1]))]=(code,int(v[6]),int(v[2]),int(v[3]))
    assert pos==len(data)
    def corners(e,f):return elements[e-1][list(prepare.FACES[f-1])]
    def key(p):return tuple(sorted(tuple(round(float(x),12) for x in q) for q in p))
    faces={}
    for e in range(1,n+1):
        for f in range(1,7):faces.setdefault(key(corners(e,f)),[]).append((e,f))
    for rows in faces.values():
        assert len(rows) in (1,2)
        if len(rows)==2:
            assert all(row not in b for row in rows)
        else:assert rows[0] in b
    for (e,f),(code,bid,pe,pf) in b.items():
        if code=='P':
            assert b[(pe,pf)]==('P',0,e,f)
            a=corners(e,f);other=corners(pe,pf);delta=other.mean(0)-a.mean(0)
            assert key(a+delta)==key(other)
            assert np.count_nonzero(np.abs(delta)>1e-10)==1
            if case==3:assert abs(abs(delta[2])-.3)<1e-12
        else:
            assert case==3 and code=='W' and bid==1 and f==2
            assert (e,2) in curves and (e,6) in curves
            assert np.max(np.abs(np.linalg.norm(corners(e,f)[:,:2],axis=1)-.075))<1e-12
    # P5 GLL geometry: straight cells + native circular-edge transfinite map.
    nodes=np.r_[-1,np.polynomial.legendre.Legendre.basis(5).deriv().roots(),1]
    bary=np.array([1/np.prod([nodes[i]-nodes[j] for j in range(6) if i!=j]) for i in range(6)])
    D=np.zeros((6,6))
    for i in range(6):
        for j in range(6):
            if i!=j:D[i,j]=bary[j]/bary[i]/(nodes[i]-nodes[j])
        D[i,i]=-sum(D[i])
    weights=2/(5*6*np.polynomial.legendre.legval(nodes,[0]*5+[1])**2)
    low=1e9;discreteLow=1e9;volume=0
    for e,p in enumerate(elements,1):
        q=p[:4,:2];dz=p[4,2]-p[0,2]
        curved=(e,2) in curves
        xy=np.zeros((6,6,2))
        for ir,r in enumerate(nodes):
            for js,s in enumerate(nodes):
                u=(r+1)/2;v=(s+1)/2
                inner=(1-v)*q[0]+v*q[3];di=q[3]-q[0]
                if curved:
                    t0=math.atan2(q[1,1],q[1,0]);t1=math.atan2(q[2,1],q[2,0])
                    angle=(t1-t0)%(2*math.pi);assert angle<math.pi/2
                    t=t0+v*angle;R=curves[(e,2)]
                    outer=R*np.array([math.cos(t),math.sin(t)])
                    do=R*angle*np.array([-math.sin(t),math.cos(t)])
                else:outer=(1-v)*q[1]+v*q[2];do=q[2]-q[1]
                xy[ir,js]=(1-u)*inner+u*outer
                du=outer-inner;dv=(1-u)*di+u*do
                det=(du[0]*dv[1]-du[1]*dv[0])*dz/8
                low=min(low,det);assert det>0
        dr=np.einsum('ij,jsk->isk',D,xy);ds=np.einsum('ij,rjk->rik',D,xy)
        jac=(dr[:,:,0]*ds[:,:,1]-dr[:,:,1]*ds[:,:,0])*dz/2
        assert np.min(jac)>0
        discreteLow=min(discreteLow,float(np.min(jac)))
        volume+=float(np.sum(jac*weights[:,None]*weights[None,:]))*2
    expected=.1**3 if case<3 else math.pi*.075**2*.3
    assert abs(volume/expected-1)<1e-6
    assert nc==(0 if case<3 else 256)
    print(f'case{case}: {n} hex, reciprocal periodic pairs, conformal faces, P5 discrete Jmin={discreteLow:.8e}, volume={volume:.12e}')

for case in (1,2,3):check_mesh(case)

def function(text,marker):
    a=text.index(marker);b=text.index('{',a)+1;depth=1
    while depth:depth+=(text[b]=='{')-(text[b]=='}');b+=1
    return text[a:b]
def serial(text):
    return re.sub(r';\s*@tile\(p_blockSize,\s*@outer,\s*@inner\)','',text.replace('@kernel ','').replace('@ restrict ',''))
periodic=(ROOT/'method2Periodic.okl').read_text()
base=(ROOT.parent/'bubbleColumn2/bubbleColumn2Equations.okl').read_text()
build=serial(function(base,'@kernel void buildEquationTerms('))
args=build.split(')\n{')[0].split('(',1)[1].split(',')
values=dict(N=2,offset=3,rhoLiquid=998.2,rhoGas=1.2,muLiquid=.001,muGas=1.8e-5,alphaFloor=1e-6,
 gasPressureEnabled=1,dragEnabled=1,mixtureDragEnabled=1,dragAlphaCutoff=0,dragSlipLimitEnabled=0,
 dragSlipMaximum=1,bubbleDiameter=.003,virtualMassEnabled=0,virtualMassCoefficient=.5,
 mixtureVirtualMassEnabled=1,explicitVirtualMassEnabled=0,gravityX=0,gravityY=0,gravityZ=-9.81)
decl=[];names=[]
for a in args:
    name=a.strip().split()[-1].lstrip('*');names.append(name)
    decl.append('double '+name+'[64]={};' if '*' in a else ('long ' if name in ['N','offset'] else 'double ')+name+'='+str(values[name])+';')
cpp=r'''
#include <cassert>
#include <cmath>
#include <iostream>
#include "periodicPhysics.hpp"
using dfloat=double;using dlong=long;
'''+serial(periodic)+build+serial(function(base,'@kernel void buildMixtureForce('))+r'''
int main(){
'''+''.join(decl)+r'''
 double x[3]={0,.075,999},y[3]={0,0,999},aa[3]={0,0,999},ux[3]={0,0,999},uy[3]={0,0,999},uz[3]={0,0,999},um[9]={};
 double driftDiv[9]={},force[9]={},gradient[9]={};
 const double a=.05,rhoM=(1-a)*rhoLiquid+a*rhoGas;
 const double w=method2Periodic::equilibriumSlip(a,rhoLiquid,rhoGas,muLiquid,bubbleDiameter,gravityZ);
 const double Ki=method2Periodic::dragKi(w,rhoLiquid,muLiquid,bubbleDiameter);
 assert(std::abs(Ki*w-(rhoGas-rhoM)*gravityZ)<1e-10);
 std::cout.precision(16);std::cout<<"Case2 terminal slip="<<w<<" UG="<<.05+(1-a)*w<<" UL="<<.05-a*w<<" G0="<<rhoM*gravityZ<<'\n';
 for(int test=1;test<=3;++test){
  const double g=test==1?0:-9.81,G=test==1?0:rhoM*g+(test==3?-10:0);
  const double ws=test==1?0:w,U=test==1?.3:.05;gravityZ=g;
  initializePeriodic(2,3,test==3,a,U,ws,.075,x,y,aa,ux,uy,uz,um);
  assert(aa[2]==999 && ux[2]==999 && uy[2]==999 && uz[2]==999);
  if(test==3)assert(uz[1]==0 && um[7]==0);
  for(int n=0;n<2;++n){
   alpha[n]=aa[n];ug[n]=ux[n];ug[n+3]=uy[n];ug[n+6]=uz[n];
   for(int f=0;f<3;++f){uv[n+f*3]=um[n+f*3];ul[n+f*3]=(uv[n+f*3]-a*ug[n+f*3])/(1-a);}
  }
'''+ 'buildEquationTerms('+','.join(names)+');'+r'''
  buildMixtureForce(2,3,1e-6,1,0,0,g,alpha,driftDiv,mixtureInterphaseAcceleration,force);
  addMeanPressure(2,3,G,rhoLiquid,rhoGas,alpha,ugSource+6,force);
  for(int n=0;n<2;++n){
   const double gasResidual=ugSource[n+6]-dragLambda[n]*ug[n+6];
   if(test<=2){assert(std::abs(gasResidual)<2e-10);assert(std::abs(force[n+6])<2e-10);}
   // Independent physical liquid equation check, beyond the implementation's UM residual.
   if(test==2)assert(std::abs(-G/rhoLiquid+g+a/(1-a)*Ki/rhoLiquid*w)<1e-12);
   if(test==3 && n==0){assert(std::abs(gasResidual-10/rhoGas)<2e-10);}
  }
 }
 // Constant-pressure gradient remains zero until the explicitly prescribed G0 is added.
 addMeanGradient(2,3,-9300,gradient);assert(gradient[6]==-9300 && gradient[7]==-9300 && gradient[8]==0);
 assert(method2Periodic::equilibriumSlip(.05,998.2,1.2,.001,.003,0)==0);
 assert(method2Periodic::equilibriumSlip(.05,998.2,1.2,.001,.003,9.81)==-w);
 // Densities and gas-transport units: actual added acceleration has no extra rhoP.
 double alphaV[3]={.05,.2,999},gasV[3]={1,2,999},fluidV[9]={};
 addMeanPressure(2,3,-123,998.2,1.2,alphaV,gasV,fluidV);
 for(int n=0;n<2;++n){assert(std::abs(gasV[n]-(n+1+123/1.2))<1e-12);assert(std::abs(fluidV[n+6]-123*((1-alphaV[n])/998.2+alphaV[n]/1.2))<1e-12);}
 assert(gasV[2]==999 && fluidV[8]==0);
}
'''
with tempfile.TemporaryDirectory() as d:
    path=Path(d)/'physics.cpp';path.write_text(cpp)
    subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-Wno-unused-parameter','-I',str(ROOT),str(path),'-o',str(path.with_suffix(''))],check=True)
    subprocess.run([str(path.with_suffix(''))],check=True)
print('Actual method2 gas/mixture kernels: case1 zero forces, case2 both phase + UM balance, case3 driving load, initialization/wall values, mobility signs/units/padding passed.')

# Native UDF extraction reads the literal session file, not included headers.
for case in (1,2,3):
    text=(ROOT/f'case{case}/case{case}.par').read_text()
    for key,val in (('testCase',str(case)),('virtualMassEnabled','0'),
                    ('virtualMassExtrapolationEnabled','0'),('gasPressureCorrectorEnabled','1'),
                    ('alphaClipEnabled','0'),('outletDampingFactor','1'),
                    ('polynomialOrder','5'),('cubaturePolynomialOrder','10')):
        assert re.search(r'^'+key+r'\s*=\s*'+val+r'\s*$',text,re.M),(case,key)
    source=(ROOT/f'case{case}/case{case}.udf').read_text()
    assert '#ifdef __okl__\n#include "periodicBoundary.oudf"' in source
    assert (ROOT/f'case{case}/periodicBoundary.oudf').read_bytes()==(ROOT/'periodicBoundary.oudf').read_bytes()
    for kernel in ('bubbleColumn2Equations.okl','bubbleColumn2Transport.okl','explicitFilter.okl'):
        assert (ROOT/f'case{case}'/kernel).read_bytes()==(ROOT.parent/'bubbleColumn2'/kernel).read_bytes()
wrapper=(ROOT/'periodic.udf').read_text()
source=function(wrapper,'inline void source(')
assert source.index('bubbleColumn2::addExplicitSources(time)')<source.index('meanPressureKernel(')
execute=function(wrapper,'void UDF_ExecuteStep(')
assert execute.index('bubbleColumn2::correctGasPressure()')<execute.index('method2Periodic::diagnostics(')
assert 'method2BaseExecuteStep(time,step)' not in execute
assert 'nrs->userSource=&method2Periodic::source' in wrapper
print('Native source hook ordering, literal OKL extraction blocks, runtime staging and baseline controls passed.')
