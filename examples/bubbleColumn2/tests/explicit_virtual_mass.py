"""Actual kernels and host EXT routing for paired last-completed-step explicit VM."""
from pathlib import Path
import re, subprocess, tempfile
case=Path(__file__).resolve().parents[1]
eq=(case/'bubbleColumn2Equations.okl').read_text();terms=(case/'bubbleColumn2Terms.hpp').read_text()
def function(s,marker):
 start=s.index(marker);end=s.index('{',start)+1;depth=1
 while depth:
  depth+=(s[end]=='{')-(s[end]=='}');end+=1
 return s[start:end]
def kernel(name):
 s=function(eq,'@kernel void '+name+'(').replace('@kernel ','').replace('@ restrict ','')
 return re.sub(r'; @tile\(p_blockSize, @outer, @inner\)','',s)
k=kernel('buildEquationTerms');args=k.split(')\n{')[0].split('(',1)[1].split(',')
values=dict(N=3,offset=5,rhoLiquid=998.2,rhoGas=1.2,muLiquid=.001,muGas=1.8e-5,
 alphaFloor=1e-6,gasPressureEnabled=1,dragEnabled=1,mixtureDragEnabled=1,
 dragAlphaCutoff=0,dragSlipLimitEnabled=0,dragSlipMaximum=1,bubbleDiameter=.003,
 virtualMassEnabled=1,virtualMassCoefficient=.5,mixtureVirtualMassEnabled=1,
 explicitVirtualMassEnabled=1,gravityX=.3,gravityY=-.7,gravityZ=-9.81)
arrays=[];decl=[];names=[]
for arg in args:
 name=arg.strip().split()[-1].lstrip('*');names.append(name)
 if '*' in arg:arrays.append(name);decl.append('double '+name+'[64];')
 else:decl.append(('long ' if name in ['N','offset'] else 'double ')+name+'='+str(values[name])+';')
call='buildEquationTerms('+','.join(names)+');'
cpp='''#include <algorithm>
#include <cassert>
#include <cmath>
#include <initializer_list>
#include <memory>
#include <vector>
using dfloat=double;using dlong=long;
bool close(double x,double y){return std::abs(x-y)<1e-10*(1+std::abs(y));}
'''+k+r'''
void pointwise(){
'''+''.join(decl)+''.join('std::fill_n('+a+',64,9876.);' for a in arrays)+r'''
 for(double fraction:{0.,1e-8,.016667,.2,.9})for(double enabled:{0.,.4,1.}){
  virtualMassEnabled=enabled;
  for(int j=0;j<N;j++){
   alpha[j]=fraction;for(int c=0;c<3;c++){
    const int id=j+c*offset;ug[id]=.3+.02*c;ul[id]=.1-.01*c;
    uv[id]=fraction*ug[id]+(1-fraction)*ul[id];gradP[id]=2+c;
    virtualMassRelativeAcceleration[id]=-.7+.1*c;
    liquidAcceleration[id]=1e12; // Must not enter fully explicit UG.
   }
   for(int c=0;c<9;c++)gradUg[j+c*offset]=.01*(1+c);
  }
'''+call+r'''
  double saved[3][3],drag[3],mobility[3],viscosity[3],stress[9*5];
  const double added=enabled*.5*rhoLiquid;
  for(int j=0;j<N;j++){
   drag[j]=dragLambda[j];mobility[j]=rhoPressureField[j];viscosity[j]=muMField[j];
   assert(close(1/rhoPressureField[j],fraction/rhoGas+(1-fraction)/rhoLiquid));
   for(int c=0;c<3;c++){
    int id=j+c*offset;double g=c==0?gravityX:(c==1?gravityY:gravityZ);
    double vm=added/rhoGas*virtualMassRelativeAcceleration[id];
    double expected=-gradP[id]/rhoGas+g+dragLambda[j]*ul[id]+vm;
    assert(close(ugSource[id],expected));saved[c][j]=ugSource[id];
    assert(close(mixtureInterphaseAcceleration[id]-mixtureDragSource[id],
      fraction*(1-rhoGas/rhoLiquid)*vm));
    // Native BDF2 has ordinary UG inertia; VM is a prescribed lagged load.
    double h=1e-5,old=.3,older=.2;
    double next=(2*old-.5*older+h*ugSource[id])/(1.5+h*dragLambda[j]);
    assert(close((1.5*next-2*old+.5*older)/h,
      -gradP[id]/rhoGas+g+dragLambda[j]*(ul[id]-next)+vm));
   }
   for(int c=0;c<9;c++)stress[j+c*offset]=gasStress[j+c*offset];
  }
  // Removing VM changes only the paired acceleration sources, not drag,
  // pressure mobility, baseline viscosity or the physical gas stress.
  virtualMassEnabled=0;
'''+call+r'''
  for(int j=0;j<N;j++){
   assert(drag[j]==dragLambda[j] && mobility[j]==rhoPressureField[j] && viscosity[j]==muMField[j]);
   for(int c=0;c<3;c++)assert(close(saved[c][j]-ugSource[j+c*offset],
      added/rhoGas*virtualMassRelativeAcceleration[j+c*offset]));
   for(int c=0;c<9;c++)assert(stress[j+c*offset]==gasStress[j+c*offset]);
  }
  for(int j=N;j<offset;j++){
   assert(rhoPressureField[j]==9876. && dragLambda[j]==9876.);
   for(int c=0;c<3;c++)assert(ugSource[j+c*offset]==9876.);
  }
 }
}
struct Mem{
 std::shared_ptr<std::vector<double>> data;long off=0,count=0;
 Mem(long n=0):data(std::make_shared<std::vector<double>>(n,0)),count(n){}
 double& operator[](long i){return data->at(off+i);}double operator[](long i)const{return data->at(off+i);}
 long size()const{return count;}
 Mem slice(long first,long n)const{auto m=*this;m.off+=first;m.count=n;return m;}
 void copyFrom(const Mem& m){for(long j=0;j<count;j++)(*this)[j]=m[j];}
};
struct LA{
 void fill(long n,double a,Mem& x){for(long j=0;j<n;j++)x[j]=a;}
 void axpby(long n,double a,const Mem& x,double b,Mem& y,long xo=0,long yo=0){for(long j=0;j<n;j++)y[j+yo]=a*x[j+xo]+b*y[j+yo];}
 void axpbyz(long n,double a,const Mem& x,double b,const Mem& y,Mem& z){for(long j=0;j<n;j++)z[j]=a*x[j]+b*y[j];}
}la;
struct Params{int explicitVirtualMassEnabled=1,mixtureForcingEnabled=1,frozenGasVelocityEnabled=0,frozenGasMixtureForcingEnabled=0;}p;
struct Mesh{long Nlocal=3;}mesh;
struct Fluid{long fieldOffsetSum=15;Mem o_coeffEXT{3},o_EXT{45};}fluid;
struct Nrs{int tstep=0;long fieldOffset=5;Mesh* meshV=&mesh;Fluid* fluid=&::fluid;}app;auto nrs=&app;
struct Platform{LA* linAlg=&la;}env;auto platform=&env;
Mem o_mixtureVirtualMassSource(15),o_previousMixtureVirtualMassSource(15);
Mem o_mixtureInterphaseAcceleration(15),o_mixtureDragSource(15);
'''+function(terms,'inline void freezeExplicitVirtualMassHistory()')+r'''
void history(){
 o_mixtureInterphaseAcceleration[0]=9;
 freezeExplicitVirtualMassHistory();assert(o_previousMixtureVirtualMassSource[0]==0);
 for(int step=1;step<=9;step++){
  nrs->tstep=step;double vm=(step%2?-1.:1.)*step*step;
  for(int c=0;c<3;c++)for(int j=0;j<5;j++){
   int id=j+5*c;
   o_mixtureDragSource[id]=3*step;
   o_mixtureInterphaseAcceleration[id]=3*step+vm*(c+1);
   fluid.o_EXT[id]=vm*(c+1)+.1*step*step*(c+1); // VM + unrelated source.
  }
  auto before=*fluid.o_EXT.data;
  freezeExplicitVirtualMassHistory();
  double weights[3]={3,-3,1};if(step==1){weights[0]=1;weights[1]=weights[2]=0;}
  if(step==2){weights[0]=2;weights[1]=-1;weights[2]=0;}
  for(int c=0;c<3;c++)for(int j=0;j<3;j++){
   double actual=0,nonVM=0;
   for(int stage=0;stage<3;stage++){
    actual+=weights[stage]*fluid.o_EXT[15*stage+5*c+j];
    int old=step-stage;
    nonVM+=weights[stage]*(old>0?.1*old*old*(c+1):0.);
   }
   assert(close(actual,vm*(c+1)+nonVM));
  }
  // Padded entries in every old EXT slot and VM buffer remain untouched.
  for(int stage=1;stage<3;stage++)for(int c=0;c<3;c++)for(int j=3;j<5;j++)
   assert(fluid.o_EXT[15*stage+5*c+j]==before[15*stage+5*c+j]);
  for(int c=0;c<3;c++)assert(o_mixtureVirtualMassSource[5*c+3]==0.);
  for(int stage=2;stage>=1;stage--)for(int j=0;j<15;j++)fluid.o_EXT[15*stage+j]=fluid.o_EXT[15*(stage-1)+j];
 }
 auto old=*fluid.o_EXT.data;p.explicitVirtualMassEnabled=0;freezeExplicitVirtualMassHistory();assert(*fluid.o_EXT.data==old);
 p.explicitVirtualMassEnabled=1;p.mixtureForcingEnabled=0;freezeExplicitVirtualMassHistory();
 for(int j=0;j<15;j++)assert(o_mixtureVirtualMassSource[j]==0);
}
int main(){pointwise();history();}
'''
with tempfile.TemporaryDirectory() as tmp:
 f=Path(tmp)/'explicit.cpp';f.write_text(cpp)
 subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror',str(f),'-o',str(f.with_suffix(''))],check=True)
 subprocess.run([str(f.with_suffix(''))],check=True)
# Compile the actual mode guards, including incompatible paired-force switches.
udf=(case/'bubbleColumn2.udf').read_text();par=(case/'bubbleColumn2.par').read_text()
a=udf.index('  nekrsCheck(p.explicitVirtualMassEnabled < 0')
b=udf.index('  nekrsCheck(p.coupledVirtualMassEnabled < 0',a)
guard=udf[a:b]
controls=r"""
#include <cassert>
#include <cstdlib>
#include <stdexcept>
struct P{int explicitVirtualMassEnabled=1,coupledVirtualMassEnabled=0,mixtureVirtualMassEnabled=1,mixtureForcingEnabled=1;}p;
struct Comm{int mpiComm(){return 0;}};
struct Env{Comm comm;}env;auto platform=&env;
#define nekrsCheck(test,...) do{if(test)throw std::runtime_error("invalid mode");}while(0)
void validate(){
"""+guard+r"""
}
int main(){
 validate();
 for(int j=0;j<5;j++){
  p=P{};
  if(j==0)p.explicitVirtualMassEnabled=2;
  if(j==1)p.explicitVirtualMassEnabled=-1;
  if(j==2)p.coupledVirtualMassEnabled=1;
  if(j==3)p.mixtureVirtualMassEnabled=0;
  if(j==4)p.mixtureForcingEnabled=0;
  bool rejected=false;try{validate();}catch(const std::runtime_error&){rejected=true;}
  assert(rejected);
  p.explicitVirtualMassEnabled=0;validate();
 }
}
"""
with tempfile.TemporaryDirectory() as tmp:
 f=Path(tmp)/'controls.cpp';f.write_text(controls)
 subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror',str(f),'-o',str(f.with_suffix(''))],check=True)
 subprocess.run([str(f.with_suffix(''))],check=True)
# Ordering and configuration are part of the diagnostic contract.
execute=udf.split('void UDF_ExecuteStep')[1]
assert execute.index('correctGasPressure();')<execute.index('updateVirtualMassHistory();')<execute.index('printStabilityMonitors(')
source=function(terms,'inline void addExplicitSources(')
assert source.index('evaluateMixtureForce();')<source.index('freezeExplicitVirtualMassHistory();')<source.index('rampMixtureDrag(')<source.index('freezeDragHistory();')
for key,value in [('polynomialOrder','5'),('cubaturePolynomialOrder','10'),('coupledVirtualMassEnabled','0'),('mixtureVirtualMassEnabled','1')]:
 assert re.search(r'^'+key+r'\s*=\s*'+value+r'\s*$',par,re.M)
assert re.search(r'\(p\.explicitVirtualMassEnabled \|\| p\.virtualMassExtrapolationEnabled\)\s*\? 1\.0 : p\.mixtureViscousCorrectionEnabled', terms)
print('Fully explicit VM: paired actual-kernel sources, ordinary gas BDF/drag, unchanged non-VM forces, last-step UM load without EXT1/2/3 extrapolation, padding, completed-step ordering and P5/cubature10 passed.')
