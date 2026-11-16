"""Execute production kernels: source conservation, momentum, closures, stress.

Requires g++; additionally compiles/executes OKL and translates CUDA when a
built OCCA installation is supplied through OCCA_DIR. No NekRS/GPU run is
claimed by these local checks.
"""
import os
from pathlib import Path
import re
import subprocess
import tempfile

case=Path(__file__).resolve().parents[1]
equations=(case/"bubbleColumnEquations.okl").read_text()
reference=(case/"referenceCase.okl").read_text()
def extract(name):
    return "@kernel void "+name+equations.split("@kernel void "+name,1)[1].split("@kernel void",1)[0]
src=reference+extract("buildExplicitMixtureDivergence")+extract("addDispersionLift")
src=src.replace("@kernel ","").replace("@ restrict ","")
src=re.sub(r";\s*@tile\(p_blockSize,\s*@outer,\s*@inner\)","",src)
prefix=r'''
#include <algorithm>
#include <cassert>
#include <cmath>
using dlong=int; using dfloat=double;
double p_REFERENCE_EO=9.81*(1000.-1.2)*.003*.003/.072;
double p_REFERENCE_EOD=p_REFERENCE_EO*pow(1+.163*pow(p_REFERENCE_EO,.757),2);
bool near(double a,double b){return std::abs(a-b)<1e-11*(1+std::abs(a)+std::abs(b));}
'''
main=r'''
int main(){
 constexpr int N=3,O=5;
 const double rl=1000,rg=1.2,mu=.00089,vm=.5,ui=.25,dt=.0001,S=.3703703703703704;
 double a[O]={0,.02,.2},q[O]={S,S,0},um[3*O]={},rate[O]={},gas[3*O]={},mix[3*O]={};
 for(int i=0;i<N;++i){um[i]=.01;um[i+O]=-.02;um[i+2*O]=.03;}
 injectionMomentum(N,O,rl,rg,vm,1e-6,ui,a,q,um,rate,gas,mix);
 for(int i=0;i<N;++i){
   assert(std::isfinite(rate[i]) && rate[i]>=0);
   double rm=(1-a[i])*rl+a[i]*rg;
   // Multiplying by rhoM recovers injected primitive momentum, including -UM.
   assert(near(rm*mix[i],-rg*q[i]*um[i]));
   assert(near(rm*mix[i+O],-rg*q[i]*um[i+O]));
   assert(near(rm*mix[i+2*O],rg*q[i]*(ui-um[i+2*O])));
   // Injection alone preserves UG=Uinj exactly, even in initially gas-free nodes.
   assert(near((ui/dt+gas[i+2*O])/(1/dt+rate[i]),ui));
   const double advanced=gas[i+2*O]/(1/dt+rate[i]);
   assert(advanced>=0 && advanced<=ui);
   assert(gas[i]==0 && gas[i+O]==0);
 }
 assert(rate[2]==0 && mix[2]==0);
 assert(rate[3]==0 && mix[3]==0 && gas[3]==0); // padded entries untouched
 // Uniform gas injection, no advection: alpha'=S must imply div(UM)=S
 // exactly, at arbitrary density ratio. This catches omitting or double
 // counting the source mass in the pressure target.
 double old[O]={},adv[O]={},d[O]={};
 for(int i=0;i<N;++i){a[i]=dt*q[i];}
 buildExplicitMixtureDivergence(N,rl,rg,1/dt,a,old,adv,d);
 addInjectionDivergence(N,rl,rg,a,q,d);
 for(int i=0;i<N;++i)assert(near(d[i],rl*q[i]/((1-a[i])*rl+a[i]*rg)));
 // At alpha=0, d=S exactly; above difference is O(dt) from density change.
 std::fill(a,a+O,0);std::fill(d,d+O,0);
 for(int i=0;i<N;++i)d[i]=(rl-rg)*q[i]/rl;
 addInjectionDivergence(N,rl,rg,a,q,d);
 for(int i=0;i<N;++i)assert(near(d[i],q[i]));
 // Hydrostatic pressure gives zero liquid force with P=0 at z=height.
 double z[O]={0,.1,.45},p[O]={};initializeReferencePressure(N,rl,9.81,.45,z,p);
 assert(near(p[2],0) && near((p[1]-p[0])/.1,-rl*9.81));
 // Newtonian stress: symmetry, zero trace and positive strain work.
 double gl[9*O]={},tau[9*O]={};
 for(int c=0;c<9;++c)for(int i=0;i<N;++i)gl[c*O+i]=.1*(c+1);
 liquidStress(N,O,mu,gl,tau);
 for(int i=0;i<N;++i){
   assert(near(tau[i]+tau[i+4*O]+tau[i+8*O],0));
   double power=0;for(int c=0;c<3;++c)for(int j=0;j<3;++j){
     assert(near(tau[i+(3*c+j)*O],tau[i+(3*j+c)*O]));
     power+=tau[i+(3*c+j)*O]*gl[i+(3*c+j)*O];
   }assert(power>=0);
 }
 double rm[O]={rl,500,1.2},weak[O]={1,-2,3},force[O]={};
 addLiquidStressForce(N,rm,weak,force);
 for(int i=0;i<N;++i)assert(near(rm[i]*force[i],-weak[i]));
 // Actual MP-LD/Tomiyama kernel: zero slip, force direction, shear lift,
 // both drag branches, all lift branches, and the alpha=0 continuation.
 double ug[3*O]={},ul[3*O]={},ga[3*O]={},out[3*O]={};
 std::fill(gl,gl+9*O,0);gl[6*O]=4;ga[0]=2;
 for(double eod:{2.,6.,12.})for(double speed:{0.,1e-5,.25,2.}){
   p_REFERENCE_EOD=eod;ug[2*O]=speed;std::fill(out,out+3*O,0);
   addDispersionLift(1,O,rl,rg,mu,.003,vm,1,3.2,1,1,a,ug,ul,ga,gl,out);
   double re=rl*.003*speed/mu;
   double cdSpeed2=std::max(std::min(16*(1+.15*pow(re,.687)),48.),
       (8./3)*p_REFERENCE_EO/(p_REFERENCE_EO+4)*re)*mu*speed/(rl*.003);
   double f=.00105*eod*eod*eod-.0159*eod*eod-.0204*eod+.474;
   double cl=eod<4?std::min(.288*tanh(.121*re),f):(eod<10.7?f:-.288);
   double expected=(-.75*3.2*rl*cdSpeed2*2-cl*rl*4*speed)/(rg+vm*rl);
   assert(std::isfinite(out[0]) && near(out[0],expected));
   assert(out[O]==0 && out[2*O]==0);
 }
}
'''
with tempfile.TemporaryDirectory() as t:
    p=Path(t);(p/"test.cpp").write_text(prefix+src+main)
    subprocess.run(["g++","-std=c++17","-Wall","-Wextra","-Werror",str(p/"test.cpp"),"-o",str(p/"test")],check=True)
    subprocess.run([str(p/"test")],check=True)
print("Production source/momentum/divergence, hydrostatic balance, liquid stress and MP-LD/Tomiyama limits passed.")

occa=Path(os.environ.get("OCCA_DIR",case.parents[1]/"3rd_party/occa"))
if not (occa/"lib/libocca.so").exists():
    print("OCCA checks skipped; set OCCA_DIR to a built OCCA installation.")
    raise SystemExit(0)
code=r'''
#include <occa.hpp>
#include <cassert>
#include <cmath>
int main(int argc,char **argv){
 occa::device d;d.setup(std::string("{mode:'Serial'}"));occa::properties p;
 p["defines/dfloat"]="double";p["defines/dlong"]="int";p["defines/p_blockSize"]=256;
 p["defines/p_REFERENCE_EO"]=1.2247785;p["defines/p_REFERENCE_EOD"]=1.734;
 for(const char* name:{"initializeReferencePressure","injectionMomentum","addInjectionDivergence","liquidStress","addLiquidStressForce"})
   d.buildKernel(argv[1],name,p);
 auto k=d.buildKernel(argv[1],"injectionMomentum",p);
 double a=0,s=.3703703703703704,r=0,u[3]={},g[3]={},m[3]={};
 auto A=d.malloc<double>(1,&a),S=d.malloc<double>(1,&s),R=d.malloc<double>(1,&r),
      U=d.malloc<double>(3,u),G=d.malloc<double>(3,g),M=d.malloc<double>(3,m);
 k(1,1,1000.,1.2,.5,1e-6,.25,A,S,U,R,G,M);
 R.copyTo(&r);G.copyTo(g);M.copyTo(m);
 assert(std::abs(g[2]-.25*r)<1e-12 && r>0);
 assert(std::abs(m[2]-1.2*s/1000*.25)<1e-12);
 d.buildKernel(argv[2],"buildEquationTerms",p);
 d.buildKernel(argv[2],"addDispersionLift",p);
}
'''
with tempfile.TemporaryDirectory() as t:
    p=Path(t);(p/"occa.cpp").write_text(code)
    subprocess.run(["g++","-std=c++17",str(p/"occa.cpp"),"-I"+str(occa/"include"),"-L"+str(occa/"lib"),"-Wl,-rpath,"+str(occa/"lib"),"-locca","-o",str(p/"occa")],check=True)
    subprocess.run([str(p/"occa"),str(case/"referenceCase.okl"),str(case/"bubbleColumnEquations.okl")],check=True,env=dict(os.environ,OCCA_CACHE_DIR=str(p/"cache")))
    for name in ("referenceCase.okl","bubbleColumnEquations.okl"):
        with (p/(name+".cu")).open("w") as output:
            subprocess.run([str(occa/"bin/occa"),"translate","-m","CUDA","-D","p_blockSize=256","-D","dfloat=double","-D","dlong=int","-D","p_REFERENCE_EO=1.2247785","-D","p_REFERENCE_EOD=1.734",str(case/name)],stdout=output,check=True)
print("OCCA Serial compilation/execution and CUDA translation passed.")
