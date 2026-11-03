"""Compare actual coupled kernels with an independent two-phase inertia solve."""
from pathlib import Path
import re
import subprocess
import tempfile

case=Path(__file__).resolve().parents[1]
source=(case/'bubbleColumn2Equations.okl').read_text()
def kernel(name):
    s='void '+name+source.split('@kernel void '+name)[1].split('@kernel')[0]
    return re.sub(r'; @tile\(p_blockSize, @outer, @inner\)','',s.replace('@ restrict ',''))

cpp=r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <initializer_list>
using dfloat=double;using dlong=long;
''' + kernel('configureCoupledVirtualMass')+kernel('addCoupledVMStress')+r'''
bool close(double x,double y){return std::abs(x-y)<1e-9*(1+std::abs(y));}
int main(){
 const long N=3,O=5;
 double alpha[O]={},ul[3*O]={},ug[3*O]={},gp[3*O]={};
 double gl[9*O]={},gm[9*O]={},lambda[O],src[3*O],drag[3*O],inter[3*O];
 double rp[O],mu[O],coef[4*O],tl[9*O],base[9*O];
 double dg[3*O],dl[3*O],db[3*O],ms[3*O];
 const double rl=998.2,floor=1e-6,gravity[3]={.3,-.5,-9.81},Ki=33;
 for(double rg:{1.2,200.,998.2})for(double a:{.00001,.001,.016667,.05,.8,.99999})
 for(double k:{0.,.1,499.1,1e6}){
  const double b=1-a;
  for(int j=0;j<O;j++){
   alpha[j]=a;lambda[j]=Ki/rg;rp[j]=mu[j]=91;
   for(int c=0;c<3;c++){
    const int id=c*O+j;
    ul[id]=.1+.04*c;ug[id]=.3-.02*c;gp[id]=30-5*c;
    drag[id]=a*(1-rg/rl)*Ki/rg*(ul[id]-ug[id]);
    inter[id]=999999;src[id]=98765;
    dg[id]=a*(.4+c);dl[id]=b*(-.2+.1*c);db[id]=.6-c;ms[id]=91;
   }
   for(int c=0;c<4;c++)coef[c*O+j]=91;
   for(int c=0;c<9;c++){
    gl[c*O+j]=.02*(c+1);gm[c*O+j]=-.03*(c+1);
    tl[c*O+j]=base[c*O+j]=91;
   }
  }
  configureCoupledVirtualMass(N,O,rg,rl,.001,.000018,floor,k,
   gravity[0],gravity[1],gravity[2],alpha,ul,gp,gl,gm,lambda,src,drag,inter,rp,mu,coef,tl,base);
  addCoupledVMStress(N,O,floor,alpha,coef,rp,dg,dl,db,src,ms);
  for(int j=0;j<N;j++){
   // Independent Gaussian elimination of the phase-volume momentum block.
   const double m00=rg+k,m01=-k,m10=-a/b*k,m11=rl+a/b*k;
   const double schur=m11-m10*m01/m00;
   assert(schur>0 && rp[j]>0);
   for(int c=0;c<3;c++){
    const int id=c*O+j;
    const double slip=ul[id]-ug[id];
    double fg=-gp[id]+rg*gravity[c]+Ki*slip+dg[id]/a;
    double fl=-gp[id]+rl*gravity[c]-a/b*Ki*slip+dl[id]/b;
    double al=(fl-m10*fg/m00)/schur;
    double ag=(fg-m01*al)/m00;
    double actualG=src[id]-lambda[j]*ug[id];
    double actualM=gravity[c]-gp[id]/rp[j]+inter[id]+ms[id]+db[id]/rp[j];
    assert(close(actualG,ag));assert(close(actualM,a*ag+b*al));
    assert(inter[id]==drag[id]); // No old acceleration can enter UM.
    // Native BDF1/BDF2 on transformed material accelerations is equivalent
    // to applying the original coupled inertia block to the BDF derivative.
    for(int order:{1,2}){
      const double h=1e-5,g0=order==1?1.:1.5;
      const double hg=order==1?-.3/h:(-2*.3+.5*.2)/h;
      const double hl=order==1?.1/h:(2*.1-.5*.05)/h;
      double gn=(actualG-hg)*h/g0;
      double ln=(al-hl)*h/g0;
      // Scale the residual by the BDF terms before cancellation, especially
      // for the deliberately ill-conditioned k=1e6, beta=1e-5 cases.
      double scaleG=1+std::abs(m00)*(std::abs(g0*gn/h)+std::abs(hg))
          +std::abs(m01)*(std::abs(g0*ln/h)+std::abs(hl))+std::abs(fg);
      double scaleL=1+std::abs(m10)*(std::abs(g0*gn/h)+std::abs(hg))
          +std::abs(m11)*(std::abs(g0*ln/h)+std::abs(hl))+std::abs(fl);
      assert(std::abs(m00*(g0*gn/h+hg)+m01*(g0*ln/h+hl)-fg)<1e-12*scaleG);
      assert(std::abs(m10*(g0*gn/h+hg)+m11*(g0*ln/h+hl)-fl)<1e-12*scaleL);
    }
   }
   // Solve the same block with a unit pressure force for both phases.
   double alP=(1-m10/m00)/schur,agP=(1-m01*alP)/m00;
   assert(close(coef[j+2*O],agP));assert(close(coef[j+3*O],alP));
   assert(close(1/rp[j],a*agP+b*alP));
   if(k==0){assert(close(1/rp[j],a/rg+b/rl));assert(close(lambda[j],Ki/rg));}
   // Positive kinetic energy for the symmetric phase-volume inertia block.
   double x=.7,y=-.2;
   double energy=a*(rg+k)*x*x-2*a*k*x*y+(b*rl+a*k)*y*y;
   assert(energy>0);
  }
  for(int j=N;j<O;j++){
   assert(rp[j]==91 && coef[j]==91 && src[j]==98765 && ms[j]==91);
   assert(tl[j]==91 && base[j]==91 && lambda[j]==Ki/rg);
  }
 }
 // Pure-phase/roundoff alpha inputs are protected; no division by alpha/beta.
 for(double a:{-1e-8,0.,1.,1.00001}){
  for(int j=0;j<O;j++){alpha[j]=a;lambda[j]=0;}
  configureCoupledVirtualMass(N,O,1.2,rl,.001,.000018,floor,499.1,
   0,0,0,alpha,ul,gp,gl,gm,lambda,src,drag,inter,rp,mu,coef,tl,base);
  for(int j=0;j<N;j++)assert(std::isfinite(rp[j]) && rp[j]>0 && std::isfinite(src[j]));
 }
 // A spatially uniform impulse cannot self-amplify when only VM is present.
 // Use the actual force kernel, then a native normalized BDF1 update.
 double maxMagnitude=0;const double dt=1e-5,a=.016667;
 double gas=1e-8,mix=0;
 for(int j=0;j<9*O;j++)gl[j]=gm[j]=0;
 for(int step=0;step<1000;step++){
  alpha[0]=a;ug[0]=gas;ul[0]=(mix-a*gas)/(1-a);
  for(int c=0;c<3*O;c++){gp[c]=0;drag[c]=0;}
  lambda[0]=0;
  configureCoupledVirtualMass(1,O,1.2,rl,.001,.000018,floor,499.1,
   0,0,0,alpha,ul,gp,gl,gm,lambda,src,drag,inter,rp,mu,coef,tl,base);
  gas=(gas/dt+src[0])/(1/dt+lambda[0]);mix+=dt*inter[0];
  maxMagnitude=std::max(maxMagnitude,std::abs(gas));
 }
 assert(maxMagnitude<=1.00000001e-8 && mix==0);
 // Retained semiimplicit gas drag / explicit mixture drag should relax a
 // homogeneous slip under the configured timestep, without VM feedback.
 gas=.3;mix=a*gas;double oldSlip=.3/(1-a);
 for(int step=0;step<5000;step++){
  double liquid=(mix-a*gas)/(1-a),slip=liquid-gas;
  alpha[0]=a;ul[0]=liquid;lambda[0]=33000./1.2;
  drag[0]=a*(1-1.2/rl)*lambda[0]*slip;
  configureCoupledVirtualMass(1,O,1.2,rl,.001,.000018,floor,499.1,
   0,0,0,alpha,ul,gp,gl,gm,lambda,src,drag,inter,rp,mu,coef,tl,base);
  gas=(gas/dt+src[0])/(1/dt+lambda[0]);mix+=dt*inter[0];
  double newSlip=std::abs((mix-gas)/(1-a));
  assert(newSlip<=oldSlip+1e-14 && std::isfinite(gas) && std::abs(gas)<.31);
  oldSlip=newSlip;
 }
 assert(oldSlip<.02);
}
'''
with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp);(p/'check.cpp').write_text(cpp)
    subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror',str(p/'check.cpp'),'-o',str(p/'check')],check=True)
    subprocess.run([str(p/'check')],check=True)
print('Coupled VM actual kernels: independent phase-block solve, pressure response, gas/mixture drag, both phase stresses, positive inertia, zero-VM and pure-phase limits, padding, and no 0D impulse amplification passed.')
