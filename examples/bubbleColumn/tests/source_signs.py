"""Physical limiting cases for actual gas/mixture/divergence source kernels."""
from pathlib import Path
import re
import subprocess
import tempfile

root=Path(__file__).resolve().parents[3]
def serial(s):
    return re.sub(r';\s*@tile\(p_blockSize,\s*@outer,\s*@inner\)','',s.replace('@kernel ','').replace('@ restrict ',''))
sources=[root/'examples/bubbleColumn/bubbleColumnEquations.okl',
         root/'src/solver/fluid/kernels/pressureAddQtl.okl',
         root/'src/solver/fluid/kernels/velocityRhsHex3D.okl']
cpp='''#include <cmath>
#include <cassert>
#include <initializer_list>
using dfloat=double;using dlong=long;
'''+''.join(serial(p.read_text()) for p in sources)+r'''
int main(){
 const int n=1,o=1;const double rl=998.2,rg=1.2,g=9.81;
 double alpha[1]={.01},um[3]={},ug[3]={},ul[3]={},ga[3]={},dq[1]={},gu[9]={},al[3]={},gp[3]={};
 double rho[1],mu[1],drift[9],stress[9],sa[1],su[3],lambda[1];
 // Physical upward z: hydrostatic pressure decreases with z. Gas must rise.
 for(double cv:{0.,.5}){
 gp[2]=-rl*g;
 buildEquationTerms(n,o,rl,rg,.001,.000018,1e-6,1.,0.,.003,1.,cv,0.,0.,-g,alpha,um,ug,ul,ga,dq,gu,al,gp,rho,mu,drift,stress,sa,su,lambda);
 assert(su[2]>0.);assert(std::abs(su[2]-(rl-rg)*g/(rg+cv*rl))<1e-12);
 }
 // Drag must dissipate slip energy for either sign of slip, including an
 // implicit timestep much longer than its relaxation time.
 gp[2]=0.;
 for(double sign:{-1.,1.}){
 ug[2]=sign*.3;ul[2]=sign*.1;
 buildEquationTerms(n,o,rl,rg,.001,.000018,1e-6,0.,1.,.003,1.,.5,0.,0.,0.,alpha,um,ug,ul,ga,dq,gu,al,gp,rho,mu,drift,stress,sa,su,lambda);
 double accel=su[2]-lambda[0]*ug[2];
 assert(lambda[0]>0.&&accel*(ug[2]-ul[2])<0.);
 double next=(ug[2]+100.*su[2])/(1.+100.*lambda[0]);
 assert(std::abs(next-ul[2])<std::abs(ug[2]-ul[2]));
 }
 // Common acceleration A: VM must not change acceleration when both phases
 // accelerate together. External gas force rho_g*A plus c*A gives rho_eff*A.
 for(double cv:{0.,.5}){
 al[2]=2.;gp[2]=-rg*2.;
 buildEquationTerms(n,o,rl,rg,.001,.000018,1e-6,1.,0.,.003,1.,cv,0.,0.,0.,alpha,um,ug,ul,ga,dq,gu,al,gp,rho,mu,drift,stress,sa,su,lambda);
 assert(std::abs(su[2]-2.)<1e-13);
 }
 // For an axial sinusoidal mode, div(alpha*tau)_z=-(4/3)*alpha*mu*k^2*u_z.
 // Positive viscous divergence on RHS must damp, not grow, that mode.
 double divStress[1]={-(4./3.)*alpha[0]*.000018*4.},force[1]={};
 addGasStress(1,1e-6,rg+.5*rl,alpha,divStress,force);assert(force[0]<0.);
 // Drift stress moves positive div(tau_d) to the negative mixture RHS.
 double divDrift[3]={0.,0.,3.},mix[3];
 buildMixtureForce(1,1,1e-6,1.,0.,0.,0.,alpha,rho,divDrift,mix);
 assert(mix[2]<0.&&std::abs(mix[2]+3./rho[0])<1e-14);
 // Advecting a decreasing alpha front upward gives positive d_t alpha and
 // positive mixture expansion when UM is small.
 double materialRhs[1]={.3},zero[1]={},div[1];
 buildDivergenceFromAlphaRhs(1,rl,rg,alpha,materialRhs,zero,div);assert(div[0]>0.);
 // Native projection sign: on [0,1], -p''=gamma*rho*d has solution
 // p=gamma*rho*d*z*(1-z)/2. Its velocity correction must have derivative +d.
 const int nn=5;double mass[5],target[5],prhs[5]={},rr[5],bf[15]={},gm[15]={},GP[15]={},vrhs[15];
 double gamma=10.,density=1000.,d=.2;
 for(int i=0;i<nn;++i){mass[i]=1.;target[i]=d;rr[i]=density;double z=i/4.;GP[i+2*nn]=-gamma*density*d*(.5-z);}
 pressureAddQtl(nn,mass,gamma,target,prhs);
 for(int i=0;i<nn;++i)assert(prhs[i]==gamma*d);
 velocityRhsHex3D(nn,nn,rr,bf,gm,GP,vrhs);
 for(int i=1;i<nn-1;++i){double divU=(vrhs[i+1+2*nn]-vrhs[i-1+2*nn])/(.5*gamma*density);assert(std::abs(divU-d)<1e-14);}
}
'''
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory);(path/'check.cpp').write_text(cpp)
    subprocess.run(['g++','-std=c++17',str(path/'check.cpp'),'-o',str(path/'check')],check=True)
    subprocess.run([str(path/'check')],check=True)
print('Physical sign checks passed: hydrostatic buoyancy, dissipative implicit drag, common-acceleration VM, viscous damping, drift force, front expansion and native projection target convention.')
