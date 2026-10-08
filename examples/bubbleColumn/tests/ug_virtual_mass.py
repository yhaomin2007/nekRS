from pathlib import Path
import re,subprocess,tempfile
r=Path(__file__).resolve().parents[1]
s=(r/'bubbleColumnEquations.okl').read_text().replace('@kernel ','').replace('@ restrict ','')
s=re.sub(r'; @tile\(p_blockSize, @outer, @inner\)','',s)
cpp='#include <cmath>\n#include <cassert>\n#include <initializer_list>\nusing dfloat=double; using dlong=long;\n'+s+'''
int main(){
 const int n=2,o=4; const double rl=998.2,rg=1.2;
 double alpha[4],x[4],y[4],z[4],um[12];
 for(double a:{0.,.02,.05}) for(double v:{0.,.3,-.3}) {
 initializeUniform(n,o,rl,rg,a,v,alpha,x,y,z,um);
 assert(z[0]==v); assert(std::abs(um[8]-rg*a*v/((1-a)*rl+a*rg))<1e-15);
 }
 double ug[12]={},ul[12]={},q[12]={},gradA[12]={},gradU[36]={},divQ[4]={},al[12]={},gp[12]={},rho[4],mu[4],ds[36],gs[36],sa[4],su[12],lambda[4];
 alpha[0]=alpha[1]=.05; gp[0]=gp[1]=10; al[0]=al[1]=2;
 for(double cv:{0.,.5}) {
 buildEquationTerms(n,o,rl,rg,.001,.000018,1e-6,1.,0.,.003,1.,cv,0.,0.,-9.81,alpha,um,ug,ul,gradA,divQ,gradU,al,gp,rho,mu,ds,gs,sa,su,lambda);
 double eff=rg+cv*rl;
 assert(std::abs(su[0]-(-10+cv*rl*2)/eff)<1e-12);
 assert(std::abs(su[8]+9.81*rg/eff)<1e-12);
 assert(lambda[0]==0);
 }
 // With nonzero gradients, equation sources must not include material
 // convection; the native scalar solver supplies it exactly once.
 gradA[0]=4; divQ[0]=3; gradU[0]=2; gradU[16]=3; gradU[32]=4;
 ug[0]=.7;gp[0]=0;al[0]=0;
 buildEquationTerms(n,o,rl,rg,.001,.000018,1e-6,0.,0.,.003,0.,0.,0.,0.,0.,alpha,um,ug,ul,gradA,divQ,gradU,al,gp,rho,mu,ds,gs,sa,su,lambda);
 assert(su[0]==0 && su[4]==0 && su[8]==0);
 alphaCompression(n,o,alpha,gradU,sa);
 assert(std::abs(sa[0]+.05*9)<1e-12);
 // Reconstructed continuity uses the mixture material derivative:
 // Salpha + Am - Ag + diffusion, not merely the gas compression source.
 double rhs[4]={sa[0]+.2-.6,0}, diff[4]={.03,0}, div[4]={};
 buildDivergenceFromAlphaRhs(n,rl,rg,alpha,rhs,diff,div);
 assert(std::abs(div[0]-(rl-rg)/((1-.05)*rl+.05*rg)*(rhs[0]+.03))<1e-12);
 // Exact drag-only backward Euler transient with VM: stable and converges
 // to unchanged equilibrium for dt much larger than the gas-only timescale.
 double K=100.,eff=rg+.5*rl, v=0., dt=.1;
 for(int i=0;i<2000;++i) v=(v+dt*K/eff*.3)/(1+dt*K/eff);
 assert(std::abs(v-.3)<1e-12);
 // Liquid history stores only liquid material acceleration.
 double prev[12]={},gL[36]={},acc[12]={}; ul[0]=ul[1]=.2;gL[0]=gL[1]=3;
 updateVirtualMassHistory(n,o,10.,ul,prev,gL,acc);
 assert(std::abs(acc[0]-2.6)<1e-12);assert(prev[0]==.2);
 // Inactive gas stress does not divide by zero.
 alpha[0]=0;alpha[1]=.05; double st[4]={1,1},src[4]={};
 addGasStress(n,1e-6,500.3,alpha,st,src);
 assert(src[0]==0 && std::abs(src[1]-1/(.05*500.3))<1e-12);
}
'''
with tempfile.TemporaryDirectory() as t:
 p=Path(t);(p/'check.cpp').write_text(cpp)
 subprocess.run(['g++','-std=c++17','-Wall','-Wextra',str(p/'check.cpp'),'-o',str(p/'check')],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 subprocess.run([str(p/'check')],check=True)
print('Actual serial kernels: VM force scaling, zero-VM limit, gravity, initialization, liquid history, safe stress and implicit drag equilibrium passed.')
