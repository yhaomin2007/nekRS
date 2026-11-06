#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
using dfloat=double;using dlong=int;
#define p_blockSize 1
#define p_Nfaces 6
#define p_Nfp 4
#define p_Nsgeo 4
#define p_NXID 0
#define p_NYID 1
#define p_NZID 2
#define p_WSJID 3
#include "equations_serial.hpp"
static bool near(double a,double b){return std::abs(a-b)<1e-11*std::max({1.,std::abs(a),std::abs(b)});}
int main(){
 constexpr int N=4,O=N;
 double a[N]={0,.03,.7,.99}, mass[N]={.2,.5,2,3};
 double ug[3*N],ul[3*N],bg[3*N],bl[3*N],vg[3*N],vl[3*N],flux[3*N],beta[N],lg[N],ll[N];
 for(int c=0;c<3;++c)for(int i=0;i<N;++i){int k=i+c*O;
  ug[k]=.05*(k+1);ul[k]=-.03*k;
  // BDF history example: historical mixture uses deliberately different alpha.
  bg[k]=mass[i]*(2*ug[k]-.5*ul[k]);bl[k]=mass[i]*(2*ul[k]-.5*ug[k]);
  vg[k]=.02*k;vl[k]=-.01*k;
 }
 drag(N,O,1,1.2,998.2,1e-3,3e-3,a,ug,ul,lg,ll);
 pressureFlux(N,O,1.2,998.2,a,mass,bg,bl,vg,vl,ug,ul,lg,ll,beta,flux);
 for(int i=0;i<N;++i){
  assert(near(beta[i],a[i]/1.2+(1-a[i])/998.2));
  for(int c=0;c<3;++c){int k=i+c*O;double slip=ul[k]-ug[k];
   assert(near(a[i]*1.2*lg[i]*slip,(1-a[i])*998.2*ll[i]*slip));
   double expected=a[i]*(bg[k]/mass[i]+vg[k]+lg[i]*slip)+(1-a[i])*(bl[k]/mass[i]+vl[k]-ll[i]*slip);
   assert(near(flux[k],expected));
  }
 }
 double grad[3*N],cross[N],rhs[N];for(int i=0;i<3*N;++i)grad[i]=.4+i*.1;
 for(int c=0;c<3;++c){
  double base[N];for(int i=0;i<N;++i){base[i]=bg[i+c*O];cross[i]=.07*i;}
  momentumLoad(N,O,c,1/1.2,1,mass,base,grad,lg,ug,ul,cross,rhs);
  for(int i=0;i<N;++i)assert(near(rhs[i],base[i]-mass[i]*grad[i+c*O]/1.2+mass[i]*lg[i]*(ul[i+c*O]-ug[i+c*O])-cross[i]));
  for(int i=0;i<N;++i)base[i]=bl[i+c*O];
  momentumLoad(N,O,c,1/998.2,-1,mass,base,grad,ll,ug,ul,cross,rhs);
  for(int i=0;i<N;++i)assert(near(rhs[i],base[i]-mass[i]*grad[i+c*O]/998.2-mass[i]*ll[i]*(ul[i+c*O]-ug[i+c*O])-cross[i]));
 }
 // Off-diagonal tensor terms and nonuniform element masses catch transposition
 // and accidentally reusing the Jw-weighted gradient as a physical derivative.
 double tensor[9*N],out[3*N];for(int k=0;k<9*N;++k)tensor[k]=(k+1)*.3;
 for(int c=0;c<3;++c){transposeFlux(N,O,c,.2,mass,tensor,out);
  for(int j=0;j<3;++j)for(int i=0;i<N;++i)assert(near(out[i+j*O],.2*tensor[i+(3*j+c)*O]/mass[i]));}
 // Hydrostatic state: beta grad(p)=g -> each phase acceleration is g-grad(p)/rho.
 double g=-9.81,dp=g/beta[1];assert(near(a[1]*(g-dp/1.2)+(1-a[1])*(g-dp/998.2),0));
 // Pressure boundary lifting uses prescribed FUTURE volume flux times c.
 int bids[6]={1,2,3,0,0,0},map[24]={};double sgeo[96]={},u[3*N]={},r[N]={};
 for(int k=0;k<24;++k){map[k]=k%N;sgeo[4*k]=1;sgeo[4*k+3]=.5;}
 for(int i=0;i<N;++i)u[i]=2*(i+1);
 pressureSurface(1,O,sgeo,map,bids,u,r);
 for(int i=0;i<N;++i)assert(near(r[i],-u[i]));
 puts("PASS: production kernels: new-alpha BDF weighting, harmonic pressure coefficient, paired drag, latest pressure loads, full transpose stress, hydrostatic balance, boundary flux");
}
