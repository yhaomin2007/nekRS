#include <cassert>
#include <cmath>
#include <cstdio>
using dfloat=double; using dlong=int;
#define p_blockSize 1
#define p_Nfaces 6
#define p_Nfp 4
#define p_Nsgeo 4
#define p_NXID 0
#define p_NYID 1
#define p_NZID 2
#define p_WSJID 3
#include "equations_serial.hpp"
static bool near(double a,double b) {return std::abs(a-b)<1e-11*std::max(1.,std::max(std::abs(a),std::abs(b)));}
int main() {
  constexpr int N=3,O=3;
  double a[N]={0,0.05,0.9},ug[3*N]={0.1,0.2,0.3,0,0,0,0.3,0.3,0.3},ul[3*N]={};
  double um[3*N],lg[N],ll[N],bg[N],bl[N],bm[N];
  reconstruct(N,O,a,ug,ul,um);
  for(int i=0;i<N;++i)assert(near(um[i+2*O],a[i]*0.3));
  drag(N,O,1,1.2,998.2,1e-3,3e-3,a,ug,ul,lg,ll);
  for(int i=0;i<N;++i) {
    assert(std::isfinite(lg[i])&&lg[i]>0&&std::isfinite(ll[i]));
    assert(near(a[i]*1.2*lg[i],(1-a[i])*998.2*ll[i]));
  }
  const double tau=1e-5/1.5;
  mobility(N,tau,1.2,998.2,a,lg,ll,bg,bl,bm);
  for(int i=0;i<N;++i) {
    assert(near((1+tau*lg[i])*bg[i]-tau*lg[i]*bl[i],tau/1.2));
    assert(near(-tau*ll[i]*bg[i]+(1+tau*ll[i])*bl[i],tau/998.2));
    assert(near(a[i]*1.2*bg[i]+(1-a[i])*998.2*bl[i],tau));
  }
  double grad[3*N]={3,7,-2,1,-4,2,-10,-20,-30},before[3*N];
  for(int i=0;i<3*N;++i) before[i]=um[i];
  correct(N,O,bg,grad,ug);correct(N,O,bl,grad,ul);reconstruct(N,O,a,ug,ul,um);
  for(int c=0;c<3;++c)for(int i=0;i<N;++i)
    assert(near(um[i+c*O],before[i+c*O]-bm[i]*grad[i+c*O]));
  // Zero-slip drag is finite; zero-alpha produces no liquid force.
  double zero[3*N]={};drag(N,O,1,1.2,998.2,1e-3,3e-3,a,zero,zero,lg,ll);
  for(int i=0;i<N;++i)assert(std::isfinite(lg[i])&&std::isfinite(ll[i]));
  assert(ll[0]==0);
  // Drag-free hydrostatic single-liquid limit: gravity predictor is canceled.
  drag(N,O,0,1.2,998.2,1e-3,3e-3,a,zero,zero,lg,ll);
  mobility(N,tau,1.2,998.2,a,lg,ll,bg,bl,bm);
  for(int i=0;i<N;++i) {zero[i+2*O]=tau*(-9.81);grad[i+2*O]=998.2*(-9.81);}
  correct(N,O,bl,grad,zero);
  for(int i=0;i<N;++i)assert(near(zero[i+2*O],0));
  // Source must retain material convection (a zero convection regression
  // would silently turn both momentum equations into diffusion equations).
  double conv[4*N]={0,0,0,2,3,4,5,6,7,8,9,10},source[N];
  phaseSource(N,O,2,-9.81,conv,source);
  for(int i=0;i<N;++i)assert(near(source[i],-conv[i+3*O]-9.81));
  // Frozen drag coupling: production partner loads converge to the 2x2
  // backward-Euler solution, preserving common gravity and total momentum.
  double aa[1]={0.05},gg[3]={0,0,0.3},llv[3]={0,0,0},cg[1],cl[1];
  drag(1,1,1,1.2,998.2,1e-3,3e-3,aa,gg,llv,cg,cl);
  for(int common=0;common<2;++common) {
    const double ag=1/tau,oldg=common?0.1:0.3,oldl=common?0.1:0.0;
    const double gravity=common?-9.81:0;
    gg[2]=oldg;llv[2]=oldl;
    double mass[1]={1};
    for(int it=0;it<100;++it) {
      double gr[1]={ag*oldg+gravity},lr[1]={ag*oldl+gravity};
      addDragLoad(1,1,2,mass,cg,llv,gr);addDragLoad(1,1,2,mass,cl,gg,lr);
      gg[2]=gr[0]/(ag+cg[0]);llv[2]=lr[0]/(ag+cl[0]);
    }
    if(common) {assert(near(gg[2],oldg+tau*gravity));assert(near(llv[2],oldl+tau*gravity));}
    else assert(near(aa[0]*1.2*gg[2]+(1-aa[0])*998.2*llv[2],aa[0]*1.2*oldg+(1-aa[0])*998.2*oldl));
  }
  // Pressure-Neumann surface subtracts prescribed incoming volume flux.
  double sgeo[6*4*4]={};int map[24],ids[6]={1,2,3,0,0,0};double rhs[N]={};
  double flux[3*N]={1,2,3,0,0,0,0,0,0};
  for(int j=0;j<24;++j) {map[j]=j%N;sgeo[4*j]=1;sgeo[4*j+3]=0.25;}
  pressureSurface(1,O,sgeo,map,ids,flux,rhs);
  for(int i=0;i<N;++i) {
    double expected=0;for(int j=0;j<24;++j)if(map[j]==i&&(ids[j/4]==1||ids[j/4]==3))expected-=0.25*flux[i];
    assert(near(rhs[i],expected));
  }
  puts("PASS: reconstruction, drag balance, finite zero-slip, phase mobilities, shared-pressure correction, hydrostatic limit, convection source, boundary flux");
}
