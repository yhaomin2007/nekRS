"""Recover actual native alpha forcing, weak diffusion, clipping and boundary RHS."""
from pathlib import Path
import re
import subprocess
import tempfile

root=Path(__file__).resolve().parents[3]
def serial(s):
    return re.sub(r';\s*@tile\(p_blockSize,\s*@outer,\s*@inner\)', '',
                  s.replace('@kernel ','').replace('@ restrict ',''))
equations=(root/'examples/bubbleColumn/bubbleColumnEquations.okl').read_text()
kernels=serial(equations.split('@kernel void reconstructGasVelocity')[0])
forcing=serial((root/'src/solver/scalar/kernels/sumMakef.okl').read_text())
cpp='''#include <cassert>
#include <cmath>
using dfloat=double;using dlong=long;
#define p_nEXT 2
#define p_nBDF 2
#define p_SUBCYCLING 0
#define p_MovingMesh 0
#define p_ADVECTION 1
'''+kernels+forcing+'''
int main() {
 const int n=2,o=2;double mass[2]={2,5},rho[2]={1,1},coeff[2]={2,-.5};
 double old[4]={.01,.02,.012,.025},adv[4]={0,0,0,0};
 // Alpha has zero native advection; RHS contains physical UG flux and HPF.
 double physical[4]={-2.,.5,-.4,.2},hpf[4]={-.03,.02,.01,-.04},fu[4];
 for(int i=0;i<4;++i)fu[i]=adv[i]+physical[i]+hpf[i];
 for(int extrapolate=0;extrapolate<2;++extrapolate) {
   double ext[2]={extrapolate?2.:1.,extrapolate?-1.:0.},load[2];
   sumMakef(n,mass,100.,ext,coeff,0,o,o,rho,old,adv,fu,load);
   // Solve (150*M + K) alpha_new = load, K is a weak stiffness matrix.
   const double a=300.02,b=-.02,c=-.02,d=750.02,det=a*d-b*c;
   double solved[2]={(load[0]*d-b*load[1])/det,(a*load[1]-c*load[0])/det};
   // An explicit field filter changes the solved field before clipping.
   double filtered[2]={solved[0]+.01,solved[1]-.02};
   double clipped[2]={filtered[0]<0?0:filtered[0],filtered[1]<0?0:filtered[1]};
   double hist[6]={clipped[0],clipped[1],old[0],old[1],old[2],old[3]},rhs[2];
   recoverAlphaRhs(n,0,o,2,100.,mass,load,hist,coeff,rhs);
   double weak[2]={.02*(solved[0]-solved[1]),.02*(solved[1]-solved[0])};
   for(int i=0;i<n;++i) {
     double expectedExplicit=ext[0]*(physical[i]+hpf[i])+ext[1]*(physical[i+o]+hpf[i+o]);
     assert(std::abs(rhs[i]-expectedExplicit)<1e-12);
     rhs[i]-=weak[i]/mass[i];
     rhs[i]+=150.*(clipped[i]-solved[i]);
     double bdf=100.*(1.5*hist[i]-2.*hist[i+o]+.5*hist[i+2*o]);
     assert(std::abs(rhs[i]-bdf)<1e-12);
   }
   // Imposed inlet equation replaces the interior RHS at masked nodes only.
   dlong ids[1]={1};rhs[1]=123.;double interior=rhs[0];
   alphaBoundaryRhs(1,ids,0,o,2,100.,1.5,hist,coeff,rhs);
   assert(rhs[0]==interior);
   assert(std::abs(rhs[1]-100.*(1.5*hist[1]-2.*hist[3]+.5*hist[5]))<1e-12);
 }
}
'''
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory);(path/'check.cpp').write_text(cpp)
    subprocess.run(['g++','-std=c++17',str(path/'check.cpp'),'-o',str(path/'check')],check=True)
    subprocess.run([str(path/'check')],check=True)
print('Actual native forcing/RHS recovery: EXT1/EXT2, HPF, weak diffusion, filtering/clipping '
      'and imposed boundary correction match the alpha BDF update.')
