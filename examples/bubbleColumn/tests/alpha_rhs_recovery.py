"""Reduced alpha RHS: transport/diffusion only; HPF/postprocessing stay in BDF diagnostic."""
from pathlib import Path
import re,subprocess,tempfile
root=Path(__file__).resolve().parents[3]
def serial(s):
 return re.sub(r';\s*@tile\(p_blockSize,\s*@outer,\s*@inner\)','',s.replace('@kernel ','').replace('@ restrict ',''))
equations=(root/'examples/bubbleColumn/bubbleColumnEquations.okl').read_text()
kernels=serial(equations.split('@kernel void reconstructGasVelocity')[0])
kernels+=serial('@kernel void buildDivergenceFromAlphaRhs'+equations.split('@kernel void buildDivergenceFromAlphaRhs',1)[1].split('@kernel void buildOutletRampFactor',1)[0])
forcing=serial((root/'src/solver/scalar/kernels/sumMakef.okl').read_text())
cpp='''#include <cassert>
#include <cmath>
using dfloat=double;using dlong=long;
#define p_nEXT 2
#define p_nBDF 2
#define p_SUBCYCLING 0
#define p_MovingMesh 0
#define p_ADVECTION 1
'''+kernels+forcing+r'''
int main(){
 const int n=2,o=2;double mass[2]={2,5},rho[2]={1,1},coeff[2]={2,-.5};
 double old[4]={.01,.02,.012,.025},adv[4]={};
 double physical[4]={-2.,.5,-.4,.2},hpf[4]={-.03,.02,.01,-.04},fu[4];
 for(int i=0;i<4;++i)fu[i]=physical[i]+hpf[i];
 for(int extrapolate=0;extrapolate<2;++extrapolate){
 double ext[2]={extrapolate?2.:1.,extrapolate?-1.:0.},load[2];
 sumMakef(n,mass,100.,ext,coeff,0,o,o,rho,old,adv,fu,load);
 // Actual native forcing includes HPF. Implicit diffusion is weak stiffness.
 const double a=300.02,b=-.02,c=-.02,d=750.02,det=a*d-b*c;
 double solved[2]={(load[0]*d-b*load[1])/det,(a*load[1]-c*load[0])/det};
 double filtered[2]={solved[0]+.01,solved[1]-.02};
 double final[2]={filtered[0]<0?0:filtered[0],filtered[1]<0?0:filtered[1]};
 double hist[6]={final[0],final[1],old[0],old[1],old[2],old[3]},rhs[2];
 extrapolateAlphaTransportRhs(n,o,2,ext,physical,rhs);
 double diffusion[2]={-.02*(solved[0]-solved[1])/mass[0],-.02*(solved[1]-solved[0])/mass[1]};
 double material[2]={.31,-.17},divRhs[2],divBdf[2];
 for(int i=0;i<n;++i){
 double expected=ext[0]*physical[i]+ext[1]*physical[i+o];
 assert(std::abs(rhs[i]-expected)<1e-14);rhs[i]+=material[i];
 }
 buildDivergenceFromAlphaRhs(n,1000.,1.,final,rhs,diffusion,divRhs);
 buildDivergenceFromAlphaBdf(n,0,o,2,100.,1.5,1000.,1.,hist,coeff,material,divBdf);
 for(int i=0;i<n;++i){
 double factor=999./(1000.-999.*final[i]);
 double transport=ext[0]*physical[i]+ext[1]*physical[i+o];
 double excludedHpf=ext[0]*hpf[i]+ext[1]*hpf[i+o];
 double excludedPost=150.*(final[i]-solved[i]);
 assert(std::abs(divRhs[i]-factor*(transport+diffusion[i]+material[i]))<1e-11);
 assert(std::abs(divRhs[i]-divBdf[i]+factor*(excludedHpf+excludedPost))<1e-11);
 assert(std::abs(divRhs[i]-divBdf[i])>1e-6);
 }
 }
 // Padding and coefficient/time-history layout.
 double padded[10]={1,2,99,99,99,3,4,99,99,99},ext[2]={2,-1},out[5]={99,99,99,99,99};
 extrapolateAlphaTransportRhs(2,5,2,ext,padded,out);
 assert(out[0]==-1&&out[1]==0&&out[2]==99);
}
'''
with tempfile.TemporaryDirectory() as d:
 p=Path(d);(p/'check.cpp').write_text(cpp)
 subprocess.run(['g++','-std=c++17',str(p/'check.cpp'),'-o',str(p/'check')],check=True)
 subprocess.run([str(p/'check')],check=True)
host=(root/'examples/bubbleColumn/bubbleColumnTerms.hpp').read_text()
reconstruction=host.split('inline void buildDivergenceFromAlphaRhs')[1].split('inline void buildDivergenceFromAlphaBdf')[0]
assert all(x not in reconstruction for x in ['o_JwF','o_alphaPostprocessRate','o_coeffBDF','Nmasked','alphaBoundaryRhs'])
assert 'o_alphaBeforePostprocess' in reconstruction
source=host.split('inline void addExplicitSources(double)')[1].split('inline void buildDivergenceFromAlphaRhs')[0]
assert source.index('subtractScalarDiffusion();')<source.index('storeAlphaTransportRhs();')
print('Actual reduced RHS/divergence kernels: EXT1/EXT2 transport + weak diffusion + UM advection only; HPF/filter/clipping corrections excluded, full BDF discrepancy predicted, padding preserved.')
