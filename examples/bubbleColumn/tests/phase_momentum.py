"""Execute production phase kernels; optional real OCCA Serial/CUDA checks."""
from pathlib import Path
import configparser
import os
import re
import subprocess
import tempfile

case = Path(__file__).resolve().parents[1]
kernel = (case / 'bubbleColumnPhaseMomentum.okl').read_text()
serial = kernel.replace('@kernel ', '').replace('@restrict ', '')
serial = re.sub(r';\s*@tile\(p_blockSize,\s*@outer,\s*@inner\)', '', serial)
prefix = r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <initializer_list>
using dfloat=double; using dlong=int;
bool close(double x,double y,double tol=2e-10) {
 return std::abs(x-y)<=tol*(1+std::max(std::abs(x),std::abs(y)));
}
'''
main = r'''
int main() {
 constexpr int N=3, O=5;
 double a[O]={.02,0,1},ug[3*O]={},ul[3*O]={},p[3*O]={};
 double rate[O]={2,3,4},sg[3*O]={},sl[3*O]={},cg[4*O]={},cl[3*O]={};
 double rg[3*O],rl[3*O],terms[18*O];
 std::fill(rg,rg+3*O,123);std::fill(rl,rl+3*O,123);
 for (int c=0;c<3;++c) for (int n=0;n<N;++n) {
   int k=n+c*O;ug[k]=.2*(c+1);ul[k]=-.03*(n+1);
   p[k]=17*(n+1);sg[k]=.04*(n+1);sl[k]=-.08*(c+1);
   cg[k+O]=.02*(n+1);cl[k]=-.07*(c+1);
 }
 for (double cvm:{0.,.5,10.}) {
   phasePredictor(N,O,1.2,998.2,cvm,1e-6,0,0,-9.81,
      a,ug,ul,p,rate,sg,sl,cg,cl,rg,rl,terms);
   for (int c=0;c<3;++c) for (int n=0;n<N;++n) {
     int k=n+c*O;double b=std::max(1-a[n],1e-6),r=a[n]/b,C=cvm*998.2;
     double K=rate[n]*(1.2+C),g=c==2?-9.81:0;
     double drag=K*(ul[k]-ug[k]);
     double fg=-p[k]+1.2*g+drag+sg[k]/std::max(a[n],1e-6);
     double fl=-p[k]+998.2*g-r*drag+sl[k]/b;
     double ag=rg[k]+cg[k+O],al=rl[k]+cl[k];
     assert(close((1.2+C)*ag-C*al,fg));
     assert(close(-r*C*ag+(998.2+r*C)*al,fl,1e-6));
     double sum=0;for(int t=0;t<6;++t)sum+=terms[k+3*t*O];
     assert(close(sum,rg[k],2e-7));
     // Interfacial drag and VM reactions cancel at ordinary alpha.
     if(n==0) {
       double vm=C*(al-ag);
       assert(close(a[n]*(drag+vm)+(1-a[n])*(-r*(drag+vm)),0));
       assert(drag*(ug[k]-ul[k])<=0); // dissipative drag
     }
   }
   assert(rg[3]==123 && rl[4]==123); // padding untouched
 }
 // Shared acceleration for equal densities: uniform pressure produces no slip.
 double zero[4*O]={};double ae[O]={.4,.2,.7};
 phasePredictor(N,O,10,10,.5,1e-6,0,0,0,ae,zero,zero,p,zero,zero,zero,
                zero,zero,rg,rl,terms);
 for(int c=0;c<3;++c)for(int n=0;n<N;++n) {
   assert(close(rg[n+c*O],-p[n+c*O]/10));assert(close(rg[n+c*O],rl[n+c*O]));
 }
 // All masks, vanishing phases, BDF coefficients, and very stiff local drag.
 for (double alpha:{0.,1e-12,.02,.5,1-1e-12,1.})
 for (double cvm:{0.,.5,100.}) for(double h:{1.,150000.})
 for (double K:{0.,10.,1e12}) for (int bits=0;bits<4;++bits) {
   double A[1]={alpha},G[3]={.3,-.5,1.2},L[3]={-.2,.4,0};
   const double U[3]={.05,-.03,.2},saved[3]={.05,-.03,.2};
   double R[1]={K/(1.2+cvm*998.2)},fG[3],fL[3],dg[3],dl[3],be[3],af[3];
   std::fill(fG,fG+3,bits&1?1:0);std::fill(fL,fL+3,bits&2?1:0);
   phaseCorrect(1,1,1.2,998.2,cvm,1e-6,h,A,U,R,fG,fL,G,L,dg,dl,be,af);
   double rho=alpha*1.2+(1-alpha)*998.2;
   double wg=alpha*1.2/rho,wl=(1-alpha)*998.2/rho;
   bool active=((bits&1)&&wg>0)||((bits&2)&&wl>0);
   for(int c=0;c<3;++c) {
     assert(U[c]==saved[c]);assert(std::isfinite(G[c])&&std::isfinite(L[c]));
     assert(close(U[c]-wg*G[c]-wl*L[c],af[c]));
     if(active)assert(std::abs(af[c])<2e-14);
     if(!(bits&1))assert(dg[c]==0);if(!(bits&2))assert(dl[c]==0);
   }
   // Full free block response satisfies the same pressure load in both rows.
   if(bits==3) {
     double r=alpha/std::max(1-alpha,1e-6),C=cvm*998.2;
     double ag=h*(1.2+C)+K,al=h*(998.2+r*C)+r*K,off=h*C+K;
     for(int c=0;c<3;++c) {
       double scale=1+std::abs(ag*dg[c])+std::abs(off*dl[c])+
                       std::abs(al*dl[c])+std::abs(r*off*dg[c]);
       assert(std::abs(ag*dg[c]-off*dl[c]-al*dl[c]+r*off*dg[c])<2e-14*scale);
     }
   }
 }
 // BDF history is independent of current solution, with new pressure added
 // to both momentum rows. Deliberate gas error gives the expected VM reaction.
 {
   double A[1]={.02},G[3],L[3],HG[3]={.7,-.1,.3},HL[3]={-.8,.2,.1};
   double RG[3]={1,2,3},RL[3]={-2,0,1},Pold[3]={-5,-5,-5},Pnew[3]={6,6,6};
   double F[3]={1,1,1},DG[3],DL[3];
   double h=150000.,C=.5*998.2,r=.02/.98;
   for(int c=0;c<3;++c){G[c]=(HG[c]+RG[c])/h;L[c]=(HL[c]+RL[c])/h;}
   phaseMomentumDefect(1,1,1.2,998.2,.5,1e-6,h,A,G,L,HG,HL,RG,RL,Pold,Pnew,F,F,DG,DL);
   for(int c=0;c<3;++c){assert(std::abs(DG[c])<1e-12);assert(std::abs(DL[c])<1e-12);}
   G[0]+=.01;
   phaseMomentumDefect(1,1,1.2,998.2,.5,1e-6,h,A,G,L,HG,HL,RG,RL,Pold,Pnew,F,F,DG,DL);
   assert(close(DG[0],(1.2+C)*h*.01/1.2));assert(close(DL[0],-r*C*h*.01/998.2));
   double fixed[3]={0,0,0};
   phaseMomentumDefect(1,1,1.2,998.2,.5,1e-6,h,A,G,L,HG,HL,RG,RL,Pold,Pnew,fixed,F,DG,DL);
   assert(DG[0]==0);assert(close(DL[0],-r*C*h*.01/998.2));
 }
 puts("Production phase kernels: VM equations, conservative reactions, drag dissipation, pressure, all masks, dilute limits, stiff response, BDF momentum defects, padding and unchanged UM passed.");
}
'''
with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp)
    (p/'test.cpp').write_text(prefix+serial+main)
    subprocess.run(['g++','-std=c++17','-O2',str(p/'test.cpp'),'-o',str(p/'test')],check=True)
    subprocess.run([str(p/'test')],check=True)

par=configparser.ConfigParser(inline_comment_prefixes=('#',),strict=True)
par.read_string('[ROOT]\n'+(case/'bubbleColumn.par').read_text())
assert [x.strip() for x in par['GENERAL']['scalars'].split(',')]==['ALPHA','UGX','UGY','UGZ','ULX','ULY','ULZ']
assert len(par['CASEDATA'])==18
udf=(case/'bubbleColumn.udf').read_text()
host=(case/'bubbleColumnPhaseMomentum.hpp').read_text()
correct=host.split('inline void correct()')[1].split('inline void reportVector')[0]
assert '->solve(' not in udf+host
assert 'nrs->fluid->o_U,' in correct and 'o_U.copyFrom' not in correct
assert udf.index('phaseMomentum::correct()')<udf.index('phaseMomentum::diagnostics(time)')
assert 'initializeMasks()' not in udf and '!elliptic' in host
assert 'rebuildMixture' not in udf+host
assert 'nrs->scalar->o_solution(liquidNames[c]).copyFrom(o_ul' in correct
assert 'phaseMomentum::singlePass' in udf
assert 'ugTerms::' not in udf and 'oneThrough::' not in udf
print('Active seven-scalar configuration, native callbacks, lazy masks, corrected phase histories and no UM reset verified.')

occa=Path(os.environ.get('OCCA_DIR',case.parents[1]/'3rd_party/occa'))
if not (occa/'lib/libocca.so').exists():
    print('OCCA runtime checks skipped: set OCCA_DIR to a built checkout.')
else:
    code=r'''
#include <occa.hpp>
#include <cassert>
int main(int argc,char **argv) {
 occa::device d;d.setup(std::string("{mode:'Serial'}"));occa::properties p;
 p["defines/dfloat"]="double";p["defines/dlong"]="int";p["defines/p_blockSize"]=256;
 for(const char *name:{"liquidStress","phasePredictor","phaseMask","phaseCorrect","phaseNonfinite","phaseMomentumDefect"})d.buildKernel(argv[1],name,p);
 auto k=d.buildKernel(argv[1],"phaseCorrect",p);
 double a[1]={.02},um[3]={.01,.02,.03},ug[3]={.3,.4,.5},ul[3]={0,0,0};
 double rate[1]={2},free[3]={1,1,1},out[3]={};
 auto A=d.malloc<double>(1,a),U=d.malloc<double>(3,um),G=d.malloc<double>(3,ug),L=d.malloc<double>(3,ul);
 auto R=d.malloc<double>(1,rate),F=d.malloc<double>(3,free),dg=d.malloc<double>(3,out),dl=d.malloc<double>(3,out);
 auto before=d.malloc<double>(3,out),after=d.malloc<double>(3,out);
 k(1,1,1.2,998.2,.5,1e-6,150000.,A,U,R,F,F,G,L,dg,dl,before,after);
 after.copyTo(out);for(double x:out)assert(std::abs(x)<1e-14);
 U.copyTo(out);for(int c=0;c<3;++c)assert(out[c]==um[c]);
}
'''
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp);(p/'occa.cpp').write_text(code)
        subprocess.run(['g++','-std=c++17',str(p/'occa.cpp'),'-I'+str(occa/'include'),'-L'+str(occa/'lib'),'-Wl,-rpath,'+str(occa/'lib'),'-locca','-o',str(p/'occa')],check=True)
        subprocess.run([str(p/'occa'),str(case/'bubbleColumnPhaseMomentum.okl')],check=True,env=dict(os.environ,OCCA_CACHE_DIR=str(p/'cache')))
        with (p/'phase.cu').open('w') as out:
            subprocess.run([str(occa/'bin/occa'),'translate','-m','CUDA','-D','p_blockSize=256','-D','dfloat=double','-D','dlong=int',str(case/'bubbleColumnPhaseMomentum.okl')],stdout=out,check=True)
    print('All production phase kernels built with OCCA Serial; correction executed; CUDA translation passed.')
