"""Execute actual explicit-target kernel and properties callback with solver doubles."""
from pathlib import Path
import re
import subprocess
import tempfile

case=Path(__file__).resolve().parents[1]
host=(case/'bubbleColumnPhaseMomentum.hpp').read_text()
props='inline void properties(double)' + host.split('inline void properties(double)')[1].split('inline bool singlePass')[0]
kernel='void buildExplicitMixtureDivergence'+(case/'bubbleColumnEquations.okl').read_text().split('@kernel void buildExplicitMixtureDivergence')[1]
kernel=re.sub(r';\s*@tile\(p_blockSize,\s*@outer,\s*@inner\)','',kernel.replace('@ restrict ',''))
# Share the tested memory/linear-algebra doubles, but execute this production callback.
prefix=(case/'tests/phase_alpha_order.py').read_text().split("source = r'''")[1].split("'''",1)[0]
prefix=prefix[:prefix.index('void snapshot(')]
prefix=prefix.replace('struct Nrs {', 'struct Nrs {int fieldOffset=O;')
prefix=prefix.replace('void copyFrom(Memory m,int n,int dst=0,int src=0){', 'void copyFrom(Memory m){copyFrom(m,count);}\n void copyFrom(Memory m,int n,int dst=0,int src=0){')
prefix=prefix.replace('Memory o_explicitTerms(std::string)', 'Memory o_solution(std::string){return o_S.slice(0,O);}\n Memory o_explicitTerms(std::string)')
prefix=prefix.replace('struct Nrs {', 'struct Fluid {Memory o_Ue{3*O},o_U{3*O},o_prop{2*O};}; Fluid fluid;\nstruct Nrs {Fluid *fluid=&::fluid;double g0=1.5;')
prefix=prefix.replace('int mixtureDivergenceForm=2,alphaCompressionEnabled=1;', 'double rhoLiquid=998.2,rhoGas=1.2;int mixtureDivergenceForm=2,alphaCompressionEnabled=1;')
extra=r'''
Memory mixtureExtrapolated(3*O),mixtureAlphaAdvection(O),o_gasAdvector(3*O),o_ug(3*O),
       o_gasTransport(4*O),o_muM(O),o_rhoM(O),o_divSource(O);
int materialCalls=0, assemblies=0;
void evaluatePointwiseTerms(){for(int i=0;i<N;++i){double a=nrs->scalar->o_S[i];o_rhoM[i]=(1-a)*p.rhoLiquid+a*p.rhoGas;o_muM[i]=.001;}}
void evaluateGasCubatureTransport(bool conservative,bool material){
 assert(!conservative && material);++materialCalls;
 double slope=nrs->scalar->o_S[1]-nrs->scalar->o_S[0];
 for(int i=0;i<N;++i)o_gasTransport[i]=o_gasAdvector[i]*slope;
}
void assembleDivergence(Memory){++assemblies;}
'''
# The real target kernel takes raw arrays; this adapter forwards memory-double views.
adapter=r'''
void buildExplicitMixtureDivergenceKernel(int n,double rl,double rg,double g,
 Memory a,Memory h,Memory adv,Memory out){
 buildExplicitMixtureDivergence(n,rl,rg,g,&a[0],&h[0],&adv[0],&out[0]);
}
'''
main=r'''
int main(){
 nrs->meshV=&mesh;
 for(int startup:{0,1})for(double rg:{1.2,998.2}){
  p.rhoGas=rg;nrs->tstep=startup?1:2;nrs->g0=startup?1:1.5;nrs->dt[0]=.1;
  auto &s=*nrs->scalar;s.o_S[0]=.1;s.o_S[1]=.2;
  double past[N]={.08,.16},older[N]={.06,.12};
  for(int i=0;i<N;++i){
   alphaHistory[i]=(startup?past[i]:2*past[i]-.5*older[i])/.1;
   fluid.o_Ue[i]=.4+.1*i;o_ug[i]=.3+i;
  }
  mixtureExtrapolated.copyFrom(fluid.o_Ue,3*O);
  // Subsequent native BC/velocity changes must not change the saved UM^EXT.
  for(int i=0;i<N;++i){fluid.o_Ue[i]=900;fluid.o_U[i]=-700;}
  properties(.2);
  for(int i=0;i<N;++i){
   double a=s.o_S[i],rho=(1-a)*p.rhoLiquid+a*rg;
   double expected=(p.rhoLiquid-rg)/rho*(nrs->g0/.1*a-alphaHistory[i]+(.4+.1*i)*.1);
   assert(std::abs(o_divSource[i]-expected)<1e-13);
   assert(o_gasAdvector[i]==o_ug[i]); // gas transport restored
   assert(fluid.o_U[i]==-700 && fluid.o_Ue[i]==900);
   assert(fluid.o_prop[O+i]==rho);
  }
 }
 assert(materialCalls==4 && assemblies==4);
 nrs->tstep=0;properties(0);assert(materialCalls==4);
 assert(o_divSource[0]==0 && o_divSource[1]==0);
}
'''
with tempfile.TemporaryDirectory() as t:
 p=Path(t);(p/'test.cpp').write_text(prefix+extra+kernel+adapter+props+main)
 subprocess.run(['g++','-std=c++17',str(p/'test.cpp'),'-o',str(p/'test')],check=True)
 subprocess.run([str(p/'test')],check=True)
udf=(case/'bubbleColumn.udf').read_text()
assert 'nrs->userProperties=&bubbleColumn::phaseMomentum::properties' in udf
begin=host.split('inline void beginStep')[1].split('inline void postScalar')[0]
assert begin.index('mixtureExtrapolated.copyFrom(nrs->fluid->o_Ue')<begin.index('addExplicitSources(time)')
assert 'divergenceTransportHistory' not in props and 'buildDivergenceFromAlphaBdf' not in props
print('Explicit target: actual production callback/kernel, BDF1/BDF2 pre-lag history, saved UM_EXT, current alpha, direct material advection, equal-density limit, startup, restored UG advector and unchanged native UM passed.')

# Optional real OKL compilation/execution and CUDA translation, no GPU required.
import os
import signal
occa=Path(os.environ.get('OCCA_DIR',case.parents[1]/'3rd_party/occa'))
if (occa/'lib/libocca.so').exists():
 probe=subprocess.run([str(occa/'bin/occa'),'version'],capture_output=True)
 if probe.returncode==-signal.SIGILL:
  print('OCCA checks unavailable: the installed OCCA executable crashes with SIGILL even for version; no new kernel was loaded.')
  raise SystemExit(0)
 probe.check_returncode()
 code=r'''
#include <occa.hpp>
#include <cassert>
#include <cmath>
int main(int argc,char **argv){
 occa::device d;d.setup(std::string("{mode:'Serial'}"));occa::properties p;
 p["defines/dfloat"]="double";p["defines/dlong"]="int";p["defines/p_blockSize"]=256;
 for(auto item:{std::pair<const char*,int>{"p_Nq",4},{"p_Np",64},{"p_cubNq",6},
     {"p_cubNp",216},{"p_Nvgeo",12},{"p_RXID",0},{"p_RYID",1},{"p_RZID",2},
     {"p_SXID",3},{"p_SYID",4},{"p_SZID",5},{"p_TXID",6},{"p_TYID",7},
     {"p_TZID",8},{"p_JWID",9},{"p_IJWID",10}})p[std::string("defines/")+item.first]=item.second;
 auto target=d.buildKernel(argv[1],"buildExplicitMixtureDivergence",p);
 d.buildKernel(argv[2],"gasMaterialCubatureProducts",p);
 double a[2]={.1,.2},history[2]={1.3,2.6},adv[2]={.04,.05},out[2]={};
 auto A=d.malloc<double>(2,a),H=d.malloc<double>(2,history),V=d.malloc<double>(2,adv),T=d.malloc<double>(2,out);
 target(2,998.2,1.2,15.,A,H,V,T);T.copyTo(out);
 for(int i=0;i<2;++i)assert(std::abs(out[i]-(998.2-1.2)/((1-a[i])*998.2+a[i]*1.2)*(15*a[i]-history[i]+adv[i]))<1e-13);
}
'''
 with tempfile.TemporaryDirectory() as t:
  p=Path(t);(p/'occa.cpp').write_text(code)
  subprocess.run(['g++','-std=c++17',str(p/'occa.cpp'),'-I'+str(occa/'include'),'-L'+str(occa/'lib'),'-Wl,-rpath,'+str(occa/'lib'),'-locca','-o',str(p/'occa')],check=True)
  subprocess.run([str(p/'occa'),str(case/'bubbleColumnEquations.okl'),str(case/'bubbleColumnTransport.okl')],check=True,env=dict(os.environ,OCCA_CACHE_DIR=str(p/'cache')))
  defines={'dfloat':'double','dlong':'int','p_blockSize':256,'p_Nq':4,'p_Np':64,'p_cubNq':6,'p_cubNp':216,'p_Nvgeo':12,'p_RXID':0,'p_RYID':1,'p_RZID':2,'p_SXID':3,'p_SYID':4,'p_SZID':5,'p_TXID':6,'p_TYID':7,'p_TZID':8,'p_JWID':9,'p_IJWID':10}
  flags=[part for k,v in defines.items() for part in ['-D',f'{k}={v}']]
  for name in ['bubbleColumnEquations.okl','bubbleColumnTransport.okl']:
   with (p/(name+'.cu')).open('w') as output:
    subprocess.run([str(occa/'bin/occa'),'translate','-m','CUDA',*flags,str(case/name)],stdout=output,check=True)
 print('New kernels compiled with OCCA Serial; explicit target executed; both OKL files translated for CUDA.')
else:
 print('OCCA checks skipped: set OCCA_DIR to a built OCCA checkout.')
