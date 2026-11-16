"""Execute actual inlet BCs in both modes, including unequal ramp durations."""
from pathlib import Path
import configparser
import subprocess
import tempfile

case=Path(__file__).resolve().parents[1]
boundary=(case/'bubbleColumnBoundary.oudf').read_text()
pre=(case/'tests/corner_boundary.py').read_text().split("pre='''")[1].split("'''",1)[0]
pre=pre.replace('#define p_STARTUP_MODE 0','')
pre=pre.replace('double uxFluid,uyFluid', 'double time,uxFluid,uyFluid')
pre+='''
#define p_INLET_RAMP_START .2
#define p_ALPHA_RAMP_DURATION .4
#define p_UG_RAMP_DURATION .1
'''
main=r'''
double smooth(double t,double duration){
 if(t<=.2)return 0;if(t>=.2+duration)return 1;
 double x=(t-.2)/duration;return 6*pow(x,5)-15*pow(x,4)+10*pow(x,3);
}
int main(){
 double mask[2]={0,1};bcData b{};b.usrwrk=mask;
 double previousAlpha=-1,previousGas=-1;
 for(double t:{0.,.1,.2,.20001,.225,.25,.275,.3,.4,.5,.6,.7,2.}){
  b.time=t;b.id=1;b.idxVol=1;
  b.fieldName="scalar alpha";udfDirichlet(&b);double a=b.sScalar;
  b.fieldName="scalar ugz";udfDirichlet(&b);double g=b.sScalar;
  double expectA=p_ALPHA_INLET*(p_STARTUP_MODE!=0?smooth(t,.4):1.);
  double expectG=p_UG_INLET*(p_STARTUP_MODE==1?smooth(t,.1):1.);
  assert(std::abs(a-expectA)<1e-14 && std::abs(g-expectG)<1e-14);
  assert(a>=previousAlpha-1e-14 && g>=previousGas-1e-14);
  previousAlpha=a;previousGas=g;
  b.fieldName="fluid velocity";udfDirichlet(&b);
  double rho=(1-a)*p_RHO_LIQUID+a*p_RHO_GAS;
  assert(b.uxFluid==0 && b.uyFluid==0);
  assert(std::abs(rho*b.uzFluid-a*p_RHO_GAS*g)<1e-14);
  // Reconstructed liquid remains stationary throughout the inlet ramp.
  assert(std::abs((rho*b.uzFluid-a*p_RHO_GAS*g)/((1-a)*p_RHO_LIQUID))<1e-14);
  for(const char *field:{"scalar ugz","fluid velocity"}){
   b.fieldName=field;b.id=1;b.idxVol=0;udfDirichlet(&b);
   double rim=!strcmp(field,"scalar ugz")?b.sScalar:b.uzFluid;
   b.id=3;udfDirichlet(&b);
   double wall=!strcmp(field,"scalar ugz")?b.sScalar:b.uzFluid;
   assert(rim==0 && wall==0);
  }
 }
}
'''
with tempfile.TemporaryDirectory() as t:
 p=Path(t);(p/'test.cpp').write_text('#include <initializer_list>\n'+pre+boundary+main)
 for mode in [0,1,2]:
  subprocess.run(['g++','-std=c++17',f'-Dp_STARTUP_MODE={mode}',str(p/'test.cpp'),'-o',str(p/'test')],check=True)
  subprocess.run([str(p/'test')],check=True)
par=configparser.ConfigParser(inline_comment_prefixes=('#',))
par.read_string('[ROOT]\n'+(case/'bubbleColumn.par').read_text())
assert par['CASEDATA'].getint('startupMode')==2
assert par['CASEDATA'].getfloat('alphaInletRampDuration')>0
assert par['CASEDATA'].getfloat('gasInletRampDuration')>0
udf=(case/'bubbleColumn.udf').read_text()
assert 'if (platform->options.getArgs("RESTART FILE NAME").empty())' in udf
assert 'if (p.startupMode==1 || p.startupMode==2)' in udf and 'initializePlumeKernel' in udf
assert 'fill(nrs->fieldOffset,p.ugInitial,nrs->scalar->o_solution("ugz"))' in udf
assert 'read("gasinitialvelocity",p.ugInitial)' in udf
print('Actual BCs: all three modes, constant UG in uniform-start mode, delayed quintic ramps, unequal alpha/UG durations, zero startup, monotonicity, final plateaus, synchronized mass-weighted UM, stationary liquid and wall/rim precedence passed.')
