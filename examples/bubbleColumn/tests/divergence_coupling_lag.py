"""Exercise the actual lag kernel/callback, padded fields and weighted output."""
from pathlib import Path
import csv, math, re, subprocess, tempfile
r=Path(__file__).resolve().parents[1]
host=(r/'bubbleColumnTerms.hpp').read_text()
kernel=(r/'bubbleColumnDiagnostics.okl').read_text().split('@kernel void divergenceCouplingLag')[1]
kernel='void divergenceCouplingLag'+kernel.replace('@ restrict ','')
kernel=re.sub(r';\s*@tile\(p_blockSize,\s*@outer,\s*@inner\)','',kernel)
callback='inline void diagnoseDivergenceCoupling'+host.split('inline void diagnoseDivergenceCoupling')[1].split('inline void printDivergenceComparison')[0]
# Reuse the reduction harness; the production callback and kernel are compiled.
pre=(r/'tests/divergence_comparison.py').read_text().split("pre='''")[1].split("'''",1)[0]
pre=pre.replace('struct Mem { double *a;', 'struct Mem { double *a; operator double*() const {return a;}')
pre=pre[:pre.index('struct Nrs')]+r'''
#include <cassert>
struct Scalar {Mem o_solution(const char*) {return {alpha};}double alpha[5]={};} scalar;
double used[15]={2./3.,-4./3.,2.,123.,123.},current[15]={};
double gradient[15]={},lag[5]={123,123,123,123,123},scratch[5]={};
Mem o_mixtureAdvectorUsed{used},o_divCouplingGradAlpha{gradient},o_divCouplingLag{lag},o_divComparisonScratch{scratch};
struct Fluid {Mem o_U{current};} fluid;
struct Nrs {Mesh *meshV;long fieldOffset=5;Scalar *scalar;Fluid *fluid;} nrsObject{&mesh,5,&scalar,&fluid},*nrs=&nrsObject;
struct Parameters {int divergenceComparisonInterval=10;double rhoLiquid=4,rhoGas=1;} p;
int gradCalls=0;
namespace opSEM {void strongGrad(Mesh*,long offset,Mem,Mem out){
 ++gradCalls;for(int i=0;i<3;++i){out.a[i]=2.;out.a[i+offset]=out.a[i+2*offset]=0.;}
}}
'''
cpp=pre+kernel+'\n#define divergenceCouplingLagKernel divergenceCouplingLag\n'+callback+r'''
int main(){
 diagnoseDivergenceCoupling(0.,0);assert(gradCalls==0&&lag[0]==123.);
 diagnoseDivergenceCoupling(.9,9);assert(gradCalls==1&&lag[0]==1.&&lag[1]==-2.&&lag[2]==3.&&lag[3]==123.);
 diagnoseDivergenceCoupling(1.,10);assert(gradCalls==2);
 // Identical velocities or a uniform alpha field must give zero mismatch.
 double a[5]={-.2,1.2,.5},u[15]={},g[15]={},out[5]={123,123,123,123,123};
 for(int c=0;c<3;++c)for(int i=0;i<3;++i){u[i+5*c]=c+1.;g[i+5*c]=c+2.;}
 double zero[15]={};
 divergenceCouplingLag(3,5,4,1,a,u,zero,g,out);
 for(int i=0;i<3;++i){double ac=std::max(0.,std::min(1.,a[i]));assert(std::abs(out[i]-60./(4.-3.*ac))<1e-12);}
 assert(out[3]==123.);
 divergenceCouplingLag(3,5,4,1,a,u,u,g,out);for(int i=0;i<3;++i)assert(out[i]==0.);
 divergenceCouplingLag(3,5,4,1,a,u,zero,zero,out);for(int i=0;i<3;++i)assert(out[i]==0.);
 divergenceCouplingLag(3,5,1,1,a,u,zero,g,out);for(int i=0;i<3;++i)assert(out[i]==0.);
}
'''
with tempfile.TemporaryDirectory() as t:
 q=Path(t);(q/'check.cpp').write_text(cpp)
 subprocess.run(['g++','-std=c++17',str(q/'check.cpp'),'-o',str(q/'check')],check=True)
 result=subprocess.run([str(q/'check')],cwd=t,text=True,capture_output=True,check=True)
 assert result.stdout.count('divUmCouplingLag')==1
 rows=list(csv.DictReader((q/'bubbleColumn_divergence_coupling.csv').open()))
 assert len(rows)==1
 row=rows[0]
 for key,value in dict(lag_min=-2,lag_max=3,lag_mean=1,lag_rms=math.sqrt(36/6)).items():
  assert abs(float(row[key])-value)<1e-12,(key,row[key],value)
# Check that the snapshot is taken at source entry and diagnostic precedes limiter.
source=host.split('inline void addExplicitSources(double)')[1]
assert source.index('o_mixtureAdvectorUsed.copyFrom')<source.index('prepareGasAdvection()')
udf=(r/'bubbleColumn.udf').read_text().split('void UDF_ExecuteStep')[1]
assert udf.index('diagnoseDivergenceCoupling')<udf.index('clipMixtureVelocity()')
assert '{"scalar04", divCouplingLag}' in udf
print('Coupling lag: actual kernel/callback, three components, sign, density clamp, padded strides, zero limits, fresh gradient, interval and weighted CSV passed.')
