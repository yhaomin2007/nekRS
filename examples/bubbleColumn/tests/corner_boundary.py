"""Compile actual boundary function and check inlet/wall agreement at rim."""
from pathlib import Path
import subprocess
import tempfile
root=Path(__file__).resolve().parents[1]
source=(root/'bubbleColumnBoundary.oudf').read_text()
host=(root/'bubbleColumnTerms.hpp').read_text()
profile='inline dfloat inletProfileWeight'+host.split('inline dfloat inletProfileWeight')[1].split('inline void registerKernels')[0]
pre='''#include <cmath>
#include <cstring>
#include <cassert>
#include <algorithm>
using dfloat=double;
#define p_ALPHA_INLET .016667
#define p_RHO_LIQUID 998.2
#define p_RHO_GAS 1.2
#define p_UG_INLET .3
struct bcData { const char *fieldName; int id,idxVol; double *usrwrk;
 double uxFluid,uyFluid,uzFluid,nx,ny,nz,pFluid,sScalar; };
#define isField(name) (!strcmp(bc->fieldName,name))
'''
main='''int main() {
 double mask[2]={0,1}; bcData b{}; b.usrwrk=mask;
 for(const char *field: {"scalar ugz", "fluid velocity"}) {
   b.fieldName=field;b.idxVol=0;b.id=1;udfDirichlet(&b);
   double inlet=(!strcmp(field,"scalar ugz"))?b.sScalar:b.uzFluid;
   b.id=3;udfDirichlet(&b);
   double wall=(!strcmp(field,"scalar ugz"))?b.sScalar:b.uzFluid;
   assert(inlet==0 && wall==inlet);
   b.id=1;b.idxVol=1;udfDirichlet(&b);
   double interior=(!strcmp(field,"scalar ugz"))?b.sScalar:b.uzFluid;
   double expected=(!strcmp(field,"scalar ugz"))?p_UG_INLET:
     p_ALPHA_INLET*p_RHO_GAS*p_UG_INLET/((1-p_ALPHA_INLET)*p_RHO_LIQUID+p_ALPHA_INLET*p_RHO_GAS);
   assert(std::abs(interior-expected)<1e-14);
 }
 b.fieldName="scalar alpha";b.id=1;b.idxVol=0;udfDirichlet(&b);
 assert(b.sScalar==p_ALPHA_INLET);
 // Actual smooth profile: flat fallback, exact wall, midpoint and bulk.
 assert(inletProfileWeight(.001,0.)==1.);
 assert(inletProfileWeight(-.001,.01)==0.);
 assert(inletProfileWeight(0.,.01)==0.);
 assert(std::abs(inletProfileWeight(.0025,.01)-.15625)<1e-14);
 assert(inletProfileWeight(.005,.01)==.5);
 assert(inletProfileWeight(.01,.01)==1.);
 assert(inletProfileWeight(.02,.01)==1.);
 double previous=-1.;
 for(int i=0;i<=100;++i) {
   mask[1]=inletProfileWeight(i*.0001,.01);
   assert(mask[1]>=previous && mask[1]>=0. && mask[1]<=1.);previous=mask[1];
   b.idxVol=1;b.id=1;b.fieldName="scalar ugz";udfDirichlet(&b);
   double ug=b.sScalar;
   assert(std::abs(ug-.3*mask[1])<1e-14);
   b.fieldName="fluid velocity";udfDirichlet(&b);
   double rho=(1-p_ALPHA_INLET)*p_RHO_LIQUID+p_ALPHA_INLET*p_RHO_GAS;
   assert(std::abs(rho*b.uzFluid-p_ALPHA_INLET*p_RHO_GAS*ug)<1e-14);
 }
}
'''
with tempfile.TemporaryDirectory() as t:
 p=Path(t);(p/'test.cpp').write_text('#include <initializer_list>\n'+pre+profile+source+main)
 subprocess.run(['g++','-std=c++17',str(p/'test.cpp'),'-o',str(p/'test')],check=True)
 subprocess.run([str(p/'test')],check=True)
print('Actual inlet profile and boundary: smooth monotone wall-to-bulk ramp, zero-width fallback, rim agreement and stationary-liquid mixture relation passed.')
