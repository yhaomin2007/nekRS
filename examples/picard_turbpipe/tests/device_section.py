"""Use v26's real OKL extraction; check the copied turbPipe BCs compile/work."""
from pathlib import Path
import subprocess,tempfile
case=Path(__file__).resolve().parents[1];repo=case.parents[1]
base=(repo/'examples/turbPipe/turbPipe.udf').read_text().split('#ifdef __okl__',1)[1].split('#endif',1)[0]
new=(case/'picard_turbpipe.udf').read_text().split('#ifdef __okl__',1)[1].split('#endif',1)[0]
assert [line.rstrip() for line in base.strip().splitlines()]==[line.rstrip() for line in new.strip().splitlines()]
with tempfile.TemporaryDirectory() as d:
 p=Path(d)
 (p/'extract.c').write_text('void extract_ifdef(const char*,const char*,const char*);\nint main(int n,char**a){extract_ifdef("__okl__",a[1],a[2]);return 0;}\n')
 subprocess.run(['gcc',str(repo/'src/platform/utils/unifdef.c'),str(p/'extract.c'),'-o',str(p/'extract')],check=True)
 subprocess.run([str(p/'extract'),str(case/'picard_turbpipe.udf'),str(p/'device.cpp')],check=True)
 extracted=(p/'device.cpp').read_text()
 assert 'udfDirichlet' in extracted and 'frozenInner' not in extracted and 'turbPipeExecuteStep' not in extracted
 stub=r'''
#include <cmath>
#include <cassert>
#include <string>
using dfloat=double;
std::string field;
bool isField(const char *f){return field==f;}
struct bcData{double uxFluid=0,uyFluid=0,uzFluid=0,pFluid=0,sScalar=0,nx=0,ny=0,nz=1,z=1;
 double *usrwrk;long idxVol=0,fieldOffset=1;int id=1;};
'''
 test=r'''
int main(){double work[3]={1,2,3};bcData bc;bc.usrwrk=work;
field="fluid velocity";udfDirichlet(&bc);assert(bc.uxFluid==1&&bc.uyFluid==2&&bc.uzFluid==3);
field="scalar s00";udfDirichlet(&bc);assert(bc.sScalar==0);bc.id=3;udfDirichlet(&bc);assert(bc.sScalar==.5);
field="fluid pressure";bc.uzFluid=-.1;udfDirichlet(&bc);assert(bc.pFluid<0.);
}
'''
 (p/'test.cpp').write_text(stub+'\n#define __okl__\n'+extracted+test)
 subprocess.run(['g++','-std=c++17',str(p/'test.cpp'),'-o',str(p/'check')],check=True)
 subprocess.run([str(p/'check')],check=True)
print('Native OKL extraction finds the boundary section; original turbPipe BCs are preserved and execute correctly.')
