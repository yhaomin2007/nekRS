"""Check the actual UDF Urst wrapper; optionally execute native OKL via OCCA."""
from pathlib import Path
import os,re,subprocess,tempfile
case=Path(__file__).resolve().parents[1];repo=case.parents[1]
h=(case/'bubbleColumn2Native.hpp').read_text()
a=h.index('inline void computeGasUrst()');b=h.index('\ninline void addExplicitSources',a)
wrapper=h[a:b]
assert 's->computeUrst()' not in h
kernel=repo/'src/app/nrs/kernels/UrstCubatureHex3D.okl'
signature=kernel.read_text().split(')\n{',1)[0]
assert signature.count(',')+1==10 and 'Woffset' in signature
# The kernel's typed scalar/memory sequence must match even without OCCA.
common=r'''
#include <cassert>
#include <string>
using dlong=int;
struct Mem { int id; };
struct Scalar { dlong vFieldOffset=13,vCubatureOffset=17;Mem o_U{3},o_relUrst{4}; } scalar;
struct Mesh { dlong Nelements=2;Mem o_cubvgeo{1},o_cubInterpT{2}; } mesh;
struct Nrs { Scalar* scalar=&::scalar;Mesh* meshV=&mesh; } app;auto nrs=&app;
Mem o_NULL{0};bool called=false;
void launchKernel(const char* name,dlong n,int relative,Mem geo,Mem interp,
                  dlong offset,dlong Woffset,dlong cubOffset,Mem U,Mem W,Mem out)
{
 assert(std::string(name)=="nrs-UrstCubatureHex3D");
 assert(n==2 && relative==0 && geo.id==1 && interp.id==2);
 assert(offset==13 && Woffset==0 && cubOffset==17);
 assert(U.id==3 && W.id==0 && out.id==4);called=true;
}
'''
with tempfile.TemporaryDirectory() as t:
 p=Path(t)/'launch.cpp';p.write_text(common+wrapper+'\nint main(){computeGasUrst();assert(called);}')
 subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror',str(p),'-o',str(p.with_suffix(''))],check=True)
 subprocess.run([str(p.with_suffix(''))],check=True)
print('Actual UG Urst wrapper: ten arguments, typed strides/memory and independent gas buffers passed.')
# Execute the same wrapper against the real OCCA native cubature kernel.
if os.getenv('OCCA_DIR'):
 occa=Path(os.environ['OCCA_DIR'])
 cpp=r'''
#include <occa.hpp>
#include <cassert>
#include <cmath>
#include <vector>
using dlong=int;
struct Scalar { dlong vFieldOffset=11,vCubatureOffset=13;occa::memory o_U,o_relUrst; } scalar;
struct Mesh { dlong Nelements=1;occa::memory o_cubvgeo,o_cubInterpT; } mesh;
struct Nrs { Scalar* scalar=&::scalar;Mesh* meshV=&mesh; } app;auto nrs=&app;
occa::memory o_NULL;occa::kernel urst;
void launchKernel(const char*,dlong n,int relative,occa::memory geo,occa::memory interp,
                  dlong offset,dlong Woffset,dlong cubOffset,occa::memory U,occa::memory W,occa::memory out)
{ urst(n,relative,geo,interp,offset,Woffset,cubOffset,U,W,out); }
'''+wrapper+r'''
int main(int argc,char **argv){
 assert(argc==2);occa::device d;d.setup(std::string("{mode:'Serial'}"));occa::properties p;
 p["defines/dfloat"]="double";p["defines/dlong"]="int";
 p["defines/p_Nq"]=2;p["defines/p_Np"]=8;p["defines/p_cubNq"]=2;p["defines/p_cubNp"]=8;p["defines/p_Nvgeo"]=12;
 const char *ids[]={"RXID","RYID","RZID","SXID","SYID","SZID","TXID","TYID","TZID","JID","JWID","IJWID"};
 for(int i=0;i<12;++i)p[std::string("defines/p_")+ids[i]]=i;
 urst=d.buildKernel(argv[1],"UrstCubatureHex3D",p);
 std::vector<double> geo(96,0),interp{1,0,0,1},u(33,-777),out(39,-999);
 for(int n=0;n<8;++n){geo[n]=2;geo[n+4*8]=3;geo[n+8*8]=4;geo[n+10*8]=.5;
  for(int c=0;c<3;++c)u[n+c*11]=(c+1)*(.2+.01*n);}
 mesh.o_cubvgeo=d.malloc<double>(geo.size(),geo.data());mesh.o_cubInterpT=d.malloc<double>(interp.size(),interp.data());
 scalar.o_U=d.malloc<double>(u.size(),u.data());scalar.o_relUrst=d.malloc<double>(out.size(),out.data());
 computeGasUrst();scalar.o_relUrst.copyTo(out.data());
 for(int c=0;c<3;++c){for(int n=0;n<8;++n)assert(std::abs(out[n+c*13]-.5*(c+2)*u[n+c*11])<1e-13);
  for(int n=8;n<13;++n)assert(out[n+c*13]==-999);}
}
'''
 with tempfile.TemporaryDirectory() as t:
  p=Path(t)/'occa.cpp';p.write_text(cpp);binary=p.with_suffix('')
  subprocess.run(['g++','-std=c++17',str(p),'-I'+str(occa/'include'),'-L'+str(occa/'lib'),'-Wl,-rpath,'+str(occa/'lib'),'-locca','-ldl','-pthread','-o',str(binary)],check=True)
  subprocess.run([str(binary),str(kernel)],check=True)
 print('Actual UDF wrapper + native UrstCubatureHex3D OCCA Serial execution: metric/velocity transform and stride padding passed.')
else:
 print('OCCA runtime portion skipped: set OCCA_DIR to enable.')
