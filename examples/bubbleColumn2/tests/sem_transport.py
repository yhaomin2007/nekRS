"""Execute actual native SEM derivative kernels and Method2 UG contraction."""
from pathlib import Path
import os,subprocess,tempfile
case=Path(__file__).resolve().parents[1];repo=case.parents[1]
h=(case/'bubbleColumn2Terms.hpp').read_text()
assert 'opSEM::strongDivergence(mesh, offset, o_qg, divergence)' in h
assert 'opSEM::strongGradVec(mesh, offset, o_ug, o_gradUg)' in h
assert 'gasMaterialConvectionKernel(mesh->Nlocal, offset, o_ug, o_gradUg, o_gasTransport)' in h
for removed in ('gasInterpolateKernel','gasProjectKernel','gasCubatureProductsKernel','alphaConservativeCubatureKernel','evaluateGasCubatureTransport','bubbleColumn2Transport.okl'):
 assert removed not in h,removed
assert 'nrs->scalar->o_coeffEXT = o_scalarCoeffEXT' in h
assert 'freezeDragHistory();' in h
assert 'virtualMassEnabled = 0' in (case/'bubbleColumn2.par').read_text()
occa=Path(os.environ['OCCA_DIR'])
cpp=r'''
#include <occa.hpp>
#include <vector>
#include <cassert>
#include <cmath>
bool close(double a,double b){return std::abs(a-b)<1e-12;}
int main(int argc,char**argv){
 assert(argc==3);occa::device d;d.setup(std::string("{mode:'Serial'}"));occa::properties p;
 p["defines/dfloat"]="double";p["defines/dlong"]="int";p["defines/p_blockSize"]=256;
 p["headers"].asArray();p["headers"] += "#include <math.h>";
 p["defines/p_Nq"]=3;p["defines/p_Np"]=27;p["defines/p_Nvgeo"]=12;
 const char* ids[]={"RXID","RYID","RZID","SXID","SYID","SZID","TXID","TYID","TZID","JWID","JID","IJWID"};
 for(int i=0;i<12;++i)p[std::string("defines/p_")+ids[i]]=i;
 auto grad=d.buildKernel(std::string(argv[1])+"/gradientVolumeHex3D.okl","gradientVolumeHex3D",p);
 auto div=d.buildKernel(std::string(argv[1])+"/divergenceVolumeHex3D.okl","divergenceVolumeHex3D",p);
 auto dot=d.buildKernel(argv[2],"gasMaterialConvection",p);
 const int N=27,O=31;double D[9]={-1.5,2,-.5,-.5,0,.5,.5,-2,1.5};
 std::vector<double> geo(12*N,0),u(3*O,777),q(3*O,777),g(9*O,888),out(4*O,999),a(O),expected(N);
 double x[3]={-1,0,1},w[3]={1./3,4./3,1./3};
 for(int k=0;k<3;++k)for(int j=0;j<3;++j)for(int i=0;i<3;++i){int n=i+3*j+9*k;
  geo[n]=geo[n+4*N]=geo[n+8*N]=1;geo[n+9*N]=w[i]*w[j]*w[k];
  a[n]=.2+.01*x[i]+.02*x[j]+.03*x[k];
  // A non-diagonal affine velocity gradient, divergence=1+2+3=6.
  u[n]=.3+x[i]+.2*x[j];u[n+O]=.4+2*x[j]+.1*x[k];u[n+2*O]=.5+3*x[k]+.3*x[i];
  expected[n]=.01*u[n]+.02*u[n+O]+.03*u[n+2*O]+6*a[n];
  for(int c=0;c<3;++c)q[n+c*O]=a[n]*u[n+c*O];
 }
 auto G=d.malloc<double>(geo.size(),geo.data()),DD=d.malloc<double>(9,D),U=d.malloc<double>(u.size(),u.data()),Q=d.malloc<double>(q.size(),q.data());
 auto GR=d.malloc<double>(g.size(),g.data()),T=d.malloc<double>(out.size(),out.data());
 div(1,G,DD,O,Q,T);T.copyTo(out.data());
 double integral=0;
 for(int n=0;n<N;++n){assert(close(out[n]/geo[n+9*N],expected[n]));integral+=out[n];}
 // Exact integral of the affine divergence on [-1,1]^3: 8*(.01*.3+.02*.4+.03*.5+6*.2).
 assert(close(integral,8*(.003+.008+.015+1.2)));
 for(int n=N;n<O;++n)assert(out[n]==999);
 for(int c=0;c<3;++c){grad(1,G,DD,O,U.slice(c*O,N),GR.slice(3*c*O,3*O));}
 GR.copyTo(g.data());
 // A one-element mesh's native runAvg divides the weakly stored derivatives by JW.
 // Duplicate interface copies undergo the same gather/inverse assembled mass operation.
 for(int c=0;c<9;++c)for(int n=0;n<N;++n)g[n+c*O]/=geo[n+9*N];
 GR.copyFrom(g.data());dot(N,O,U,GR,T);T.copyTo(out.data());
 for(int n=0;n<N;++n){double expectedU[3]={u[n]+.2*u[n+O],2*u[n+O]+.1*u[n+2*O],.3*u[n]+3*u[n+2*O]};
  for(int c=0;c<3;++c){assert(close(out[n+(c+1)*O],expectedU[c]));
   if(std::abs(u[n+c*O])>.1)assert(!close(out[n+(c+1)*O],expectedU[c]+6*u[n+c*O]));}
 }
 for(int c=0;c<3;++c)for(int n=N;n<O;++n)assert(out[n+(c+1)*O]==999);
}
'''
with tempfile.TemporaryDirectory() as t:
 p=Path(t)/'sem.cpp';p.write_text(cpp);binary=p.with_suffix('')
 subprocess.run(['g++','-std=c++17',str(p),'-I'+str(occa/'include'),'-L'+str(occa/'lib'),'-Wl,-rpath,'+str(occa/'lib'),'-locca','-ldl','-pthread','-o',str(binary)],check=True)
 subprocess.run([str(binary),str(repo/'src/core/kernels'),str(case/'bubbleColumn2Equations.okl')],check=True)
print('Actual native SEM gradient/divergence and UG contraction passed: compression, analytic derivatives, gas-volume integral, JW normalization, material UG form, padding and routing.')
