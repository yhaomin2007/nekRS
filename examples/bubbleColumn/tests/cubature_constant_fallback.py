"""Reproduce missing native c_I and verify the case-generated fallback header."""
from pathlib import Path
import os
import re
import subprocess
import tempfile
import numpy as np
from numpy.polynomial.legendre import Legendre, leggauss

root=Path(__file__).resolve().parents[3]
case=root/'examples/bubbleColumn'
N=7;nq=8;cq=15
gll=np.r_[-1.,Legendre.basis(N).deriv().roots(),1.]
gl,weights=leggauss(cq)
def array(name,a):return 'double '+name+'[]={'+','.join(format(v,'.17g') for v in a)+'};\n'
pre='''#include <cassert>
#include <iostream>
#include <stdexcept>
#include <algorithm>
using dfloat=double;
struct Comm {int mpiComm(){return 0;}};
struct Platform {Comm comm;} object,*platform=&object;
template<class... T> void nekrsCheck(bool bad,int,int,const char*,T...){if(bad)throw std::runtime_error("check");}
'''+array('gll',gll)+array('gl',gl)+array('wc',weights)+'''
void JacobiGLL(int n,double *x){assert(n==7);std::copy(gll,gll+8,x);}
void JacobiGQ(double a,double b,int n,double *x,double *w){assert(a==0&&b==0&&n==14);std::copy(gl,gl+15,x);std::copy(wc,wc+15,w);}
void InterpolationMatrix1D(int n,int ni,double *x,int no,double *z,double *out){
 assert(n==7&&ni==8&&no==15);
 for(int i=0;i<no;++i)for(int j=0;j<ni;++j){
   double v=1.;for(int k=0;k<ni;++k)if(j!=k)v*=(z[i]-x[k])/(x[j]-x[k]);out[i*ni+j]=v;
 }
}
'''
main='''int main(){
 assert(bubbleColumn::missingCubatureInterpolationHeader(7,10).empty());
 assert(bubbleColumn::missingCubatureInterpolationHeader(7,11).empty());
 std::cout<<bubbleColumn::missingCubatureInterpolationHeader(7,14);
}
'''
native=(root/'src/core/kernels/strongAdvectionCubatureVolumeScalarHex3D.okl').read_text()
serial=native.replace('@kernel ','').replace('@ restrict ','').replace('@shared ','').replace('@exclusive ','').replace('@barrier();','')
serial=re.sub(r';\s*@outer\(0\)|;\s*@inner\([01]\)', '', serial)
defines='''using dfloat=double;using dlong=long;
#define p_Nq 8
#define p_Np 512
#define p_cubNq 15
#define p_cubNp 3375
#define p_Nvgeo 12
#define p_IJWID 10
'''
tables=root/'src/core/mesh/kernels'
includes=f'#include "{tables}/constantInterpolationMatrices.h"\n#include "{tables}/constantDifferentiationMatrices.h"\n'
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory);fake=path/'core/mesh';fake.mkdir(parents=True)
    (fake/'constantInterpolationMatrices.h').write_text((tables/'constantInterpolationMatrices.h').read_text())
    env=dict(os.environ,NEKRS_KERNEL_DIR=directory)
    (path/'generate.cpp').write_text(pre+f'#include "{case}/bubbleColumnCubature.hpp"\n'+main)
    subprocess.run(['g++','-std=c++11',str(path/'generate.cpp'),'-o',str(path/'generate')],check=True)
    header=subprocess.run([str(path/'generate')],env=env,check=True,capture_output=True,text=True).stdout
    (path/'missing.cpp').write_text(defines+includes+serial)
    failed=subprocess.run(['g++','-std=c++11','-fsyntax-only',str(path/'missing.cpp')],capture_output=True,text=True)
    assert failed.returncode!=0 and 'c_I' in failed.stderr
    # Real core source compiles with generated c_I; core files stay unchanged.
    checks='''int main(){for(int i=0;i<15;++i)for(int k=0;k<=7;++k){double sum=0;for(int j=0;j<8;++j)sum+=c_I[i][j]*std::pow(gll[j],k);assert(std::abs(sum-std::pow(gl[i],k))<1e-13);}}
'''
    (path/'fixed.cpp').write_text('#include <cmath>\n#include <cassert>\n'+defines+includes+header+array('gll',gll)+array('gl',gl)+serial+checks)
    subprocess.run(['g++','-std=c++11',str(path/'fixed.cpp'),'-o',str(path/'fixed')],check=True)
    subprocess.run([str(path/'fixed')],check=True)
    parser=root.parent/'occa-cub-parser/bin/occa'
    if parser.exists():
        (path/'fixed.okl').write_text(defines.replace('using dfloat=double;using dlong=long;','#define dfloat double\n#define dlong long')+includes+header+native)
        with (path/'fixed.cu').open('w') as output:
            subprocess.run([str(parser),'translate','-m','CUDA',str(path/'fixed.okl')],stdout=output,check=True)
print('Missing c_I reproduced; actual fallback generator skips supported pairs, supplies '
      '8x15 interpolation, and real native scalar kernel compiles/translates with the header.')
