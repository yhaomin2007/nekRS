"""Run the actual standalone module/kernel using OCCA Serial."""
from pathlib import Path
import subprocess
import tempfile
import os
import numpy as np
from numpy.polynomial.legendre import Legendre, legvander

case=Path(__file__).resolve().parents[1]
occa=Path(os.environ.get('OCCA_DIR', str(case.parents[1]/'3rd_party/occa')))
N=7;nq=N+1;np3=nq**3
x=np.r_[-1.,Legendre.basis(N).deriv().roots(),1.]
L=legvander(x,N);V=L.copy();V[:,2:]=L[:,2:]-L[:,:-2]
def literal(a):return '{'+','.join(format(v,'.17g') for v in np.asarray(a).ravel())+'}'
stub=r'''
#pragma once
#include <occa.hpp>
#include <vector>
#include <stdexcept>
#include <cstdio>
using dfloat=double;using dlong=long;
using deviceKernelProperties=occa::properties;
struct Comm {int mpiComm(){return 0;}int mpiRank(){return 0;}};
struct Options {bool registering=false;
 void getArgs(const char*,int &n){n=7;}
 bool compareArgs(const char*,const char*){return registering;}};
struct LA {std::vector<double> matrixInverse(int n,const std::vector<double>&a){
 auto b=a;std::vector<double>out(n*n,0.);for(int i=0;i<n;++i)out[i*n+i]=1.;
 for(int k=0;k<n;++k){int pivot=k;for(int i=k+1;i<n;++i)if(std::abs(b[i*n+k])>std::abs(b[pivot*n+k]))pivot=i;
 for(int j=0;j<n;++j){std::swap(b[k*n+j],b[pivot*n+j]);std::swap(out[k*n+j],out[pivot*n+j]);}
 double d=b[k*n+k];for(int j=0;j<n;++j){b[k*n+j]/=d;out[k*n+j]/=d;}
 for(int i=0;i<n;++i)if(i!=k){double v=b[i*n+k];for(int j=0;j<n;++j){b[i*n+j]-=v*b[k*n+j];out[i*n+j]-=v*out[k*n+j];}}}
 return out;}};
struct Requests {occa::device *device;std::string file;occa::properties props;
 void add(const std::string&,const std::string&f,const occa::properties&p){file=f;props=p;}
 occa::kernel load(const std::string&,const std::string&name){return device->buildKernel(file,name,props);}};
struct Platform {Comm comm;Options options;LA *linAlg;occa::device device;Requests kernelRequests;} object,*platform=&object;
struct mesh_t {int N;long Nlocal,Nelements;std::vector<double>r;};
template<class... T>void nekrsCheck(bool bad,int,int,const char*,T...){if(bad)throw std::runtime_error("check");}
template<class T>struct deviceMemory {occa::memory memory;
 void resize(long n){memory=platform->device.malloc<T>(n);}
 void copyFrom(const std::vector<T>&a){memory.copyFrom(a.data());}
 operator occa::memory()const{return memory;}};
'''
main='''#include <cassert>
#include <cmath>
#include "explicitFilter.hpp"
int main(){LA la;platform->linAlg=&la;occa::properties deviceProps;deviceProps["mode"]="Serial";platform->device.setup(deviceProps);
platform->kernelRequests.device=&platform->device;
deviceKernelProperties props;props["defines/p_Nq"]=8;props["defines/p_Np"]=512;
props["defines/dfloat"]="double";props["defines/dlong"]="long";
platform->options.registering=true;
userExplicitFilter::registerKernels(props,"'''+str(case/'explicitFilter.okl')+'''");
platform->options.registering=false;userExplicitFilter::registerKernels(props);
mesh_t mesh{7,1024,2,'''+literal(np.tile(x,nq*nq))+'''};
'''
for modes in (1,2,3):
    for strength in (0.,.05,.25,1.):
        sigma=np.ones(nq)
        sigma[nq-modes:]-=strength*(np.arange(1,modes+1)/modes)**2
        F=(V*sigma)@np.linalg.inv(V)
        # Mixed bubble modes plus a linear component exercise transpose and
        # all three axes, endpoint preservation, and nontrivial modal weights.
        field=np.einsum('k,j,i->kji',V[:,7],V[:,6],V[:,5])+.2*np.einsum('k,j,i->kji',V[:,0],V[:,1],V[:,0])
        expected=np.einsum('ka,jb,ic,abc->kji',F,F,F,field)
        data=np.r_[field.ravel(),2*field.ravel(),[99.,98.]]
        target=np.r_[expected.ravel(),2*expected.ravel(),[99.,98.]]
        main+='''{std::vector<double>data='''+literal(data)+''',expected='''+literal(target)+''';
auto mem=platform->device.malloc<double>(data.size(),data.data());
userExplicitFilter::explicitFilterApply("test",&mesh,'''+str(modes)+','+str(strength)+''',mem);
mem.copyTo(data.data());for(size_t i=0;i<data.size();++i)assert(std::abs(data[i]-expected[i])<2e-12);
}
'''
main+='''bool bad=false;auto mem=platform->device.malloc<double>(1024);
try{userExplicitFilter::explicitFilterApply("bad",&mesh,7,.1,mem);}catch(const std::runtime_error&){bad=true;}assert(bad);
}
'''
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory);(path/'udf.hpp').write_text(stub)
    (path/'check.cpp').write_text(main)
    subprocess.run(['g++','-std=c++11',str(path/'check.cpp'),'-I'+str(path),
        '-I'+str(case),'-I'+str(occa/'include'),'-L'+str(occa/'lib'),
        '-Wl,-rpath,'+str(occa/'lib'),'-locca','-o',str(path/'check')],check=True)
    env=dict(os.environ,OCCA_CACHE_DIR=str(path/'cache'))
    subprocess.run([str(path/'check')],check=True,env=env)
    with (path/'filter.cu').open('w') as output:
        subprocess.run([str(occa/'bin/occa'),'translate','-m','CUDA',
            '-D','p_Nq=8','-D','p_Np=512','-D','dfloat=double','-D','dlong=long',
            str(case/'explicitFilter.okl')],stdout=output,check=True)
print('Actual module and kernel: OCCA Serial matches bubble-basis reference for two elements, modes/strength/cache changes, padding and invalid modes; CUDA translation passed.')
