"""Production anisotropic flux + real native weak SEM operators on two hexes."""
from pathlib import Path
import ctypes
import json
import os
import subprocess
import tempfile
import numpy as np
from numpy.polynomial.legendre import Legendre

case=Path(__file__).resolve().parents[1]
root=case.parents[1]
header=(case/'bubbleColumnAlphaStabilization.hpp').read_text()

def function(text,signature):
    start=text.index(signature); brace=text.index('{',start); depth=1; end=brace+1
    while depth:
        depth+=(text[end]=='{')-(text[end]=='}');end+=1
    return text[start:end]

occa=Path(os.environ.get('OCCA_DIR',root/'3rd_party/occa'))
if not (occa/'lib/libocca.so').exists():
    raise SystemExit('Build OCCA first or set OCCA_DIR: real weak-operator checks are required.')

nq=4;np3=nq**3;ne=2;N=ne*np3;offset=N+5;nvgeo=12
x=np.r_[-1.,Legendre.basis(nq-1).deriv().roots(),1.]
w=2/((nq-1)*nq*Legendre.basis(nq-1)(x)**2)
bary=np.array([1/np.prod(x[i]-np.delete(x,i)) for i in range(nq)])
D=np.zeros((nq,nq))
for i in range(nq):
    for j in range(nq):
        if i!=j:D[i,j]=bary[j]/bary[i]/(x[i]-x[j])
    D[i,i]=-D[i].sum()
geo=np.zeros((ne,nvgeo,np3));localMass=np.zeros(N);coord=np.zeros((3,N));ids=[];G=[]
Gr=np.kron(np.eye(nq),np.kron(np.eye(nq),D))
Gs=np.kron(np.eye(nq),np.kron(D,np.eye(nq)))
Gt=np.kron(D,np.kron(np.eye(nq),np.eye(nq)))
for e,(left,right) in enumerate([(-1.,0.),(0.,2.)]):
    for k in range(nq):
        for j in range(nq):
            for i in range(nq):
                node=i+nq*j+nq*nq*k;n=e*np3+node
                coord[:,n]=[left+(x[i]+1)*(right-left)/2,x[j],x[k]]
                geo[e,0,node]=2/(right-left);geo[e,4,node]=1;geo[e,8,node]=1
                geo[e,10,node]=(right-left)/2*w[i]*w[j]*w[k]
                localMass[n]=geo[e,10,node]
                ids.append((e*(nq-1)+i)+(2*(nq-1)+1)*(j+nq*k))
    G.append(np.vstack([geo[e,0,:,None]*Gr, Gs, Gt]))
ids=np.asarray(ids);globalN=ids.max()+1
mass=np.bincount(ids,weights=localMass,minlength=globalN)
alpha=.2+.03*np.sin(2*coord[0])+.02*coord[1]**2+.01*coord[2]**3
ug=np.zeros((3,offset));ug[:,:N]=np.vstack([.03*np.cos(coord[1]),.02*np.sin(coord[0]),.3+.04*coord[1]])

code=r'''
#include <occa.hpp>
#include <vector>
#include <limits>
#include <algorithm>
using dfloat=double;
#define JWID 10
'''+function(header,'inline dfloat geometryBound(')+r'''
extern "C" double bound(int nq,int nvgeo,int ne,const double *geo,const double *D) {
 return geometryBound(nq,nvgeo,ne,std::vector<double>(geo,geo+ne*nvgeo*nq*nq*nq),std::vector<double>(D,D+nq*nq));
}
extern "C" void run(int ne,int offset,const double *geo,const double *D,const double *alpha,const double *ug,
 double cw,double sw,double eps,double *gradient,double *flux,double *load) {
 static occa::device device;
 static occa::kernel grad,weak,tensor;
 if(!device.isInitialized()) {
   device.setup(std::string("{mode:'Serial'}"));occa::properties p;
   p["defines/dfloat"]="double";p["defines/dlong"]="int";p["defines/p_blockSize"]=256;
   p["defines/p_Nq"]=4;p["defines/p_Np"]=64;p["defines/p_Nvgeo"]=12;
   p["defines/p_RXID"]=0;p["defines/p_RYID"]=1;p["defines/p_RZID"]=2;
   p["defines/p_SXID"]=3;p["defines/p_SYID"]=4;p["defines/p_SZID"]=5;
   p["defines/p_TXID"]=6;p["defines/p_TYID"]=7;p["defines/p_TZID"]=8;p["defines/p_JWID"]=10;
   grad=device.buildKernel(GRAD_FILE,"gradientVolumeHex3D",p);
   weak=device.buildKernel(WEAK_FILE,"wDivergenceVolumeHex3D",p);
   tensor=device.buildKernel(TENSOR_FILE,"alphaStabilizationFlux",p);
   device.buildKernel(TENSOR_FILE,"alphaStabilizationDiagnostics",p);
 }
 int N=ne*64;
 auto og=device.malloc<double>(ne*12*64,geo),od=device.malloc<double>(16,D);
 auto oa=device.malloc<double>(offset,alpha),ou=device.malloc<double>(3*offset,ug);
 auto gradientBuffer=device.malloc<double>(3*offset,gradient),fluxBuffer=device.malloc<double>(3*offset,flux);
 auto loadBuffer=device.malloc<double>(offset,load);
 grad(ne,og,od,offset,oa,gradientBuffer);
 tensor(N,offset,cw,sw,eps,og,ou,gradientBuffer,fluxBuffer);
 weak(ne,og,od,offset,fluxBuffer,loadBuffer);
 gradientBuffer.copyTo(gradient);fluxBuffer.copyTo(flux);loadBuffer.copyTo(load);
}
'''
code=code.replace('GRAD_FILE',json.dumps(str(root/'src/core/kernels/gradientVolumeHex3D.okl')))
code=code.replace('WEAK_FILE',json.dumps(str(root/'src/core/kernels/wDivergenceVolumeHex3D.okl')))
code=code.replace('TENSOR_FILE',json.dumps(str(case/'bubbleColumnAlphaStabilization.okl')))
P=ctypes.POINTER(ctypes.c_double)
def pointer(a):return a.ctypes.data_as(P)

with tempfile.TemporaryDirectory() as tmp:
    tmp=Path(tmp);(tmp/'run.cpp').write_text(code)
    subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC',str(tmp/'run.cpp'),'-I'+str(occa/'include'),'-L'+str(occa/'lib'),'-Wl,-rpath,'+str(occa/'lib'),'-locca','-o',str(tmp/'run.so')],check=True)
    lib=ctypes.CDLL(str(tmp/'run.so'))
    lib.run.argtypes=[ctypes.c_int,ctypes.c_int,P,P,P,P,ctypes.c_double,ctypes.c_double,ctypes.c_double,P,P,P]
    lib.bound.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_int,P,P];lib.bound.restype=ctypes.c_double
    actualBound=lib.bound(nq,nvgeo,ne,pointer(geo),pointer(D))
    expectedBound=max(np.sum(Ge**2*np.tile(localMass[e*np3:(e+1)*np3],3)[:,None]/localMass[e*np3:(e+1)*np3][None,:]) for e,Ge in enumerate(G))
    np.testing.assert_allclose(actualBound,expectedBound,rtol=2e-14)

    def evaluate(a,u,cw,sw,eps):
        a=np.r_[a,np.full(offset-N,777.)].astype(float)
        gradient=np.full((3,offset),777.);flux=gradient.copy();load=np.full(offset,777.)
        lib.run(ne,offset,pointer(geo),pointer(D),pointer(a),pointer(u),cw,sw,eps,pointer(gradient),pointer(flux),pointer(load))
        unweighted=np.hstack([(Ge@a[e*np3:(e+1)*np3]).reshape(3,np3) for e,Ge in enumerate(G)])
        np.testing.assert_allclose(gradient[:,:N],unweighted*localMass,atol=2e-15)
        projected=u[:,:N]*np.sum(u[:,:N]*unweighted,axis=0)/(np.sum(u[:,:N]**2,axis=0)+eps**2)
        expectedFlux=cw*unweighted+(sw-cw)*projected
        np.testing.assert_allclose(flux[:,:N],expectedFlux,atol=1e-18)
        expectedLoad=np.hstack([Ge.T@(np.tile(localMass[e*np3:(e+1)*np3],3)*expectedFlux[:,e*np3:(e+1)*np3].ravel()) for e,Ge in enumerate(G)])
        np.testing.assert_allclose(load[:N],expectedLoad,atol=2e-18)
        assert np.all(gradient[:,N:]==777) and np.all(flux[:,N:]==777) and np.all(load[N:]==777)
        assembled=np.bincount(ids,weights=-load[:N],minlength=globalN)
        rate=assembled/mass
        # Native host code uses gather-add + inverse assembled mass at each copy.
        raw=rate[ids]
        assert abs(np.dot(localMass,raw))<1e-15
        energy=np.dot(localMass*a[:N],raw)
        dissipation=np.sum(localMass*np.sum(unweighted*expectedFlux,axis=0))
        np.testing.assert_allclose(energy,-dissipation,atol=2e-17)
        assert energy<=1e-17
        return flux[:,:N],raw

    for cw,sw in [(0,0),(2e-5,0),(0,3e-5),(2e-5,3e-5),(5e-5,5e-5)]:
        evaluate(alpha,ug,cw,sw,1e-6)
        _,raw=evaluate(np.full(N,.2),ug,cw,sw,1e-6);assert max(abs(raw))<1e-15
    vertical=np.zeros_like(ug);vertical[2,:N]=.3
    cwflux,_=evaluate(alpha,vertical,2e-5,0,1e-6)
    swflux,_=evaluate(alpha,vertical,0,2e-5,1e-6)
    assert np.max(abs(cwflux[2]))<1e-15
    assert np.max(abs(swflux[:2]))==0
    zero=np.zeros_like(ug)
    _,zeroStream=evaluate(alpha,zero,0,2e-5,1e-6);assert np.max(abs(zeroStream))==0
    cwZero,_=evaluate(alpha,zero,2e-5,0,1e-6)
    iso,_=evaluate(alpha,vertical,2e-5,2e-5,1e-6)
    np.testing.assert_allclose(cwZero,iso,atol=1e-18)
    # Bound controls the generalized assembled isotropic eigenvalues.
    K=np.zeros((globalN,globalN))
    for e,Ge in enumerate(G):
        Ke=Ge.T@(np.tile(localMass[e*np3:(e+1)*np3],3)[:,None]*Ge)
        ie=ids[e*np3:(e+1)*np3];np.add.at(K,(ie[:,None],ie[None,:]),Ke)
    eig=np.linalg.eigvalsh(K/np.sqrt(mass[:,None]*mass[None,:]))
    assert eig[-1]<=actualBound*(1+1e-14)
    # Also verify positive variable-JW/sheared geometry in the trace helper.
    variable=geo.copy();variable[:,1,:]=.13;variable[:,5,:]=-.07
    variable[:,10,:]*=1+.1*np.sin(coord[0]).reshape(ne,np3)
    variableG=[]
    for e in range(ne):
        variableG.append(np.vstack([sum(variable[e,3*d+c,:,None]*mat for d,mat in enumerate([Gr,Gs,Gt])) for c in range(3)]))
    expected=max(np.sum(Ge**2*np.tile(variable[e,10,:],3)[:,None]/variable[e,10,:][None,:]) for e,Ge in enumerate(variableG))
    np.testing.assert_allclose(lib.bound(nq,nvgeo,ne,pointer(variable),pointer(D)),expected,rtol=2e-14)
    with (tmp/'stabilization.cu').open('w') as out:
        subprocess.run([str(occa/'bin/occa'),'translate','-m','CUDA','-D','p_blockSize=256','-D','dfloat=double','-D','dlong=int','-D','p_Np=64','-D','p_Nvgeo=12','-D','p_JWID=10',str(case/'bubbleColumnAlphaStabilization.okl')],stdout=out,check=True)
print('Actual OCCA Serial gradient/tensor/weak divergence: unequal element sizes, CG shared mass, constant preservation, nonpositive energy, zero net load, orientation, zero speed, isotropy, padding, geometric spectral bound and CUDA translation passed.')

# Verify the production integration route and absence of momentum/core changes.
phase=(case/'bubbleColumnPhaseMomentum.hpp').read_text();udf=(case/'bubbleColumn.udf').read_text()
assert phase.index('addExplicitSources(time);')<phase.index('alphaStabilization::addSources();')<phase.index('predictorKernel(N,offset')
add=function(header,'inline void addSources()')
assert add.count('rhsHistory.apply(rawRhs,usedRhs,1,true)')==1
assert 'o_alphaExplicitDiffusion' in add and 'o_alphaSource' in add and 'o_explicitTerms("alpha")' in add
assert 'o_divergenceTransportEXT' not in add and 'o_convectionEXT' not in add
assert 'if (!enabled()) return;' in add
assert 'o_U.copyFrom' not in header and '->solve(' not in header
assert 'strongGrad(mesh,offset,nrs->scalar->o_solution("alpha"),weightedGradient,false)' in header
assert 'axmy(N,-1.0,mesh->o_invLMM,rawRhs)' in header
assert 'alphaStabilization::diagnostics(time)' in udf
assert 'mode<0 || mode>6' in header
print('Legacy constant-mode source and Method1 target receive the same EXT sample once; off path verified.')

# Execute the actual source hook, with known variable-step EXT weights. The
# real ExplicitHistory itself is exercised by time_extrapolation.py.
prefix=r'''
#include <vector>
#include <map>
#include <string>
#include <cassert>
#include <cmath>
using dfloat=double;using dlong=int;
struct Mem {double *p;int count;Mem slice(int off,int n){return {p+off,n};}
 int size(){return count;}void copyTo(double *x){for(int i=0;i<count;++i)x[i]=p[i];}};
struct LA {void fill(int n,double v,Mem a){for(int i=0;i<n;++i)a.p[i]=v;}
 void axpby(int n,double x,Mem a,double y,Mem b){for(int i=0;i<n;++i)b.p[i]=x*a.p[i]+y*b.p[i];}};
LA la;struct Platform {LA *linAlg=&la;} object,*platform=&object;
double state[10]={},coeff[2]={1,0},nativeRhs[5]={5,5,5,123,123};
struct Scalar {Mem o_S{state,10},o_coeffBDF{coeff,2};int fieldOffsetSum=5;
 std::vector<int>fieldOffsetScan{0};std::map<std::string,int>nameToIndex{{"alpha",0}};
 Mem o_explicitTerms(const char *name){assert(std::string(name)=="alpha");return {nativeRhs,5};}} scalar;
struct Mesh {int Nlocal=3;} mesh;
struct Nrs {Mesh *meshV=&mesh;Scalar *scalar;double dt[1]={.01};} nr{&mesh,&scalar},*nrs=&nr;
double raw[5]={},used[5]={},alphaSource[5]={1,1,1,123,123},targetPart[5]={.5,.5,.5,123,123},hist[5]={};
Mem rawRhs{raw,5},usedRhs{used,5},o_alphaSource{alphaSource,5},o_alphaExplicitDiffusion{targetPart,5},historyRate{hist,5};
int mode=1;bool active=true;bool enabled(){return active;}bool due(){return false;}
void evaluateDiffusion(){raw[0]=-.2;raw[1]=.1;raw[2]=.1;}
struct History {int calls=0;void apply(Mem sample,Mem out,int components,bool extrapolate){
 assert(components==1&&extrapolate);++calls;double previous[3]={-.1,.04,.06};
 for(int i=0;i<3;++i)out.p[i]=2.3*sample.p[i]-1.3*previous[i];}} rhsHistory;
'''
main=r'''
int main(){
 addSources();assert(rhsHistory.calls==1);
 double previous[3]={-.1,.04,.06};
 for(int i=0;i<3;++i){double r=2.3*raw[i]-1.3*previous[i];
 assert(std::abs(alphaSource[i]-1-r)<1e-15);assert(std::abs(nativeRhs[i]-5-r)<1e-15);
 assert(std::abs(targetPart[i]-.5-r)<1e-15);}
 assert(std::abs(used[0]+used[1]+used[2])<1e-15);
 assert(alphaSource[3]==123&&nativeRhs[4]==123&&targetPart[3]==123);
 double saved[3]={alphaSource[0],nativeRhs[0],targetPart[0]};active=false;
 addSources();assert(rhsHistory.calls==1);
 assert(alphaSource[0]==saved[0]&&nativeRhs[0]==saved[1]&&targetPart[0]==saved[2]);
}
'''
with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp);(p/'hook.cpp').write_text(prefix+add+main)
    subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror',str(p/'hook.cpp'),'-o',str(p/'hook')],check=True)
    subprocess.run([str(p/'hook')],check=True)
print('Actual source hook: variable-step EXT used once, matching alpha and continuity target increments, zero-net load, padding and bitwise off behavior passed.')
