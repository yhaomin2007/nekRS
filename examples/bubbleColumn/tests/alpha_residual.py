"""Execute residual stabilization with real SEM kernels and nonlinear weak solves."""
from pathlib import Path
import ctypes,json,os,subprocess,tempfile
import numpy as np
from numpy.polynomial.legendre import Legendre

case=Path(__file__).resolve().parents[1];root=case.parents[1]
occa=Path(os.environ.get('OCCA_DIR',root/'3rd_party/occa'))
assert (occa/'lib/libocca.so').exists(),'Set OCCA_DIR to a built checkout.'
nq=4;np3=64;ne=2;N=128;offset=N+3
x=np.r_[-1.,Legendre.basis(3).deriv().roots(),1.];w=2/(12*Legendre.basis(3)(x)**2)
b=np.array([1/np.prod(x[i]-np.delete(x,i)) for i in range(nq)])
D=np.zeros((nq,nq))
for i in range(nq):
    for j in range(nq):
        if i!=j:D[i,j]=b[j]/b[i]/(x[i]-x[j])
    D[i,i]=-D[i].sum()
Dr=np.kron(np.eye(4),np.kron(np.eye(4),D));Ds=np.kron(np.eye(4),np.kron(D,np.eye(4)));Dt=np.kron(D,np.kron(np.eye(4),np.eye(4)))
geo=np.zeros((2,12,64));coord=np.zeros((3,N));ids=[];G=[]
for e,(left,right) in enumerate([(-.005,0.),(0.,.01)]):
    for k in range(4):
        for j in range(4):
            for i in range(4):
                n=i+4*j+16*k;coord[:,e*64+n]=[left+(x[i]+1)*(right-left)/2,.005*x[j],.005*x[k]]
                geo[e,0,n]=2/(right-left);geo[e,4,n]=200;geo[e,8,n]=200
                geo[e,10,n]=(right-left)/2*.005**2*w[i]*w[j]*w[k]
                ids.append(e*3+i+7*(j+4*k))
    G.append(np.vstack([geo[e,0,:,None]*Dr,200*Ds,200*Dt]))
ids=np.asarray(ids);ng=ids.max()+1
massLocal=geo[:,10,:].ravel();mass=np.bincount(ids,weights=massLocal,minlength=ng)
ug=np.zeros((3,offset));ug[2,:N]=.3
diffusion=np.full(offset,1e-5)
alpha=.008+.003*np.tanh(-coord[2]/.001)+.0004*np.sin(700*coord[0])
code=r'''
#include <occa.hpp>
extern "C" void run(int ne,int offset,const double *geo,const double *D,const double *alpha,const double *ug,
 const double *diffusion,const double *history,const double *transport,double h,double dt,int order,
 int cw,int supg,double cwScale,double supgScale,double gradFloor,double cap,double epsilon,
 double *fields,double *cwLoad,double *supgLoad) {
 static occa::device d;static occa::kernel grad,div,weak,flux,stab;
 if(!d.isInitialized()) {
  d.setup(std::string("{mode:'Serial'}"));occa::properties p;
  p["defines/dfloat"]="double";p["defines/dlong"]="int";p["defines/p_blockSize"]=256;
  p["defines/p_Nq"]=4;p["defines/p_Np"]=64;p["defines/p_Nvgeo"]=12;
  const char *names[]={"p_RXID","p_RYID","p_RZID","p_SXID","p_SYID","p_SZID","p_TXID","p_TYID","p_TZID"};
  for(int i=0;i<9;++i)p[std::string("defines/")+names[i]]=i;
  p["defines/p_JWID"]=10;
  grad=d.buildKernel(GRAD,"gradientVolumeHex3D",p);div=d.buildKernel(DIV,"divergenceVolumeHex3D",p);
  weak=d.buildKernel(WEAK,"wDivergenceVolumeHex3D",p);
  flux=d.buildKernel(STAB,"alphaResidualGradient",p);stab=d.buildKernel(STAB,"alphaResidualFlux",p);
  d.buildKernel(STAB,"alphaResidualMask",p);
 }
 int N=ne*64;
 auto og=d.malloc<double>(ne*12*64,geo),od=d.malloc<double>(16,D),oa=d.malloc<double>(offset,alpha);
 auto ou=d.malloc<double>(3*offset,ug),oD=d.malloc<double>(offset,diffusion);
 auto oh=d.malloc<double>(offset,history),ot=d.malloc<double>(offset,transport);
 auto gradient=d.malloc<double>(3*offset),baseFlux=d.malloc<double>(3*offset),lap=d.malloc<double>(offset),q=d.malloc<double>(offset);
 auto fc=d.malloc<double>(3*offset),fs=d.malloc<double>(3*offset),of=d.malloc<double>(11*offset,fields);
 auto lc=d.malloc<double>(offset,cwLoad),ls=d.malloc<double>(offset,supgLoad);
 grad(ne,og,od,offset,oa,gradient);flux(N,offset,og,gradient,oD,baseFlux);
 div(ne,og,od,offset,baseFlux,lap);div(ne,og,od,offset,ou,q);
 stab(N,offset,h,dt,order,cw,supg,cwScale,supgScale,gradFloor,cap,epsilon,og,oa,ou,gradient,lap,q,oh,ot,oD,fc,fs,of);
 weak(ne,og,od,offset,fc,lc);weak(ne,og,od,offset,fs,ls);
 of.copyTo(fields);lc.copyTo(cwLoad);ls.copyTo(supgLoad);
}
'''
for key,path in {'GRAD':root/'src/core/kernels/gradientVolumeHex3D.okl','DIV':root/'src/core/kernels/divergenceVolumeHex3D.okl','WEAK':root/'src/core/kernels/wDivergenceVolumeHex3D.okl','STAB':case/'bubbleColumnAlphaResidual.okl'}.items():code=code.replace(key,json.dumps(str(path)))
P=ctypes.POINTER(ctypes.c_double)
def ptr(a):return a.ctypes.data_as(P)
def local(a):return np.r_[a,np.zeros(offset-N)].astype(float)
with tempfile.TemporaryDirectory() as tmp:
    tmp=Path(tmp);(tmp/'run.cpp').write_text(code)
    subprocess.run(['g++','-std=c++17','-shared','-fPIC','-O2',str(tmp/'run.cpp'),'-I'+str(occa/'include'),'-L'+str(occa/'lib'),'-Wl,-rpath,'+str(occa/'lib'),'-locca','-o',str(tmp/'run.so')],check=True)
    lib=ctypes.CDLL(str(tmp/'run.so'))
    lib.run.argtypes=[ctypes.c_int,ctypes.c_int]+[P]*7+[ctypes.c_double,ctypes.c_double]+[ctypes.c_int]*3+[ctypes.c_double]*5+[P]*3

    def evaluate(a,hist,transport,h=150000.,dt=1e-5,cw=1,supg=1,order=3,cap=1e-3):
        fields=np.full((11,offset),777.);lc=np.full(offset,777.);ls=lc.copy()
        a=local(a);hist=local(hist);transport=local(transport)
        lib.run(ne,offset,ptr(geo),ptr(D),ptr(a),ptr(ug),ptr(diffusion),ptr(hist),ptr(transport),h,dt,order,cw,supg,1.,1.,1e-5,cap,1e-6,ptr(fields),ptr(lc),ptr(ls))
        gradients=np.hstack([(Ge@a[e*64:(e+1)*64]).reshape(3,64) for e,Ge in enumerate(G)])
        lap=np.hstack([sum((Ge[d*64:(d+1)*64]@(1e-5*gradients[d,e*64:(e+1)*64]) for d in range(3))) for e,Ge in enumerate(G)])
        R=h*a[:N]-hist[:N]-transport[:N]-lap
        np.testing.assert_allclose(fields[8,:N],R,atol=2e-12)
        assert np.all(fields[:,N:]==777.) and np.all(lc[N:]==777.) and np.all(ls[N:]==777.)
        assert fields[4,:N].min()>=0 and fields[4,:N].max()<=cap
        assert fields[5,:N].min()>=0
        # Actual native weak loads preserve the constant test function.
        assert abs(lc[:N].sum())<1e-17 and abs(ls[:N].sum())<1e-17
        assert np.dot(a[:N],lc[:N])>=-1e-17
        return fields[:,:N],lc[:N]+ls[:N]

    # A cubic with zero Neumann gradient on every face, exact GLL representation.
    truth=np.zeros(N)
    for e in range(ne):
        zz=coord[2,e*64:(e+1)*64]/.005
        truth[e*64:(e+1)*64]=.008+.001*(zz**3/3-zz)
    lapTruth=np.hstack([sum(Ge[d*64:(d+1)*64]@(1e-5*(Ge@truth[e*64:(e+1)*64]).reshape(3,64)[d]) for d in range(3)) for e,Ge in enumerate(G)])
    h=150000.;hist=h*truth;transport=-lapTruth
    f,load=evaluate(truth,hist,transport)
    assert max(abs(f[8]))<1e-11 and max(abs(f[4]))<1e-12 and max(abs(load))<1e-17
    # Nonzero BDF residual activates a spatially varying coefficient.
    hist=h*alpha;transport=np.zeros(N)
    f,load=evaluate(alpha,hist,transport)
    assert f[4].max()>0 and np.ptp(f[4])>1e-10
    assert max(abs(f[5]*f[8]))>0
    f0,load0=evaluate(alpha,hist,transport,cw=0,supg=0)
    assert not np.any(f0[4:6]) and max(abs(load0))==0
    f7,_=evaluate(alpha,hist,transport,supg=0,order=7)
    free=(f[4]>1e-14)&(f[4]<1e-4)
    assert free.any();np.testing.assert_allclose(f7[4,free]/f[4,free],.25,rtol=1e-8)
    savedUg=ug.copy();ug[:]=0
    zero,_=evaluate(alpha,hist-1.,transport)
    assert not np.any(zero[4])
    ug[:]=savedUg
    # Discrete weak nonlinear problem with native mass + baseline stiffness.
    K=np.zeros((ng,ng))
    for e,Ge in enumerate(G):
        ke=Ge.T@(np.tile(massLocal[e*64:(e+1)*64],3)[:,None]*Ge)*1e-5
        ie=ids[e*64:(e+1)*64];np.add.at(K,(ie[:,None],ie[None,:]),ke)
    H=K+np.diag(h*mass);Hinv=np.linalg.inv(H)
    def assemble(load):return np.bincount(ids,weights=load,minlength=ng)
    old=np.zeros(ng);old[ids]=alpha
    old[ids]=truth
    hist=h*old;transport=np.zeros(ng);transport[ids]=-lapTruth
    base=mass*(hist+transport)
    a=Hinv@base+1e-5*np.sin(np.arange(ng)*.3)
    def equation(a):
        ff,ll=evaluate(a[ids],hist[ids],transport[ids]);r=H@a+assemble(ll)-base
        return np.sqrt(np.dot(r/mass,r)/mass.sum()),ff,ll
    # Matrix-free Newton/GMRES, matching the UDF algorithm (mass-weighted norm).
    def dot(a,b):return np.dot(mass*a,b)/mass.sum()
    def length(a):return np.sqrt(dot(a,a))
    constrained=np.zeros(ng,dtype=bool)
    def residual(a):
        ff,ll=evaluate(a[ids],hist[ids],transport[ids]);r=(H@a+assemble(ll)-base)/mass
        r[constrained]=0
        return r,ff,ll
    def direction(a,r):
        beta=length(r);V=[-r/beta];A=np.zeros((21,20));g=np.zeros(21);g[0]=beta
        eps=np.sqrt(np.finfo(float).eps)*max(length(a),1e-4)
        for j in range(20):
            trial,_,_=residual(a+eps*V[j]);z=(trial-r)/(eps*h)
            for repeat in range(2):
                for i in range(j+1):
                    c=dot(V[i],z);A[i,j]+=c;z-=c*V[i]
            A[j+1,j]=length(z)
            y=np.linalg.lstsq(A[:j+2,:j+1],g[:j+2],rcond=None)[0]
            error=np.linalg.norm(A[:j+2,:j+1]@y-g[:j+2])
            if error<.05*beta or A[j+1,j]<1e-12:break
            V.append(z/A[j+1,j])
        return sum(y[i]*V[i] for i in range(len(y)))/h
    for with_inlet in [False,True]:
        if with_inlet:constrained[ids[np.isclose(coord[2],-.005)]]=True
        a=Hinv@base+1e-5*np.sin(np.arange(ng)*.3)
        exact=np.zeros(ng);exact[ids]=truth;a[constrained]=exact[constrained]
        fixed=a[constrained].copy()
        r,ff,ll=residual(a);start=norm=length(r)
        for it in range(30):
            if norm<1e-7+1e-5*start:break
            delta=direction(a,r);accepted=False
            for omega in [2.**(-k) for k in range(21)]:
                trial=a+omega*delta;rr,tf,tl=residual(trial);new=length(rr)
                if new<1e-7+1e-5*start or new<(1-1e-4*omega)*norm:
                    a,norm,r,ff,ll=trial,new,rr,tf,tl;accepted=True;break
            assert accepted,'Newton line search failed in manufactured SEM test'
        assert norm<=1e-7+1e-5*start,(start,norm,it)
        np.testing.assert_array_equal(a[constrained],fixed)
        np.testing.assert_allclose(a,exact,atol=1e-8)
        print('Manufactured two-element solve, inlet Dirichlet=',with_inlet,':',it,'Newton iterations, RMS',start,'->',norm)
    for mode in [(1,0),(0,1)]:evaluate(alpha,hist[ids],transport[ids],cw=mode[0],supg=mode[1])
    with (tmp/'residual.cu').open('w') as out:
        subprocess.run([str(occa/'bin/occa'),'translate','-m','CUDA','-D','p_blockSize=256','-D','dfloat=double','-D','dlong=int','-D','p_Np=64','-D','p_Nvgeo=12','-D','p_JWID=10',str(case/'bubbleColumnAlphaResidual.okl')],stdout=out,check=True)
print('Production residual kernels: correct BDF/EXT residual, zero-residual consistency, nonuniform coefficient, cap, high-order reduction, conservative weak SUPG/CW, nonlinear convergence, padding and CUDA translation passed.')

host=(case/'bubbleColumnAlphaResidual.hpp').read_text();phase=(case/'bubbleColumnPhaseMomentum.hpp').read_text()
assert 'direction();' in host and 'baseResidual.copyFrom' in host
assert 'nrs->scalar->solve(' not in host and 'compute=' not in host
assert 'baselineRhs.copyFrom(nativeRhs' in host
assert 'lagSolution(' not in host and 'makeForcing(' not in host
assert phase.index('residual::solve(time,tstep)')<phase.index('postProcessAlpha(time,tstep)')
assert 'o_alphaExplicitDiffusion' in host and 'usedRhs.copyFrom(totalWeak' in host
assert 'No unconverged alpha is accepted' in host
