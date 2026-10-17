"""Real OCCA Serial arrays + production helper + native v26 inner-step driver.

Elliptic solves/BCs/AVM are controlled test doubles; this is not a pipe CFD run.
Native sumMakef kernels and solution-history lagging are exercised unchanged.
"""
from pathlib import Path
import os, re, subprocess, tempfile, csv
case=Path(__file__).resolve().parents[1]
repo=case.parents[1]
occa=repo.parent/'occa-cub-parser'

def function(path, signature):
    text=path.read_text();start=text.index(signature);brace=text.index('{', start)
    depth=1;end=brace+1
    while depth:
        depth+=(text[end]=='{')-(text[end]=='}');end+=1
    return text[start:end]

def serial(text):
    return re.sub(r';\s*@tile\(p_blockSize,\s*@outer,\s*@inner\)','',
                  text.replace('@kernel ','').replace('@ restrict ',''))

prefix=r'''
#include <occa.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cassert>
#include <functional>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cstddef>
#include <iostream>
using dfloat=double;using dlong=long;
struct Options {std::map<std::string,std::string> map;
 bool compareArgs(std::string k,std::string v){return map[k]==v;}
 void getArgs(std::string k,int &v){if(map.count(k)&&!map[k].empty())v=std::stoi(map[k]);}
};
struct Comm {int mpiComm(){return 0;}int mpiRank(){return 0;}};
struct Timer {void tic(std::string,int=0){}void toc(std::string){}};
double *ptr(const occa::memory &m){return const_cast<double*>(static_cast<const double*>(m.ptr()));}
struct LA {
 double min(long n,const occa::memory&a,int){return *std::min_element(ptr(a),ptr(a)+n);}
 double max(long n,const occa::memory&a,int){return *std::max_element(ptr(a),ptr(a)+n);}
 double innerProd(long n,const occa::memory&a,const occa::memory&b,int){double r=0;for(long i=0;i<n;++i)r+=ptr(a)[i]*ptr(b)[i];return r;}
 void axpby(long n,double a,const occa::memory&x,double b,const occa::memory&y){for(long i=0;i<n;++i)ptr(y)[i]=a*ptr(x)[i]+b*ptr(y)[i];}
 void axmy(long n,double a,const occa::memory&x,const occa::memory&y){for(long i=0;i<n;++i)ptr(y)[i]*=a*ptr(x)[i];}
} la;
struct Pool {occa::device *d;occa::memory reserve(size_t n){return d->malloc(n);}
 template<class T>occa::memory reserve(size_t n){return d->malloc<T>(n);}};
struct Platform {Options options;Comm comm;Timer timer;LA *linAlg=&la;occa::device device;
 Pool deviceMemoryPool{&device};occa::kernel copyDoubleToFloatKernel,copyFloatToDoubleKernel;
 bool verbose(){return false;}struct App *app=nullptr;
} object,*platform=&object;
#include "deviceMemory.hpp"
struct Boundary {deviceMemory<dfloat> o_usrwrk;bool hasOutflow(std::string){return true;}};
struct App {Boundary *bc;};
template<class...T>void nekrsCheck(bool bad,int,int,const char *msg,T...){if(bad)throw std::runtime_error(msg);}
occa::memory alloc(long n,double base=0){std::vector<double>a(n);for(long i=0;i<n;++i)a[i]=base+.001*i;return platform->device.malloc<double>(n,a.data());}
struct Mesh {long Nlocal=3;double volume=6;occa::memory o_LMM;};
struct Fluid {long fieldOffset=5,fieldOffsetSum=15;
 occa::memory o_div,o_U,o_P,o_Ue,o_relUrst,o_prop,o_JwF,o_EXT,o_ADV,o_coeffBDF,o_coeffEXT,o_coeffEXTP;
 int lagCount=0,solveCount=0;
 void lagSolution();void solve(double,int);
 void applyDirichlet(double);void updateZeroNormalMask(){}
};
struct Cvode {void solve(double,double,int){}};
struct Scalar {long fieldOffsetSum=5;occa::memory o_S,o_Se,o_prop,o_JwF,o_EXT,o_ADV,o_coeffBDF,o_coeffEXT;
 Cvode *cvode=nullptr;bool anyEllipticSolver=true;int lagCount=0,solveCount=0;
 void lagSolution();void solve(double,int);void applyDirichlet(double);
};
struct Geom {void integrate(){}void lagSolution(){}void applyDirichlet(double){}void solve(double,int){}};
struct Neknek {void updateBoundary(int,int,double*,double){}void fixCoupledSurfaceFlux(){}};
struct nrs_t {Fluid *fluid;Scalar *scalar;Mesh *meshV;Geom *geom=nullptr;Neknek *neknek=nullptr;
 int outerCorrector=0,checkpointStep=0,tstep=0,Nscalar=1,advectionSubcycingSteps=0,timeStepConverged=1;
 double dt[3]={.01,.01,.01},timePrevious=0,g0=1.;occa::memory o_coeffBDF;
 std::function<void(double)> userSource=nullptr,userProperties=nullptr,userDivergence=nullptr;
 std::function<void(double,int)> postScalar=nullptr,preFluid=nullptr;
 std::function<bool(int)>userConvergenceCheck=nullptr;
 bool fieldsToSolveContains(std::string){return false;}void adjustFlowRate(int,double){}
 void evaluateProperties(double);void evaluateDivergence(double){}
 bool runInnerStep(std::function<bool(int)>,int,bool);
} nrsObject,*nrs=&nrsObject;
namespace nek {int output=0;void ifoutfld(int n){output=n;}}
struct Udf {std::function<void(double,int)> executeStep;} udf;
double setPrecision(double d,int){return d;}
'''
# The unreachable neknek branch requires its normal method signature.
prefix=prefix.replace('void fixCoupledSurfaceFlux(){}', 'void fixCoupledSurfaceFlux(const occa::memory&,long,const occa::memory&){}')
prefix=prefix.replace('occa::memory o_div,o_U,o_P,', 'occa::memory o_EToB,o_div,o_U,o_P,')
methods=r'''
void Fluid::applyDirichlet(double){
 ptr(o_U)[0]=ptr(platform->app->bc->o_usrwrk)[0];ptr(o_Ue)[0]=ptr(o_U)[0];
 ptr(o_P)[2]=-.5*ptr(o_U)[12]*ptr(o_U)[12];
}
void Scalar::applyDirichlet(double){ptr(o_S)[0]=.2;ptr(o_Se)[0]=.2;}
void Scalar::solve(double,int){++solveCount;for(int i=1;i<3;++i)
 ptr(o_S)[i]=ptr(o_JwF)[i]/(100*nrs->g0+ptr(o_prop)[i])+.001*ptr(o_Se)[i];}
void Fluid::solve(double,int){++solveCount;
 const double bc=ptr(o_P)[2];for(int i=0;i<2;++i)ptr(o_P)[i]=.01*ptr(o_JwF)[i]+bc;
 for(int c=0;c<3;++c)for(int i=0;i<3;++i)if(c||i)
 ptr(o_U)[c*5+i]=(ptr(o_JwF)[c*5+i]-.1*ptr(o_P)[i])/(100*nrs->g0)+.001*ptr(o_Ue)[c*5+i];
}
void nrs_t::evaluateProperties(double){
 // Native AVM timing: update coefficients after scalar solve using current U,S.
 for(int i=0;i<3;++i)ptr(scalar->o_prop)[i]=.1+.03*std::abs(ptr(scalar->o_S)[i])+ .02*std::abs(ptr(fluid->o_U)[i]);
}
'''
# Actual solution lag functions; no second history push is allowed.
lagu=function(repo/'src/solver/fluid/fluidSolver.cpp','void fluidSolver_t::lagSolution()').replace('fluidSolver_t::','Fluid::')
lags=function(repo/'src/solver/scalar/scalarSolver.cpp','void scalar_t::lagSolution()').replace('scalar_t::','Scalar::')
lagu=lagu.replace('{','{\n++lagCount;',1);lags=lags.replace('{','{\n++lagCount;',1)
driver=function(repo/'src/app/nrs/nrs.cpp','bool nrs_t::runInnerStep(')
wrapper=function(case/'picard_turbpipe.udf','void UDF_ExecuteStep(')
wrapper='''int commits=0,checkpoints=0;
void turbPipeExecuteStep(double,int t){if(t>0){++commits;if(nrs->checkpointStep)++checkpoints;}
ptr(platform->app->bc->o_usrwrk)[0]=1+.01*ptr(nrs->fluid->o_U)[12];}
'''+wrapper
kernels='\n#define p_nBDF 3\n#define p_nEXT 3\n#define p_ADVECTION 1\n#define p_SUBCYCLING 0\n#define p_MovingMesh 0\n'
for kind in ['fluid','scalar']:
 kernels+=serial((repo/f'src/solver/{kind}/kernels/sumMakef.okl').read_text()).replace('sumMakef(',kind+'SumMakef(')
main=r'''
void prepare(Fluid &f,Scalar &s,Mesh &mesh,int step){
 const double bdf[3][3]={{1,0,0},{2,-.5,0},{3,-1.5,1./3}};
 const double ext[3][3]={{1,0,0},{2,-1,0},{3,-3,1}};
 int order=std::min(step,3);f.o_coeffBDF.copyFrom(bdf[order-1],3);f.o_coeffEXT.copyFrom(ext[order-1],3);
 nrs->g0=order==1?1:order==2?1.5:11./6;
 f.o_Ue.copyFrom(f.o_U,15);s.o_Se.copyFrom(s.o_S,5);f.o_relUrst.copyFrom(f.o_U,15);
 // This is the sole userSource entry. Real initInnerStep constructs RHS next.
 if(nrs->userSource)nrs->userSource(nrs->timePrevious);
 for(int i=0;i<15;++i){ptr(f.o_EXT)[i]=.01;ptr(f.o_ADV)[i]=.02*ptr(f.o_U)[i];}
 for(int i=0;i<5;++i){ptr(s.o_EXT)[i]=.015;ptr(s.o_ADV)[i]=.03*ptr(s.o_S)[i];}
 double rho[5]={1,1,1,1,1};
 fluidSumMakef(3,ptr(mesh.o_LMM),100,ptr(f.o_coeffEXT),ptr(f.o_coeffBDF),5,15,5,
 rho,ptr(f.o_U),ptr(f.o_ADV),ptr(f.o_EXT),ptr(f.o_JwF));
 scalarSumMakef(3,ptr(mesh.o_LMM),100,ptr(s.o_coeffEXT),ptr(s.o_coeffBDF),0,5,5,
 rho,ptr(s.o_S),ptr(s.o_ADV),ptr(s.o_EXT),ptr(s.o_JwF));
 // Native sumMakef pushes these explicit histories once, before runInnerStep.
 for(int k=3;k>1;--k){f.o_EXT.copyFrom(f.o_EXT,15,(k-1)*15,(k-2)*15);f.o_ADV.copyFrom(f.o_ADV,15,(k-1)*15,(k-2)*15);
 s.o_EXT.copyFrom(s.o_EXT,5,(k-1)*5,(k-2)*5);s.o_ADV.copyFrom(s.o_ADV,5,(k-1)*5,(k-2)*5);}
}
int main(int argc,char**argv){
 const int passes=std::stoi(argv[1]);occa::properties props;props["mode"]="Serial";platform->device.setup(props);
 for(auto key:{"FLUID PRESSURE INITIAL GUESS","FLUID VELOCITY INITIAL GUESS","SCALAR00 INITIAL GUESS"})platform->options.map[key]="PREVIOUS";
 Boundary boundary;boundary.o_usrwrk.resize(15);std::vector<double> bc(15,1.);boundary.o_usrwrk.copyFrom(bc);
 App app{&boundary};platform->app=&app;
 Mesh mesh;double weights[3]={1,2,3};mesh.o_LMM=platform->device.malloc<double>(3,weights);
 Fluid f;Scalar s;
 f.o_div=alloc(5);f.o_U=alloc(45,.3);f.o_P=alloc(5,.1);f.o_Ue=alloc(15);f.o_relUrst=alloc(15);
 f.o_prop=alloc(10,.1);f.o_JwF=alloc(15);f.o_EXT=alloc(45,.01);f.o_ADV=alloc(45,.02);
 f.o_coeffBDF=alloc(3);f.o_coeffEXT=alloc(3);
 s.o_S=alloc(15,.2);s.o_Se=alloc(5);s.o_prop=alloc(10,.1);s.o_JwF=alloc(5);s.o_EXT=alloc(15,.015);s.o_ADV=alloc(15,.03);
 s.o_coeffBDF=f.o_coeffBDF;s.o_coeffEXT=f.o_coeffEXT;
 nrs->fluid=&f;nrs->scalar=&s;nrs->meshV=&mesh;nrs->o_coeffBDF=f.o_coeffBDF;
 // Seeded higher history slots exercise a restart-like initial state too.
 frozenInner::enabled=passes!=0;frozenInner::iterations=passes?passes:1;
 frozenInner::configure();udf.executeStep=&UDF_ExecuteStep;
 UDF_ExecuteStep(0.,0);
 for(int step=1;step<=6;++step){
 nrs->tstep=step;nrs->timePrevious=(step-1)*.01;nrs->timeStepConverged=false;nrs->checkpointStep=step%2==0;
 prepare(f,s,mesh,step);
 for(int it=1;;++it){
 auto check=[&](int i){return nrs->userConvergenceCheck?nrs->userConvergenceCheck(i):true;};
 if(nrs->runInnerStep(check,it,true))break;
 assert(it<passes);
 }
 assert(f.lagCount==step&&s.lagCount==step);assert(commits==step&&checkpoints==step/2);
 assert(f.solveCount==step*(passes?passes:1)&&s.solveCount==f.solveCount);
 assert(nek::output==(step%2==0));
 }
 // Identical final fields, histories, forcing histories and AVM coefficient state.
 std::ofstream out("result.txt");out<<std::setprecision(17);
 for(auto field:{f.o_U,f.o_P,s.o_S,f.o_EXT,s.o_EXT,f.o_ADV,s.o_ADV,s.o_prop})
 for(size_t i=0;i<field.size();++i)out<<ptr(field)[i]<<'\n';
 // Guard must detect history corruption, not silently erase it on restore.
 if(passes){ptr(f.o_U)[15]+=.1;bool failed=false;try{frozenInner::convergenceCheck(passes+1);}catch(const std::runtime_error&){failed=true;}assert(failed);}
}
'''
cpp=prefix+methods+lagu+lags+'\n#include "frozenInner.hpp"\n'+wrapper+driver+kernels+main
with tempfile.TemporaryDirectory() as directory:
 path=Path(directory)
 (path/'nekrsSys.hpp').write_text('#pragma once\n')
 (path/'check.cpp').write_text(cpp)
 subprocess.run(['g++','-std=c++17',str(path/'check.cpp'),'-I'+str(path),'-I'+str(case),
                 '-I'+str(repo/'src/platform'),'-I'+str(occa/'include'),'-L'+str(occa/'lib'),
                 '-Wl,-rpath,'+str(occa/'lib'),'-locca','-o',str(path/'check')],check=True)
 expected=None
 for passes in [0,1,2,4]:
  run=path/str(passes);run.mkdir()
  result=subprocess.run([str(path/'check'),str(passes)],cwd=run,text=True,capture_output=True,check=True)
  actual=(run/'result.txt').read_text()
  if expected is None:expected=actual
  else:assert actual==expected,f'Final state differs for {passes} passes'
  if passes:
   rows=list(csv.DictReader((run/'picard_repeatability.csv').open()))
   # One additional row is emitted by the deliberate corruption test.
   assert len(rows)==6*passes+1
   for row in rows[:-1]:
    assert all(float(row[k])==0 for k in ['u_max','p_max','s_max','history_max','frozen_max','property_max'])
  print(f'PASS: native inner driver, {passes or "disabled"} passes; final arrays identical.')
print('Real Serial arrays + native history/sumMakef/inner driver: BDF1/2/3, frozen RHS, '
      'AVM before/after state, boundary recycling once, checkpoints retained, corruption detected. '
      'Elliptic solves are mocked; full GPU/MPI pipe comparison remains required.')
