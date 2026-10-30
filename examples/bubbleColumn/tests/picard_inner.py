"""Production bubble Picard + real OCCA/deviceMemory + native driver/forcing.

Elliptic solves and spatial physics are controlled doubles; not a CFD run.
"""
from pathlib import Path
import os,re,subprocess,tempfile,csv
case=Path(__file__).resolve().parents[1];repo=case.parents[1]
occa=Path(os.environ.get('OCCA_DIR',repo/'3rd_party/occa'))
def function(path, signature):
 s=path.read_text();a=s.index(signature);b=s.index('{',a);d=1;e=b+1
 while d:d+=(s[e]=='{')-(s[e]=='}');e+=1
 return s[a:e]
def serial(s):return re.sub(r';\s*@tile\(p_blockSize,\s*@outer,\s*@inner\)','',s.replace('@kernel ','').replace('@ restrict ',''))
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
#include <fstream>
#include <iomanip>
using dfloat=double;using dlong=long;using MPI_Comm=int;
struct Options{std::map<std::string,std::string>m;bool compareArgs(std::string k,std::string v){return m[k]==v;}};
struct Comm{int mpiComm(){return 0;}int mpiRank(){return 0;}};
struct Timer{void tic(std::string,int=0){}void toc(std::string){}};
double *ptr(const occa::memory&m){return const_cast<double*>(static_cast<const double*>(m.ptr()));}
struct LA{
 void fill(long n,double x,occa::memory &a){std::fill(ptr(a),ptr(a)+n,x);}
 double min(long n,const occa::memory&a,int){return *std::min_element(ptr(a),ptr(a)+n);}
 double max(long n,const occa::memory&a,int){return *std::max_element(ptr(a),ptr(a)+n);}
 void axpby(long n,double a,const occa::memory&x,double b,occa::memory&y){for(int i=0;i<n;++i)ptr(y)[i]=a*ptr(x)[i]+b*ptr(y)[i];}
 void axmy(long n,double a,const occa::memory&x,occa::memory&y){for(int i=0;i<n;++i)ptr(y)[i]*=a*ptr(x)[i];}
 double innerProd(long n,const occa::memory&a,const occa::memory&b,int){double v=0;for(int i=0;i<n;++i)v+=ptr(a)[i]*ptr(b)[i];return v;}
}la;
struct Pool{occa::device*d;occa::memory reserve(size_t n){return d->malloc(n);}template<class T>occa::memory reserve(size_t n){return d->malloc<T>(n);}};
int umKernelCalls=0;
struct Kernel{
 std::string name;
 static void adv(const occa::memory &u,const occa::memory &advector,occa::memory out){
   ++umKernelCalls;
   for(int c=0;c<3;++c)for(int i=0;i<3;++i){int j=c*5+i;
     assert(ptr(u)[j]==ptr(advector)[j]); // both factors from same iterate
     ptr(out)[j]=.02*ptr(u)[i]*ptr(u)[j];
   }
 }
 template<class...A>void operator()(A...args){auto t=std::make_tuple(args...);
 if constexpr(sizeof...(A)==8){assert(name=="nrs-UrstHex3D");std::get<7>(t).copyFrom(std::get<5>(t),15);}
 else if constexpr(sizeof...(A)==10){
   if(name=="nrs-UrstCubatureHex3D")std::get<9>(t).copyFrom(std::get<7>(t),15);
   else{assert(name=="core-strongAdvectionCubatureVolumeHex3D");adv(std::get<7>(t),std::get<8>(t),std::get<9>(t));}
 }else if constexpr(sizeof...(A)==7){assert(name=="core-strongAdvectionVolumeHex3D");adv(std::get<4>(t),std::get<5>(t),std::get<6>(t));}
 }
};
struct Requests{Kernel load(std::string name){return {name};}};
struct Boundary{bool hasOutflow(std::string){return true;}};
struct App{Boundary*bc;};
struct Platform{Options options;Comm comm;Timer timer;occa::device device;LA*linAlg=&la;Pool deviceMemoryPool{&device};
 occa::kernel copyDoubleToFloatKernel,copyFloatToDoubleKernel;Requests kernelRequests;App*app;bool verbose(){return false;}
}object,*platform=&object;
#define launchKernel(name,...) platform->kernelRequests.load(name)(__VA_ARGS__)
#include "deviceMemory.hpp"
template<class...T>void nekrsCheck(bool bad,int,int,const char*msg,T...args){if(bad){char buffer[1024];std::snprintf(buffer,sizeof(buffer),msg,args...);throw std::runtime_error(buffer);}}
occa::memory alloc(long n,double base=0){std::vector<double>a(n);for(int i=0;i<n;++i)a[i]=base+.0001*i;return platform->device.malloc<double>(n,a.data());}
struct Mesh{long Nlocal=3,Nelements=1;double volume=6;occa::memory o_LMM,o_Jw,o_vgeo,o_D,o_cubvgeo,o_cubInterpT,o_cubDiffInterpT,o_cubProjectT;}mesh;
struct Fluid{std::function<occa::memory(double,int)>userAdvectionTerm;long fieldOffset=5,fieldOffsetSum=15,cubatureOffset=5;
 occa::memory o_U,o_P,o_Ue,o_relUrst,o_prop,o_JwF,o_EXT,o_ADV,o_coeffBDF,o_coeffEXT,o_coeffEXTP,o_div,o_EToB;
 int lagCount=0,solves=0;void lagSolution();void applyDirichlet(double);void solve(double,int);void updateZeroNormalMask(){}
 deviceMemory<double>o_explicitTerms(){return deviceMemory<double>(o_EXT);}
};
struct Cvode{void solve(double,double,int){}};
struct Scalar{long fieldOffsetSum=20;std::vector<long>fieldOffsetScan{0,5,10,15};
 std::map<std::string,int>nameToIndex{{"alpha",0},{"ugx",1},{"ugy",2},{"ugz",3}};
 occa::memory o_S,o_Se,o_prop,o_JwF,o_EXT,o_ADV,o_coeffBDF,o_coeffEXT;bool anyEllipticSolver=true;Cvode*cvode=nullptr;
 int lagCount=0,solves=0;void lagSolution();void applyDirichlet(double);void solve(double,int);
 deviceMemory<double>o_solution(std::string n){return deviceMemory<double>(o_S.slice(fieldOffsetScan[nameToIndex[n]],5));}
};
struct Geom{void integrate(){}void lagSolution(){}void applyDirichlet(double){}void solve(double,int){}};
struct Neknek{void updateBoundary(int,int,double*,double){}void fixCoupledSurfaceFlux(const occa::memory&,long,const occa::memory&){}};
struct nrs_t{Fluid*fluid;Scalar*scalar;Mesh*meshV=&mesh;Geom*geom=nullptr;Neknek*neknek=nullptr;
 long fieldOffset=5;int tstep=0,Nscalar=4,advectionSubcycingSteps=0,checkpointStep=0,timeStepConverged=1,outerCorrector=1;
 double dt[3]={.01,.01,.01},timePrevious=0,g0=1;occa::memory o_coeffBDF;
 std::function<void(double)>userSource,userProperties,userDivergence;
 std::function<void(double,int)>postScalar,preFluid;
 std::function<bool(int)>userConvergenceCheck;
 void evaluateProperties(double t){userProperties(t);}void evaluateDivergence(double t){userDivergence(t);}
 bool fieldsToSolveContains(std::string){return false;}void adjustFlowRate(int,double){}
 bool runInnerStep(std::function<bool(int)>,int,bool);
}nr,*nrs=&nr;
namespace nek{int output=0;void ifoutfld(int i){output=i;}}
struct Udf{std::function<void(double,int)>executeStep;}udf;
double setPrecision(double x,int){return x;}
namespace opSEM{void strongDivergence(Mesh*,long,const occa::memory&u,occa::memory&out,bool){for(int i=0;i<3;++i)ptr(out)[i]=.01*ptr(u)[10+i];}}
namespace bubbleColumn{
struct Parameters{double gasPressureGradientFilterEnabled=1,gasPressureGradientFilterWeight=.3,subtractAlphaDiffusion=0;
 int mixtureDivergenceForm=2;double virtualMassEnabled=1,divergenceExtrapolationEnabled=0;}p;
bool picardSkipSourceHistory=false,pressureGradientFilterInitialized=true;
int pressureGradientFilterStep=0,alphaTransportHistoryStep=-1;
std::function<void()>picardPressureGradientUpdate,picardAfterScalarPhysics,picardAlphaRhs;
occa::memory o_divergenceUrstOverride;
deviceMemory<double>o_gradP,o_gradPSpatial,o_ulPrevious,o_liquidAcceleration,o_dragLambda,o_alphaTransportRhsHistory;
deviceMemory<double>o_mixtureAdvectorUsed,o_alphaSource,o_alphaTransportOnly,o_alphaRelativeRhs,o_alphaMixtureRhs,o_mixtureForce;
double cumulativeAlphaFilterDeltaVolume=0,cumulativeAlphaClipDeltaVolume=0,cumulativeQgMaskDeltaIntegral[3]={},cumulativeQgMaskDeltaMagnitudeIntegral=0;
int sourceShifts=0,vmCommits=0;
void assembleDivergence(deviceMemory<double>&){}
void prepareGasAdvection(){}
void updateVirtualMassHistory(){
 for(int c=0;c<3;++c)for(int i=0;i<3;++i){int k=c*5+i;
 double ul=ptr(nrs->fluid->o_U)[k]-.01*ptr(nrs->scalar->o_S)[5+k];
 o_liquidAcceleration.ptr()[k]=(ul-o_ulPrevious.ptr()[k])*100;
 o_ulPrevious.ptr()[k]=ul;
 }
}
void evaluatePointwiseTerms(){
 for(int i=0;i<3;++i)o_gradPSpatial.ptr()[i]=.01*ptr(nrs->fluid->o_P)[i];
 if(picardPressureGradientUpdate)picardPressureGradientUpdate();
 else if(p.gasPressureGradientFilterEnabled==0){o_gradP.copyFrom(o_gradPSpatial,15);}
 else if(pressureGradientFilterStep!=nrs->tstep){for(int i=0;i<3;++i)o_gradP.ptr()[i]=.7*o_gradP.ptr()[i]+.3*o_gradPSpatial.ptr()[i];pressureGradientFilterStep=nrs->tstep;}
 for(int i=0;i<3;++i){
 double a=ptr(nrs->scalar->o_S)[i],g=ptr(nrs->scalar->o_S)[15+i];
 o_alphaSource.ptr()[i]=-.02*a*g;o_alphaTransportOnly.ptr()[i]=o_alphaSource.ptr()[i];
 o_alphaRelativeRhs.ptr()[i]=.02*a*(ptr(nrs->fluid->o_U)[10+i]-g);
 ptr(nrs->fluid->o_prop)[i]=.1+.01*a;ptr(nrs->fluid->o_prop)[5+i]=10-2*a;
 o_dragLambda.ptr()[i]=.1+.02*std::abs(g);
 }
}
void prepareRelativeDivergenceRhs(){}
void subtractScalarDiffusion(){}
void evaluateMixtureForce(){for(int c=0;c<3;++c)for(int i=0;i<3;++i)
 o_mixtureForce.ptr()[c*5+i]=.03*ptr(nrs->scalar->o_S)[i]+.01*ptr(nrs->scalar->o_S)[(c+1)*5+i]-.01*ptr(nrs->fluid->o_U)[c*5+i];}
void addExplicitSources(double){
 o_mixtureAdvectorUsed.copyFrom(nrs->fluid->o_U,15);evaluatePointwiseTerms();evaluateMixtureForce();
 for(int i=0;i<3;++i){
 ptr(nrs->scalar->o_EXT)[i]=o_alphaSource.ptr()[i];
 for(int c=0;c<3;++c){int k=c*5+i;double g=ptr(nrs->scalar->o_S)[(c+1)*5+i];
 ptr(nrs->scalar->o_EXT)[(c+1)*5+i]=-.02*g*g-.01*o_gradP.ptr()[i]+.01*ptr(nrs->fluid->o_U)[k]+.005*o_liquidAcceleration.ptr()[k];
 ptr(nrs->fluid->o_EXT)[k]=o_mixtureForce.ptr()[k];
 }
 }
 if(picardSkipSourceHistory&&p.virtualMassEnabled){for(int c=0;c<3;++c)for(int i=0;i<3;++i){int k=c*5+i;
 double expected=(ptr(nrs->fluid->o_U)[k]-.01*ptr(nrs->scalar->o_S)[5+k]-o_ulPrevious.ptr()[k])*100;
 assert(std::abs(expected-o_liquidAcceleration.ptr()[k])<1e-12);}}
 if(!picardSkipSourceHistory){++sourceShifts;alphaTransportHistoryStep=nrs->tstep;for(int i=0;i<3;++i)o_alphaTransportRhsHistory.ptr()[i]=o_alphaRelativeRhs.ptr()[i];}
}
void postProcessGasFlux(){for(int i=0;i<3;++i)ptr(nrs->scalar->o_S)[15+i]+=1e-8;
 cumulativeQgMaskDeltaIntegral[2]+=1e-8;cumulativeQgMaskDeltaMagnitudeIntegral+=1e-8;}
void postProcessAlpha(double,int){for(int i=0;i<3;++i)ptr(nrs->scalar->o_S)[i]+=1e-7;
 cumulativeAlphaFilterDeltaVolume+=1e-7;cumulativeAlphaClipDeltaVolume+=2e-7;}
void updateProperties(double){if(picardAfterScalarPhysics)picardAfterScalarPhysics();else{postProcessGasFlux();evaluatePointwiseTerms();}
 if(picardAlphaRhs)picardAlphaRhs();else o_alphaMixtureRhs.copyFrom(o_alphaTransportRhsHistory,3);
 for(int i=0;i<3;++i)ptr(nrs->fluid->o_div)[i]=.8*o_alphaMixtureRhs.ptr()[i];}
void updateDivergence(double){}
}
'''
# Actual production module is compiled against native deviceMemory with strict
# mutable-reference linAlg signatures, not a hand-reimplemented Picard loop.
methods=r'''
void Fluid::applyDirichlet(double){ptr(o_U)[0]=.05;ptr(o_Ue)[0]=.05;ptr(o_P)[2]=-.5*ptr(o_U)[12]*ptr(o_U)[12];}
void Scalar::applyDirichlet(double){ptr(o_S)[0]=.01;ptr(o_Se)[0]=.01;ptr(o_S)[15]=.3;ptr(o_Se)[15]=.3;}
void Scalar::solve(double,int){++solves;for(int c=0;c<4;++c)for(int i=0;i<3;++i)if(i||c==1||c==2){
 double lambda=c?bubbleColumn::o_dragLambda.ptr()[i]:0.;
 ptr(o_S)[c*5+i]=ptr(o_JwF)[c*5+i]/(ptr(mesh.o_LMM)[i]*(100*nrs->g0+lambda));}}
void Fluid::solve(double,int){++solves;
 double expectedConvection[15]={};
 for(int c=0;c<3;++c)for(int i=0;i<3;++i)expectedConvection[c*5+i]=.02*ptr(o_U)[i]*ptr(o_U)[c*5+i];
 for(int i=0;i<2;++i)ptr(o_P)[i]=.001*ptr(o_JwF)[i]/ptr(o_prop)[5+i]+.01*ptr(o_div)[i];
 for(int c=0;c<3;++c)for(int i=0;i<3;++i)if(c||i)
 ptr(o_U)[c*5+i]=(ptr(o_JwF)[c*5+i]/ptr(mesh.o_Jw)[i]-.01*ptr(o_P)[i]/ptr(o_prop)[5+i])/(100*nrs->g0);
 if(bubbleColumn::picard::corrector){
 for(int c=0;c<3;++c)for(int i=0;i<3;++i){int j=c*5+i;
 double expected=bubbleColumn::picard::baseU.ptr()[j]+ptr(mesh.o_Jw)[i]*bubbleColumn::o_mixtureForce.ptr()[j];
 if(bubbleColumn::picard::umConvection)expected-=ptr(mesh.o_Jw)[i]*expectedConvection[j];
 assert(std::abs(ptr(o_JwF)[j]-expected)<1e-12);
 }}
}
'''
lagu=function(repo/'src/solver/fluid/fluidSolver.cpp','void fluidSolver_t::lagSolution()').replace('fluidSolver_t::','Fluid::').replace('{','{++lagCount;',1)
lags=function(repo/'src/solver/scalar/scalarSolver.cpp','void scalar_t::lagSolution()').replace('scalar_t::','Scalar::').replace('{','{++lagCount;',1)
driver=function(repo/'src/app/nrs/nrs.cpp','bool nrs_t::runInnerStep(')
execute=function(case/'bubbleColumn.udf','void UDF_ExecuteStep(')
execute=execute[:execute.index('  // Measure the fluid update')]+r'''
  ++accepted; if(nrs->checkpointStep)++outputs;
  if(bubbleColumn::p.virtualMassEnabled){bubbleColumn::updateVirtualMassHistory();++bubbleColumn::vmCommits;}
}
'''
kernels='\n#define p_nBDF 3\n#define p_nEXT 3\n#define p_ADVECTION 1\n#define p_SUBCYCLING 0\n#define p_MovingMesh 0\n'
for kind in ['fluid','scalar']:
 kernels+=serial((repo/f'src/solver/{kind}/kernels/sumMakef.okl').read_text()).replace('sumMakef(',kind+'SumMakef(')
main=r'''
int main(int argc,char**argv){
 int mode=std::stoi(argv[1]),passes=std::stoi(argv[2]);
 occa::properties prop;prop["mode"]="Serial";platform->device.setup(prop);
 Boundary bc;App app{&bc};platform->app=&app;
 double w[3]={1,2,3},jw[3]={2,3,5};mesh.o_LMM=platform->device.malloc<double>(3,w);mesh.o_Jw=platform->device.malloc<double>(3,jw);
 Fluid f;Scalar s;nrs->fluid=&f;nrs->scalar=&s;
 f.o_U=alloc(45,.05);f.o_Ue=alloc(15);f.o_P=alloc(5,.1);f.o_relUrst=alloc(15);f.o_div=alloc(5);
 f.o_prop=alloc(10,.1);f.o_EXT=alloc(45);f.o_ADV=alloc(45);f.o_JwF=alloc(15);f.o_coeffEXT=alloc(3);f.o_coeffBDF=alloc(3);
 s.o_S=alloc(60,.01);for(int i=15;i<18;++i)ptr(s.o_S)[i]=.3;
 s.o_Se=alloc(20);s.o_prop=alloc(40,.1);s.o_JwF=alloc(20);s.o_EXT=alloc(60);s.o_ADV=alloc(60);
 // Scalar and fluid EXT coefficients intentionally have different values.
 s.o_coeffEXT=alloc(3);s.o_coeffBDF=f.o_coeffBDF;nrs->o_coeffBDF=f.o_coeffBDF;
 using namespace bubbleColumn;
 for(auto field:{&o_alphaSource,&o_alphaTransportOnly,&o_alphaRelativeRhs,&o_alphaMixtureRhs,&o_dragLambda}){field->resize(5);std::fill(field->ptr(),field->ptr()+field->size(),0.);}
 for(auto field:{&o_gradP,&o_gradPSpatial,&o_ulPrevious,&o_liquidAcceleration,&o_mixtureAdvectorUsed,&o_mixtureForce,&o_alphaTransportRhsHistory}){field->resize(15);std::fill(field->ptr(),field->ptr()+field->size(),0.);}
 o_ulPrevious.copyFrom(f.o_U,15);
 nrs->userSource=&addExplicitSources;nrs->postScalar=&postProcessAlpha;nrs->userProperties=&updateProperties;nrs->userDivergence=&updateDivergence;
 if(std::getenv("PICARD_TEST_CUBATURE"))platform->options.m["ADVECTION TYPE"]="CUBATURE";
 if(std::getenv("PICARD_TEST_NO_VM"))p.virtualMassEnabled=0;
 if(std::getenv("PICARD_TEST_NO_PRESSURE_EMA"))p.gasPressureGradientFilterEnabled=0;
 platform->options.m["EQUATION TYPE"]="NAVIERSTOKES";
 if(std::getenv("PICARD_TEST_UM_CONVECTION"))picard::umConvection=1;
 picard::mode=mode;picard::maxPasses=passes;picard::acceptAtCap=argc>3?std::stoi(argv[3]):0;picard::configure();
 udf.executeStep=&UDF_ExecuteStep;
 if(argc>4){picard::rtol=1e-30;picard::alphaAtol=picard::ugAtol=picard::umAtol=picard::pressureAtol=picard::divAtol=1e-30;}
 int nativeAdvectionCalls=0;
 for(int step=1;step<=6;++step){
 nrs->tstep=step;nrs->timePrevious=(step-1)*.01;nrs->checkpointStep=step%2==0;nrs->timeStepConverged=false;
 double bdf[3][3]={{1,0,0},{2,-.5,0},{3,-1.5,1./3}},ext[3][3]={{1,0,0},{2,-1,0},{3,-3,1}},scalarCoeff[3]={1,0,0};
 int order=std::min(step,3);nrs->g0=order==1?1:order==2?1.5:11./6;
 f.o_coeffBDF.copyFrom(bdf[order-1],3);f.o_coeffEXT.copyFrom(ext[order-1],3);s.o_coeffEXT.copyFrom(scalarCoeff,3);
 f.o_Ue.copyFrom(f.o_U,15);s.o_Se.copyFrom(s.o_S,20);f.o_relUrst.copyFrom(f.o_U,15);
 for(int i=0;i<15;++i)ptr(f.o_EXT)[i]=0;for(int i=0;i<20;++i)ptr(s.o_EXT)[i]=0;
 nrs->userSource(nrs->timePrevious);++nativeAdvectionCalls;
 for(int c=0;c<3;++c)for(int i=0;i<3;++i){int j=c*5+i;ptr(f.o_ADV)[j]=.02*ptr(f.o_U)[i]*ptr(f.o_U)[j];ptr(f.o_EXT)[j]+=.00003;}
 for(int c=0;c<4;++c)for(int i=0;i<3;++i)ptr(s.o_EXT)[c*5+i]+=.00002;
 double rho[20];std::fill(rho,rho+20,1.);
 double hpfU[45],hpfS[60],expectedBaseU[15]={},expectedBaseS[20]={},noAdv[45]={},noAdvBase[15]={};
 std::fill(hpfU,hpfU+45,.00003);std::fill(hpfS,hpfS+60,.00002);
 // Independent native-forcing reference with HPF and native EXT UM ADV.
 fluidSumMakef(3,ptr(mesh.o_Jw),100,ptr(f.o_coeffEXT),ptr(f.o_coeffBDF),5,15,5,rho,ptr(f.o_U),ptr(f.o_ADV),hpfU,expectedBaseU);
 for(int c=0;c<4;++c)scalarSumMakef(3,ptr(mesh.o_LMM),100,ptr(s.o_coeffEXT),ptr(s.o_coeffBDF),5*c,20,5,rho,ptr(s.o_S),ptr(s.o_ADV),hpfS,expectedBaseS);
 fluidSumMakef(3,ptr(mesh.o_Jw),100,ptr(f.o_coeffEXT),ptr(f.o_coeffBDF),5,15,5,rho,ptr(f.o_U),noAdv,hpfU,noAdvBase);
 fluidSumMakef(3,ptr(mesh.o_Jw),100,ptr(f.o_coeffEXT),ptr(f.o_coeffBDF),5,15,5,rho,ptr(f.o_U),ptr(f.o_ADV),ptr(f.o_EXT),ptr(f.o_JwF));
 for(int c=0;c<4;++c)scalarSumMakef(3,ptr(mesh.o_LMM),100,ptr(s.o_coeffEXT),ptr(s.o_coeffBDF),5*c,20,5,rho,ptr(s.o_S),ptr(s.o_ADV),ptr(s.o_EXT),ptr(s.o_JwF));
 for(int k=3;k>1;--k){f.o_EXT.copyFrom(f.o_EXT,15,(k-1)*15,(k-2)*15);f.o_ADV.copyFrom(f.o_ADV,15,(k-1)*15,(k-2)*15);
 s.o_EXT.copyFrom(s.o_EXT,20,(k-1)*20,(k-2)*20);s.o_ADV.copyFrom(s.o_ADV,20,(k-1)*20,(k-2)*20);}
 std::vector<double> nativeAdv(ptr(f.o_ADV),ptr(f.o_ADV)+45);
 for(int it=1;;++it){auto check=[&](int n){return nrs->userConvergenceCheck?nrs->userConvergenceCheck(n):true;};
 bool done=nrs->runInnerStep(check,it,true);
 assert(std::equal(nativeAdv.begin(),nativeAdv.end(),ptr(f.o_ADV)));
 if(mode){for(int c=0;c<3;++c)for(int i=0;i<3;++i){int j=c*5+i;double expected=expectedBaseU[j];
 if(picard::umConvection)expected=noAdvBase[j];
 assert(std::abs(picard::baseU.ptr()[j]-expected)<1e-11);
 }
 for(int c=0;c<4;++c)for(int i=0;i<3;++i)assert(std::abs(picard::baseS.ptr()[5*c+i]-expectedBaseS[5*c+i])<1e-11);}
 if(done)break;assert(it<passes);}
 assert(f.lagCount==step&&s.lagCount==step&&nativeAdvectionCalls==step);
 assert(accepted==step&&outputs==step/2&&vmCommits==step*int(p.virtualMassEnabled!=0));
 assert(std::abs(cumulativeAlphaFilterDeltaVolume-step*1e-7)<1e-18);
 assert(std::abs(cumulativeAlphaClipDeltaVolume-step*2e-7)<1e-18);
 assert(std::abs(cumulativeQgMaskDeltaMagnitudeIntegral-step*1e-8)<1e-18);
 assert(sourceShifts==step);
 if(picard::corrector)for(int i=0;i<3;++i)assert(std::abs(o_gradP.ptr()[i]-(p.gasPressureGradientFilterEnabled?(.7*picard::pressureAnchor.ptr()[i]+.3*.01*ptr(f.o_P)[i]):.01*ptr(f.o_P)[i]))<1e-14);
 }
 std::ofstream out("result.txt");out<<std::setprecision(17);
 for(auto field:{f.o_U,f.o_P,s.o_S,f.o_EXT,s.o_EXT,f.o_ADV,s.o_ADV})for(size_t i=0;i<field.size();++i)out<<ptr(field)[i]<<'\n';
 if(mode){ptr(f.o_U)[15]+=.1;bool caught=false;try{picard::convergence(picard::pass+1);}catch(const std::runtime_error&){caught=true;}assert(caught);}
}
'''
cpp=prefix+'\n#include "bubbleColumnPicard.hpp"\n'+methods+lagu+lags+'\nint accepted=0,outputs=0;\n'+execute+driver+kernels+main
with tempfile.TemporaryDirectory() as d:
 path=Path(d);(path/'nekrsSys.hpp').write_text('#pragma once\n');(path/'check.cpp').write_text(cpp)
 subprocess.run(['g++','-std=c++17',str(path/'check.cpp'),'-I'+str(path),'-I'+str(case),'-I'+str(repo/'src/platform'),'-I'+str(occa/'include'),'-L'+str(occa/'lib'),'-Wl,-rpath,'+str(occa/'lib'),'-locca','-o',str(path/'check')],check=True)
 baseline=None
 for mode,passes in [(0,1),(1,1),(1,4),(2,1),(2,12)]:
  run=path/f'{mode}_{passes}';run.mkdir()
  result=subprocess.run([str(path/'check'),str(mode),str(passes)],cwd=run,capture_output=True,text=True)
  if result.returncode:print(result.stdout[-4000:]);print(result.stderr);raise RuntimeError((mode,passes))
  state=(run/'result.txt').read_text()
  if mode==0:baseline=state
  elif mode==1 or passes==1:assert state==baseline,(mode,passes)
  else:
   rows=list(csv.DictReader((run/'bubbleColumn_picard.csv').open()))[:-1]
   accepted=[x for x in rows if x['status']=='converged'];assert len(accepted)==6
   assert all(float(x['history_max'])==0 for x in rows)
   assert state!=baseline
  print(f'PASS mode={mode} passes={passes}: history, frozen UM convection, RHS weights and single commits.')
 for flag in ['PICARD_TEST_CUBATURE','PICARD_TEST_NO_VM','PICARD_TEST_NO_PRESSURE_EMA']:
  run=path/flag;run.mkdir()
  r=subprocess.run([str(path/'check'),'2','12'],cwd=run,capture_output=True,text=True,env=dict(os.environ,**{flag:'1'}))
  assert r.returncode==0,(flag,r.stdout[-2000:],r.stderr)
  print('PASS',flag)
 # Enabled convection: one-pass predictor unchanged; correctors update convection.
 for cubature in [False,True]:
  for passes in [1,12]:
   run=path/('um_'+str(cubature)+'_'+str(passes));run.mkdir()
   env=dict(os.environ,PICARD_TEST_UM_CONVECTION='1')
   if cubature:env['PICARD_TEST_CUBATURE']='1'
   r=subprocess.run([str(path/'check'),'2',str(passes)],cwd=run,capture_output=True,text=True,env=env)
   assert r.returncode==0,(cubature,passes,r.stdout[-3000:],r.stderr)
   state=(run/'result.txt').read_text()
   if passes==1:assert state==baseline
   else:
    assert state!=baseline and state!=(path/'2_12'/'result.txt').read_text()
    rows=list(csv.DictReader((run/'bubbleColumn_picard.csv').open()))[:-1]
    assert len([x for x in rows if x['status']=='converged'])==6
    assert all(float(x['history_max'])==0 for x in rows)
   print('PASS iterated UM convection:', 'cubature' if cubature else 'GLL', 'passes',passes)
 for invalidMode in [0,1]:
  r=subprocess.run([str(path/'check'),str(invalidMode),'2'],cwd=path,capture_output=True,text=True,
                   env=dict(os.environ,PICARD_TEST_UM_CONVECTION='1'))
  assert r.returncode!=0 and 'requires picardMode=2' in r.stderr
 for allow in [0,1]:
  run=path/f'cap{allow}';run.mkdir()
  r=subprocess.run([str(path/'check'),'2','2',str(allow),'strict'],cwd=run,capture_output=True,text=True)
  assert (r.returncode==0)==bool(allow),(allow,r.stdout,r.stderr)
  if not allow:assert "reached maxPasses without convergence" in r.stderr
print('Production Picard + native inner driver/BDF1-3 forcing on real OCCA Serial arrays passed; elliptic/spatial physics are controlled doubles, not full GPU/MPI validation.')
