"""Actual callback + history kernel + native scalar forcing: convection only fluid EXT."""
from pathlib import Path
import re,subprocess,tempfile
case=Path(__file__).resolve().parents[1];repo=case.parents[1]
terms=(case/'bubbleColumn2Terms.hpp').read_text();okl=(case/'bubbleColumn2Equations.okl').read_text()
def function(s,marker):
 start=s.index(marker);end=s.index('{',start)+1;depth=1
 while depth:
  depth+=(s[end]=='{')-(s[end]=='}');end+=1
 return s[start:end]
def serial(s):
 return re.sub(r';\s*@tile\(p_blockSize,\s*@outer,\s*@inner\)','',s.replace('@kernel ','').replace('@ restrict ',''))
kernel=serial(function(okl,'@kernel void extrapolateScalarConvection('))
native=serial((repo/'src/solver/scalar/kernels/sumMakef.okl').read_text())
callback=function(terms,'inline void addExplicitSources(')
cpp=r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>
using dfloat=double;using dlong=long;
#define p_nBDF 2
#define p_nEXT 3
#define p_ADVECTION 1
#define p_SUBCYCLING 0
#define p_MovingMesh 0
bool close(double a,double b){return std::abs(a-b)<1e-10*(1+std::abs(b));}
'''+kernel+native+r'''
struct Mem{
 std::shared_ptr<std::vector<double>> data;long off=0,count=0;
 Mem(long n=0):data(std::make_shared<std::vector<double>>(n,0)),count(n){}
 double& operator[](long i){return data->at(off+i);}double operator[](long i)const{return data->at(off+i);}
 double* ptr(){return data->data()+off;}
 long size()const{return count;}
 Mem slice(long a,long n)const{auto m=*this;m.off+=a;m.count=n;return m;}
 void copyFrom(const Mem& m,long n,long dst=0,long src=0){for(long j=0;j<n;j++)(*this)[dst+j]=m[src+j];}
};
struct LA{
 void axpby(long n,double a,const Mem& x,double b,Mem y){for(long j=0;j<n;j++)y[j]=a*x[j]+b*y[j];}
}la;
struct Platform{LA* linAlg=&la;}env;auto platform=&env;
struct Mesh{long Nlocal=2;}mesh;
struct Scalar{
 long fieldOffsetSum=18;
 std::map<std::string,int> nameToIndex{{"alpha",0},{"ugx",1},{"ugy",2},{"ugz",3}};
 std::vector<long> fieldOffsetScan{0,5,9,14};
 Mem o_EXT{54};
 Mem o_explicitTerms(const std::string& n){return o_EXT.slice(fieldOffsetScan.at(nameToIndex.at(n)),3);}
}scalar;
struct Fluid{Mem o_coeffEXT{3},rhs{9};Mem o_explicitTerms(){return rhs;}}fluid;
struct Nrs{int tstep=0;long fieldOffset=3;Mesh* meshV=&mesh;Scalar* scalar=&::scalar;Fluid* fluid=&::fluid;}app;auto nrs=&app;
struct Params{int frozenGasVelocityEnabled=0,gasSolveAfterFluid=0;}p;
Mem o_alphaSource(3),o_ugSource(9),o_mixtureForce(9),o_gasTransport(12);
Mem o_nativeTransportAdvection(18);
Mem o_scalarConvectionCurrent(12),o_scalarConvectionEXT(12),o_scalarConvectionHistory(36);
int scalarConvectionHistoryStep=-1;
std::array<double,4> nonlinear{},extra{};
double rawNative[4][2]={{.8,-.2},{-.4,.7},{.2,-.5},{-.1,.6}};
double masses[2]={2,5};
void prepareGasAdvection(){}
void evaluatePointwiseTerms(){
 for(int n=0;n<2;n++){
  o_gasTransport[n]=nonlinear[0];o_alphaSource[n]=-nonlinear[0];
  for(int c=0;c<3;c++)o_ugSource[n+c*3]=extra[c+1];
 }
}
void applyGasVirtualMassExtrapolation(){}
void captureGasPressureSource(){}
void subtractScalarDiffusion(){for(int n=0;n<2;n++)o_alphaSource[n]+=extra[0];}
void evaluateMixtureForce(){for(int n=0;n<2;n++)for(int c=0;c<3;c++)o_mixtureForce[n+c*3]=.6+c;}
void freezeExplicitVirtualMassHistory(){}
void rampMixtureDrag(int){}
void freezeDragHistory(){}
void captureAlphaAdvectionDiagnostics(){}
void evaluateGasCubatureTransport(bool conservative){
 assert(!conservative);
 for(int n=0;n<2;n++){
  o_gasTransport[n]=12345; // Compression is NOT full conservative alpha convection.
  for(int c=0;c<3;c++)o_gasTransport[n+(c+1)*3]=nonlinear[c+1];
 }
}
void evaluateNativeAlphaAdvection(Mem& out,bool assembled,const char* name,bool mixture){
 assert(assembled && mixture);
 int f=scalar.nameToIndex.at(name);double avg=(2*rawNative[f][0]+5*rawNative[f][1])/7;
 for(int n=0;n<2;n++)out[scalar.fieldOffsetScan[f]+n]=avg;
}
void extrapolateScalarConvectionKernel(long N,long offset,int stages,int use,int first,int advance,
 Mem coeff,Mem current,Mem history,Mem ext){
 // Host must use the live FLUID coefficients, not scalar EXT1 or constants.
 assert(coeff.data==fluid.o_coeffEXT.data);
 extrapolateScalarConvection(N,offset,stages,use,first,advance,coeff.ptr(),current.ptr(),history.ptr(),ext.ptr());
}
'''+callback+r'''
std::array<double,3> lagrange(const std::array<double,3>& times,double target){
 std::array<double,3> out{};
 for(int j=0;j<3;j++){out[j]=1;for(int k=0;k<3;k++)if(j!=k)out[j]*=(target-times[k])/(times[j]-times[k]);}
 return out;
}
int main(){
 std::fill(scalar.o_EXT.data->begin(),scalar.o_EXT.data->end(),777);
 std::fill(fluid.rhs.data->begin(),fluid.rhs.data->end(),888);
 std::fill(o_scalarConvectionHistory.data->begin(),o_scalarConvectionHistory.data->end(),999);
 std::fill(o_scalarConvectionEXT.data->begin(),o_scalarConvectionEXT.data->end(),999);
 std::vector<std::array<double,4>> reference;
 std::array<double,3> times{};double t=0;
 // Multiple initial CFL probes must not create or shift time history.
 for(int probe=0;probe<2;probe++){
  for(int f=0;f<4;f++){nonlinear[f]=3+f+probe;extra[f]=.1*f;}
  auto history=*o_scalarConvectionHistory.data;
  addExplicitSources(0);
  assert(scalarConvectionHistoryStep==-1 && *o_scalarConvectionHistory.data==history);
  for(int f=0;f<4;f++)assert(o_scalarConvectionEXT[3*f]==nonlinear[f]);
 }
 for(int step=1;step<=8;step++){
  nrs->tstep=step;
  times[2]=times[1];times[1]=times[0];times[0]=t;
  double dt=step<5?.02:(step%2?.013:.037); // Exercise unequal timesteps too.
  std::array<double,3> beta=step==1?std::array<double,3>{1,0,0}:
      step==2?std::array<double,3>{2,-1,0}:lagrange(times,t+dt);
  for(int j=0;j<3;j++)fluid.o_coeffEXT[j]=beta[j];
  for(int f=0;f<4;f++){
   nonlinear[f]=std::sin(11*t+f)+(.4+f)*step*step;
   // Strongly varying signed non-convective sources expose accidental EXT.
   extra[f]=(step%2?1.:-1.)*(100+f)*step*step;
  }
  reference.insert(reference.begin(),nonlinear);
  if(step==1)while(reference.size()<3)reference.push_back(nonlinear);
  if(reference.size()>3)reference.resize(3);
  for(int repeat=0;repeat<2;repeat++){
   // Refresh the current sample without consuming a second history level.
   if(repeat){nonlinear[0]+=.37;reference[0]=nonlinear;}
   addExplicitSources(t);
   assert(scalarConvectionHistoryStep==step);
   for(int f=0;f<4;f++){
    double expected=0;for(int j=0;j<3;j++)expected+=beta[j]*reference[j][f];
    assert(close(o_scalarConvectionEXT[3*f],expected));
    // The initial/native cancellation uses unequal element-local ADV loads.
    double U[36],RHO[18],ADV[54],BF[18]={},BDF[2]={2,-.5},EXT1[3]={1,0,0};
    std::fill_n(U,36,.25);std::fill_n(RHO,18,1.);std::fill_n(ADV,54,30000.);
    int at=scalar.fieldOffsetScan[f];
    for(int n=0;n<2;n++)ADV[at+n]=rawNative[f][n];
    sumMakef(2,masses,1/dt,EXT1,BDF,at,18,2,RHO,U,ADV,scalar.o_EXT.ptr(),BF);
    double time=7*(2*.25-.5*.25)/dt;
    // Conservative alpha and material UG both use EXT convection, latest
    // non-convective sources, ordinary BDF, and exact shared-node cancellation.
    assert(close(BF[at]+BF[at+1],time+7*(extra[f]-expected)));
    for(int j=0;j<3;j++)for(int n=0;n<2;n++)assert(close(o_scalarConvectionHistory[j*12+3*f+n],reference[j][f]));
   }
   // No modification to native fluid coefficients or forcing components.
   for(int j=0;j<3;j++)assert(fluid.o_coeffEXT[j]==beta[j]);
   for(int c=0;c<3;c++)for(int n=0;n<2;n++)assert(close(fluid.rhs[c*3+n],.6+c));
   for(int c=0;c<3;c++)assert(fluid.rhs[c*3+2]==888);
   // Packed and native scalar padding remains untouched.
   for(int f=0;f<4;f++){
    assert(o_scalarConvectionEXT[3*f+2]==999);
    for(int j=0;j<3;j++)assert(o_scalarConvectionHistory[j*12+3*f+2]==999);
   }
   for(int n:{2,3,4,7,8,11,12,13,16,17})assert(scalar.o_EXT[n]==777);
  }
  t+=dt;
 }
 // The selected mode uses whatever native EXT order is configured, including 1.
 double c[1]={1},x[12]={},h[12]={},y[12]={};x[0]=3;
 extrapolateScalarConvection(2,3,1,1,1,1,c,x,h,y);assert(y[0]==3);
}
'''
with tempfile.TemporaryDirectory() as directory:
 p=Path(directory)/'check.cpp';p.write_text(cpp)
 subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-Wno-unknown-pragmas','-Wno-unused-parameter',str(p),'-o',str(p.with_suffix(''))],check=True)
 subprocess.run([str(p.with_suffix(''))],check=True)
# Field packing must preserve conservative alpha before overwriting transport[0].
assert callback.index('o_scalarConvectionCurrent.copyFrom(o_gasTransport, Nlocal);')<callback.index('evaluateGasCubatureTransport(false);')
assert 'nrs->scalar->o_coeffEXT = o_scalarCoeffEXT;' in terms
assert 'nrs->fluid->o_coeffEXT.copyFrom' not in terms
print('Actual callback/history/native forcing passed: complete convection EXT1/2/3, variable dt, startup probes, repeated steps, conservative-alpha packing, UG material convection, source isolation, unequal shared-node mass cancellation, BDF and padding.')
