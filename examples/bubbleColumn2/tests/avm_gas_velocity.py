"""Compile actual AVM integration, verify UG input and per-scalar parameter routing."""
from pathlib import Path
import subprocess,tempfile
case=Path(__file__).resolve().parents[1]
s=(case/'bubbleColumn2Terms.hpp').read_text()
helper='inline void applyGasVelocityAVM()'+s.split('inline void applyGasVelocityAVM()')[1].split('inline void updateProperties')[0]
cpp=r'''
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
using dfloat=double;
struct Mem{
 std::shared_ptr<std::vector<double>> v;
 Mem(int n=0):v(std::make_shared<std::vector<double>>(n,0)){}
};
Mem o_ug(18),fluidUM(18);
struct Scalar{
 std::map<std::string,int> nameToIndex{{"alpha",0},{"ugx",1},{"ugy",2},{"ugz",3}};
 std::vector<Mem> solution{Mem(6),Mem(6),Mem(6),Mem(6)},diff{Mem(6),Mem(6),Mem(6),Mem(6)};
 Mem o_solution(std::string n){return solution[nameToIndex.at(n)];}
 Mem o_diffusionCoeff(std::string n){return diff[nameToIndex.at(n)];}
} scalar;
struct Mesh{int N=5,Nlocal=4;}mesh;
struct Nrs{Scalar* scalar=&::scalar;Mesh* meshV=&mesh;int fieldOffset=6;} solver;
auto nrs=&solver;
std::string scalarDigitStr(int n){return "0"+std::to_string(n);}
struct Options{
 std::map<std::string,std::string> strings;
 std::map<std::string,double> values;
 bool compareArgs(std::string k,std::string v){return strings[k]==v;}
 void getArgs(std::string k,double& v){if(values.count(k))v=values[k];}
};
struct LA{
 void axpby(int n,double a,const Mem& x,double b,Mem& y){for(int i=0;i<n;i++)y.v->at(i)=a*x.v->at(i)+b*y.v->at(i);}
 std::vector<std::pair<double,double>> minMax(int n,std::initializer_list<Mem> ms,int){
  std::vector<std::pair<double,double>> r;for(auto m:ms){auto a=std::minmax_element(m.v->begin(),m.v->begin()+n);r.push_back({*a.first,*a.second});}return r;
 }
}la;
struct Comm{int mpiComm(){return 0;}int mpiRank(){return 0;}};
struct Platform{Options options;Comm comm;LA* linAlg=&la;bool verbose(){return true;}}env;
auto platform=&env;
int calls=0;
namespace avm{
 Mem viscosity(long offset,const Mem& velocity,const Mem& field,double noise,double scaling,double decay,double width,bool c0){
  assert(offset==6 && velocity.v==o_ug.v && velocity.v!=fluidUM.v);
  int index=-1;for(int i=0;i<4;i++)if(field.v==scalar.solution[i].v)index=i;
  assert(index>=0);std::string p="SCALAR"+scalarDigitStr(index)+" ";
  auto &opts=platform->options;
  assert(noise==opts.values[p+"REGULARIZATION AVM ABSOLUTE TOL"]);
  assert(scaling==opts.values[p+"REGULARIZATION AVM SCALING COEFF"]);
  assert(decay==opts.values[p+"REGULARIZATION AVM DECAY THRESHOLD"]);
  assert(width==opts.values[p+"REGULARIZATION AVM ACTIVATION WIDTH"]);
  assert(c0==(index%2==0));calls++;
  Mem out(6);for(int i=0;i<6;i++)out.v->at(i)=velocity.v->at(i)*scaling;return out;
 }
}
#define nekrsCheck(test,...) do{if(test)throw std::runtime_error("bad order");}while(0)
'''+helper+r'''
int main(){
 for(int i=0;i<18;i++){o_ug.v->at(i)=.3;fluidUM.v->at(i)=.001;}
 for(int j=0;j<4;j++){
  std::string p="SCALAR"+scalarDigitStr(j)+" ";
  platform->options.values[p+"REGULARIZATION AVM ABSOLUTE TOL"]=1e-6*(j+1);
  platform->options.values[p+"REGULARIZATION AVM SCALING COEFF"]=j+1;
  platform->options.values[p+"REGULARIZATION AVM DECAY THRESHOLD"]=2+j;
  platform->options.values[p+"REGULARIZATION AVM ACTIVATION WIDTH"]=1+j;
  platform->options.strings[p+"REGULARIZATION AVM C0"]=j%2==0?"TRUE":"FALSE";
 }
 for(int mask=0;mask<16;mask++)for(int repeat=0;repeat<2;repeat++){
  int expected=0;calls=0;
  for(int j=0;j<4;j++){
   bool on=(mask>>j)&1;expected+=on;
   platform->options.strings["SCALAR"+scalarDigitStr(j)+" REGULARIZATION METHOD"]=on?"AVM_AVERAGED_MODAL_DECAY":(j%2?"HPFRT":"NONE");
   for(int n=0;n<6;n++)scalar.diff[j].v->at(n)=2+repeat; // new baseline per callback
  }
  applyGasVelocityAVM();assert(calls==expected);
  for(int j=0;j<4;j++)for(int n=0;n<6;n++){
   double val=2+repeat+(((mask>>j)&1)&&n<4?.3*(j+1):0);
   assert(std::abs(scalar.diff[j].v->at(n)-val)<1e-12);
  }
 }
 mesh.N=4;bool rejected=false;try{applyGasVelocityAVM();}catch(const std::runtime_error&){rejected=true;}assert(rejected);
}
'''
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp);(p/'check.cpp').write_text(cpp)
 subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror',str(p/'check.cpp'),'-o',str(p/'check')],check=True)
 subprocess.run([str(p/'check')],check=True,stdout=subprocess.DEVNULL)
assert 'nrs->scalar->applyAVM();' not in s
assert '#include "avm.hpp"' in s
print('Actual UG AVM integration: UG pointer/stride, scalar sensors/options, all enable combinations, baseline preservation, padding, repeated resets, verbose branch and order guard passed.')
