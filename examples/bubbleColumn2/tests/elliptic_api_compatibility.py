"""Compile the actual corrector adapter against both elliptic solve APIs."""
from pathlib import Path
import subprocess
import tempfile

case=Path(__file__).resolve().parents[1]
cpp=r'''
#include <cassert>
#include <memory>
#include <vector>
#include "bubbleColumn2Elliptic.hpp"
struct Mem {
 std::shared_ptr<double> value;
 explicit Mem(double x):value(std::make_shared<double>(x)){}
};
// Signatures match NekRS-LS: setters take const occa::memory&, solve takes
// const RHS reference and solution by value. No four-argument overload exists.
struct Modern {
 Mem diff{0},lambda{0};std::vector<int> order;
 void coeff0HLM(const Mem& x){diff=x;order.push_back(0);}
 void coeff1HLM(const Mem& x){lambda=x;order.push_back(1);}
 void solve(const Mem& rhs,Mem x){
  order.push_back(2);*x.value=*rhs.value/(*diff.value+*lambda.value);
 }
};
struct Legacy {
 int calls=0;
 void solve(const Mem& diff,const Mem& lambda,const Mem& rhs,Mem x){
  calls++;*x.value=*rhs.value/(*diff.value+*lambda.value);
 }
};
int main(){
 Modern modern;Legacy legacy;
 for(double coefficient:{2.,7.}){
  Mem diff(coefficient),lambda(3.),rhs(20.),a(0.),b(0.);
  bubbleColumn2::solveGasPressureCorrection(&modern,diff,lambda,rhs,a,0);
  bubbleColumn2::solveGasPressureCorrection(&legacy,diff,lambda,rhs,b,0);
  assert(*a.value==20./(coefficient+3.) && *a.value==*b.value);
  assert(modern.diff.value==diff.value && modern.lambda.value==lambda.value);
 }
 assert((modern.order==std::vector<int>{0,1,2,0,1,2}));
 assert(legacy.calls==2);
}
'''
with tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp);(path/'check.cpp').write_text(cpp)
    subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-I',str(case),str(path/'check.cpp'),'-o',str(path/'check')],check=True)
    subprocess.run([str(path/'check')],check=True)
print('Corrector elliptic adapter: NekRS-LS two-argument API and legacy four-argument API compile and use refreshed coefficients in order.')
