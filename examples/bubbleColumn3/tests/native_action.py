#!/usr/bin/env python3
"""Check the production explicit-coefficient native action/assembly contract.

The small adapter models solve() unbinding coefficients and Ax() returning
unassembled element contributions. It is not a CUDA/native elliptic run.
"""
from pathlib import Path
import subprocess,tempfile
root=Path(__file__).resolve().parents[1]
text=(root/'bubbleColumn3Pressure.hpp').read_text()
start=text.index('inline void nativePressureAction(');brace=text.index('{',start)
level=1;end=brace+1
while level:
    level+=(text[end]=='{')-(text[end]=='}');end+=1
function=text[start:end]
assert 'pressureSolver->op(' not in text
assert text.count('nativePressureAction(')==3 # definition + both native actions
# Confirm the adapter's lifetime assumption against this branch's native code.
repo=root.parents[1]
native=(repo/'src/core/elliptic/elliptic.cpp').read_text()
solve=native.split('void elliptic::_solve(',1)[1].split('void checkConfig',1)[0]
assert 'elliptic->o_lambda0 = nullptr' in solve
assert 'elliptic->o_lambda1 = nullptr' in solve
source=r'''
#include <vector>
#include <cassert>
#include <cstdio>
namespace occa {struct memory {std::vector<double> values;};}
using occa::memory;
static memory pressureDiffusion{{4}},o_NULL;
struct Elliptic {
 bool bound=false;int calls=0;
 void solve(){bound=false;}
 void Ax(const memory& coefficient,const memory& mass,const memory& q,memory& r){
  assert(!bound);assert(&coefficient==&pressureDiffusion);assert(&mass==&o_NULL);
  ++calls;r.values={2*q.values[0],3*q.values[1],5*q.values[2],7*q.values[3]};
 }
};
static Elliptic instance;static Elliptic* pressureSolver=&instance;
struct Mesh{int oogs=12;}mesh;
struct Nrs{Mesh* meshV=&mesh;}systemState;
static Nrs* nrs=&systemState;
static const int ogsDfloat=0,ogsAdd=1;
static int assemblies=0;
namespace oogs {void startFinish(memory& r,int fields,int stride,int dtype,int operation,int handle){
 assert(fields==1&&stride==0&&dtype==ogsDfloat&&operation==ogsAdd&&handle==12);
 ++assemblies;double sum=r.values[1]+r.values[2];r.values[1]=r.values[2]=sum;
}}
FUNCTION
int main(){memory q{{1,2,2,3}},out;
 for(int i=0;i<2;++i){instance.solve();nativePressureAction(q,out);
 assert(out.values==std::vector<double>({2,16,16,21}));}
 assert(instance.calls==2&&assemblies==2);
 puts("PASS: production native action explicitly binds coefficients after solve and assembles exactly once");
}
'''.replace('FUNCTION',function)
with tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp);(path/'test.cpp').write_text(source)
    subprocess.run(['c++','-std=c++17','-Wall','-Wextra','-Werror',str(path/'test.cpp'),'-o',str(path/'test')],check=True)
    subprocess.run([str(path/'test')],check=True)
