#!/usr/bin/env python3
"""CPU production-kernel and core dispatch tests. No NekRS installation needed."""
from pathlib import Path
import re,subprocess,tempfile
root=Path(__file__).resolve().parents[1]
repo=root.parents[1]
def extract(source,marker):
    start=source.index(marker);b=source.index('{',start);end=b+1;depth=1
    while depth:
        depth+=(source[end]=='{')-(source[end]=='}');end+=1
    return source[start:end]
def serial(source):
    source=source.replace('@kernel','').replace('@restrict','').replace('@atomic','')
    source=re.sub(r';\s*@tile\([^;\n]*?\)\)',')',source)
    return re.sub(r';\s*@(?:outer|inner)\(\d+\)','',source)
with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp)
    (p/'equations_serial.hpp').write_text(serial((root/'bubbleColumn4Equations.okl').read_text()))
    def run(file):
        subprocess.run(['c++','-std=c++17','-Wall','-Wextra','-Werror','-I'+tmp,str(file),'-o',str(p/'test')],check=True)
        subprocess.run([str(p/'test')],check=True)
    run(root/'tests/kernels.cpp')
    dispatch=extract((repo/'src/solver/fluid/fluidSolver.cpp').read_text(),'void fluidSolver_t::solve(double time, int stage)')
    prefix='''
#include <cassert>
#include <functional>
#include <stdexcept>
#include <vector>
#define EXIT_FAILURE 1
#define nekrsCheck(condition,comm,code,fmt,...) do {if(condition) throw std::runtime_error(fmt);}while(0)
struct fluidSolver_t {
 std::function<void(double,int)> userPressureSolve,userVelocitySolve;
 std::vector<int> ellipticSolver;int* ellipticSolverP=nullptr;
 std::vector<int> calls;
 void solve(double,int);
 void solvePressure(double t,int s){assert(t==2&&s==1);calls.push_back(1);}
 void solveVelocity(double t,int s){assert(t==2&&s==1);calls.push_back(2);}
};
'''
    suffix='''
int main(){
 fluidSolver_t f;f.solve(2,1);assert((f.calls==std::vector<int>{1,2}));f.calls.clear();
 double latestPressure=0;
 f.userPressureSolve=[&](double t,int s){assert(t==2&&s==1);latestPressure=42;f.calls.push_back(3);};
 f.userVelocitySolve=[&](double,int){assert(latestPressure==42);f.calls.push_back(4);f.calls.push_back(5);};
 f.solve(2,1);assert((f.calls==std::vector<int>{3,4,5}));
 f.userVelocitySolve=nullptr;bool rejected=false;
 try{f.solve(2,1);}catch(const std::runtime_error&){rejected=true;}assert(rejected);
 f.userVelocitySolve=[](double,int){};f.ellipticSolver.push_back(1);rejected=false;
 try{f.solve(2,1);}catch(const std::runtime_error&){rejected=true;}assert(rejected);
 puts("PASS: actual core fluid solve dispatch, latest pressure before both momenta, native regression, missing-pair and conflicting-solver guards");
}
'''
    (p/'dispatch.cpp').write_text(prefix+dispatch+suffix);run(p/'dispatch.cpp')
    # The numerical full-stress host test includes actual production callbacks.
    host=(root/'tests/stress_adapter.cpp').read_text()
    header=(root/'bubbleColumn4Solve.hpp').read_text()
    functions='\n'.join(extract(header,m) for m in [
        'inline void crossStress(', 'inline void viscousAcceleration(', 'inline void solvePhase('])
    (p/'stress.cpp').write_text(host.replace('// PRODUCTION_FUNCTIONS',functions))
    run(p/'stress.cpp')

    host=(root/'tests/pressure_adapter.cpp').read_text()
    (p/'pressure.cpp').write_text(host.replace('// PRODUCTION_FUNCTIONS',extract(header,'inline void pressureStep(')))
    run(p/'pressure.cpp')
