"""Compile actual implicit coefficient callback and verify scalar routing."""
from pathlib import Path
import subprocess,tempfile
r=Path(__file__).resolve().parents[1]
s=(r/'bubbleColumnTerms.hpp').read_text()
hook=s.split('inline occa::memory implicitGasDrag')[1].split('inline void updateDivergence(')[0]
cpp='''#include <cassert>
#include <map>
#include <string>
namespace occa { using memory=int; }
struct Parameters { int alphaCompressionMode; double dragEnabled; } p;
struct Scalar { std::map<std::string,int> nameToIndex{{"alpha",0}}; } scalar;
struct Nrs { Scalar *scalar; } object{&scalar},*nrs=&object;
int o_alphaCompressionRate=11,o_dragLambda=22,o_NULL=0;
'''+ 'inline occa::memory implicitGasDrag'+hook+'''
int main() {
 for(int mode=0;mode<=2;++mode) for(int drag=0;drag<=1;++drag) {
   p.alphaCompressionMode=mode;p.dragEnabled=drag;
   assert(implicitGasDrag(0,0)==(mode==2?11:0));
   for(int i=1;i<=3;++i) assert(implicitGasDrag(0,i)==(drag?22:0));
 }
}
'''
with tempfile.TemporaryDirectory() as t:
 p=Path(t);(p/'check.cpp').write_text(cpp)
 subprocess.run(['g++','-std=c++17',str(p/'check.cpp'),'-o',str(p/'check')],check=True)
 subprocess.run([str(p/'check')],check=True)
# Verify the frozen coefficient is not rebuilt during post-solve properties.
properties=s.split('inline void updateProperties(double)')[1].split('inline occa::memory implicitGasDrag')[0]
assert 'prepareGasAdvection' not in properties
source=s.split('inline void addExplicitSources(double)')[1].split('inline void buildDivergenceFromAlphaRhs')[0]
assert source.index('if (p.alphaCompressionMode == 2)')<source.index('subtractScalarDiffusion();')
assert 'o_alphaCompressionRate, o_alphaSource' in s
print('Actual implicit callback: alpha reaction only in mode 2; UG drag unchanged. Frozen-rate and explicit-source routing checked.')
