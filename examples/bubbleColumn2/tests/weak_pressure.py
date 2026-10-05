"""Translate actual OKL pressure kernels to serial C++; check two-element SBP loads."""
from pathlib import Path
import re, subprocess, tempfile
root=Path(__file__).resolve().parents[1]
repo=root.parents[1]
def translate(s):
 s=s.replace('@kernel ', '').replace('@ restrict ', '').replace('@shared ', '')
 # Exclusive arrays have one instance per (i,j) lane.
 names=re.findall(r'@exclusive dfloat (\w+)\[p_Nq\]',s)
 for name in names:
  s=s.replace('@exclusive dfloat '+name+'[p_Nq]', 'dfloat '+name+'[p_Nq][p_Nq][p_Nq]')
  s=re.sub(r'\b'+name+r'\[([^\]]+)\](?!\[)', name+r'[j][i][\1]',s)
 s=re.sub(r'; @tile\(p_blockSize, @outer, @inner\)', '', s)
 s=re.sub(r'; @(?:outer|inner)\(\d\)', '', s)
 s=s.replace('@barrier();','')
 return s
core=[]
for name in ['wGradientVolumeHex3D','gradientVolumeHex3D']:
 core.append(translate((repo/'src/core/kernels'/f'{name}.okl').read_text()))
case=(root/'bubbleColumn2Equations.okl').read_text()
case=case[case.index('@kernel void prepareGasPressure'):case.index('#include "bubbleColumn2PressureBoundary.okl"')]
body=translate((root/'bubbleColumn2PressureBoundary.okl').read_text())
cpp=r'''
#include <algorithm>
#include <cmath>
#include <cassert>
#include <cstdio>
using dfloat=double;using dlong=long;
#define p_Nq 3
#define p_Np 27
#define p_Nfp 9
#define p_Nfaces 6
#define p_Nvgeo 10
#define p_RXID 0
#define p_RYID 1
#define p_RZID 2
#define p_SXID 3
#define p_SYID 4
#define p_SZID 5
#define p_TXID 6
#define p_TYID 7
#define p_TZID 8
#define p_JWID 9
#define p_Nsgeo 4
#define p_WSJID 0
#define p_NXID 1
#define p_NYID 2
#define p_NZID 3
'''+''.join(core)+translate(case)+body+r'''
int main(){
 constexpr int off=54;
 double vg[540]={},sg[432]={},D[]={-1.5,2,-.5,-.5,0,.5,.5,-2,1.5};
 double w[]={1./3,4./3,1./3},xi[]={-1,0,1};
 long vm[108];int bc[12];double alpha[off],P[off],B[off],BP[off],gB[162],load[162];
 for(int e=0;e<2;e++){
 for(int k=0;k<3;k++)for(int j=0;j<3;j++)for(int i=0;i<3;i++){
 int n=i+3*j+9*k;
 vg[e*270+p_RXID*27+n]=2;vg[e*270+p_SYID*27+n]=1;vg[e*270+p_TZID*27+n]=1;
 vg[e*270+p_JWID*27+n]=.5*w[i]*w[j]*w[k];
 }
 for(int f=0;f<6;f++){
 bc[e*6+f]=((e==0 && f==2)||(e==1 && f==4))?0:1;
 for(int b=0;b<3;b++)for(int a=0;a<3;a++){
 int i,j,k;double nx=0,ny=0,nz=0,ws;
 if(f==0||f==5){i=a;j=b;k=f==0?0:2;nz=f==0?-1:1;ws=.5*w[a]*w[b];}
 else if(f==1||f==3){i=a;j=f==1?0:2;k=b;ny=f==1?-1:1;ws=.5*w[a]*w[b];}
 else{i=f==2?2:0;j=a;k=b;nx=f==2?1:-1;ws=w[a]*w[b];}
 int sk=e*54+f*9+a+3*b;vm[sk]=e*27+i+3*j+9*k;
 sg[sk*4]=ws;sg[sk*4+1]=nx;sg[sk*4+2]=ny;sg[sk*4+3]=nz;
 }
 }
 }
 for(int varied=0;varied<2;varied++)for(int linear=0;linear<2;linear++){
 for(int e=0;e<2;e++)for(int k=0;k<3;k++)for(int j=0;j<3;j++)for(int i=0;i<3;i++){
 int id=e*27+i+3*j+9*k;double x=e+(xi[i]+1)/2;
 alpha[id]=varied?.1+.02*x:.1;P[id]=linear?5+2*x+3*xi[j]-xi[k]:5;
 }
 prepareGasPressure(off,1.2,1,alpha,P,B,BP);
 wGradientVolumeHex3D(2,vg,D,off,BP,load);
 gradientVolumeHex3D(2,vg,D,off,B,gB);
 finishGasPressureVolume(off,off,P,gB,load);
 for(int c=0;c<3;c++)gasPressureBoundary(2,c,sg,vm,bc,BP,load+c*off);
 // Gather shared interface nodes and compare against direct strong load.
 for(int c=0;c<3;c++)for(int e=0;e<2;e++)for(int k=0;k<3;k++)for(int j=0;j<3;j++)for(int i=0;i<3;i++){
 if(e==1 && i==0)continue;
 int n=i+3*j+9*k,id=e*27+n;double got=load[c*off+id];
 double mass=vg[e*270+p_JWID*27+n];
 if(e==0 && i==2){int next=27+3*j+9*k;got+=load[c*off+next];mass+=vg[270+p_JWID*27+3*j+9*k];}
 double dp=linear?(c==0?2:(c==1?3:-1)):0;
 assert(std::abs(got+mass*B[id]*dp)<1e-12);
 }
 // Pressure offset must not affect physical load (constant-pressure case above).
 }
 puts("Actual weak volume/coefficient and face kernels: two-element assembly, constant/nonzero pressure, linear pressure, variable alpha and all components passed.");
}
'''
with tempfile.TemporaryDirectory() as t:
 f=Path(t)/'test.cpp';f.write_text(cpp)
 subprocess.run(['g++','-std=c++17','-O0','-Wall','-Wno-unknown-pragmas',str(f),'-o',str(Path(t)/'test')],check=True)
 subprocess.run([str(Path(t)/'test')],check=True)
