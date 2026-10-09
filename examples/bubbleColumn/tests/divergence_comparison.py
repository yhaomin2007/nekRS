"""Compile actual statistics routine against host reductions with nonuniform mass."""
from pathlib import Path
import subprocess,tempfile,csv,math
r=Path(__file__).resolve().parents[1]
s=(r/'bubbleColumnTerms.hpp').read_text()
body=s.split('inline void printDivergenceComparison(double time, int tstep)')[1].split('inline occa::memory implicitGasDrag')[0]
pre='''#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <vector>
using dfloat=double;using dlong=long;
struct Mem { double *a; Mem slice(long start,long) {return {a+start};}
 void copyFrom(Mem b,long n) {std::copy(b.a,b.a+n,a);} };
struct LA {
 double min(long n,Mem a,int) {return *std::min_element(a.a,a.a+n);}
 double max(long n,Mem a,int) {return *std::max_element(a.a,a.a+n);}
 double innerProd(long n,Mem a,Mem b,int) {double v=0;for(long i=0;i<n;++i)v+=a.a[i]*b.a[i];return v;}
 void axmy(long n,double c,Mem a,Mem b) {for(long i=0;i<n;++i)b.a[i]*=c*a.a[i];}
} la;
struct Comm {int mpiComm(){return 0;}int mpiRank(){return 0;}};
struct Platform { LA *linAlg;Comm comm;} platformObject{&la},*platform=&platformObject;
double mass[3]={1,2,3};
struct Mesh {long Nlocal=3;double volume=6;Mem o_LMM{mass};} mesh;
struct Nrs {Mesh *meshV;long fieldOffset=3;} nrsObject{&mesh},*nrs=&nrsObject;
struct Parameters {int divergenceComparisonInterval=10;} p;
double data[9]={1,2,4, 2,0,1, -1,2,3},scratch[3];
Mem o_divComparison{data},o_divComparisonScratch{scratch};
'''
cpp=pre+'inline void printDivergenceComparison(double time,int tstep)'+body+'''
int main(){printDivergenceComparison(.5,9);printDivergenceComparison(.5,10);}
'''
with tempfile.TemporaryDirectory() as t:
 q=Path(t);(q/'check.cpp').write_text(cpp)
 subprocess.run(['g++','-std=c++17',str(q/'check.cpp'),'-o',str(q/'check')],check=True)
 result=subprocess.run([str(q/'check')],cwd=t,check=True,text=True,capture_output=True)
 assert result.stdout.count('divUmCompare')==1
 rows=list(csv.DictReader((q/'bubbleColumn_divergence_compare.csv').open()))
 assert len(rows)==1
 row=rows[0]
 for prefix,values in [('m0',[1,2,4]),('m1',[2,0,1]),('diff',[-1,2,3])]:
  expected={'min':min(values),'max':max(values),'mean':sum(w*v for w,v in zip([1,2,3],values))/6,
   'rms':math.sqrt(sum(w*v*v for w,v in zip([1,2,3],values))/6)}
  for key,value in expected.items():assert abs(float(row[prefix+'_'+key])-value)<1e-12
 assert float(row['diff_maxabs'])==3
 assert abs(float(row['diff_relative_rms'])-math.sqrt(36/57))<1e-12
print('Actual divergence statistics: signed difference, min/max, weighted mean/RMS, relative RMS, CSV mapping and interval passed.')
