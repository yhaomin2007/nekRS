// Execute NekRS's actual weak pressure-gradient kernel with OCCA Serial.
#include <occa.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
int main(int argc,char**argv){
 if(argc!=2)throw std::runtime_error("usage: weak_pressure path/to/wGradientVolumeHex3D.okl");
 occa::device device;device.setup(std::string("{mode:'Serial'}"));occa::properties props;
 props["defines/dfloat"]="double";props["defines/dlong"]="int";
 for(auto kv:std::vector<std::pair<std::string,int>>{{"p_Nq",3},{"p_Np",27},{"p_Nvgeo",11},
 {"p_RXID",0},{"p_RYID",1},{"p_RZID",2},{"p_SXID",3},{"p_SYID",4},{"p_SZID",5},
 {"p_TXID",6},{"p_TYID",7},{"p_TZID",8},{"p_JWID",9}})props["defines/"+kv.first]=kv.second;
 auto kernel=device.buildKernel(argv[1],"wGradientVolumeHex3D",props);
 const int N=27;std::vector<double> D={-1.5,2,-.5,-.5,0,.5,.5,-2,1.5},w={1./3,4./3,1./3},x={-1,0,1};
 std::vector<double> geometry(11*N),pressure(N),out(3*N),expected(3*N);
 for(int k=0;k<3;++k)for(int j=0;j<3;++j)for(int i=0;i<3;++i){
  const int n=k*9+j*3+i;geometry[n]=geometry[n+4*N]=geometry[n+8*N]=1;
  geometry[n+9*N]=w[i]*w[j]*w[k];pressure[n]=5+2*x[i]+3*x[j]+4*x[k];
 }
 for(int k=0;k<3;++k)for(int j=0;j<3;++j)for(int i=0;i<3;++i){const int n=k*9+j*3+i;
  for(int a=0;a<3;++a){
   expected[n]+=D[3*a+i]*w[a]*w[j]*w[k]*pressure[k*9+j*3+a];
   expected[n+N]+=D[3*a+j]*w[i]*w[a]*w[k]*pressure[k*9+a*3+i];
   expected[n+2*N]+=D[3*a+k]*w[i]*w[j]*w[a]*pressure[a*9+j*3+i];
  }
 }
 auto vg=device.malloc<double>(geometry.size(),geometry.data()),d=device.malloc<double>(D.size(),D.data());
 auto p=device.malloc<double>(N,pressure.data()),g=device.malloc<double>(3*N);
 kernel(1,vg,d,N,p,g);g.copyTo(out.data());double error=0;
 for(int i=0;i<3*N;++i)error=std::max(error,std::abs(out[i]-expected[i]));
 // Interior node: weak load is -M grad(p), not +M grad(p).
 for(int c=0;c<3;++c)error=std::max(error,std::abs(out[13+c*N]+std::pow(4./3,3)*(2+c)));
 if(error>1e-12)throw std::runtime_error("native weak pressure gradient mismatch");
 std::cout<<"PASS: actual native weak-gradient kernel, all volume/boundary loads and pressure sign; max error="<<error<<'\n';
}
