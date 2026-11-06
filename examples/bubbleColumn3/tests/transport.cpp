// Optional real OCCA runtime test: compile/link against the installed libocca.
#include <occa.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
using V=std::vector<double>;
int main(int argc,char**argv) {
  if(argc!=2)throw std::runtime_error("usage: transport path/to/bubbleColumn3Transport.okl");
  occa::device device;device.setup(std::string("{mode: 'Serial'}"));
  occa::properties props;
  props["defines/dfloat"]="double";props["defines/dlong"]="int";
  for(auto kv:std::vector<std::pair<std::string,int>>{{"p_Nq",3},{"p_Np",27},{"p_cubNq",5},{"p_cubNp",125},
       {"p_Nvgeo",11},{"p_RXID",0},{"p_RYID",1},{"p_RZID",2},{"p_SXID",3},{"p_SYID",4},{"p_SZID",5},
       {"p_TXID",6},{"p_TYID",7},{"p_TZID",8},{"p_JWID",9},{"p_IJWID",10},{"p_blockSize",64}})
    props["defines/"+kv.first]=kv.second;
  auto interp=device.buildKernel(argv[1],"gasInterpolate",props);
  auto material=device.buildKernel(argv[1],"gasCubatureProducts",props);
  auto conservative=device.buildKernel(argv[1],"alphaConservativeCubature",props);
  auto project=device.buildKernel(argv[1],"gasProject",props);
  const int N=27,C=125;
  V gll={-1,0,1},wgll={1./3,4./3,1./3};
  const double q=std::sqrt(10./7);
  V gl={-std::sqrt(5+2*q)/3,-std::sqrt(5-2*q)/3,0,std::sqrt(5-2*q)/3,std::sqrt(5+2*q)/3};
  V weights={(322-13*std::sqrt(70.))/900,(322+13*std::sqrt(70.))/900,128./225,
       (322+13*std::sqrt(70.))/900,(322-13*std::sqrt(70.))/900};
  V I(15),D(25),bary(5,1),fields(4*N),vgeo(11*N),cvgeo(11*C),out(4*N),alpha(4*N);
  for(int a=0;a<3;++a)for(int i=0;i<5;++i) {
    double l=1;for(int b=0;b<3;++b)if(b!=a)l*=(gl[i]-gll[b])/(gll[a]-gll[b]);
    I[i+a*5]=l;
  }
  for(int i=0;i<5;++i)for(int j=0;j<5;++j)if(i!=j)bary[i]/=(gl[i]-gl[j]);
  for(int i=0;i<5;++i)for(int j=0;j<5;++j)if(i!=j) {D[i*5+j]=bary[j]/(bary[i]*(gl[i]-gl[j]));D[i*5+i]-=D[i*5+j];}
  for(int k=0;k<3;++k)for(int j=0;j<3;++j)for(int i=0;i<3;++i) {
    int n=k*9+j*3+i;double x=gll[i],y=gll[j],z=gll[k];
    fields[n]=0.2+0.02*x;fields[n+N]=x;fields[n+2*N]=2*y;fields[n+3*N]=3*z;
    vgeo[n]=vgeo[n+4*N]=vgeo[n+8*N]=1;
    vgeo[n+10*N]=1/(wgll[i]*wgll[j]*wgll[k]);
  }
  for(int k=0;k<5;++k)for(int j=0;j<5;++j)for(int i=0;i<5;++i) {
    int n=k*25+j*5+i;cvgeo[n]=cvgeo[n+4*C]=cvgeo[n+8*C]=1;
    cvgeo[n+9*C]=weights[i]*weights[j]*weights[k];
  }
  auto mem=[&](V&v){return device.malloc<double>(v.size(),v.data());};
  auto oI=mem(I),oD=mem(D),oF=mem(fields),ov=mem(vgeo),ocv=mem(cvgeo);
  auto cube=device.malloc<double>(4*C),prod=device.malloc<double>(4*C),oOut=device.malloc<double>(4*N),oAlpha=device.malloc<double>(4*N);
  interp(1,N,C,oI,oF,cube);material(1,C,oD,ocv,cube,prod);project(1,N,C,oI,ov,prod,oOut);
  conservative(1,C,oD,ocv,cube,prod);project(1,N,C,oI,ov,prod,oAlpha);
  oOut.copyTo(out.data());oAlpha.copyTo(alpha.data());
  double error=0;
  for(int k=0;k<3;++k)for(int j=0;j<3;++j)for(int i=0;i<3;++i) {
    const int n=k*9+j*3+i;
    // div(alpha*u)=6*alpha+0.02*x, u.grad(u)=(x,4y,9z).
    error=std::max(error,std::abs(alpha[n]-(1.2+0.14*gll[i])));
    error=std::max(error,std::abs(out[n+N]-gll[i]));
    error=std::max(error,std::abs(out[n+2*N]-4*gll[j]));
    error=std::max(error,std::abs(out[n+3*N]-9*gll[k]));
  }
  if(error>1e-11)throw std::runtime_error("manufactured cubature transport failed: "+std::to_string(error));
  std::cout<<"PASS: real OCCA interpolation, conservative alpha, material phase convection, and projection; max error="<<error<<'\n';
}
