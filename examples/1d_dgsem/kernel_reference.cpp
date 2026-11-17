// Periodic 1D reference embedded in a 3D tensor-product mesh.
// Runs the actual production DG OKL and native weak divergence kernels.
// Test-only dense MPI exchange replaces OGS; production uses sparse face OGS.
#include <occa.hpp>
#include <mpi.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <stdexcept>
#include <cstdio>
using V=std::vector<double>;
#ifndef DG_NQ
#define DG_NQ 4
#endif
constexpr int nq=DG_NQ,np=nq*nq*nq,nfp=nq*nq,ns=19,nv=12;
constexpr double pi=3.14159265358979323846;
double globalSum(double v){double s;MPI_Allreduce(&v,&s,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);return s;}
int main(int argc,char **argv){
 MPI_Init(&argc,&argv);
 if(argc!=10) {std::cerr<<"usage: kernels weak nx flux steps dt reference.csv output-prefix mode\n";MPI_Abort(MPI_COMM_WORLD,2);}
int rank,size;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
 occa::device d;d.setup(std::string(argv[9]));occa::properties props;
 for(auto kv:std::vector<std::pair<std::string,int>>{{"p_Nq",nq},{"p_Np",np},{"p_Nfp",nfp},{"p_Nfaces",6},{"p_Nsgeo",ns},{"p_Nvgeo",nv},{"p_RXID",0},{"p_RYID",1},{"p_RZID",2},{"p_SXID",3},{"p_SYID",4},{"p_SZID",5},{"p_TXID",6},{"p_TYID",7},{"p_TZID",8},{"p_JWID",10},{"p_NXID",0},{"p_NYID",1},{"p_NZID",2},{"p_WSJID",10},{"p_blockSize",64}}) props["defines/"+kv.first]=kv.second;
 props["defines/dfloat"]="double";props["defines/dlong"]="int";
 auto vf=d.buildKernel(argv[1],"dgVolumeFlux",props),tr=d.buildKernel(argv[1],"dgTrace",props),
 sf=d.buildKernel(argv[1],"dgSurfaceFlux",props),lift=d.buildKernel(argv[1],"dgLift",props),
 rk=d.buildKernel(argv[1],"dgRK",props),
 weak=d.buildKernel(argv[2],"wDivergenceVolumeHex3D",props);
 V nodes(nq),weights(nq),deriv(nq*nq),bary(nq,1.);
 const int degree=nq-1;
 auto legendre=[&](double x){double a=1,b=x;for(int k=2;k<=degree;++k){double c=((2*k-1)*x*b-(k-1)*a)/k;a=b;b=c;}return std::pair<double,double>(b,a);};
 for(int i=0;i<nq;++i){
  double x=-std::cos(pi*i/degree);
  if(i>0 && i<degree)for(int it=0;it<30;++it){auto pp=legendre(x);x-=(x*pp.first-pp.second)/((degree+1)*pp.first);}
  nodes[i]=x;double pn=legendre(x).first;weights[i]=2./(degree*(degree+1)*pn*pn);
 }
 for(int i=0;i<nq;++i)for(int j=0;j<nq;++j)if(i!=j)bary[i]/=nodes[i]-nodes[j];
 for(int i=0;i<nq;++i)for(int j=0;j<nq;++j)if(i!=j){deriv[i*nq+j]=bary[j]/bary[i]/(nodes[i]-nodes[j]);deriv[i*nq+i]-=deriv[i*nq+j];}
 {
  const int fluxType=std::stoi(argv[4]),steps=std::stoi(argv[5]);
  const double dt=std::stod(argv[6]);
  const int nx=std::stoi(argv[3]),ne=nx/size,N=ne*np,NF=ne*6*nfp,GN=nx*nfp;
  if(nx%size || ne<1 || nq<2 || fluxType<0 || fluxType>2 || steps<4 || steps%4 || !std::isfinite(dt) || dt<=0) MPI_Abort(MPI_COMM_WORLD,2);
  V refX(nx*nq),refInitial(nx*nq),refRhs(nx*nq),refFinal(nx*nq);
  std::ifstream input(argv[7]);
  for(int q=0;q<nx*nq;++q) {
    if(!(input>>refX[q]>>refInitial[q]>>refRhs[q]>>refFinal[q])) MPI_Abort(MPI_COMM_WORLD,3);
  }
  assert(nx%size==0);double dx=1.0/nx;
  V a(N),u(3*N),mass(N),vg(ne*np*nv),sg(NF*ns),x(N);
  std::vector<int> map(NF),inc(6*N,-1),bid(6*ne,3),ids(NF,-1);
  for(int e=0;e<ne;++e){
   int eg=rank*ne+e;
   for(int k=0;k<nq;++k)for(int j=0;j<nq;++j)for(int i=0;i<nq;++i){
    int l=i+nq*(j+nq*k),n=e*np+l,b=e*np*nv+l;
    x[n]=(eg+.5*(nodes[i]+1))*dx;mass[n]=dx/8*weights[i]*weights[j]*weights[k];
    vg[b]=2/dx;vg[b+4*np]=2;vg[b+8*np]=2;vg[b+10*np]=mass[n];u[n]=1;
   }
   for(int f=0;f<6;++f)for(int v=0;v<nfp;++v){
    int i=v%nq,j=v/nq,l=0,axis=0;double normal=0,sj=0;
    if(f==0){l=i+nq*j;axis=2;normal=-1;sj=dx/4;}
    if(f==5){l=i+nq*(j+nq*(nq-1));axis=2;normal=1;sj=dx/4;}
    if(f==1){l=i+nq*nq*j;axis=1;normal=-1;sj=dx/4;}
    if(f==3){l=i+nq*((nq-1)+nq*j);axis=1;normal=1;sj=dx/4;}
    if(f==2){l=(nq-1)+nq*(i+nq*j);axis=0;normal=1;sj=.25;}
    if(f==4){l=nq*(i+nq*j);axis=0;normal=-1;sj=.25;}
    int fn=(e*6+f)*nfp+v;map[fn]=e*np+l;inc[6*map[fn]+f]=fn;
    sg[fn*ns+axis]=normal;sg[fn*ns+10]=sj*weights[i]*weights[j];
    if(f==2 || f==4){bid[e*6+f]=0;ids[fn]=((eg+(f==2))%nx)*nfp+v;}
   }
  }
  auto A=d.malloc<double>(N),Old=d.malloc<double>(N),Increment=d.malloc<double>(N),U=d.malloc<double>(3*N,u.data()),
   M=d.malloc<double>(N,mass.data()),VG=d.malloc<double>(vg.size(),vg.data()),SG=d.malloc<double>(sg.size(),sg.data()),
   D=d.malloc<double>(nq*nq,deriv.data()),F=d.malloc<double>(3*N),R=d.malloc<double>(N),T=d.malloc<double>(2*NF),
   S=d.malloc<double>(2*NF),Face=d.malloc<double>(NF),B=d.malloc<double>(NF);
  auto Map=d.malloc<int>(NF,map.data()),Inc=d.malloc<int>(6*N,inc.data()),Bid=d.malloc<int>(6*ne,bid.data());
  auto vol=[&](){A.copyTo(a.data());double v=0;for(int i=0;i<N;++i)v+=mass[i]*a[i];return globalSum(v);};
  auto eval=[&](double inlet=0.0,double backflow=0.0){
   vf(N,N,A,U,F);weak(ne,VG,D,N,F,R);tr(NF,N,Map,SG,A,U,T);
   V t(2*NF),sum(2*NF),send(2*GN,0),recv(2*GN);T.copyTo(t.data());
   for(int i=0;i<NF;++i)if(ids[i]>=0)for(int c=0;c<2;++c)send[ids[i]+c*GN]+=t[i+c*NF];
   MPI_Allreduce(send.data(),recv.data(),2*GN,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
   for(int i=0;i<NF;++i)for(int c=0;c<2;++c)sum[i+c*NF]=ids[i]>=0?recv[ids[i]+c*GN]:t[i+c*NF];
   S.copyFrom(sum.data());sf(NF,inlet,backflow,fluxType,Bid,SG,T,S,Face,B);lift(N,Inc,M,Face,R);
   V r(N),b(NF);R.copyTo(r.data());B.copyTo(b.data());double balance=0;
   for(int i=0;i<N;++i)balance+=mass[i]*r[i];for(double v:b)balance+=v;
   double gb=globalSum(balance);
   if(!std::isfinite(gb) || std::abs(gb)>2e-10) MPI_Abort(MPI_COMM_WORLD,8);
   double net=0;for(double v:b)net+=v;return globalSum(net);
  };
  // Use element-local initial values, including separate interface traces.
  double coordinateError=0;
  for(int e=0;e<ne;++e)for(int k=0;k<nq;++k)for(int j=0;j<nq;++j)for(int i=0;i<nq;++i){
    int n=e*np+i+nq*(j+nq*k),q=(rank*ne+e)*nq+i;
    a[n]=refInitial[q];coordinateError=std::max(coordinateError,std::abs(x[n]-refX[q]));
  }
  if(coordinateError>1e-13) MPI_Abort(MPI_COMM_WORLD,4);
  A.copyFrom(a.data());double initial=vol();eval();
  V initialRhs(N);R.copyTo(initialRhs.data());
  double rhsError=0;
  for(int e=0;e<ne;++e)for(int k=0;k<nq;++k)for(int j=0;j<nq;++j)for(int i=0;i<nq;++i){
    int n=e*np+i+nq*(j+nq*k),q=(rank*ne+e)*nq+i;
    rhsError=std::max(rhsError,std::abs(initialRhs[n]-refRhs[q]));
  }
  // Export each rank independently; preserve every element-local trace.
  // Publish by rename only after the complete snapshot has reached disk.
  auto snapshot=[&](int step){
    A.copyTo(a.data());
    const int index=step/(steps/4);
    const std::string file=std::string(argv[8])+".snapshot"+std::to_string(index)+".rank"+std::to_string(rank)+".csv";
    const std::string temporary=file+".tmp";
    std::ofstream out(temporary);
    if(!out) MPI_Abort(MPI_COMM_WORLD,9);
    out<<"step,time,element,i,j,k,x,mass,scalar\n"<<std::setprecision(17);
    for(int e=0;e<ne;++e)for(int k=0;k<nq;++k)for(int j=0;j<nq;++j)for(int i=0;i<nq;++i){
      const int n=e*np+i+nq*(j+nq*k);
      out<<step<<","<<step*dt<<","<<rank*ne+e<<","<<i<<","<<j<<","<<k<<","<<x[n]<<","<<mass[n]<<","<<a[n]<<"\n";
    }
    out.close();
    if(!out || std::rename(temporary.c_str(),file.c_str())) MPI_Abort(MPI_COMM_WORLD,9);
    const double currentMass=vol();
    if(rank==0)std::cout<<std::setprecision(17)<<"EXPORT snapshot="<<index<<" step="<<step<<" time="<<step*dt
      <<" mass="<<currentMass<<" massDrift="<<currentMass-initial<<std::endl;
  };
  snapshot(0);
  for(int step=0;step<steps;++step){
    Old.copyFrom(A);
    for(int stage=0;stage<4;++stage){eval();rk(N,stage,dt,Old,R,Increment,A);}
    if((step+1)%(steps/4)==0) snapshot(step+1);
  }
  double final=vol(),finalError=0;
  std::ofstream out(std::string(argv[8])+".rank"+std::to_string(rank)+".csv");
  if(!out) MPI_Abort(MPI_COMM_WORLD,5);
  out<<"element,i,j,k,x,initial,rhs,final,reference_final\n"<<std::setprecision(17);
  for(int e=0;e<ne;++e)for(int k=0;k<nq;++k)for(int j=0;j<nq;++j)for(int i=0;i<nq;++i){
    int n=e*np+i+nq*(j+nq*k),q=(rank*ne+e)*nq+i;
    if(!std::isfinite(a[n]) || !std::isfinite(initialRhs[n])) MPI_Abort(MPI_COMM_WORLD,6);
    finalError=std::max(finalError,std::abs(a[n]-refFinal[q]));
    out<<rank*ne+e<<","<<i<<","<<j<<","<<k<<","<<x[n]<<","<<refInitial[q]<<","
       <<initialRhs[n]<<","<<a[n]<<","<<refFinal[q]<<"\n";
  }
  double local[2]={rhsError,finalError},global[2];
  MPI_Allreduce(local,global,2,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
  if(rank==0)std::cout<<std::setprecision(16)<<"nx="<<nx<<" flux="<<fluxType<<" ranks="<<size
    <<" RHS max error="<<global[0]<<" final max error="<<global[1]<<" mass drift="<<final-initial<<"\n";
  if(!std::isfinite(final) || global[0]>1e-9 || global[1]>1e-9 || std::abs(final-initial)>1e-10) MPI_Abort(MPI_COMM_WORLD,7);
 }
 MPI_Finalize();
}
