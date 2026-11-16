// Exercise production DG OKL + native weak divergence on actual OCCA Serial.
// Test-only dense MPI exchange replaces OGS; production uses sparse face OGS.
#include <occa.hpp>
#include <mpi.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
using V=std::vector<double>;
constexpr int nq=3,np=27,nfp=9,ns=19,nv=12;
constexpr double pi=3.14159265358979323846;
double globalSum(double v){double s;MPI_Allreduce(&v,&s,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);return s;}
int main(int argc,char **argv){
 MPI_Init(&argc,&argv);int rank,size;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
 occa::device d;d.setup(std::string("{mode:'Serial'}"));occa::properties props;
 for(auto kv:std::vector<std::pair<std::string,int>>{{"p_Nq",nq},{"p_Np",np},{"p_Nfp",nfp},{"p_Nfaces",6},{"p_Nsgeo",ns},{"p_Nvgeo",nv},{"p_RXID",0},{"p_RYID",1},{"p_RZID",2},{"p_SXID",3},{"p_SYID",4},{"p_SZID",5},{"p_TXID",6},{"p_TYID",7},{"p_TZID",8},{"p_JWID",10},{"p_NXID",0},{"p_NYID",1},{"p_NZID",2},{"p_WSJID",10},{"p_blockSize",64}}) props["defines/"+kv.first]=kv.second;
 props["defines/dfloat"]="double";props["defines/dlong"]="int";
 auto vf=d.buildKernel(argv[1],"dgVolumeFlux",props),tr=d.buildKernel(argv[1],"dgTrace",props),
 sf=d.buildKernel(argv[1],"dgSurfaceFlux",props),lift=d.buildKernel(argv[1],"dgLift",props),
 rk=d.buildKernel(argv[1],"dgRK",props),lim=d.buildKernel(argv[1],"dgLimit",props),
 weak=d.buildKernel(argv[2],"wDivergenceVolumeHex3D",props);
 double errors[2]={};
 for(int level=0;level<2;++level){
  const int nx=4*(1<<level),ne=nx/size,N=ne*np,NF=ne*6*nfp,GN=nx*nfp;
  assert(nx%size==0);double dx=1.0/nx;
  V a(N),u(3*N),mass(N),vg(ne*np*nv),sg(NF*ns),x(N),weights={1./3,4./3,1./3};
  V deriv={-1.5,2,-.5,-.5,0,.5,.5,-2,1.5};
  std::vector<int> map(NF),inc(6*N,-1),bid(6*ne,3),ids(NF,-1);
  for(int e=0;e<ne;++e){
   int eg=rank*ne+e;
   for(int k=0;k<nq;++k)for(int j=0;j<nq;++j)for(int i=0;i<nq;++i){
    int l=i+nq*(j+nq*k),n=e*np+l,b=e*np*nv+l;
    x[n]=(eg+.5*i)*dx;mass[n]=dx/8*weights[i]*weights[j]*weights[k];
    vg[b]=2/dx;vg[b+4*np]=2;vg[b+8*np]=2;vg[b+10*np]=mass[n];u[n]=1;
   }
   for(int f=0;f<6;++f)for(int v=0;v<nfp;++v){
    int i=v%nq,j=v/nq,l=0,axis=0;double normal=0,sj=0;
    if(f==0){l=i+nq*j;axis=2;normal=-1;sj=dx/4;}
    if(f==5){l=i+nq*(j+nq*2);axis=2;normal=1;sj=dx/4;}
    if(f==1){l=i+nq*nq*j;axis=1;normal=-1;sj=dx/4;}
    if(f==3){l=i+nq*(2+nq*j);axis=1;normal=1;sj=dx/4;}
    if(f==2){l=2+nq*(i+nq*j);axis=0;normal=1;sj=.25;}
    if(f==4){l=nq*(i+nq*j);axis=0;normal=-1;sj=.25;}
    int fn=(e*6+f)*nfp+v;map[fn]=e*np+l;inc[6*map[fn]+f]=fn;
    sg[fn*ns+axis]=normal;sg[fn*ns+10]=sj*weights[i]*weights[j];
    if(f==2 || f==4){bid[e*6+f]=0;ids[fn]=((eg+(f==2))%nx)*nfp+v;}
   }
  }
  auto A=d.malloc<double>(N),Old=d.malloc<double>(N),U=d.malloc<double>(3*N,u.data()),
   M=d.malloc<double>(N,mass.data()),VG=d.malloc<double>(vg.size(),vg.data()),SG=d.malloc<double>(sg.size(),sg.data()),
   D=d.malloc<double>(9,deriv.data()),F=d.malloc<double>(3*N),R=d.malloc<double>(N),T=d.malloc<double>(2*NF),
   S=d.malloc<double>(2*NF),Face=d.malloc<double>(NF),B=d.malloc<double>(NF),Bad=d.malloc<double>(ne),Limited=d.malloc<double>(ne);
  auto Map=d.malloc<int>(NF,map.data()),Inc=d.malloc<int>(6*N,inc.data()),Bid=d.malloc<int>(6*ne,bid.data());
  auto vol=[&](){A.copyTo(a.data());double v=0;for(int i=0;i<N;++i)v+=mass[i]*a[i];return globalSum(v);};
  auto eval=[&](double inlet=0.0,double backflow=0.0){
   vf(N,N,A,U,F);weak(ne,VG,D,N,F,R);tr(NF,N,Map,SG,A,U,T);
   V t(2*NF),sum(2*NF),send(2*GN,0),recv(2*GN);T.copyTo(t.data());
   for(int i=0;i<NF;++i)if(ids[i]>=0)for(int c=0;c<2;++c)send[ids[i]+c*GN]+=t[i+c*NF];
   MPI_Allreduce(send.data(),recv.data(),2*GN,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
   for(int i=0;i<NF;++i)for(int c=0;c<2;++c)sum[i+c*NF]=ids[i]>=0?recv[ids[i]+c*GN]:t[i+c*NF];
   S.copyFrom(sum.data());sf(NF,inlet,backflow,Bid,SG,T,S,Face,B);lift(N,Inc,M,Face,R);
   V r(N),b(NF);R.copyTo(r.data());B.copyTo(b.data());double balance=0;
   for(int i=0;i<N;++i)balance+=mass[i]*r[i];for(double v:b)balance+=v;
   assert(std::abs(globalSum(balance))<2e-12);
  };
  auto limit=[&](){lim(ne,1,M,A,Bad,Limited);V bad(ne);Bad.copyTo(bad.data());for(double v:bad)assert(v==0);};
  // Constant-state preservation catches wrong derivative transpose / face sign.
  std::fill(a.begin(),a.end(),.2);A.copyFrom(a.data());eval();V r(N);R.copyTo(r.data());for(double v:r)assert(std::abs(v)<1e-12);
  // Non-divergence-free velocity: conservative transport must contain
  // -alpha*div(u), with no separately added compression source.
  for(int i=0;i<N;++i)u[i]=1+.1*std::sin(2*pi*x[i]);U.copyFrom(u.data());eval();R.copyTo(r.data());
  for(int e=0;e<ne;++e)for(int k=0;k<nq;++k)for(int j=0;j<nq;++j)for(int i=0;i<nq;++i){
   double du=0;for(int q=0;q<nq;++q)du+=deriv[3*i+q]*u[e*np+q+nq*(j+nq*k)]*2/dx;
   assert(std::abs(r[e*np+i+nq*(j+nq*k)]+.2*du)<1e-11);
  }
  std::fill(u.begin(),u.begin()+N,1.0);U.copyFrom(u.data());
  // Physical inflow/outflow and top backflow: integrated numerical boundary
  // flux balances the change of gas volume, including across MPI partitions.
  if(rank==0)bid[4]=1;
  if(rank==size-1)bid[(ne-1)*6+2]=2;
  Bid.copyFrom(bid.data());std::fill(a.begin(),a.end(),0);A.copyFrom(a.data());eval(.2);
  V bd(NF);B.copyTo(bd.data());double net=0;for(double v:bd)net+=v;
  assert(std::abs(globalSum(net)+.2)<1e-12);
  std::fill(u.begin(),u.begin()+N,-1.0);U.copyFrom(u.data());eval(.2,.1);
  B.copyTo(bd.data());net=0;for(double v:bd)net+=v;assert(std::abs(globalSum(net)+.1)<1e-12);
  if(rank==0)bid[4]=0;if(rank==size-1)bid[(ne-1)*6+2]=0;
  Bid.copyFrom(bid.data());std::fill(u.begin(),u.begin()+N,1.0);U.copyFrom(u.data());
  // Smooth periodic advection: accuracy must improve with refinement.
  for(int i=0;i<N;++i)a[i]=.3+.1*std::sin(2*pi*x[i]);A.copyFrom(a.data());double initial=vol();
  const double end=.2;int steps=std::ceil(end/(.005*dx));double dt=end/steps;
  for(int step=0;step<steps;++step){
   Old.copyFrom(A);eval();rk(N,0.,dt,Old,R,A);limit();
   eval();rk(N,.75,dt,Old,R,A);limit();
   eval();rk(N,1./3,dt,Old,R,A);limit();
  }
  double final=vol();assert(std::abs(final-initial)<1e-12);
  double err=0;for(int i=0;i<N;++i)err+=mass[i]*std::pow(a[i]-(.3+.1*std::sin(2*pi*(x[i]-end))),2);
  errors[level]=std::sqrt(globalSum(err));
  // Limiter preserves the element mean, suppresses nodal overshoots.
  std::fill(a.begin(),a.end(),.2);for(int e=0;e<ne;++e){a[e*np]=-.1;a[e*np+np-1]=1.1;}
  A.copyFrom(a.data());initial=vol();limit();final=vol();assert(std::abs(final-initial)<1e-13);
  for(double v:a)assert(v>=-1e-14 && v<=1+1e-14);
  // Sharp periodic packet: boundedness and global volume with limiter active.
  for(int i=0;i<N;++i)a[i]=(x[i]>.2 && x[i]<.55)?.5:0;
  A.copyFrom(a.data());initial=vol();
  for(int step=0;step<40;++step){
   Old.copyFrom(A);eval();rk(N,0.,dt,Old,R,A);limit();
   eval();rk(N,.75,dt,Old,R,A);limit();
   eval();rk(N,1./3,dt,Old,R,A);limit();
  }
  final=vol();assert(std::abs(final-initial)<1e-12);for(double v:a)assert(v>=-1e-12 && v<=1+1e-12);
 }
 assert(errors[1]<errors[0]/3);
 if(rank==0)std::cout<<"PASS ranks="<<size<<" smooth L2="<<errors[0]<<","<<errors[1]<<"; constant state, periodic mass, sharp packet, limiter mean preservation\n";
 MPI_Finalize();
}
