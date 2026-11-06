#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

namespace method3Projection {
struct Result { int iterations=0; double initial=0, residual=0; bool converged=false; };
// Restarted flexible GCR. Dot supplies a global, duplicate-aware inner product.
// Recompute the true residual at every restart and before accepting convergence.
template<class Apply,class Precondition,class Dot>
Result solve(const std::vector<double>& b,std::vector<double>& x,
             Apply apply,Precondition precondition,Dot dot,
             int maximum,int restart,double tolerance)
{
  Result out;
  std::vector<double> r=b, ax(b.size());
  const auto norm=[&](const std::vector<double>& v){
    const double d=dot(v,v);
    return std::isfinite(d)&&d>=0?std::sqrt(d):std::numeric_limits<double>::infinity();
  };
  apply(x,ax);
  for(size_t i=0;i<b.size();++i) r[i]-=ax[i];
  out.initial=norm(r);out.residual=out.initial;
  const double rhsNorm=norm(b);
  if(!std::isfinite(out.initial)||!std::isfinite(rhsNorm)) return out;
  const double target=tolerance*std::max(out.initial,rhsNorm);
  while(out.iterations<maximum) {
    if(std::isfinite(out.residual)&&out.residual<=target) {out.converged=true;break;}
    if(!std::isfinite(out.residual)) break;
    std::vector<std::vector<double>> z,q;
    for(int k=0;k<restart && out.iterations<maximum;++k) {
      std::vector<double> v(b.size()),w(b.size());
      precondition(r,v);apply(v,w);
      // Two passes limit loss of orthogonality for the near-null pressure modes.
      for(int pass=0;pass<2;++pass) for(size_t j=0;j<q.size();++j) {
        const double c=dot(w,q[j]);
        for(size_t i=0;i<b.size();++i) {w[i]-=c*q[j][i];v[i]-=c*z[j][i];}
      }
      const double wn=norm(w);
      if(!(wn>std::numeric_limits<double>::min()) || !std::isfinite(wn)) break;
      for(size_t i=0;i<b.size();++i) {w[i]/=wn;v[i]/=wn;}
      const double c=dot(r,w);
      for(size_t i=0;i<b.size();++i) {x[i]+=c*v[i];r[i]-=c*w[i];}
      z.push_back(std::move(v));q.push_back(std::move(w));++out.iterations;
      if(norm(r)<=target) break;
    }
    apply(x,ax);
    for(size_t i=0;i<b.size();++i) r[i]=b[i]-ax[i];
    out.residual=norm(r);
    if(z.empty()) break;
  }
  out.converged=std::isfinite(out.residual)&&out.residual<=target;
  return out;
}
}
