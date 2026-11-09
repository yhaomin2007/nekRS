#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

namespace method3Projection {
struct ResidualSample { int iteration; double norm; };
struct Result { int iterations=0; double initial=0, residual=0, rhsNorm=0, target=0; bool converged=false;
  const char* reason="maximumIterations";
  std::vector<ResidualSample> residualHistory;
};
// Restarted flexible GCR. The caller supplies vectors and operations on its backend.
// Recompute the true residual at every restart and before accepting convergence.
// On success, the last apply call evaluated the accepted x.
template<class Vector,class Apply,class Precondition,class Dot,class Axpby,class Scale,
         class Direction,class Image,class Orthogonalize>
Result solveWithWorkspace(const Vector& b,Vector& x,Vector& r,Vector& ax,
                          Apply apply,Precondition precondition,Dot dot,Axpby axpby,Scale scale,
                          Direction direction,Image image,Orthogonalize orthogonalize,
                          int maximum,int restart,double tolerance,double absoluteTolerance)
{
  Result out;
  const auto norm=[&](const Vector& v){
    const double d=dot(v,v);
    return std::isfinite(d)&&d>=0?std::sqrt(d):std::numeric_limits<double>::infinity();
  };
  apply(x,ax);
  axpby(1,b,0,r);
  axpby(-1,ax,1,r);
  out.initial=norm(r);out.residual=out.initial;
  out.residualHistory.push_back({0,out.initial});
  const double rhsNorm=norm(b);
  out.rhsNorm=rhsNorm;
  if(!std::isfinite(out.initial)||!std::isfinite(rhsNorm)) {out.reason="nonfiniteResidual";return out;}
  const double target=std::max(absoluteTolerance,tolerance*rhsNorm);out.target=target;
  while(out.iterations<maximum) {
    if(std::isfinite(out.residual)&&out.residual<=target) {out.converged=true;break;}
    if(!std::isfinite(out.residual)) {out.reason="nonfiniteResidual";break;}
    const int cycleStart=out.iterations;
    for(int k=0;k<restart && out.iterations<maximum;++k) {
      auto&& v=direction(k);
      auto&& w=image(k);
      precondition(r,v);apply(v,w);
      // Two passes limit loss of orthogonality for the near-null pressure modes.
      if(k) for(int pass=0;pass<2;++pass) orthogonalize(k,v,w);
      const double wn=norm(w);
      if(!(wn>std::numeric_limits<double>::min()) || !std::isfinite(wn)) {out.reason="directionBreakdown";break;}
      scale(1/wn,w);
      scale(1/wn,v);
      const double c=dot(r,w);
      axpby(c,v,1,x);
      axpby(-c,w,1,r);
      ++out.iterations;
      if(norm(r)<=target) break;
    }
    apply(x,ax);
    axpby(1,b,0,r);
    axpby(-1,ax,1,r);
    out.residual=norm(r);
    out.residualHistory.push_back({out.iterations,out.residual});
    if(out.iterations==cycleStart) break;
  }
  out.converged=std::isfinite(out.residual)&&out.residual<=target;
  if(out.converged) out.reason="converged";
  return out;
}

// Host backend used by the small algebraic regression cases.
template<class Apply,class Precondition,class Dot>
Result solve(const std::vector<double>& b,std::vector<double>& x,
             Apply apply,Precondition precondition,Dot dot,
             int maximum,int restart,double tolerance,double absoluteTolerance=0)
{
  using Vector=std::vector<double>;
  Vector r(b.size()),ax(b.size()),coefficients(restart);
  std::vector<Vector> z(restart,Vector(b.size())),q(restart,Vector(b.size()));
  const auto axpby=[](double a,const Vector& u,double b,Vector& v) {
    for(size_t i=0;i<v.size();++i) v[i]=a*u[i]+b*v[i];
  };
  const auto scale=[](double a,Vector& v) {for(auto& value:v) value*=a;};
  const auto orthogonalize=[&](int k,Vector& v,Vector& w) {
    for(int j=0;j<k;++j) coefficients[j]=dot(w,q[j]);
    for(int j=0;j<k;++j) {
      axpby(-coefficients[j],q[j],1,w);
      axpby(-coefficients[j],z[j],1,v);
    }
  };
  return solveWithWorkspace(b,x,r,ax,apply,precondition,dot,axpby,scale,
      [&](int k)->Vector& {return z[k];},[&](int k)->Vector& {return q[k];},
      orthogonalize,maximum,restart,tolerance,absoluteTolerance);
}
}
