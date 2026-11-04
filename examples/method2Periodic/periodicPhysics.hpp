#pragma once
#include <cmath>
#include <stdexcept>
namespace method2Periodic {
// Same Schiller-Naumann Cd*Re closure and per-gas-volume Ki as method2.
inline double dragKi(double slip, double rhoL, double muL, double diameter)
{
  const double Re = rhoL * diameter * std::abs(slip) / muL;
  const double CdRe = Re < 1000 ? 24 * (1 + 0.15 * std::pow(Re, 0.687)) : 0.44 * Re;
  return 0.75 * CdRe * muL / (diameter * diameter);
}
inline double equilibriumSlip(double alpha, double rhoL, double rhoG,
                              double muL, double diameter, double gravityZ)
{
  const double rhoM = (1-alpha)*rhoL + alpha*rhoG;
  const double target = (rhoG-rhoM)*gravityZ;
  if (target == 0) return 0;
  double lo=0, hi=1;
  while (dragKi(hi,rhoL,muL,diameter)*hi < std::abs(target)) {
    hi *= 2;
    if (hi > 1e6) throw std::runtime_error("Cannot bracket terminal slip");
  }
  for (int i=0;i<100;++i) {
    const double mid=(lo+hi)/2;
    if (dragKi(mid,rhoL,muL,diameter)*mid < std::abs(target)) lo=mid; else hi=mid;
  }
  return std::copysign((lo+hi)/2,target);
}
}
