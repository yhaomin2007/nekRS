"""Check conservative phase exchange, pressure response and core solve dispatch."""
from pathlib import Path
import re
import subprocess
import tempfile

case = Path(__file__).resolve().parents[1]
root = case.parents[1]
kernels = (case / 'bubbleColumnPhaseMomentum.okl').read_text()
kernels = kernels.replace('@kernel ', '').replace('@ restrict ', '')
kernels = re.sub(r'; @tile\(p_blockSize, @outer, @inner\)', '', kernels)
source = '#include <cmath>\n#include <cassert>\n#include <initializer_list>\nusing dfloat = double; using dlong = long;\n'
header = (root / 'src/solver/fluid/fluidSolver.hpp').read_text()
start = header.index('  void solve(double time, int stage) override')
end = header.index('  };', start) + len('  };')
source += '''
struct Base { virtual void solve(double, int) = 0; };
struct Solver : Base {
 bool pressureFromVelocity = false;
 int events = 0;
 void applyDirichlet(double, bool onlyPressure) {assert(onlyPressure); events = 1;}
 void solvePressure(double, int) { events = 10 * events + 2; }
 void solveVelocity(double, int) { events = 10 * events + 3; }
''' + header[start:end] + '\n};\n'
source += kernels + r'''
void near(double actual, double expected, double tol=1e-11) {
  assert(std::abs(actual - expected) < tol * (1.0 + std::abs(expected)));
}
int main() {
  Solver solver;
  solver.solve(0.0, 1); assert(solver.events == 23);
  solver.events = 0; solver.pressureFromVelocity = true;
  solver.solve(0.0, 1); assert(solver.events == 12);

  constexpr double rl = 998.2, rg = 1.2, dt = 0.0001, g = -9.81;
  const double ug[] = {0.4, -0.2, 0.3}, ul[] = {0.01, 0.12, 0.05};
  const double aLext[] = {0.3, 1.7, -3.2}, gradP[] = {10., -10., 9810.};
  const double conv[] = {0.1, -0.5, 3.0}, divStress[] = {1.0, -2.0, 4.0};
  const double gradUl[] = {1.0, 2.0, 3.0, -1.0, 2.0, 0.0, 0.3, -0.6, 5.0};
  for (double a : {0.0, 0.016667, 0.6, 0.999999}) {
    const double beta = 1 - a, rm = a * rg + beta * rl;
    double material[3], stress[9];
    phaseLiquidTransport(1, 1, 1e-3, &a, ul, gradUl, material, stress);
    near(material[0], ul[0]+2*ul[1]+3*ul[2]);
    near(stress[0]+stress[4]+stress[8], 0.0);
    near(stress[1], beta*1e-3);
    for (double cVM : {0.0, 0.5 * rl}) {
      const double eff = rg + cVM, k = 15.0, dragRate = k / eff;
      double gasAccel[3], liquidRhs[3], duG[3], duL[3], uGNew[3], uLNew[3], umOld[3], umNew[3];
      for (int i=0; i<3; ++i) {
        const double gravity = i==2 ? g : 0.0;
        gasAccel[i] = (rg*gravity+k*(ul[i]-ug[i])+cVM*aLext[i])/eff;
      }
      phaseLiquidRhs(1, 1, rl, eff, cVM, 1e-6, 0, 0, g, &a, ug, ul, &dragRate,
                     gasAccel, aLext, conv, divStress, liquidRhs);
      phasePressureResponse(1, 1, rl, eff, cVM, 1e-6, dt, &a, gradP, duG, duL);
      for (int i=0; i<3; ++i) {
        const double gravity = i==2 ? g : 0.0;
        // Summing phase forces cancels equal/opposite drag and VM.
        near(a*rg*gasAccel[i]+beta*rl*(liquidRhs[i]+conv[i]), rm*gravity+divStress[i]);
        near(duG[i], -dt*gradP[i]/eff);
        uGNew[i] = ug[i]+duG[i]; uLNew[i] = ul[i]+duL[i];
      }
      phaseMixtureVelocity(1, 1, rl, rg, &a, ug, ul, umOld);
      phaseMixtureVelocity(1, 1, rl, rg, &a, uGNew, uLNew, umNew);
      for (int i=0; i<3; ++i) {
        near(umOld[i], (a*rg*ug[i]+beta*rl*ul[i])/rm);
        // The native Poisson operator uses mobility 1/rho_m.
        near(umNew[i]-umOld[i], -dt*gradP[i]/rm);
      }
    }
  }
}
'''
with tempfile.TemporaryDirectory() as directory:
    cpp = Path(directory) / 'phase.cpp'
    cpp.write_text(source)
    exe = cpp.with_suffix('')
    subprocess.run(['c++', '-std=c++17', str(cpp), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
print('Phase momentum: convection/stress, conservative drag/VM, mixture pressure mobility, and both core dispatches passed.')
