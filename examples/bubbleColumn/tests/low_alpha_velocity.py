"""Exercise actual reconstruction/postprocess kernels with the cutoff switch."""
from pathlib import Path
import re
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / "bubbleColumnEquations.okl").read_text()
source = source.replace("@kernel ", "").replace("@ restrict ", "")
source = re.sub(r"; @tile\(p_blockSize, @outer, @inner\)", "", source)
cpp = """#include <cmath>
#include <cassert>
#include <initializer_list>
using dfloat=double; using dlong=long;
""" + source + """
int main() {
  const int n=6,o=8;
  double a[o]={-.01,0.,1e-6,5e-5,5.5e-4,.02};
  for(int enabled: {0,1}) for(int smooth: {0,1}) {
    double x[o],y[o],z[o],ug[3*o];
    for(int i=0;i<o;++i) {x[i]=.3;y[i]=-.4;z[i]=.5;}
    for(int i=0;i<3*o;++i) ug[i]=123.;
    reconstructGasVelocity(n,o,enabled,1e-6,a,x,y,z,ug);
    for(int i=0;i<n;++i) {
      double active=(!enabled || a[i]>1e-6)?1.:0.;
      assert(ug[i]==.3*active && ug[o+i]==-.4*active && ug[2*o+i]==.5*active);
    }
    assert(ug[n]==123. && ug[o+n]==123. && ug[2*o+n]==123.);
    postProcessGasFlux(n,enabled,smooth,0.,1.,1e-6,1e-4,1e-3,a,x,y,z);
    for(int i=0;i<n;++i) {
      double weight=(!enabled || a[i]>1e-6)?1.:0.;
      if(enabled && smooth && a[i]<=1e-4) weight=0.;
      else if(enabled && smooth && a[i]<1e-3) {
        double s=(a[i]-1e-4)/(1e-3-1e-4);weight*=s*s*(3-2*s);
      }
      assert(std::abs(x[i]-.3*weight)<1e-14);
      assert(std::abs(y[i]+.4*weight)<1e-14);
      assert(std::abs(z[i]-.5*weight)<1e-14);
    }
    assert(x[n]==.3 && y[n]==-.4 && z[n]==.5);
  }
  // The low-alpha switch must not bypass independent magnitude clipping.
  double x[o]={3.},y[o]={4.},z[o]={0.};
  postProcessGasFlux(1,0,1.,1.,1.,1e-6,1e-4,1e-3,a,x,y,z);
  assert(std::abs(x[0]-.6)<1e-14 && std::abs(y[0]-.8)<1e-14);
}
"""
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory)
    (path / "check.cpp").write_text(cpp)
    subprocess.run(["g++", "-std=c++17", str(path / "check.cpp"),
                    "-o", str(path / "check")], check=True)
    subprocess.run([str(path / "check")], check=True)
print("Low-alpha UG switch: reconstruction, hard/smooth postprocess, negative/zero "
      "alpha, all components, padding and independent magnitude clipping passed.")
