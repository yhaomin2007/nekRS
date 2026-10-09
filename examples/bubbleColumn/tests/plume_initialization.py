"""Exercise the restored plume kernel with direct UG and stationary liquid."""
from pathlib import Path
import re
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / "bubbleColumnEquations.okl").read_text()
kernel = "void initializePlume" + source.split("@kernel void initializePlume")[1].split("@kernel")[0]
kernel = kernel.replace("@ restrict ", "")
kernel = re.sub(r"; @tile\(p_blockSize, @outer, @inner\)", "", kernel)
cpp = """#include <cmath>
#include <cassert>
#include <initializer_list>
using dfloat=double; using dlong=long;
""" + kernel + """
int main() {
  const int n=5,o=7;
  double coord[o]={0.,.04,.05,.06,1.,0.,0.};
  for(double speed: {0.,.3,-.3}) {
    double alpha[o],x[o],y[o],z[o],u[3*o];
    for(int i=0;i<o;++i) alpha[i]=x[i]=y[i]=z[i]=123.;
    for(int i=0;i<3*o;++i) u[i]=123.;
    initializePlume(n,o,998.2,1.2,.016667,speed,.05,.01,coord,alpha,x,y,z,u);
    for(int i=0;i<n;++i) {
      double f=.5*(1-std::tanh((coord[i]-.05)/.01));
      double a=.016667*f,rho=(1-a)*998.2+a*1.2;
      assert(std::abs(alpha[i]-a)<1e-15 && x[i]==0 && y[i]==0);
      assert(std::abs(z[i]-speed*f)<1e-15);
      assert(u[i]==0 && u[o+i]==0);
      assert(std::abs(rho*u[2*o+i]-1.2*a*z[i])<1e-14);
      if(i) assert(alpha[i]<=alpha[i-1]);
    }
    assert(alpha[2]==.016667/2 && z[2]==speed/2);
    assert(alpha[0]>.999*.016667 && alpha[4]==0);
    assert(alpha[n]==123. && z[n]==123. && u[2*o+n]==123.);
  }
}
"""
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory)
    (path / "check.cpp").write_text(cpp)
    subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    str(path / "check.cpp"), "-o", str(path / "check")], check=True)
    subprocess.run([str(path / "check")], check=True)
print("Restored tanh plume: alpha/UG profile, signed speed, stationary liquid, "
      "density-averaged mixture velocity and padding passed.")
