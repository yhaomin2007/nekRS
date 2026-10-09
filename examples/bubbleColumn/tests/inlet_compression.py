"""Compile the actual inlet mask and exercise explicit/frozen-implicit terms."""
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
  const int n=5, offset=7;
  double z[offset]={0., .009, .01, .010001, .1, 123., 456.};
  double alpha[offset]={.02,.03,.04,.05,.06,0.,0.};
  double grad[9*offset]={};
  for(int i=0;i<n;++i) grad[i]=(i%2 ? -2. : 3.);
  for(double thickness: {0., .01, .2}) {
    double explicitSource[offset]={}, rate[offset]={};
    alphaCompression(n,offset,alpha,grad,explicitSource);
    for(int i=0;i<n;++i) rate[i]=grad[i];
    explicitSource[n]=rate[n]=123.;
    maskAlphaCompressionInlet(n,thickness,z,explicitSource);
    maskAlphaCompressionInlet(n,thickness,z,rate);
    for(int i=0;i<n;++i) {
      double expected=(thickness>0 && z[i]<=thickness) ? 0. : grad[i];
      assert(rate[i]==expected);
      assert(std::abs(explicitSource[i]+alpha[i]*expected)<1e-14);
      // Reconstruction uses the frozen masked rate with the updated alpha.
      double updatedAlpha=2*alpha[i];
      assert(std::abs(-rate[i]*updatedAlpha-2*explicitSource[i])<1e-14);
    }
    assert(explicitSource[n]==123. && rate[n]==123.);
  }
}
"""
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory)
    (path / "check.cpp").write_text(cpp)
    subprocess.run(["g++", "-std=c++17", str(path / "check.cpp"),
                    "-o", str(path / "check")], check=True)
    subprocess.run([str(path / "check")], check=True)

host = (root / "bubbleColumnTerms.hpp").read_text()
evaluation = host.split("inline void evaluatePointwiseTerms()")[1].split(
    "inline void prepareGasAdvection()")[0]
assert evaluation.index("alphaCompressionKernel(mesh->Nlocal") < evaluation.index(
    "maskAlphaCompressionInletKernel(mesh->Nlocal")
assert "p.alphaCompressionMode == 1 && p.alphaCompressionInletThickness > 0.0" in evaluation
preparation = host.split("inline void prepareGasAdvection()")[1].split(
    "inline void postProcessGasFlux")[0]
assert preparation.index("opSEM::strongDivergence") < preparation.index(
    "maskAlphaCompressionInletKernel(nrs->meshV->Nlocal")
assert "o_alphaCompressionRate, o_alphaSource" in evaluation
print("Inlet compression mask: off, inclusive threshold, exterior, signed rates, "
      "padding and explicit/implicit reconstruction consistency passed.")
