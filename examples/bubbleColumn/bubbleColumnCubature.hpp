#pragma once

#include <fstream>
#include <sstream>
#include <iomanip>
#include <regex>
#include <vector>
#include <cstdlib>
#include <iterator>

namespace bubbleColumn {

// Native scalar cubature advection uses c_I instead of its runtime matrix
// argument. Supplement only combinations absent from the installed table.
inline std::string missingCubatureInterpolationHeader(int N, int cubN)
{
  const int nq = N + 1, cq = cubN + 1;
  const char *kernelDirectory = std::getenv("NEKRS_KERNEL_DIR");
  nekrsCheck(!kernelDirectory, platform->comm.mpiComm(), EXIT_FAILURE,
             "%s", "NEKRS_KERNEL_DIR is required for cubature table inspection.\n");
  const std::string path = std::string(kernelDirectory)
      + "/core/mesh/constantInterpolationMatrices.h";
  std::ifstream input(path);
  nekrsCheck(!input.is_open(), platform->comm.mpiComm(), EXIT_FAILURE,
             "Cannot inspect installed interpolation constants: %s\n", path.c_str());
  const std::string table((std::istreambuf_iterator<char>(input)),
                          std::istreambuf_iterator<char>());
  const std::regex pair("p_Nq\\s*==\\s*([0-9]+)\\s*&&\\s*p_cubNq\\s*==\\s*([0-9]+)");
  for (std::sregex_iterator it(table.begin(), table.end(), pair), end; it != end; ++it) {
    if (std::stoi((*it)[1].str()) == nq && std::stoi((*it)[2].str()) == cq)
      return {};
  }

  std::vector<dfloat> gll(nq), gl(cq), glWeights(cq), interpolation(nq * cq);
  JacobiGLL(N, gll.data());
  JacobiGQ(0.0, 0.0, cubN, gl.data(), glWeights.data());
  InterpolationMatrix1D(N, nq, gll.data(), cq, gl.data(), interpolation.data());
  std::ostringstream header;
  header << std::scientific << std::setprecision(17)
         << "#if p_Nq == " << nq << " && p_cubNq == " << cq << "\n"
         << "const dfloat c_I[" << cq << "][" << nq << "] = {\n";
  for (int i = 0; i < cq; ++i) {
    header << '{';
    for (int j = 0; j < nq; ++j) {
      if (j) header << ',';
      header << interpolation[i * nq + j];
    }
    header << '}';
    if (i + 1 < cq) header << ',';
    header << '\n';
  }
  header << "};\n#endif\n";
  return header.str();
}

} // namespace bubbleColumn
