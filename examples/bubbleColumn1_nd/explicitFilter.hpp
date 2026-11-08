#pragma once

#include "udf.hpp"
#include <cmath>
#include <map>
#include <string>
#include <vector>

// Standalone v26 port of the user's explicitFilterApply. No dependency on
// bubbleColumn parameters, scratch buffers, or core lowPassFilterSetup.
namespace userExplicitFilter {

struct FilterContainer {
  mesh_t *mesh = nullptr;
  int degree = -1;
  int nModes = 0;
  dfloat strength = 0.0;
  deviceMemory<dfloat> matrix;
};

static std::map<std::string, FilterContainer> filterMap;
static occa::kernel filterTensor3;
static int kernelDegree = -1;

inline void registerKernels(deviceKernelProperties &properties,
                            const std::string &source = "explicitFilter.okl")
{
  platform->options.getArgs("POLYNOMIAL DEGREE", kernelDegree);
  const std::string request = "userExplicitFilter::tensor";
  if (platform->options.compareArgs("REGISTER ONLY", "TRUE")) {
    platform->kernelRequests.add(request, source, properties);
  } else {
    filterTensor3 = platform->kernelRequests.load(request, "filterTensor3");
  }
}

// Row-major F=V*diag(sigma)*inverse(V). The supplied kernel reads F as
// fMT[input*nq+output], so setup transposes this matrix before upload.
inline std::vector<dfloat> filterMatrix(const std::vector<dfloat> &nodes,
                                       int nModes, dfloat strength)
{
  const int nq = nodes.size();
  nekrsCheck(nq < 2 || nModes < 1 || nModes > nq - 2
                 || !std::isfinite(strength) || strength < 0.0 || strength > 1.0,
             platform->comm.mpiComm(), EXIT_FAILURE,
             "%s", "Explicit filter requires N>=2, 1<=modes<=N-1, and finite strength in [0,1].\n");
  std::vector<dfloat> V(nq * nq), L(nq), sigma(nq, 1.0);
  for (int j = 0; j < nq; ++j) {
    L[0] = 1.0;
    L[1] = nodes[j];
    for (int k = 2; k < nq; ++k)
      L[k] = ((2.0*k-1.0)*nodes[j]*L[k-1]-(k-1.0)*L[k-2])/k;
    V[j*nq] = L[0];
    V[j*nq+1] = L[1];
    for (int k = 2; k < nq; ++k) V[j*nq+k] = L[k]-L[k-2];
  }
  const int k0 = nq - nModes;
  for (int k = k0; k < nq; ++k) {
    const dfloat ratio = static_cast<dfloat>(k-k0+1)/nModes;
    sigma[k] = 1.0-strength*ratio*ratio;
  }
  const auto inverse = platform->linAlg->matrixInverse(nq, V);
  std::vector<dfloat> F(nq * nq, 0.0);
  for (int i = 0; i < nq; ++i)
    for (int j = 0; j < nq; ++j)
      for (int k = 0; k < nq; ++k)
        F[i*nq+j] += V[i*nq+k]*sigma[k]*inverse[k*nq+j];
  return F;
}

inline void explicitFilterApply(const std::string &tag, mesh_t *mesh,
                                int nModes, dfloat strength, occa::memory field)
{
  nekrsCheck(!mesh || mesh->N != kernelDegree,
             platform->comm.mpiComm(), EXIT_FAILURE,
             "%s", "Explicit filter mesh degree must match its registered kernel degree.\n");
  nekrsCheck(nModes < 1 || nModes > mesh->N - 1
                 || !std::isfinite(strength) || strength < 0.0 || strength > 1.0,
             platform->comm.mpiComm(), EXIT_FAILURE,
             "%s", "Explicit filter requires 1<=modes<=N-1 and finite strength in [0,1].\n");
  nekrsCheck(field.byte_size() < mesh->Nlocal * sizeof(dfloat),
             platform->comm.mpiComm(), EXIT_FAILURE,
             "%s", "Explicit filter field is smaller than mesh Nlocal.\n");
  if (strength == 0.0) return;
  auto &entry = filterMap[tag];
  if (entry.mesh != mesh || entry.degree != mesh->N
      || entry.nModes != nModes || entry.strength != strength) {
    const int nq = mesh->N + 1;
    std::vector<dfloat> nodes(nq);
    for (int j = 0; j < nq; ++j) nodes[j] = mesh->r[j];
    const auto F = filterMatrix(nodes, nModes, strength);
    std::vector<dfloat> transpose(nq * nq);
    for (int i = 0; i < nq; ++i)
      for (int j = 0; j < nq; ++j) transpose[i*nq+j] = F[j*nq+i];
    entry.matrix.resize(nq*nq);
    entry.matrix.copyFrom(transpose);
    entry.mesh = mesh;
    entry.degree = mesh->N;
    entry.nModes = nModes;
    entry.strength = strength;
    if (platform->comm.mpiRank() == 0) {
      printf("explicit filter (%s): modes=%d strength=%g (Legendre bubble basis)\n",
             tag.c_str(), nModes, strength);
      printf("filt trn (%s):", tag.c_str());
      for (int k = 0; k < nq; ++k) {
        const dfloat ratio = k < nq - nModes ? 0.0
            : static_cast<dfloat>(k-(nq-nModes)+1)/nModes;
        printf(" %7.4f", 1.0-strength*ratio*ratio);
      }
      printf("\n");
    }
  }
  filterTensor3(mesh->Nelements, static_cast<occa::memory>(entry.matrix), field);
}

} // namespace userExplicitFilter
