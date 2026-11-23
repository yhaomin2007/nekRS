#pragma once

namespace bubbleColumn {
// Physical-time products only; native scalar EXT1 carries the selected result.
struct ExplicitHistory {
  deviceMemory<dfloat> samples;
  int step = -1;
  bool initialized() const { return samples.isInitialized(); }
  void apply(const occa::memory &sample, deviceMemory<dfloat> &result,
             int components, bool extrapolate)
  {
    const dlong offset = nrs->fieldOffset, N = nrs->meshV->Nlocal;
    const dlong stride = components * offset;
    const int states = static_cast<int>(nrs->fluid->o_coeffEXT.size());
    if (nrs->tstep < 1) { result.copyFrom(sample, stride); return; }
    if (!initialized()) {
      samples.resize(states * stride);
      // Missing user histories on restart use constant continuation of the
      // first sample. Native startup coefficients already reduce EXT order.
      for (int h = 0; h < states; ++h)
        samples.copyFrom(sample, stride, h * stride, 0);
    } else if (step != nrs->tstep) {
      for (int h = states - 1; h > 0; --h)
        samples.copyFrom(samples, stride, h * stride, (h - 1) * stride);
    }
    samples.copyFrom(sample, stride);
    step = nrs->tstep;
    if (!extrapolate) { result.copyFrom(sample, stride); return; }
    std::vector<dfloat> weights(states);
    nrs->fluid->o_coeffEXT.copyTo(weights.data());
    for (int c = 0; c < components; ++c) {
      auto output = result.slice(c * offset, N);
      platform->linAlg->fill(N, 0.0, output);
      for (int h = 0; h < states; ++h) {
        auto input = samples.slice(h * stride + c * offset, N);
        platform->linAlg->axpby(N, weights[h], input, 1.0, output);
      }
    }
  }
};
} // namespace bubbleColumn
