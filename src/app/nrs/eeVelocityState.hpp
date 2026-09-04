#pragma once

#include "deviceMemory.hpp"

#include <algorithm>

// Velocity-only numerical state for the Euler-Euler gas phase.
//
// This deliberately owns no pressure field. The E-E formulation keeps one
// shared pressure while each phase owns an independent momentum predictor and
// Helmholtz velocity solve.  The existing fluidSolver_t remains the liquid
// phase; this structure is the storage contract for the additional gas phase.
//
// Milestone-1 rule: none of these buffers may alias fluidSolver_t evolving
// velocity state (o_U, o_Ue, o_ADV, o_EXT, o_JwF, o_relUrst).
struct eeVelocityState_t
{
  deviceMemory<dfloat> o_U;       // BDF/EXT velocity history, state 0 is current
  deviceMemory<dfloat> o_Ue;      // extrapolated velocity
  deviceMemory<dfloat> o_ADV;     // explicit advection history
  deviceMemory<dfloat> o_EXT;     // other explicit forcing history
  deviceMemory<dfloat> o_JwF;     // assembled momentum predictor forcing
  deviceMemory<dfloat> o_relUrst; // convecting/reference-space velocity history

  dlong fieldOffset = 0;
  dlong fieldOffsetSum = 0;
  dlong cubatureOffset = 0;
  int nHistory = 0;
  int nEXT = 0;
  int nUrstStates = 0;

  void resize(dlong fieldOffset_,
              int dim,
              dlong cubatureOffset_,
              int nBDF,
              int nEXT_,
              int nUrstStates_)
  {
    fieldOffset = fieldOffset_;
    fieldOffsetSum = dim * fieldOffset;
    cubatureOffset = cubatureOffset_;
    nHistory = std::max(1, std::max(nBDF, nEXT_));
    nEXT = std::max(1, nEXT_);
    nUrstStates = std::max(1, nUrstStates_);

    o_U.resize(nHistory * fieldOffsetSum);
    o_Ue.resize(fieldOffsetSum);
    o_ADV.resize(nEXT * fieldOffsetSum);
    o_EXT.resize(nEXT * fieldOffsetSum);
    o_JwF.resize(fieldOffsetSum);
    o_relUrst.resize(nUrstStates * dim * cubatureOffset);
  }

  deviceMemory<dfloat> currentVelocity() const
  {
    return o_U.slice(0, fieldOffsetSum);
  }
};
