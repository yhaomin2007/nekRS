#pragma once

// Local declaration of NekRS' native advection-subcycling entry point.
// The implementation is linked from NekRS core; keeping this declaration
// next to the UDF avoids depending on the private src/core/advsub include path.
void advectionSubcyclingRK(mesh_t *_mesh,
                           mesh_t *meshV,
                           double time,
                           dfloat *dt,
                           int Nsubsteps,
                           const occa::memory &o_coeffBDF,
                           int nEXT,
                           int nFields,
                           const occa::kernel &kernel,
                           oogs_t *_gsh,
                           dlong _meshOffset,
                           dlong _fieldOffset,
                           dlong cubatureOffset,
                           dlong fieldOffsetSum,
                           const occa::memory &o_divUMesh,
                           const occa::memory &o_Urst,
                           const occa::memory &o_U,
                           occa::memory &o_out);
