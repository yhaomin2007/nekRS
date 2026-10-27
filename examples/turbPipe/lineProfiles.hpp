#pragma once

#include "pointInterpolation.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// Shared by the native turbPipe UDF and the Picard wrapper that includes it.
namespace pipeLines {
static int enabled = 1, interval = 100, pointsPerLine = 201;
static int lastWrittenStep = -1;
static std::unique_ptr<pointInterpolation_t> interpolator;
static std::vector<dfloat> x, y, z, coordinate;
static std::vector<std::string> names, axes;
static deviceMemory<dfloat> packed, sampled;

inline void configure()
{
  platform->par->extract("casedata", "lineprofilesenabled", enabled);
  platform->par->extract("casedata", "lineprofileinterval", interval);
  platform->par->extract("casedata", "lineprofilepoints", pointsPerLine);
  nekrsCheck((enabled != 0 && enabled != 1) || interval < 1 || pointsPerLine < 2,
             platform->comm.mpiComm(), EXIT_FAILURE, "%s",
             "Line profiles require enabled=0/1, interval>=1 and points>=2.\n");
}

inline void initialize()
{
  auto mesh = nrs->meshV;
  const auto comm = platform->comm.mpiComm();
  const dfloat xmin = platform->linAlg->min(mesh->Nlocal, mesh->o_x, comm);
  const dfloat xmax = platform->linAlg->max(mesh->Nlocal, mesh->o_x, comm);
  const dfloat ymin = platform->linAlg->min(mesh->Nlocal, mesh->o_y, comm);
  const dfloat ymax = platform->linAlg->max(mesh->Nlocal, mesh->o_y, comm);
  const dfloat zmin = platform->linAlg->min(mesh->Nlocal, mesh->o_z, comm);
  const dfloat zmax = platform->linAlg->max(mesh->Nlocal, mesh->o_z, comm);
  const dfloat xc = (xmin+xmax)/2, yc = (ymin+ymax)/2;
  if (platform->comm.mpiRank() == 0) {
    // Three x diameters, one orthogonal y diameter, and the axial centerline.
    // Inset endpoints by 1e-6 of the span to avoid geometric boundary ambiguity.
    for (int line = 0; line < 5; ++line) {
      const char *labels[] = {"x_z25", "x_z50", "x_z75", "y_z50", "axial"};
      for (int i = 0; i < pointsPerLine; ++i) {
        const dfloat f = 1e-6 + (1-2e-6)*static_cast<dfloat>(i)/(pointsPerLine-1);
        const dfloat xp = line < 3 ? xmin+f*(xmax-xmin) : xc;
        const dfloat yp = line == 3 ? ymin+f*(ymax-ymin) : yc;
        const dfloat zp = line == 4 ? zmin+f*(zmax-zmin)
                                   : zmin+(line < 3 ? (line+1)*0.25 : 0.5)*(zmax-zmin);
        x.push_back(xp); y.push_back(yp); z.push_back(zp);
        names.push_back(labels[line]);
        axes.push_back(line < 3 ? "x" : line == 3 ? "y" : "z");
        coordinate.push_back(line < 3 ? xp : line == 3 ? yp : zp);
      }
    }
  }
  // All ranks participate, while rank zero owns the requested points (gabls pattern).
  interpolator.reset(new pointInterpolation_t(mesh, comm));
  interpolator->setPoints(x, y, z);
  interpolator->find(pointInterpolation_t::VerbosityLevel::Detailed);
  int missing = 0;
  for (const auto code : interpolator->data().code)
    missing += code == pointInterpolation_t::CODE_NOT_FOUND;
  int globalMissing = 0;
  MPI_Allreduce(&missing, &globalMissing, 1, MPI_INT, MPI_SUM, comm);
  nekrsCheck(globalMissing != 0, comm, EXIT_FAILURE,
             "Line profiles: %d points were not found in the pipe mesh.\n", globalMissing);
}

inline void write(double time, int step)
{
  if (!enabled || step == lastWrittenStep) return;
  if (step != 0 && step % interval != 0 && !nrs->lastStep) return;
  if (!interpolator) initialize();
  auto mesh = nrs->meshV;
  const auto offset = nrs->fluid->fieldOffset;
  const dlong np = x.size();
  packed.resize(5*offset);
  platform->linAlg->fill(packed.size(), 0.0, packed);
  packed.copyFrom(nrs->fluid->o_U, 3*offset);
  packed.copyFrom(nrs->fluid->o_P, mesh->Nlocal, 3*offset);
  packed.copyFrom(nrs->scalar->o_solution("s00"), mesh->Nlocal, 4*offset);
  sampled.resize(std::max<dlong>(1, 5*np));
  interpolator->eval(5, offset, packed, np, sampled);
  if (platform->comm.mpiRank() == 0) {
    std::vector<dfloat> values(5*np);
    sampled.copyTo(values);
    std::ostringstream name;
    name << "lineProfiles_step" << std::setfill('0') << std::setw(6) << step << ".csv";
    std::ofstream out(name.str());
    nekrsCheck(!out, platform->comm.mpiComm(), EXIT_FAILURE,
               "Cannot open line profile file %s.\n", name.str().c_str());
    out << "step,time,line,point,axis,coordinate,x,y,z,ux,uy,uz,p,s00\n";
    out << std::scientific << std::setprecision(17);
    for (dlong i = 0; i < np; ++i) {
      out << step << ',' << time << ',' << names[i] << ',' << i%pointsPerLine
          << ',' << axes[i] << ',' << coordinate[i] << ',' << x[i] << ',' << y[i] << ',' << z[i];
      for (int field = 0; field < 5; ++field) out << ',' << values[field*np+i];
      out << '\n';
    }
    out.close();
    nekrsCheck(!out, platform->comm.mpiComm(), EXIT_FAILURE,
               "Failed writing line profile file %s.\n", name.str().c_str());
    printf("pipeLines step=%d time=%.10e lines=5 pointsPerLine=%d file=%s\n",
           step, time, pointsPerLine, name.str().c_str());
  }
  lastWrittenStep = step;
  // Release gslib resources while MPI is still active.
  if (nrs->lastStep) interpolator.reset();
}
} // namespace pipeLines
