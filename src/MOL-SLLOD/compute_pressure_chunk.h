/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(pressure/chunk,ComputePressureChunk);
// clang-format on
#else

#ifndef LMP_COMPUTE_PRESSURE_CHUNK_H
#define LMP_COMPUTE_PRESSURE_CHUNK_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputePressureChunk : public Compute {
 public:
  ComputePressureChunk(class LAMMPS *, int, char **);
  virtual ~ComputePressureChunk() override;
  void init() override;
  double compute_scalar() override;
  void compute_vector() override;
  void reset_extra_compute_fix(const char *) override;

 protected:
  double boltz, nktv2p, inv_volume;
  int nvirial, dimension;
  double **vptr;
  double *kspace_virial;
  Compute *temperature;
  char *id_temp;
  double virial[9];    // ordering: xx,yy,zz,xy,xz,yz,yx,zx,zy
  int pairhybridflag;
  class Pair *pairhybrid;
  int keflag, pairflag, bondflag, angleflag, dihedralflag, improperflag;
  int fixflag, kspaceflag;

  void virial_compute(int, int);

 private:
  char *pstyle;
  int nsub;
};

}    // namespace LAMMPS_NS

#endif
#endif
