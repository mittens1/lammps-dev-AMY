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
ComputeStyle(pressure/mol,ComputePressureMol);
// clang-format on
#else

#ifndef LMP_COMPUTE_PRESSURE_MOL_H
#define LMP_COMPUTE_PRESSURE_MOL_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputePressureMol : public Compute {
 public:
  ComputePressureMol(class LAMMPS *, int, char **);
  virtual ~ComputePressureMol() override;
  void init() override;
  double compute_scalar() override;
  void compute_vector() override;
  void reset_extra_compute_fix(const char *) override;
  void pair_setup_callback(int, int) override;
  void pair_tally_callback(int, int, int, int,
      double, double, double, double, double, double) override;

 protected:
  double boltz, nktv2p, inv_volume;
  int dimension;
  double *kspace_virial;
  Compute *temperature;
  char *id_temp;
  double virial[9];    // ordering: xx,yy,zz,xy,xz,yz,yx,zx,zy
  double pair_virial[9];
  int keflag, pairflag;
  int kspaceflag;

  void virial_compute(int, int);

 private:
  bigint did_setup;
};

}    // namespace LAMMPS_NS

#endif
#endif
