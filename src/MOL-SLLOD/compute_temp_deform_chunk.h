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
ComputeStyle(temp/deform/chunk,ComputeTempDeformChunk);
// clang-format on
#else

#ifndef LMP_COMPUTE_TEMP_DEFORM_CHUNK_H
#define LMP_COMPUTE_TEMP_DEFORM_CHUNK_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeTempDeformChunk : public Compute {
 public:
  ComputeTempDeformChunk(class LAMMPS *, int, char **);
  ~ComputeTempDeformChunk() override;
  void init() override;
  void setup() override;
  double compute_scalar() override;
  void compute_vector() override;

  void remove_bias(int, double *) override;
  void remove_bias_all() override;
  void restore_bias(int, double *) override;
  void restore_bias_all() override;

  double memory_usage() override;

 private:
  int nmax;
  double adof, cdof, tfactor;

  double **vcm, **vcmall;
  double **&vthermal = array;

  void vcm_thermal_compute();
  void dof_compute();
  void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif
