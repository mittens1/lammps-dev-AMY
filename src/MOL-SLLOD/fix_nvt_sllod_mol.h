/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   www.cs.sandia.gov/~sjplimp/lammps.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(nvt/sllod/chunk,FixNVTSllodMol);
// clang-format on
#else

#ifndef LMP_FIX_NVT_SLLOD_MOL_H
#define LMP_FIX_NVT_SLLOD_MOL_H

#include "fix_nh.h"

namespace LAMMPS_NS {

class FixNVTSllodMol : public FixNH {
 public:
  FixNVTSllodMol(class LAMMPS *, int, char **);
  ~FixNVTSllodMol();

  void init() override;
  void setup(int) override;

 private:
  int nondeformbias;
  // int nchunk, maxchunk;
  int kickflag;           // Whether to apply the streaming profile on init
  double **vcm, **vcmall;
  // double *massproc, *masstotal;

  void nh_v_temp() override;
  void vcm_thermal_compute();

  // class ComputeChunkAtom *cchunk;
  // class ComputeVCMChunk *cvcm;
};

}    // namespace LAMMPS_NS

#endif
#endif
