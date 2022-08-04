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
FixStyle(nvt/sllod/chunk,FixNVTSllodChunk);
// clang-format on
#else

#ifndef LMP_FIX_NVT_SLLOD_CHUNK_H
#define LMP_FIX_NVT_SLLOD_CHUNK_H

#include "fix_nh.h"

namespace LAMMPS_NS {

class FixNVTSllodChunk : public FixNH {
 public:
  FixNVTSllodChunk(class LAMMPS *, int, char **);
  ~FixNVTSllodChunk();

  void init() override;

 private:
  int nondeformbias;
  int nchunk, maxchunk;
  double **vcm, **vcmall;
  double *massproc, *masstotal;

  void nh_v_temp() override;
  void vcm_compute();

  class ComputeChunkAtom *cchunk;
  class ComputeVCMChunk *cvcm;
};

}    // namespace LAMMPS_NS

#endif
#endif
