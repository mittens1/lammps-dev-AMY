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

#ifdef PAIR_CLASS
// clang-format off
PairStyle(lj/cut/chunk,PairLJCutChunk);
// clang-format on
#else

#ifndef LMP_PAIR_LJ_CUT_CHUNK_H
#define LMP_PAIR_LJ_CUT_CHUNK_H

#include "pair.h"

namespace LAMMPS_NS {

class PairLJCutChunk : public Pair {
 public:
  PairLJCutChunk(class LAMMPS *);
  ~PairLJCutChunk() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_restart_settings(FILE *) override;
  void read_restart_settings(FILE *) override;
  void write_data(FILE *) override;
  void write_data_all(FILE *) override;
  double single(int, int, int, int, double, double, double, double &) override;
  void born_matrix(int, int, int, int, double, double, double, double &, double &) override;
  void *extract(const char *, int &) override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;

  void compute_inner() override;
  void compute_middle() override;
  void compute_outer(int, int) override;

 protected:
  double cut_global;
  double **cut;
  double **epsilon, **sigma;
  double **lj1, **lj2, **lj3, **lj4, **offset;
  double *cut_respa;
  
  int nchunk, maxchunk, nmax;
  char* idchunk;
  char* idcom;
  class ComputeChunkAtom *cchunk;
  class ComputeCOMChunk *ccom;
  int* chunk_ID; // Deliberately not calling this ichunk to avoid confusing it with the compute's attribute
  virtual void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif
