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

#ifdef FIX_CLASS
// clang-format off
FixStyle(deform,FixDeform);
// clang-format on
#else

#ifndef LMP_FIX_DEFORM_H
#define LMP_FIX_DEFORM_H

#include "fix.h"

namespace LAMMPS_NS {

class FixDeform : public Fix {
 public:
  int remapflag;     // whether x,v are remapped across PBC
  int dimflag[6];    // which dims are deformed

  FixDeform(class LAMMPS *, int, char **);
  ~FixDeform() override;
  int setmask() override;
  void init() override;
  void pre_exchange() override;
  void post_integrate() override;
  void post_integrate_respa(int, int) override;
  void end_of_step() override;
  void write_restart(FILE *) override;
  void restart(char *buf) override;
  double memory_usage() override;

 protected:
  void update_box();
  double calc_xz_correction(double);

  int triclinic, scaleflag, flipflag;
  int flip, flipxy, flipxz, flipyz;
  double *h_rate, *h_ratelo;
  int end_flag;                  // 1 if box update at end_of_step, 0 if post_integrate
  int varflag;                   // 1 if VARIABLE option is used, 0 if not
  int kspace_flag;               // 1 if KSpace invoked, 0 if not
  std::vector<Fix *> rfix;       // pointers to rigid fixes
  class Irregular *irregular;    // for migrating atoms after box flips

  int nlevels_respa;
  double *step_respa;
  int nloop0_respa;
  int kspace_level_respa;
  bigint nsteps, nsteps_total;
  int allow_flip_change;        // For rRESPA - flip can only change in outer step
  int need_flip_change;
  double dt;

  double TWOPI;

  struct Set {
    int style, substyle;
    double flo, fhi, ftilt;
    double dlo, dhi, dtilt;
    double scale, vel, rate;
    double amplitude, tperiod;
    double lo_initial, hi_initial;
    double lo_start, hi_start, lo_stop, hi_stop, lo_target, hi_target;
    double tilt_initial, tilt_start, tilt_stop, tilt_target, tilt_flip;
    double tilt_min, tilt_max;
    double vol_initial, vol_start;
    int fixed, dynamic1, dynamic2;
    char *hstr, *hratestr;
    int hvar, hratevar;
  };
  Set *set;

  void options(int, char **);

  // For correctness checking of set[i].style
  friend class FixNVTSllod;
  friend class FixNVTSllodMol;
  friend class FixNVTAsllodMol;
};

}    // namespace LAMMPS_NS

#endif
#endif
