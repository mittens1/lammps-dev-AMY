// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Pieter in 't Veld (SNL)
------------------------------------------------------------------------- */

#include "fix_nvt_sllod.h"

#include "atom.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "fix_deform.h"
#include "group.h"
#include "math_extra.h"
#include "modify.h"

#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

enum{NOBIAS,BIAS};
/* ---------------------------------------------------------------------- */

FixNVTSllod::FixNVTSllod(LAMMPS *lmp, int narg, char **arg) :
  FixNH(lmp, narg, arg)
{
  if (!tstat_flag)
    error->all(FLERR,"Temperature control must be used with fix nvt/sllod");
  if (pstat_flag)
    error->all(FLERR,"Pressure control can not be used with fix nvt/sllod");

  // default values
  p_sllod = 1;
  peculiar = 0;

  if (mtchain_default_flag) {
    mtchain = 1;

    // Fix allocation of chain thermostats so that size_vector is correct
    int ich;
    delete[] eta;
    delete[] eta_dot;
    delete[] eta_dotdot;
    delete[] eta_mass;
    eta = new double[mtchain];

    // add one extra dummy thermostat, set to zero

    eta_dot = new double[mtchain+1];
    eta_dot[mtchain] = 0.0;
    eta_dotdot = new double[mtchain];
    for (ich = 0; ich < mtchain; ich++) {
      eta[ich] = eta_dot[ich] = eta_dotdot[ich] = 0.0;
    }
    eta_mass = new double[mtchain];

    // Default mtchain in fix_nh is 3.
    size_vector -= 2*2*(3-mtchain);
  }

  for (int i = 0; i < narg; ++i) {
    if (strcmp(arg[i], "p_sllod")==0) {
      if (++i >= narg) error->all(FLERR, "Illegal fix nvt/sllod command");
      p_sllod = utils::logical(FLERR, arg[i], false, lmp);
    } else if (strcmp(arg[i], "peculiar")==0) {
      if (++i >= narg) error->all(FLERR, "Illegal fix nvt/sllod command");
      peculiar = utils::logical(FLERR, arg[i], false, lmp);
    }
  }

  // create a new compute temp style
  // id = fix-ID + temp

  id_temp = utils::strdup(std::string(id) + "_temp");
  if (peculiar) modify->add_compute(fmt::format("{} {} temp",
                                  id_temp,group->names[igroup]));
  else modify->add_compute(fmt::format("{} {} temp/deform",
                                  id_temp,group->names[igroup]));

  tcomputeflag = 1;
}

/* ---------------------------------------------------------------------- */

void FixNVTSllod::init()
{
  FixNH::init();

  if (!peculiar && !temperature->tempbias)
    error->all(FLERR,"Temperature for fix nvt/sllod does not have a bias");

  nondeformbias = 0;
  if (strcmp(temperature->style,"temp/deform") != 0) nondeformbias = 1;

  // check fix deform remap settings

  int i;
  for (i = 0; i < modify->nfix; i++)
    if (strncmp(modify->fix[i]->style,"deform",6) == 0) {
      if (!peculiar && (dynamic_cast<FixDeform *>( modify->fix[i]))->remapflag != Domain::V_REMAP)
        error->all(FLERR,"Using fix nvt/sllod with inconsistent fix deform "
                   "remap option");
      if (peculiar && (dynamic_cast<FixDeform *>( modify->fix[i]))->remapflag != Domain::NO_REMAP)
        error->all(FLERR,"Using fix nvt/sllod with inconsistent fix deform "
                   "remap option");
      break;
    }
  if (i == modify->nfix)
    error->all(FLERR,"Using fix nvt/sllod with no fix deform defined");

  // Apply initial kick if we can
  if (!peculiar && !nondeformbias) {
    temperature->remove_bias_all();
    temperature->restore_bias_all();
    temperature->restore_bias_all();
  }
}

/* ----------------------------------------------------------------------
   perform half-step scaling of velocities
-----------------------------------------------------------------------*/

void FixNVTSllod::nh_v_temp()
{
  // remove and restore bias = streaming velocity = Hrate*lamda + Hratelo
  // thermostat thermal velocity only
  // vdelu = SLLOD correction = Hrate*Hinv*vthermal
  // for non temp/deform BIAS:
  //   calculate temperature since some computes require temp
  //   computed on current nlocal atoms to remove bias

  if (which == BIAS) temperature->compute_scalar();

  double **v = atom->v;
  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  double h_two[6],vdelu[3];
  MathExtra::multiply_shape_shape(domain->h_rate,domain->h_inv,h_two);

  if (peculiar) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        if (which == BIAS) temperature->remove_bias(i,v[i]);
        v[i][0] = v[i][0]*factor_eta;
        v[i][1] = v[i][1]*factor_eta;
        v[i][2] = v[i][2]*factor_eta;
        if (which == BIAS) temperature->restore_bias(i,v[i]);
      }
    }
  } else {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        if (!p_sllod) temperature->remove_bias(i,v[i]);
        vdelu[0] = h_two[0]*v[i][0] + h_two[5]*v[i][1] + h_two[4]*v[i][2];
        vdelu[1] = h_two[1]*v[i][1] + h_two[3]*v[i][2];
        vdelu[2] = h_two[2]*v[i][2];
        if (p_sllod) temperature->remove_bias(i,v[i]);
        v[i][0] = v[i][0]*factor_eta - dthalf*vdelu[0];
        v[i][1] = v[i][1]*factor_eta - dthalf*vdelu[1];
        v[i][2] = v[i][2]*factor_eta - dthalf*vdelu[2];
        temperature->restore_bias(i,v[i]);
      }
    }
  }
}

void FixNVTSllod::nve_v()
{
  double dtfm, dtf2, inv_mass;
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  double h_two[6];
  MathExtra::multiply_shape_shape(domain->h_rate,domain->h_inv,h_two);

  double fac_vu[3];
  if (peculiar) {
    dtf2 = 0.5*dtf;
    fac_vu[0] = exp(-h_two[0]*dtf2);
    fac_vu[1] = exp(-h_two[1]*dtf2);
    fac_vu[2] = exp(-h_two[2]*dtf2);
  }
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      if (rmass) inv_mass = 1 / rmass[i];
      else inv_mass = 1 / mass[type[i]];

      if (peculiar) {
        // First half step with SLLOD force. Quarter step x since no dependants
        if (which == BIAS) temperature->remove_bias(i,v[i]);
        v[i][0] *= fac_vu[0];
        v[i][1] *= fac_vu[1];
        v[i][2] *= fac_vu[2];
        if (p_sllod) {
          v[i][2] += dtf2*(f[i][0]*inv_mass - h_two[2]*h_two[2]*x[i][2]);
          v[i][1] += dtf2*(f[i][1]*inv_mass - h_two[3]*v[i][2] -
                           h_two[1]*h_two[1]*x[i][1]);
          v[i][0] += dtf*(f[i][0]*inv_mass - h_two[5]*v[i][1] -
                          h_two[4]*v[i][2]) - h_two[0]*h_two[0]*x[i][0];
        } else {
          v[i][2] += dtf2*f[i][2]*inv_mass;
          v[i][1] += dtf2*(f[i][1]*inv_mass - h_two[3]*v[i][2]);
          v[i][0] += dtf*(f[i][0]*inv_mass - h_two[5]*v[i][1] - h_two[4]*v[i][2]);
        }
      } else {
        // Half step velocity
        v[i][0] += dtfm*f[i][0];
        v[i][1] += dtfm*f[i][1];
        v[i][2] += dtfm*f[i][2];
      }

      // 2nd half step SLLOD force
      if (peculiar) {
        if (p_sllod) {
          v[i][1] += dtf2*(f[i][1]*inv_mass - h_two[3]*v[i][2] -
                           h_two[1]*h_two[1]*x[i][1]);
          v[i][2] += dtf2*(f[i][0]*inv_mass - h_two[2]*h_two[2]*x[i][2]);
        } else {
          v[i][1] += dtf2*(f[i][1]*inv_mass - h_two[3]*v[i][2]);
          v[i][2] += dtf2*f[i][2]*inv_mass;
        }
        v[i][0] *= fac_vu[0];
        v[i][1] *= fac_vu[1];
        v[i][2] *= fac_vu[2];
        if (which == BIAS) temperature->restore_bias(i,v[i]);
      }
    }
  }
}


/* ----------------------------------------------------------------------
   perform full-step update of positions
-----------------------------------------------------------------------*/

void FixNVTSllod::nve_x()
{
  double **x = atom->x;
  double **v = atom->v;
  int *mask = atom->mask;
  double vstream[3], h_two[6], xfac[3];
  double dtv2 = dtv*0.5;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  // x update by full step only for atoms in group
  
  if (peculiar) {
    MathExtra::multiply_shape_shape(domain->h_rate,domain->h_inv,h_two);
    // xfac[0] = 1 / (1 - dthalf*h_two[0]);
    // xfac[1] = 1 / (1 - dthalf*h_two[1]);
    // xfac[2] = 1 / (1 - dthalf*h_two[2]);
    xfac[0] = exp(h_two[0]*dtv2);
    xfac[1] = exp(h_two[1]*dtv2);
    xfac[2] = exp(h_two[2]*dtv2);
  }

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      if (peculiar) {
        x[i][0] *= xfac[0];
        x[i][1] *= xfac[1];
        x[i][2] *= xfac[2];
        x[i][2] += dtv2 * v[i][2];
        x[i][1] += dtv2 * (v[i][1] + h_two[3]*x[i][2]);
        x[i][0] += dtv * (v[i][0] + h_two[5]*x[i][1] + h_two[4]*x[i][2]);
        x[i][1] += dtv2 * (v[i][1] + h_two[3]*x[i][2]);
        x[i][2] += dtv2 * v[i][2];
        x[i][0] *= xfac[0];
        x[i][1] *= xfac[1];
        x[i][2] *= xfac[2];

        // First half step - solve for streaming velocity at t+dt/2
        // x[i][2] += dthalf * v[i][2];
        // x[i][2] *= xfac[2];
        // vstream[2] = h_two[2]*x[i][2];

        // x[i][1] += dthalf * (v[i][1] + h_two[3]*x[i][2]);
        // x[i][1] *= xfac[1];
        // vstream[1] = h_two[1]*x[i][1] + h_two[3]*x[i][2];

        // x[i][0] += dthalf * (v[i][0] + h_two[5]*x[i][1] + h_two[4]*x[i][2]);
        // x[i][0] *= xfac[0];
        // vstream[0] = h_two[0]*x[i][0] + h_two[5]*x[i][1] + h_two[4]*x[i][2];

        // // 2nd half step - use streaming velocity from t+dt/2
        // x[i][0] += dthalf * (v[i][0] + vstream[0]);
        // x[i][1] += dthalf * (v[i][1] + vstream[1]);
        // x[i][2] += dthalf * (v[i][2] + vstream[2]);
      } else {
        x[i][0] += dtv * v[i][0];
        x[i][1] += dtv * v[i][1];
        x[i][2] += dtv * v[i][2];
      }
    }
  }
}

