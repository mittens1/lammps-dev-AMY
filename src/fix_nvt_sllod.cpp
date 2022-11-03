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

// from FixNH:
enum{NOBIAS,BIAS};

// from FixDeform:
enum{NONE=0,FINAL,DELTA,SCALE,VEL,ERATE,TRATE,VOLUME,WIGGLE,VARIABLE};
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
      auto def = dynamic_cast<FixDeform *>(modify->fix[i]);
      if (!peculiar && def->remapflag != Domain::V_REMAP)
        error->all(FLERR,"Using fix nvt/sllod with inconsistent fix deform "
                   "remap option");
      if (peculiar && def->remapflag != Domain::NO_REMAP)
        error->all(FLERR,"Using fix nvt/sllod with inconsistent fix deform "
                   "remap option");
      bool elongation = false;
      for (int j = 0; j < 3; ++j) {
        if (def->set[i].style) {
          elongation = true;
          if (def->set[j].style != TRATE)
            error->all(FLERR,"fix nvt/sllod requires the trate style for "
                "x/y/z deformation");
        }
      }
      for (int j = 3; j < 6; ++j) {
        if (def->set[j].style && def->set[j].style != ERATE) {
          if (elongation)
            error->all(FLERR,"fix nvt/sllod requires the erate style for "
                "xy/xz/yz deformation under mixed shear/extensional flow");
          else
            error->warning(FLERR,
                "Using non-constant shear rate with fix nvt/sllod");
        }
      }
      if (def->set[5].style && def->set[5].rate != 0.0 &&
          (def->set[3].style || domain->yz != 0.0) &&
          (def->set[4].style != ERATE || def->set[5].style != ERATE
           || (def->set[3].style && def->set[3].style != ERATE))
          )
        error->warning(FLERR,"Shearing xy with a yz tilt is only handled "
            "correctly if fix deform uses the erate style for xy, xz and yz");
      if (def->end_flag)
        error->warning(FLERR,"SLLOD equations of motion require box deformation"
            " to occur with position updates to be strictly correct. Set the N"
            " parameter of fix deform to 0 to enable this.");
      break;
    }
  if (i == modify->nfix)
    error->all(FLERR,"Using fix nvt/sllod with no fix deform defined");

  if (!peculiar)
    error->warning(FLERR,"fix nvt/sllod will produce incorrect energy "
        "dissipation if the peculiar flag is not set");

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

  double grad_u[6];
  double* h_rate = domain->h_rate;
  double* h_inv = domain->h_inv;
  MathExtra::multiply_shape_shape(h_rate, h_inv, grad_u);

  if (peculiar) {
    double dt4 = 0.5*dthalf;
    double vfac[3];
    vfac[0] = exp(-grad_u[0]*dt4);
    vfac[1] = exp(-grad_u[1]*dt4);
    vfac[2] = exp(-grad_u[2]*dt4);
    if (which == BIAS) temperature->remove_bias_all();
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        if (p_sllod) {
          // Add dt4*p-SLLOD force separately so that pure shear is identical
          // between SLLOD and p-SLLOD. Using dt4*(SLLOD_force + p-SLLOD_force)
          // causes numerical divergence.
          v[i][0] *= vfac[0];
          v[i][1] *= vfac[1];
          v[i][2] *= vfac[2];
          v[i][2] -= dt4*grad_u[2]*grad_u[2]*x[i][2];
          v[i][1] -= dt4*grad_u[3]*v[i][2] + dt4*grad_u[1]*grad_u[1]*x[i][1];
          v[i][0] -= dt4*(grad_u[5]*v[i][1] + grad_u[4]*v[i][2])
                     + dt4*grad_u[0]*grad_u[0]*x[i][0];
          v[i][0] *= factor_eta;
          v[i][1] *= factor_eta;
          v[i][2] *= factor_eta;
          v[i][0] -= dt4*(grad_u[5]*v[i][1] + grad_u[4]*v[i][2])
                     + dt4*grad_u[0]*grad_u[0]*x[i][0];
          v[i][1] -= dt4*grad_u[3]*v[i][2] + dt4*grad_u[1]*grad_u[1]*x[i][1];
          v[i][2] -= dt4*grad_u[2]*grad_u[2]*x[i][2];
          v[i][0] *= vfac[0];
          v[i][1] *= vfac[1];
          v[i][2] *= vfac[2];
        } else {
          v[i][0] *= vfac[0];
          v[i][1] *= vfac[1];
          v[i][2] *= vfac[2];
          v[i][1] -= dt4*grad_u[3]*v[i][2];
          v[i][0] -= dt4*(grad_u[5]*v[i][1] + grad_u[4]*v[i][2]);
          v[i][0] *= factor_eta;
          v[i][1] *= factor_eta;
          v[i][2] *= factor_eta;
          v[i][0] -= dt4*(grad_u[5]*v[i][1] + grad_u[4]*v[i][2]);
          v[i][1] -= dt4*grad_u[3]*v[i][2];
          v[i][0] *= vfac[0];
          v[i][1] *= vfac[1];
          v[i][2] *= vfac[2];
        }
      }
    }
    if (which == BIAS) temperature->restore_bias_all();
  } else {
    double vdelu[3];
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        if (!p_sllod) temperature->remove_bias(i,v[i]);
        vdelu[0] = grad_u[0]*v[i][0] + grad_u[5]*v[i][1] + grad_u[4]*v[i][2];
        vdelu[1] = grad_u[1]*v[i][1] + grad_u[3]*v[i][2];
        vdelu[2] = grad_u[2]*v[i][2];
        if (p_sllod) temperature->remove_bias(i,v[i]);
        v[i][0] = v[i][0]*factor_eta - dthalf*vdelu[0];
        v[i][1] = v[i][1]*factor_eta - dthalf*vdelu[1];
        v[i][2] = v[i][2]*factor_eta - dthalf*vdelu[2];
        temperature->restore_bias(i,v[i]);
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
  double grad_u[6], xfac[3];
  double dtv2 = dtv*0.5;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  // x update by full step only for atoms in group
  
  if (peculiar) {
    double* h_rate = domain->h_rate;
    double* h_inv = domain->h_inv;
    MathExtra::multiply_shape_shape(h_rate, h_inv, grad_u);
    xfac[0] = exp(grad_u[0]*dtv2);
    xfac[1] = exp(grad_u[1]*dtv2);
    xfac[2] = exp(grad_u[2]*dtv2);
  }

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      if (peculiar) {
        x[i][0] *= xfac[0];
        x[i][1] *= xfac[1];
        x[i][2] *= xfac[2];
        x[i][1] += dtv2 * grad_u[3]*x[i][2];
        x[i][0] += dtv2 * (grad_u[5]*x[i][1] + grad_u[4]*x[i][2]);
        x[i][0] += dtv * v[i][0];
        x[i][1] += dtv * v[i][1];
        x[i][2] += dtv * v[i][2];
        x[i][0] += dtv2 * (grad_u[5]*x[i][1] + grad_u[4]*x[i][2]);
        x[i][1] += dtv2 * grad_u[3]*x[i][2];
        x[i][0] *= xfac[0];
        x[i][1] *= xfac[1];
        x[i][2] *= xfac[2];
      } else {
        x[i][0] += dtv * v[i][0];
        x[i][1] += dtv * v[i][1];
        x[i][2] += dtv * v[i][2];
      }
    }
  }
}

