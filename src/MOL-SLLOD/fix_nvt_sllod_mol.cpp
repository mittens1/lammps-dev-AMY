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
   Contributing author: Emily Kahl (Uni of QLD)
------------------------------------------------------------------------- */

#include "fix_nvt_sllod_mol.h"

#include "atom.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "fix_deform.h"
#include "fix_property_molecule.h"
#include "group.h"
#include "math_extra.h"
#include "modify.h"
#include "memory.h"

#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixNVTSllodMol::FixNVTSllodMol(LAMMPS *lmp, int narg, char **arg) :
  FixNH(lmp, narg, arg)
{
  if (!tstat_flag)
    error->all(FLERR,"Temperature control must be used with fix nvt/sllod/mol");
  if (pstat_flag)
    error->all(FLERR,"Pressure control can not be used with fix nvt/sllod/mol");

  // default values

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


  kickflag = 0;

  int iarg = 3;

  while (iarg < narg) {
    if (strcmp(arg[iarg++], "kick")==0) {
      if (iarg >= narg) error->all(FLERR,"Invalid fix nvt/sllod/mol command");
      if (strcmp(arg[iarg], "yes")==0) {
        kickflag = 1;
      } else if (strcmp(arg[iarg], "no")==0) {
        kickflag = 0;
      } else error->all(FLERR,"Invalid fix nvt/sllod/mol command");
    }
    ++iarg;
  }

  // create a new compute temp style
  // id = fix-ID + temp

  id_temp = utils::strdup(std::string(id) + "_temp");
  modify->add_compute(fmt::format("{} {} temp/deform/mol",
                                  id_temp,group->names[igroup]));
  tcomputeflag = 1;
  vcm = nullptr;
  vcmall = nullptr;
}

/* ---------------------------------------------------------------------- */
FixNVTSllodMol::~FixNVTSllodMol() {
  // property_molecule may have already been destroyed
  if (atom->property_molecule != nullptr)
    atom->property_molecule->destroy_permolecule(&vcmall);   // Also destroys vcm
}

/* ---------------------------------------------------------------------- */

void FixNVTSllodMol::init() {
  FixNH::init();

  if (!temperature->tempbias)
    error->all(FLERR,"Temperature for fix nvt/sllod/mol does not have a bias");

  nondeformbias = 0;
  if (strcmp(temperature->style,"temp/deform/mol") != 0) nondeformbias = 1;

  // check fix deform remap settings

  int i;
  for (i = 0; i < modify->nfix; i++)
    if (strncmp(modify->fix[i]->style,"deform",6) == 0) {
      if ((dynamic_cast<FixDeform *>( modify->fix[i]))->remapflag != Domain::V_REMAP)
        error->all(FLERR,"Using fix nvt/sllod/mol with inconsistent fix deform "
                   "remap option");
      break;
    }
  if (i == modify->nfix)
    error->all(FLERR,"Using fix nvt/sllod/mol with no fix deform defined");

  if (atom->property_molecule == nullptr)
    error->all(FLERR, "fix nvt/sllod/mol requires a fix property/molecule to be defined with the com option");

  // TODO: maybe just register with fix property/molecule that we need COM to avoid this?
  if (!atom->property_molecule->com_flag)
    error->all(FLERR, "fix nvt/sllod/mol requires a fix property/molecule to be defined with the com option");

  // Set up handling of vcm memory
  atom->property_molecule->register_permolecule("nvt/sllod/mol:vcmall", &vcmall, Atom::DOUBLE, 3);
  atom->property_molecule->register_permolecule("nvt/sllod/mol:vcm", &vcm, Atom::DOUBLE, 3);
  
}

void FixNVTSllodMol::setup(int vflag) {
  FixNH::setup(vflag);

  // Check for fix property/molecule
  if (atom->property_molecule == nullptr)
    error->all(FLERR,"fix nvt/sllod/mol requires a fix property/molecule to be defined");

  // Apply kick if necessary
  if (kickflag) {
    // Call remove_bias first to calculate biases
    // temperature->compute_scalar(); // compute_scalar already called in FixNH::setup()
    temperature->remove_bias_all();

    // Restore twice to apply streaming profile
    temperature->restore_bias_all();
    temperature->restore_bias_all();

    // Don't kick again if multi-step run
    kickflag = 0;
  }

}

/* ----------------------------------------------------------------------
   perform half-step scaling of velocities
-----------------------------------------------------------------------*/

void FixNVTSllodMol::nh_v_temp() {
  // remove and restore bias = streaming velocity = Hrate*lamda + Hratelo
  // thermostat thermal velocity only
  // vdelu = SLLOD correction = Hrate*Hinv*vthermal
  // for non temp/deform BIAS:
  //   calculate temperature since some computes require temp
  //   computed on current nlocal atoms to remove bias

  if (nondeformbias) temperature->compute_scalar();

  // Remove bias from all atoms at once to avoid re-calculating the COM positions
  temperature->remove_bias_all();

  // Use molecular centre-of-mass velocity when calculating SLLOD correction
  vcm_thermal_compute();
  tagint *molecule = atom->molecule;
  int m;


  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  double h_two[6],vdelu[3],*vcom;
  MathExtra::multiply_shape_shape(domain->h_rate,domain->h_inv,h_two);

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      m = molecule[i]-1;
      if (m < 0) vcom = v[i];  // CoM velocity of single atom is just v[i]
      else vcom = vcmall[m];
      // NOTE: This uses the thermal velocity of the molecule centre-of-mass in all cases
      vdelu[0] = h_two[0]*vcom[0] + h_two[5]*vcom[1] + h_two[4]*vcom[2];
      vdelu[1] = h_two[1]*vcom[1] + h_two[3]*vcom[2];
      vdelu[2] = h_two[2]*vcom[2];
      v[i][0] = v[i][0] - vcom[0] + vcom[0]*factor_eta - dthalf*vdelu[0];
      v[i][1] = v[i][1] - vcom[1] + vcom[1]*factor_eta - dthalf*vdelu[1];
      v[i][2] = v[i][2] - vcom[2] + vcom[2]*factor_eta - dthalf*vdelu[2];
    }
  }
  temperature->restore_bias_all();
}

/* calculate COM thermal velocity. 
 * Pre: atom velocities should have streaming bias removed
 *      COM positions should already be computed when removing biases
 */
void FixNVTSllodMol::vcm_thermal_compute() {
  int m;
  double massone;

  tagint *molecule = atom->molecule;
  tagint nmolecule = atom->property_molecule->nmolecule;


  // zero local per-molecule values

  for (int i = 0; i < nmolecule; i++){
    vcm[i][0] = vcm[i][1] = vcm[i][2] = 0.0;
  }

  // compute VCM for each molecule

  double **v = atom->v;
  int *mask = atom->mask;
  int *type = atom->type;

  imageint *image = atom->image;
  int xbox, ybox, zbox;
  double v_adjust[3];

  double *mass = atom->mass;
  double *rmass = atom->rmass;
  double *molmass = atom->property_molecule->mass;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      m = molecule[i]-1;
      if (m < 0) continue;
      if (rmass) {
        massone = rmass[i];
      } else {
        massone = mass[type[i]];
      }
      // Adjust the velocity to reflect the thermal velocity 
      vcm[m][0] += v[i][0] * massone;
      vcm[m][1] += v[i][1] * massone;
      vcm[m][2] += v[i][2] * massone;
    }

  MPI_Allreduce(&vcm[0][0],&vcmall[0][0],3*nmolecule,MPI_DOUBLE,MPI_SUM,world);

  for (int m = 0; m < nmolecule; m++) {
    if (molmass[m] > 0.0) {
      vcmall[m][0] /= molmass[m];
      vcmall[m][1] /= molmass[m];
      vcmall[m][2] /= molmass[m];
    } else {
      vcmall[m][0] = vcmall[m][1] = vcmall[m][2] = 0.0;
    }
  }
}
