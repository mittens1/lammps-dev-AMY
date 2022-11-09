// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Emily Kahl, Stephen Sanderson, Shern Tee (Uni of QLD)
------------------------------------------------------------------------- */

#include "compute_temp_mol.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "fix_property_mol.h"
#include "force.h"
#include "group.h"
#include "memory.h"
#include "modify.h"
#include "update.h"

#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeTempMol::ComputeTempMol(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg), vcm(nullptr), vcmall(nullptr), molprop(nullptr),
  id_molprop(nullptr)
{
  if (narg != 4) error->all(FLERR,"Illegal compute temp/mol command");

  scalar_flag = vector_flag = 1;
  size_vector = 6;
  extscalar = 0;
  extvector = 1;  // TODO(SS): if thermo_modify norm yes is set, then the vector will be divided by the number of atoms, which is incorrect.
  tempflag = 1;
  tempbias = 0;
  maxbias = 0;
  array_flag = 0;

  adof = domain->dimension;
  cdof = 0.0;

  // vector data
  vector = new double[size_vector];

  // per-atom allocation
  nmax = 0;

  id_molprop = utils::strdup(arg[3]);
}

/* ---------------------------------------------------------------------- */

ComputeTempMol::~ComputeTempMol()
{
  delete [] vector;

  // property_molecule may have already been destroyed
  molprop = dynamic_cast<FixPropertyMol*>(modify->get_fix_by_id(id_molprop));
  if (molprop != nullptr) {
    molprop->destroy_permolecule(&vcmall);
    molprop->destroy_permolecule(&vcm);
  }
  delete [] id_molprop;
}

/* ---------------------------------------------------------------------- */

void ComputeTempMol::init()
{
  // Get id of molprop
  molprop = dynamic_cast<FixPropertyMol*>(modify->get_fix_by_id(id_molprop));
  if (molprop == nullptr) // TODO(SS): Check that this fails when given an incorrect fix type
    error->all(FLERR, "Compute temp/mol could not find a fix property/mol with id {}", id_molprop);
  if (!molprop->mass_flag)
    error->all(FLERR, "Compute temp/mol requires fix property/mol with the mass or com flag");
  if (igroup != molprop->igroup)
    error->all(FLERR, "Fix property/mol must be defined for the same group as compute temp/mol");

  molprop->register_permolecule("temp/mol:vcmall", &vcmall, Atom::DOUBLE, 3);
  molprop->register_permolecule("temp/mol:vcm", &vcm, Atom::DOUBLE, 3);
}

void ComputeTempMol::setup()
{
  dynamic = 0;
  if (dynamic_user || group->dynamic[igroup]) dynamic = 1;
  dof_compute();
}


/* ---------------------------------------------------------------------- */

double ComputeTempMol::compute_scalar()
{
  int i;
  invoked_scalar = update->ntimestep;

  tagint molmax = molprop->molmax;
  double *molmass = molprop->mass;

  // calculate global temperature

  double **v = atom->v;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  tagint m;

  // Calculate the thermal velocity (total minus streaming) of all molecules
  double ke_singles[6];
  vcm_compute(ke_singles);

  // Tally up the molecule COM velocities to get the kinetic temperature
  double t = ke_singles[0]+ke_singles[1]+ke_singles[2];
  for (m = 0; m < molmax; m++) {
    t += (vcmall[m][0]*vcmall[m][0] + vcmall[m][1]*vcmall[m][1] + vcmall[m][2]*vcmall[m][2]) *
          molmass[m];
  }

  // final temperature
  if (dynamic)
    dof_compute();
  if (dof < 0.0)
    error->all(FLERR,"Temperature compute degrees of freedom < 0");
  scalar = t*tfactor;
  return scalar;
}

/* ---------------------------------------------------------------------- */

void ComputeTempMol::compute_vector()
{
  int i;

  invoked_vector = update->ntimestep;

  tagint molmax = molprop->molmax;
  double *molmass = molprop->mass;

  double **v = atom->v;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  tagint m;

  double massone,t[6];
  for (i = 0; i < 6; i++) t[i] = 0.0;

  double ke_singles[6];
  vcm_compute(ke_singles);

  // Tally up the molecule COM velocities to get the kinetic temperature
  // No need for MPI reductions, since every processor knows the molecule VCMs
  for (m = 0; m < molmax; m++) {
      t[0] += molmass[m] * vcmall[m][0] * vcmall[m][0];
      t[1] += molmass[m] * vcmall[m][1] * vcmall[m][1];
      t[2] += molmass[m] * vcmall[m][2] * vcmall[m][2];
      t[3] += molmass[m] * vcmall[m][0] * vcmall[m][1];
      t[4] += molmass[m] * vcmall[m][0] * vcmall[m][2];
      t[5] += molmass[m] * vcmall[m][1] * vcmall[m][2];
  }
  // final KE. Include contribution from single atoms if there are any
  for (i = 0; i < 6; i++) vector[i] = (t[i]+ke_singles[i])*force->mvv2e;
}


/* ----------------------------------------------------------------------
   Degrees of freedom for molecular temperature
------------------------------------------------------------------------- */

void ComputeTempMol::dof_compute()
{
  // TODO(SS): fix_dof will be incorrect for rigid molecules, since we only care
  //           about CoM momentum. Ignoring it for now, but maybe look
  //           into calculating the number of intermolecular constraints which
  //           should be counted.
  adjust_dof_fix();
  if (fix_dof != 0)
    error->warning(FLERR,"Ignoring dof constraints due to fixes in compute "
        "temp/mol. These must be accounted for manually since intramolecular "
        "constraints should be ignored.");

  // Count atoms in the group that aren't part of a molecule
  int *mask = atom->mask;
  bigint nsingle_local = 0, nsingle;
  for (int i = 0; i < atom->nlocal; ++i)
    if (mask[i] & groupbit && atom->molecule[i] == 0)
      ++nsingle_local;
  MPI_Allreduce(&nsingle_local,&nsingle,1,MPI_LMP_BIGINT,MPI_SUM,world);

  // Make sure molecule count is up to date
  if (molprop->dynamic_group && molprop->count_step != update->ntimestep)
    molprop->count_molecules();
  // Calculate dof from number of molecules with at least 1 atom in the group
  dof = domain->dimension * (molprop->nmolecule + nsingle);

  dof -= extra_dof; // + fix_dof;
  if (dof > 0)
    tfactor = force->mvv2e / (dof * force->boltz);
  else
    tfactor = 0.0;
}

/* ----------------------------------------------------------------------
   Calculate centre-of-mass velocity for each molecule.
   Can be safely called mid-step since doesn't set invoked flag.
  --------------------------------------------------------------------*/

void ComputeTempMol::vcm_compute(double *ke_singles)
{
  tagint m;
  double massone;
  double unwrap[3];

  // molid = 1 to molmax for included atoms, 0 for excluded atoms
  tagint *molecule = atom->molecule;
  tagint molmax = molprop->molmax;

  // Update molecular masses if required
  // Also grows vcm and vcmall if needed
  if (molprop->dynamic_group && molprop->mass_step != update->ntimestep)
    molprop->mass_compute();
  double *molmass = molprop->mass;

  // Reallocation handled by fix property/molecule.
  // Make sure size is up to date
  size_array_rows = molmax;

  // zero local per-molecule values
  for (m = 0; m < molmax; m++){
    vcm[m][0] = vcm[m][1] = vcm[m][2] = 0.0;
  }

  // compute VCM for each molecule

  double **x = atom->x;
  double **v = atom->v;
  int *mask = atom->mask;
  int *type = atom->type;

  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int nlocal = atom->nlocal;

  double ke_local[6];
  for (int i = 0; i < 6; ++i)
    ke_local[i] = 0.0;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      if (rmass) massone = rmass[i];
      else massone = mass[type[i]];
      m = molecule[i]-1;
      if (m < 0) {
        if (ke_singles != nullptr) {
          ke_local[0] += v[i][0]*v[i][0]*massone;
          ke_local[1] += v[i][1]*v[i][1]*massone;
          ke_local[2] += v[i][2]*v[i][2]*massone;
          ke_local[3] += v[i][0]*v[i][1]*massone;
          ke_local[4] += v[i][0]*v[i][2]*massone;
          ke_local[5] += v[i][1]*v[i][2]*massone;
        }
        continue;
      }
      vcm[m][0] += v[i][0] * massone;
      vcm[m][1] += v[i][1] * massone;
      vcm[m][2] += v[i][2] * massone;
    }

  double ke_total = 0;
  if (molmax > 0) MPI_Allreduce(&vcm[0][0],&vcmall[0][0],3*molmax,MPI_DOUBLE,MPI_SUM,world);
  if (ke_singles != nullptr) MPI_Allreduce(ke_local,ke_singles,6,MPI_DOUBLE,MPI_SUM,world);
  for (m = 0; m < molmax; m++) {
    if (molmass[m] > 0.0) {
      vcmall[m][0] /= molmass[m];
      vcmall[m][1] /= molmass[m];
      vcmall[m][2] /= molmass[m];
    } else {
      vcmall[m][0] = vcmall[m][1] = vcmall[m][2] = 0.0;
    }
  }
}

/* ----------------------------------------------------------------------
   memory usage of local data
------------------------------------------------------------------------- */

double ComputeTempMol::memory_usage()
{
  double bytes = 0;
  // vcm and vcmall not allocated if property_molecule is nullptr
  if (molprop != nullptr)
    bytes += (bigint) molprop->molmax * 6 * sizeof(double);

  return bytes;
}
