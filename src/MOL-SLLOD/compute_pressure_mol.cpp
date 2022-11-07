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

#include "compute_pressure_mol.h"

#include "angle.h"
#include "atom.h"
#include "bond.h"
#include "comm.h"
#include "dihedral.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "fix_property_molecule.h"
#include "force.h"
#include "improper.h"
#include "kspace.h"
#include "modify.h"
#include "pair.h"
#include "pair_hybrid.h"
#include "update.h"

#include <cctype>
#include <cstring>
using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputePressureMol::ComputePressureMol(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg), id_temp(nullptr)
{
  if (narg < 4) error->all(FLERR,"Illegal compute pressure command");
  if (igroup) error->all(FLERR,"Compute pressure must use group all");

  scalar_flag = vector_flag = 1;
  size_vector = 9;
  extscalar = 0;
  extvector = 0;
  pressflag = 1;
  timeflag = 1;

  if (strcmp(arg[3],"NULL") == 0) id_temp = nullptr;
  else {
    id_temp = utils::strdup(arg[3]);

    int icompute = modify->find_compute(id_temp);
    if (icompute < 0)
      error->all(FLERR,"Could not find compute pressure temperature ID");
    if (modify->compute[icompute]->tempflag == 0)
      error->all(FLERR,"Compute pressure temperature ID does not "
                 "compute temperature");
  }

  // process optional args

  if (narg == 4) {
    keflag = 1;
    pairflag = 1;
    kspaceflag = 1;
  } else {
    keflag = 0;
    pairflag = 0;
    kspaceflag = 0;
    int iarg = 4;
    while (iarg < narg) {
      if (strcmp(arg[iarg],"ke") == 0) keflag = 1;
      else if (strcmp(arg[iarg],"pair") == 0) pairflag = 1;
      else if (strcmp(arg[iarg],"kspace") == 0) kspaceflag = 1;
      else if (strcmp(arg[iarg],"virial") == 0) {
        pairflag = 1;
        kspaceflag = 1;
      }
      else error->all(FLERR,"Illegal compute pressure command");
      iarg++;
    }
  }

  // error check

  if (keflag && id_temp == nullptr)
    error->all(FLERR,"Compute pressure requires temperature ID "
               "to include kinetic energy");

  vector = new double[size_vector];
}

/* ---------------------------------------------------------------------- */

ComputePressureMol::~ComputePressureMol()
{
  if (force && force->pair) force->pair->del_tally_callback(this);
  delete [] id_temp;
  delete [] vector;
}

/* ---------------------------------------------------------------------- */

void ComputePressureMol::init()
{
  if (pairflag) {
    if (force->pair == nullptr)
      error->all(FLERR, "Trying to use compute pressure/mol without pair style");
    else
      force->pair->add_tally_callback(this);
  }

  if (comm->me == 0) {
    if (pairflag && force->pair->single_enable == 0 || force->pair->manybody_flag)
      error->warning(FLERR,"Compute pressure/mol used with incompatible pair style");

    if (kspaceflag && force->kspace)
      error->warning(FLERR,"Compute pressure/mol does not yet handle kspace forces");

    // Check for fixes that contribute to the virial (currently not handled)
    for (auto &ifix : modify->get_fix_list())
      if (ifix->thermo_virial)
        error->warning(FLERR,
            "Compute pressure/mol does not account for fix virial "
            "contributions. This warning can be safely ignored for fixes that "
            "only contribute intramolecular forces");

  }
  did_setup = -1;

  boltz = force->boltz;
  nktv2p = force->nktv2p;
  dimension = domain->dimension;

  // set temperature compute, must be done in init()
  // fixes could have changed or compute_modify could have changed it

  if (keflag) {
    int icompute = modify->find_compute(id_temp);
    if (icompute < 0)
      error->all(FLERR,"Could not find compute pressure temperature ID");
    temperature = modify->compute[icompute];
  }


  // Tally callback doesn't work with fdotr virial.
  // Could theoretically use it for molecular virial if it existed.

  if (pairflag) force->pair->no_virial_fdotr_compute = 1;

  // flag Kspace contribution separately, since not summed across procs

  if (kspaceflag && force->kspace) kspace_virial = force->kspace->virial;
  else kspace_virial = nullptr;

}

/* ----------------------------------------------------------------------
   compute total pressure, averaged over Pxx, Pyy, Pzz
------------------------------------------------------------------------- */

double ComputePressureMol::compute_scalar()
{
  invoked_scalar = update->ntimestep;
  if (did_setup != invoked_scalar || update->vflag_global != invoked_scalar)
    error->all(FLERR,"Virial was not tallied on needed timestep");

  // invoke temperature if it hasn't been already

  if (keflag) {
    if (temperature->invoked_scalar != update->ntimestep)
      temperature->compute_scalar();
  }

  if (dimension == 3) {
    inv_volume = 1.0 / (domain->xprd * domain->yprd * domain->zprd);
    virial_compute(3,3);
    if (keflag)
      scalar = (temperature->dof * boltz * temperature->scalar +
                virial[0] + virial[1] + virial[2]) / 3.0 * inv_volume * nktv2p;
    else
      scalar = (virial[0] + virial[1] + virial[2]) / 3.0 * inv_volume * nktv2p;
  } else {
    inv_volume = 1.0 / (domain->xprd * domain->yprd);
    virial_compute(2,2);
    if (keflag)
      scalar = (temperature->dof * boltz * temperature->scalar +
                virial[0] + virial[1]) / 2.0 * inv_volume * nktv2p;
    else
      scalar = (virial[0] + virial[1]) / 2.0 * inv_volume * nktv2p;
  }

  return scalar;
}

/* ----------------------------------------------------------------------
   compute pressure tensor
   assume KE tensor has already been computed
------------------------------------------------------------------------- */

void ComputePressureMol::compute_vector()
{
  invoked_vector = update->ntimestep;
  if (did_setup != invoked_vector || update->vflag_global != invoked_vector)
    error->all(FLERR,"Virial was not tallied on needed timestep");

  if (force->kspace && kspace_virial && force->kspace->scalar_pressure_flag)
    error->all(FLERR,"Must use 'kspace_modify pressure/scalar no' for "
               "tensor components with kspace_style msm");

  int i;
  double ke_tensor[9];
  if (keflag) {
    // invoke temperature if it hasn't been already
    if (temperature->invoked_vector != update->ntimestep)
      temperature->compute_vector();

    // The kinetic energy tensor is symmetric by definition,
    // but we still need the full 9 elements,
    // so copy them and duplicate as necessary
    double *temp_tensor = temperature->vector;
    for(i=0; i < 6; i++)
      ke_tensor[i] = temp_tensor[i];
    ke_tensor[6] = temp_tensor[3];
    ke_tensor[7] = temp_tensor[4];
    ke_tensor[8] = temp_tensor[5];
  }

  if (dimension == 3) {
    inv_volume = 1.0 / (domain->xprd * domain->yprd * domain->zprd);
    virial_compute(9,3);
    if (keflag) {
      for (int i = 0; i < 9; i++)
        vector[i] = (ke_tensor[i] + virial[i]) * inv_volume * nktv2p;
    } else {
      for (int i = 0; i < 9; i++)
        vector[i] = virial[i] * inv_volume * nktv2p;
    }
  } else {
    inv_volume = 1.0 / (domain->xprd * domain->yprd);
    virial_compute(4,2);

    if (keflag) {
      vector[0] = (ke_tensor[0] + virial[0]) * inv_volume * nktv2p;
      vector[1] = (ke_tensor[1] + virial[1]) * inv_volume * nktv2p;
      vector[2] = (ke_tensor[3] + virial[3]) * inv_volume * nktv2p;
      vector[3] = (ke_tensor[6] + virial[6]) * inv_volume * nktv2p;
    } else {
      vector[0] = virial[0] * inv_volume * nktv2p;
      vector[1] = virial[1] * inv_volume * nktv2p;
      vector[2] = virial[3] * inv_volume * nktv2p;
      vector[3] = virial[6] * inv_volume * nktv2p;
    }
    vector[4] = vector[5] = vector[6] = vector[7] = vector[8] = 0.0;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePressureMol::virial_compute(int n, int ndiag)
{
  int i,j;
  double v[9],*vcomponent;

  for (i = 0; i < n; i++) v[i] = 0.0;

  // sum contributions to virial from pair forces

  if (pairflag)
    for (i = 0; i < n; i++) v[i] += pair_virial[i];

  // sum virial across procs

  MPI_Allreduce(v,virial,n,MPI_DOUBLE,MPI_SUM,world);

  // KSpace virial contribution is already summed across procs
  // TODO(SS): calculate this correctly (probably need a kspace_virial_mol)

  if (kspace_virial)
    for (i = 0; i < n; i++) virial[i] += kspace_virial[i];

  // LJ long-range tail correction, only if pair contributions are included
  // TODO(SS): Check that this is correct

  if (force->pair && pairflag && force->pair->tail_flag)
    for (i = 0; i < ndiag; i++) virial[i] += force->pair->ptail * inv_volume;
}

/* ---------------------------------------------------------------------- */

void ComputePressureMol::reset_extra_compute_fix(const char *id_new)
{
  delete [] id_temp;
  id_temp = utils::strdup(id_new);
}

void ComputePressureMol::pair_setup_callback(int eflag, int vflag) {
  if (did_setup == update->ntimestep || !matchstep(update->ntimestep)) return;
  did_setup = update->ntimestep;

  for (int d = 0; d < 9; d++)
    pair_virial[d] = 0.0;

  // Make sure CoM is up to date
  if (atom->property_molecule->comstep != update->ntimestep)
    atom->property_molecule->com_compute();
}

/* ----------------------------------------------------------------------
   tally molecular virials into global accumulator
   have delx, dely, delz and fpair (which gives fx, fy, fz)
   get delcomx, delcomy, delcomz (molecule centre-of-mass separation)
   from atom->property_molecule
------------------------------------------------------------------------- */

void ComputePressureMol::pair_tally_callback(int i, int j, int nlocal,
    int newton_pair, double evdwl, double ecoul, double fpair,
    double delx, double dely, double delz)
{
  // Virial does not need to be tallied if we didn't do setup this step
  if (did_setup != update->ntimestep) return;

  if (atom->property_molecule == nullptr ||
      !atom->property_molecule->com_flag)
    error->all(FLERR, "calculation of the molecular virial requires a fix property/molecule to be defined with the com option");

  double delcom[3], v[9];
  double **com = atom->property_molecule->com;

  tagint mol_i = atom->molecule[i]-1;
  tagint mol_j = atom->molecule[j]-1;

  // com is stored in unwrapped coordinates, so need to map near each other
  // NOTE: this assumes that an atom is always closer to the CoM of the
  //       molecule it belongs to than to any image of that CoM. This may not
  //       hold for molecules that are longer than half the box length.
  // TODO(SS): This can probably be fixed by attaching image flags to CoM coords.
  double *com_i, *com_j, com_tmp_i[3], com_tmp_j[3];
  if (mol_i < 0) com_i = atom->x[i];
  else {
    com_tmp_i[0] = com[mol_i][0];
    com_tmp_i[1] = com[mol_i][1];
    com_tmp_i[2] = com[mol_i][2];
    domain->remap_near(com_tmp_i, atom->x[i]);
    com_i = com_tmp_i;
  }
  if (mol_j < 0) com_j = atom->x[j];
  else {
    com_tmp_j[0] = com[mol_j][0];
    com_tmp_j[1] = com[mol_j][1];
    com_tmp_j[2] = com[mol_j][2];
    domain->remap_near(com_tmp_j, atom->x[j]);
    com_j = com_tmp_j;
  }

  for (int d = 0; d < 3; d++) {
    delcom[d] = com_i[d] - com_j[d];
  }

  v[0] = delcom[0]*delx*fpair;
  v[1] = delcom[1]*dely*fpair;
  v[2] = delcom[2]*delz*fpair;
  v[3] = delcom[0]*dely*fpair;
  v[4] = delcom[0]*delz*fpair;
  v[5] = delcom[1]*delz*fpair;
  v[6] = delcom[1]*delx*fpair;
  v[7] = delcom[2]*delx*fpair;
  v[8] = delcom[2]*dely*fpair;

  if (newton_pair) {
    pair_virial[0] += v[0];
    pair_virial[1] += v[1];
    pair_virial[2] += v[2];
    pair_virial[3] += v[3];
    pair_virial[4] += v[4];
    pair_virial[5] += v[5];
    pair_virial[6] += v[6];
    pair_virial[7] += v[7];
    pair_virial[8] += v[8];
  } else {
    if (i < nlocal) {
      pair_virial[0] += 0.5*v[0];
      pair_virial[1] += 0.5*v[1];
      pair_virial[2] += 0.5*v[2];
      pair_virial[3] += 0.5*v[3];
      pair_virial[4] += 0.5*v[4];
      pair_virial[5] += 0.5*v[5];
      pair_virial[6] += 0.5*v[6];
      pair_virial[7] += 0.5*v[7];
      pair_virial[8] += 0.5*v[8];
    }
    if (j < nlocal) {
      pair_virial[0] += 0.5*v[0];
      pair_virial[1] += 0.5*v[1];
      pair_virial[2] += 0.5*v[2];
      pair_virial[3] += 0.5*v[3];
      pair_virial[4] += 0.5*v[4];
      pair_virial[5] += 0.5*v[5];
      pair_virial[6] += 0.5*v[6];
      pair_virial[7] += 0.5*v[7];
      pair_virial[8] += 0.5*v[8];
    }
  }
}
