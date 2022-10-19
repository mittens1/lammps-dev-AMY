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

#include "compute_temp_deform_mol.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "fix_deform.h"
#include "fix_property_molecule.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "update.h"

#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeTempDeformMol::ComputeTempDeformMol(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg), vcm(nullptr), vcmall(nullptr)
{
  if (narg != 3) error->all(FLERR,"Illegal compute temp/deform/mol command");

  scalar_flag = vector_flag = 1;
  size_vector = 9;
  extscalar = 0;
  extvector = 1;
  tempflag = 1;
  tempbias = 1;
  maxbias = 0;
  vbiasall = nullptr;
  vthermal = nullptr;


  adof = domain->dimension;
  cdof = 0.0;

  // vector data
  vector = new double[size_vector];


  array_flag = 1;
  size_array_cols = 3;
  size_array_rows = 0;
  size_array_rows_variable = 1;
  extarray = 0;

  // per-atom allocation
  nmax = 0;
}

/* ---------------------------------------------------------------------- */

ComputeTempDeformMol::~ComputeTempDeformMol()
{
  delete [] vector;
  memory->destroy(vbiasall);
  memory->destroy(vthermal);

  // property_molecule may have already been destroyed
  if (atom->property_molecule != nullptr) {
    atom->property_molecule->destroy_permolecule(vcm);
    atom->property_molecule->destroy_permolecule(vcmall);
  }
}

/* ---------------------------------------------------------------------- */

void ComputeTempDeformMol::init()
{
  if (atom->property_molecule == nullptr || 
      !atom->property_molecule->com_flag)
    error->all(FLERR, "compute temp/deform/mol requires a fix property/molecule to be defined with the com option");

  atom->property_molecule->register_permolecule("temp/deform/mol:vcmall", &vcmall, Atom::DOUBLE, 3);
  atom->property_molecule->register_permolecule("temp/deform/mol:vcm", &vcm, Atom::DOUBLE, 3);

  auto fixes = modify->get_fix_by_style("^deform");
  if (fixes.size() > 0) {
    if ((dynamic_cast<FixDeform *>(fixes[0]))->remapflag == Domain::X_REMAP && comm->me == 0)
      error->warning(FLERR, "Using compute temp/deform with inconsistent fix deform remap option");
  } else
    error->warning(FLERR, "Using compute temp/deform with no fix deform defined");
}

void ComputeTempDeformMol::setup()
{
  // Make sure fix property/molecule exists
  if (atom->property_molecule == nullptr || 
      !atom->property_molecule->com_flag)
    error->all(FLERR, "compute temp/deform/mol requires a fix property/molecule to be defined with the com option");
}


/* ---------------------------------------------------------------------- */

double ComputeTempDeformMol::compute_scalar()
{
  int i;
  invoked_scalar = update->ntimestep;

  tagint nmolecule = atom->property_molecule->nmolecule;
  tagint *molecule = atom->molecule;

  // Update COM if it isn't already (generally should be)
  if (atom->property_molecule->comstep != update->ntimestep)
    atom->property_molecule->com_compute();
  double **com = atom->property_molecule->com;
  double *molmass = atom->property_molecule->mass;
 
  // lamda = COM position in triclinic lamda coords
  // vstream = COM streaming velocity = Hrate*lamda + Hratelo. Will be the same
  //           for each atom in the molecule
  // vthermal = thermal velocity = v - vstream
  double lamda[3], vstream_mol[3], vstream_atom[3];

  double *h_rate = domain->h_rate;
  double *h_ratelo = domain->h_ratelo;

  // calculate global temperature

  double **v = atom->v;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  tagint m;
  int xbox, ybox, zbox;
  imageint *image = atom->image;

  if (nlocal > nmax) allocate();

  double t = 0.0;
  int mycount = 0;
 
  for (i = 0; i < atom->nlocal; i++) {
    if (mask[i] & groupbit) {
      m = molecule[i]-1;
      if (m < 0) continue;

      // Calculate streaming velocity at the molecule's centre of mass 
      domain->x2lamda(com[m], lamda);
      vstream_mol[0] = h_rate[0] * lamda[0] + h_rate[5] * lamda[1] + h_rate[4] * lamda[2] + h_ratelo[0];
      vstream_mol[1] = h_rate[1] * lamda[1] + h_rate[3] * lamda[2] + h_ratelo[1];
      vstream_mol[2] = h_rate[2] * lamda[2] + h_ratelo[2];

      // Now calculate the atomic streaming velocity at the unwrapped coordinates
      xbox = (image[i] & IMGMASK) - IMGMAX;
      ybox = (image[i] >> IMGBITS & IMGMASK) - IMGMAX;
      zbox = (image[i] >> IMG2BITS) - IMGMAX;
      vstream_atom[0] = xbox*domain->h_rate[0] + ybox*domain->h_rate[5] + zbox*domain->h_rate[4];
      vstream_atom[1] = ybox*domain->h_rate[1] + zbox*domain->h_rate[3];
      vstream_atom[2] = zbox*domain->h_rate[2];

      // Calculate the thermal velocity of the atom in its unwrapped position. Need to 
      // add the new atomic streaming velocity, but subtract the COM streaming velocity
      vthermal[i][0] = v[i][0] + vstream_atom[0] - vstream_mol[0];
      vthermal[i][1] = v[i][1] + vstream_atom[1] - vstream_mol[1];
      vthermal[i][2] = v[i][2] + vstream_atom[2] - vstream_mol[2];
    }
  }
  // Calculate the thermal velocity (total minus streaming) of all molecules
  vcm_thermal_compute();

  // Tally up the molecule COM velocities to get the kinetic temperature
  for (m = 0; m < nmolecule; m++) {
    t += (vcmall[m][0]*vcmall[m][0] + vcmall[m][1]*vcmall[m][1] + vcmall[m][2]*vcmall[m][2]) * 
          molmass[m];
  } 

  // final temperature
  dof_compute();
  if (dof < 0.0)
    error->all(FLERR,"Temperature compute degrees of freedom < 0");
  scalar = t*tfactor;
  return scalar;
}

/* ---------------------------------------------------------------------- */

void ComputeTempDeformMol::compute_vector()
{
  int i;

  invoked_vector = update->ntimestep;

  tagint *molecule = atom->molecule;
  tagint nmolecule = atom->property_molecule->nmolecule;
  double **com = atom->property_molecule->com;
  double *molmass = atom->property_molecule->mass;

  // Make sure com is up to date
  if (atom->property_molecule->comstep != update->ntimestep)
    atom->property_molecule->com_compute();

  // lamda = COM position in triclinic lamda coords
  // vstream = COM streaming velocity = Hrate*lamda + Hratelo. Will be the same
  //           for each atom in the molecule
  double lamda[3], vstream_mol[3], vstream_atom[3];

  double *h_rate = domain->h_rate;
  double *h_ratelo = domain->h_ratelo;

  double **v = atom->v;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  tagint m;

  if (nlocal > nmax) allocate();

  int xbox, ybox, zbox;
  imageint *image = atom->image;

  double massone,t[6];
  for (i = 0; i < 6; i++) t[i] = 0.0;

  // calculate KE tensor, removing COM streaming velocity
  for (i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      m = molecule[i]-1;
      if (m < 0) continue;

      // Calculate streaming velocity at the molecule's centre of mass 
      domain->x2lamda(com[m], lamda);
      vstream_mol[0] = h_rate[0] * lamda[0] + h_rate[5] * lamda[1] + h_rate[4] * lamda[2] + h_ratelo[0];
      vstream_mol[1] = h_rate[1] * lamda[1] + h_rate[3] * lamda[2] + h_ratelo[1];
      vstream_mol[2] = h_rate[2] * lamda[2] + h_ratelo[2];

      // Now calculate the atomic streaming velocity at the unwrapped coordinates
      xbox = (image[i] & IMGMASK) - IMGMAX;
      ybox = (image[i] >> IMGBITS & IMGMASK) - IMGMAX;
      zbox = (image[i] >> IMG2BITS) - IMGMAX;
      vstream_atom[0] = xbox*domain->h_rate[0] + ybox*domain->h_rate[5] + zbox*domain->h_rate[4];
      vstream_atom[1] = ybox*domain->h_rate[1] + zbox*domain->h_rate[3];
      vstream_atom[2] = zbox*domain->h_rate[2];

      // Calculate the thermal velocity of the atom in its unwrapped position. Need to 
      // add the new atomic streaming velocity, but subtract the COM streaming velocity
      vthermal[i][0] = v[i][0] + vstream_atom[0] - vstream_mol[0];
      vthermal[i][1] = v[i][1] + vstream_atom[1] - vstream_mol[1];
      vthermal[i][2] = v[i][2] + vstream_atom[2] - vstream_mol[2];
    }
  }
  // Calculate the thermal velocity (total minus streaming) of all molecules
  vcm_thermal_compute();

  // Tally up the molecule COM velocities to get the kinetic temperature
  // No need for MPI reductions, since every processor knows the molecule VCMs
  for (m = 0; m < nmolecule; m++) {
      t[0] += molmass[m] * vcmall[m][0] * vcmall[m][0];
      t[1] += molmass[m] * vcmall[m][1] * vcmall[m][1];
      t[2] += molmass[m] * vcmall[m][2] * vcmall[m][2];
      t[3] += molmass[m] * vcmall[m][0] * vcmall[m][1];
      t[4] += molmass[m] * vcmall[m][0] * vcmall[m][2];
      t[5] += molmass[m] * vcmall[m][1] * vcmall[m][2];
  }
  // final KE
  for (i = 0; i < 6; i++) vector[i] = t[i]*force->mvv2e;
}


/* ----------------------------------------------------------------------
   Degrees of freedom for molecular temperature
------------------------------------------------------------------------- */

void ComputeTempDeformMol::dof_compute()
{
  // TODO: This will be incorrect for rigid molecules, since we only care about
  //       CoM momentum
  adjust_dof_fix();

  // TODO: nmolecule is currently the max. molecule index, but some indices
  //       could be skipped which would make this incorrect
  dof = domain->dimension * atom->property_molecule->nmolecule;

  // This will vary on the type of constraint
  // e.g. if they're bond constraints then they're irrelevant to
  // the molecular temperature
  dof -= extra_dof + fix_dof;
  if (dof > 0)
    tfactor = force->mvv2e / (dof * force->boltz);
  else
    tfactor = 0.0;
}

/* ----------------------------------------------------------------------
   calculate thermal centre-of-mass velocity (lab-frame minus streaming) 
   for each molecule.
   PRE: com_compute() must have completed
  --------------------------------------------------------------------*/

void ComputeTempDeformMol::vcm_thermal_compute()
{
  tagint m;
  double massone;
  double unwrap[3];

  // molid = 1 to nmolecule for included atoms, 0 for excluded atoms
  tagint *molecule = atom->molecule;
  tagint nmolecule = atom->property_molecule->nmolecule;
  double *molmass = atom->property_molecule->mass;

  // Reallocation handled by fix property/molecule.
  // Make sure size is up to date
  size_array_rows = nmolecule;

  // zero local per-molecule values
  for (m = 0; m < nmolecule; m++){
    vcm[m][0] = vcm[m][1] = vcm[m][2] = 0.0;
  }

  // compute VCM for each molecule

  double **x = atom->x;
  double **v = atom->v;
  int *mask = atom->mask;
  int *type = atom->type;

  imageint *image = atom->image;
  double v_adjust[3];

  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      m = molecule[i]-1;
      if (m < 0) continue;
      if (rmass) massone = rmass[i];
      else massone = mass[type[i]];
      vcm[m][0] += vthermal[i][0] * massone;
      vcm[m][1] += vthermal[i][1] * massone;
      vcm[m][2] += vthermal[i][2] * massone;
    }

  MPI_Allreduce(&vcm[0][0],&vcmall[0][0],3*nmolecule,MPI_DOUBLE,MPI_SUM,world);
  for (m = 0; m < nmolecule; m++) {
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
   remove velocity bias from atom I to leave thermal velocity
------------------------------------------------------------------------- */

void ComputeTempDeformMol::remove_bias(int i, double *v)
{
  double lamda[3], vstream_mol[3], vstream_atom[3];
  double *h_rate = domain->h_rate;
  double *h_ratelo = domain->h_ratelo;
  int xbox, ybox, zbox;
  imageint *image = atom->image;
  double **com = atom->property_molecule->com;

  tagint m = atom->molecule[i]-1;
  if (m < 0) return;

  domain->x2lamda(com[m], lamda);
  vstream_mol[0] = h_rate[0] * lamda[0] + h_rate[5] * lamda[1] + h_rate[4] * lamda[2] + h_ratelo[0];
  vstream_mol[1] = h_rate[1] * lamda[1] + h_rate[3] * lamda[2] + h_ratelo[1];
  vstream_mol[2] = h_rate[2] * lamda[2] + h_ratelo[2];

  // Now calculate the atomic streaming velocity at the unwrapped coordinates
  xbox = (image[i] & IMGMASK) - IMGMAX;
  ybox = (image[i] >> IMGBITS & IMGMASK) - IMGMAX;
  zbox = (image[i] >> IMG2BITS) - IMGMAX;
  vstream_atom[0] = xbox*domain->h_rate[0] + ybox*domain->h_rate[5] + zbox*domain->h_rate[4];
  vstream_atom[1] = ybox*domain->h_rate[1] + zbox*domain->h_rate[3];
  vstream_atom[2] = zbox*domain->h_rate[2];

  // Calculate the thermal velocity of the atom in its unwrapped position. Need to 
  // add the new atomic streaming velocity, but subtract the COM streaming velocity
  vbias[0] = vstream_mol[0] - vstream_atom[0];
  vbias[1] = vstream_mol[1] - vstream_atom[1];
  vbias[2] = vstream_mol[2] - vstream_atom[2];
  v[0] = v[0] - vbias[0];
  v[1] = v[1] - vbias[1];
  v[2] = v[2] - vbias[2];
}

/* ----------------------------------------------------------------------
   remove velocity bias from all atoms to leave thermal velocity
------------------------------------------------------------------------- */

void ComputeTempDeformMol::remove_bias_all()
{
  double lamda[3], vstream_mol[3], vstream_atom[3];
  double *h_rate = domain->h_rate;
  double *h_ratelo = domain->h_ratelo;
  int xbox, ybox, zbox;
  imageint *image = atom->image;

  if (atom->nmax > maxbias) {
    memory->destroy(vbiasall);
    maxbias = atom->nmax;
    memory->create(vbiasall, maxbias, 3, "temp/deform:vbiasall");
  }

  tagint m;
  tagint *molecule = atom->molecule;
  double **com = atom->property_molecule->com;

  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      m = molecule[i]-1;
      if (m < 0) continue;
      domain->x2lamda(com[m], lamda);
      vstream_mol[0] = h_rate[0] * lamda[0] + h_rate[5] * lamda[1] + h_rate[4] * lamda[2] + h_ratelo[0];
      vstream_mol[1] = h_rate[1] * lamda[1] + h_rate[3] * lamda[2] + h_ratelo[1];
      vstream_mol[2] = h_rate[2] * lamda[2] + h_ratelo[2];

      // Now calculate the atomic streaming velocity at the unwrapped coordinates
      xbox = (image[i] & IMGMASK) - IMGMAX;
      ybox = (image[i] >> IMGBITS & IMGMASK) - IMGMAX;
      zbox = (image[i] >> IMG2BITS) - IMGMAX;
      vstream_atom[0] = xbox*domain->h_rate[0] + ybox*domain->h_rate[5] + zbox*domain->h_rate[4];
      vstream_atom[1] = ybox*domain->h_rate[1] + zbox*domain->h_rate[3];
      vstream_atom[2] = zbox*domain->h_rate[2];

      // Calculate the thermal velocity of the atom in its unwrapped position. Need to 
      // add the new atomic streaming velocity, but subtract the COM streaming velocity
      vbiasall[i][0] = vstream_mol[0] - vstream_atom[0];
      vbiasall[i][1] = vstream_mol[1] - vstream_atom[1];
      vbiasall[i][2] = vstream_mol[2] - vstream_atom[2];
    
      v[i][0] -= vbiasall[i][0];
      v[i][1] -= vbiasall[i][1];
      v[i][2] -= vbiasall[i][2];
    }
}

/* ----------------------------------------------------------------------
   add back in velocity bias to atom I removed by remove_bias()
   assume remove_bias() was previously called
------------------------------------------------------------------------- */

void ComputeTempDeformMol::restore_bias(int i, double *v)
{
  double lamda[3];
  double *h_rate = domain->h_rate;
  double *h_ratelo = domain->h_ratelo;

  tagint m = atom->molecule[i]-1;
  if (m < 0) return;

  v[0] += vbias[0];
  v[1] += vbias[1];
  v[2] += vbias[2];
}

/* ----------------------------------------------------------------------
   add back in velocity bias to all atoms removed by remove_bias_all()
   assume remove_bias_all() was previously called
------------------------------------------------------------------------- */

void ComputeTempDeformMol::restore_bias_all()
{
  double lamda[3], vbias[3];
  double *h_rate = domain->h_rate;
  double *h_ratelo = domain->h_ratelo;

  tagint m;
  tagint *molecule = atom->molecule;

  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      m = molecule[i]-1;
      if (m < 0) continue;
 
      v[i][0] += vbiasall[i][0];
      v[i][1] += vbiasall[i][1];
      v[i][2] += vbiasall[i][2];
    }
}


/* ----------------------------------------------------------------------
   free and reallocate per-atom data
------------------------------------------------------------------------- */

void ComputeTempDeformMol::allocate()
{
  nmax = atom->nlocal;
  memory->grow(vthermal,nmax,3,"temp/deform/mol:vthermal");
}

/* ----------------------------------------------------------------------
   memory usage of local data
------------------------------------------------------------------------- */

double ComputeTempDeformMol::memory_usage()
{
  double bytes = (bigint) nmax * 3 * sizeof(double);
  // vcm and vcmall not allocated if property_molecule is nullptr
  if (atom->property_molecule != nullptr)
    bytes += (bigint) atom->property_molecule->nmolecule * 6 * sizeof(double);
  
  return bytes;
}
