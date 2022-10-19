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

#include "fix_property_molecule.h"

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "memory.h"
#include "update.h"

#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;


/* ---------------------------------------------------------------------- */

FixPropertyMolecule::FixPropertyMolecule(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), mass(nullptr), com(nullptr), massproc(nullptr),
    comproc(nullptr)
{
  if (narg < 4) error->all(FLERR, "Illegal fix property/atom command");

  int iarg = 3;

  mass_flag = 0;
  com_flag = 0;

  while (iarg < narg) {
    if (strcmp(arg[iarg], "mass") == 0) {
      mass_flag = 1;
      iarg++;
    } else if (strcmp(arg[iarg], "com") == 0) {
      if (com_flag) continue;
      mass_flag = 1;
      com_flag = 1;
      iarg++;
    } else error->all(FLERR, "Illegal fix property/atom command");
  }

  nmax = 0;
  nmolecule = 1;

  if (atom->property_molecule != nullptr)
    error->all(FLERR, "Illegal redefinition of fix property/molecule");
  atom->property_molecule = this;

  comstep = -1;

  if (mass_flag) {
    register_permolecule("property/molecule:mass", &mass, Atom::DOUBLE, 0);
    register_permolecule("property/molecule:massproc", &massproc, Atom::DOUBLE, 0);
  }
  if (com_flag)  {
    register_permolecule("property/molecule:com", &com, Atom::DOUBLE, 3);
    register_permolecule("property/molecule:comproc", &comproc, Atom::DOUBLE, 3);
  }

  array_flag = 1;
  size_array_cols = 4;
  size_array_rows_variable = 1;
}

/* ---------------------------------------------------------------------- */

FixPropertyMolecule::~FixPropertyMolecule()
{
  atom->property_molecule = nullptr;
  for (auto &item : permolecule) mem_destroy(item);
}

/* ---------------------------------------------------------------------- */

int FixPropertyMolecule::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE;
  mask |= PRE_FORCE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixPropertyMolecule::register_permolecule( std::string name, void *address,
    int datatype, int cols) {
  if (address == nullptr) return;

  for (auto &item : permolecule) {
    if (address == item.address) return;
  }
  permolecule.emplace_back(PerMolecule{name, address, datatype, cols});
  if (nmax > 0) {
    auto &item = permolecule.back();
    mem_create(item);
  }
}

void FixPropertyMolecule::destroy_permolecule(void *address) {
  auto item = permolecule.begin();
  while (item != permolecule.end()) {
    if (item->address == address) {
      mem_destroy(*item);
      item = permolecule.erase(item);
    } else ++item;
  }
}

/* ---------------------------------------------------------------------- */

void FixPropertyMolecule::init()
{
  // Error if system doesn't track molecule ids.
  // Check here since atom_style could change before run.

  if (!atom->molecule_flag)
    error->all(FLERR, "Fix property/molecule when atom_style does not define a molecule attribute");
}

void FixPropertyMolecule::setup_pre_force(int vflag) {

  // This assumes number of molecules won't change during a run
  // If something like fix gcmc could change this, maybe add a flag for that?
  // Recalculating nmolecule each step requires comm, so probably not ideal if
  // it can be avoided
  // Needs to be run before main setup() calls, since those could rely on the
  // memory being allocated (eg. for fix nvt/sllod/molecule with kick yes)
  grow_permolecule();
  if (com_flag) com_compute();
}

void FixPropertyMolecule::setup_pre_force_respa(int vflag, int ilevel) {

  // TODO: Any reason to check other levels for new number of molecules?
  if (ilevel == 0) setup_pre_force(vflag);
}

/* ----------------------------------------------------------------------
   Calculate number of molecules and grow permolecule arrays if needed
------------------------------------------------------------------------- */

void FixPropertyMolecule::grow_permolecule() {
  // Calculate number of molecules
  // TODO: maybe take an input value for how much to grow by,
  //       or 0 if nmolecule should be calculated?
  //       This could all change if we handle molecule ownership.
  tagint *molecule = atom->molecule;
  int nlocal = atom->nlocal;
  tagint maxone = -1;
  for (int i = 0; i < nlocal; i++)
    if (molecule[i] > maxone) maxone = molecule[i];
  tagint maxall;
  MPI_Allreduce(&maxone, &maxall, 1, MPI_LMP_TAGINT, MPI_MAX, world);
  if (maxall > MAXSMALLINT)
    error->all(FLERR, "Molecule IDs too large for fix property/molecule");
  nmolecule = maxall;
  
  // Grow arrays as needed
  if (nmax < nmolecule) {
    nmax = nmolecule;
    for (auto &item : permolecule) mem_grow(item);

    // Recompute mass if number of molecules has changed
    // Assumes mass is constant.
    if (mass_flag) mass_compute();
  }

  size_array_rows = static_cast<int>(nmolecule);
}

/* ----------------------------------------------------------------------
   Update total mass of each molecule
------------------------------------------------------------------------- */

void FixPropertyMolecule::mass_compute() {
  if (nmolecule == 0) return;
  double massone;
  for (tagint m = 0; m < nmolecule; ++m)
    massproc[m] = 0.0;

  for (int i = 0; i < atom->nlocal; ++i) {
    tagint m = atom->molecule[i]-1;
    if (m < 0) continue;
    if (atom->rmass) massone = atom->rmass[i];
    else massone = atom->mass[atom->type[i]];
    massproc[m] += massone;
  }
  MPI_Allreduce(massproc,mass,nmolecule,MPI_DOUBLE,MPI_SUM,world);
}

/* ----------------------------------------------------------------------
   Update COM before force calculation so it can be used to tally the
   molecular virial
------------------------------------------------------------------------- */

void FixPropertyMolecule::pre_force(int vflag) {

  // NOTE: This is quite specific to COM. Probably best to add a general
  // framework in future to handle where in the run each per-molecule
  // vector/array should be recalculated, and to combine the tallying and MPI
  // communication where possible
  // Alternatively, property/molecule could just handle memory allocation, and
  // let other code do the actual calculation
  if (com_flag) com_compute();
}

// TODO: Not sure how often this actually needs to be called, or if there are
// smart performance things we could do
void FixPropertyMolecule::pre_force_respa(int vflag, int /*ilevel*/, int /*iloop*/) {
  pre_force(vflag);
}

/* ----------------------------------------------------------------------
   Calculate center of mass of each molecule in unwrapped coords
------------------------------------------------------------------------- */

void FixPropertyMolecule::com_compute() {
  comstep = update->ntimestep;
  if (nmolecule == 0) return; // Prevent segfault if no molecules exit

  int nlocal = atom->nlocal;
  tagint *molecule = atom->molecule;

  int *type = atom->type;
  imageint *image = atom->image;
  double *amass = atom->mass;
  double *rmass = atom->rmass;
  double **x = atom->x;
  double **v = atom->v;
  double massone, unwrap[3];

  for (int m = 0; m < nmolecule; ++m) {
    comproc[m][0] = 0.0;
    comproc[m][1] = 0.0;
    comproc[m][2] = 0.0;
  }

  for (int i = 0; i < nlocal; ++i) {
    tagint m = molecule[i]-1;
    if (m < 0) continue;
    if (rmass) massone = rmass[i];
    else massone = amass[type[i]];

    domain->unmap(x[i],image[i],unwrap);
    comproc[m][0] += unwrap[0] * massone;
    comproc[m][1] += unwrap[1] * massone;
    comproc[m][2] += unwrap[2] * massone;
  }

  MPI_Allreduce(&comproc[0][0],&com[0][0],3*nmolecule,MPI_DOUBLE,MPI_SUM,world);

  for (int m = 0; m < nmolecule; ++m) {
    // Some molecule ids could be skipped (not assigned atoms)
    if (mass[m] > 0.0) {
      com[m][0] /= mass[m];
      com[m][1] /= mass[m];
      com[m][2] /= mass[m];
    } else {
      com[m][0] = com[m][1] = com[m][2] = 0.0;
    }
  }
}


/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double FixPropertyMolecule::memory_usage()
{
  double bytes = 0.0;
  if (mass_flag) bytes += nmax * sizeof(double);
  if (com_flag)  bytes += nmax * 3 * sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   basic array output, almost no error checking
------------------------------------------------------------------------- */

double FixPropertyMolecule::compute_array(int imol, int col)
{
  if (imol > static_cast<int>(nmolecule))
    error->all(FLERR, fmt::format("Cannot request info for molecule {} from fix/property/molecule (nmolecule = {})", imol, nmolecule));
  if (col == 3) return mass[imol];
  else return com[imol][col];
}

/* ----------------------------------------------------------------------
   memory handling for permolecule data
------------------------------------------------------------------------- */

template<typename T> inline
void FixPropertyMolecule::mem_create_impl(PerMolecule &item) {
  if (item.cols == 0)
    memory->create(*(T**)item.address, nmax, item.name.c_str());
  else if (item.cols > 0)
    memory->create(*(T***)item.address, nmax, item.cols, item.name.c_str());
}
void FixPropertyMolecule::mem_create(PerMolecule &item) {

  if      (item.datatype == Atom::DOUBLE) mem_create_impl<double>(item);
  else if (item.datatype == Atom::INT)    mem_create_impl<int>(item);
  else if (item.datatype == Atom::BIGINT) mem_create_impl<bigint>(item);

}

template<typename T> inline
void FixPropertyMolecule::mem_grow_impl(PerMolecule &item) {
  if (item.cols == 0)
    memory->grow(*(T**)item.address, nmax, item.name.c_str());
  else if (item.cols > 0)
    memory->grow(*(T***)item.address, nmax, item.cols, item.name.c_str());
}
void FixPropertyMolecule::mem_grow(PerMolecule &item) {
  if      (item.datatype == Atom::DOUBLE) mem_grow_impl<double>(item);
  else if (item.datatype == Atom::INT)    mem_grow_impl<int>(item);
  else if (item.datatype == Atom::BIGINT) mem_grow_impl<bigint>(item);
}

template<typename T> inline
void FixPropertyMolecule::mem_destroy_impl(PerMolecule &item) {
  if (item.cols == 0) 
    memory->destroy(*(T**)item.address);
  else if (item.cols > 0)
    memory->destroy(*(T***)item.address);
}
void FixPropertyMolecule::mem_destroy(PerMolecule &item) {
  if      (item.datatype == Atom::DOUBLE) mem_destroy_impl<double>(item);
  else if (item.datatype == Atom::INT)    mem_destroy_impl<int>(item);
  else if (item.datatype == Atom::BIGINT) mem_destroy_impl<bigint>(item);
}
