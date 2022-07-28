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

#include "compute_pressure_chunk.h"

#include "angle.h"
#include "atom.h"
#include "bond.h"
#include "comm.h"
#include "compute_chunk_atom.h"
#include "dihedral.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
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

ComputePressureChunk::ComputePressureChunk(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg),
  vptr(nullptr), id_temp(nullptr), pstyle(nullptr)
{
  if (narg < 5) error->all(FLERR,"Illegal compute pressure command");
  if (igroup) error->all(FLERR,"Compute pressure must use group all");

  scalar_flag = vector_flag = 1;
  size_vector = 9;
  extscalar = 0;
  extvector = 0;
  pressflag = 1;
  timeflag = 1;

  // store temperature ID used by pressure computation
  // insure it is valid for temperature computation
  idchunk = strdup(arg[3]);
  int icompute = modify->find_compute(idchunk);
  if (icompute < 0)
    error->all(FLERR,"Chunk/atom compute does not exist for "
               "compute pressure/chunk");
  cchunk = dynamic_cast<ComputeChunkAtom *>( modify->compute[icompute]);

  if (strcmp(cchunk->style,"chunk/atom") != 0)
    error->all(FLERR,"Compute pressure/chunk does not use chunk/atom compute");

  if (strcmp(arg[4],"NULL") == 0) id_temp = nullptr;
  else {
    id_temp = utils::strdup(arg[4]);

    int icompute = modify->find_compute(id_temp);
    if (icompute < 0)
      error->all(FLERR,"Could not find compute pressure temperature ID");
    if (modify->compute[icompute]->tempflag == 0)
      error->all(FLERR,"Compute pressure temperature ID does not "
                 "compute temperature");
  }

  // process optional args

  pairhybridflag = 0;
  if (narg == 5) {
    keflag = 1;
    pairflag = 1;
    bondflag = angleflag = dihedralflag = improperflag = 1;
    kspaceflag = fixflag = 1;
  } else {
    keflag = 0;
    pairflag = 0;
    bondflag = angleflag = dihedralflag = improperflag = 0;
    kspaceflag = fixflag = 0;
    int iarg = 5;
    while (iarg < narg) {
      if (strcmp(arg[iarg],"ke") == 0) keflag = 1;
      else if (strcmp(arg[iarg],"pair/hybrid") == 0) {
        if (lmp->suffix)
          pstyle = utils::strdup(fmt::format("{}/{}",arg[++iarg],lmp->suffix));
        else
          pstyle = utils::strdup(arg[++iarg]);

        nsub = 0;

        if (narg > iarg) {
          if (isdigit(arg[iarg][0])) {
            nsub = utils::inumeric(FLERR,arg[iarg],false,lmp);
            ++iarg;
            if (nsub <= 0)
              error->all(FLERR,"Illegal compute pressure command");
          }
        }

        // check if pair style with and without suffix exists

        pairhybrid = (Pair *) force->pair_match(pstyle,1,nsub);
        if (!pairhybrid && lmp->suffix) {
          pstyle[strlen(pstyle) - strlen(lmp->suffix) - 1] = '\0';
          pairhybrid = (Pair *) force->pair_match(pstyle,1,nsub);
        }

        if (!pairhybrid)
          error->all(FLERR,"Unrecognized pair style in compute pressure command");

        pairhybridflag = 1;
      }
      else if (strcmp(arg[iarg],"pair") == 0) pairflag = 1;
      else if (strcmp(arg[iarg],"bond") == 0) bondflag = 1;
      else if (strcmp(arg[iarg],"angle") == 0) angleflag = 1;
      else if (strcmp(arg[iarg],"dihedral") == 0) dihedralflag = 1;
      else if (strcmp(arg[iarg],"improper") == 0) improperflag = 1;
      else if (strcmp(arg[iarg],"kspace") == 0) kspaceflag = 1;
      else if (strcmp(arg[iarg],"fix") == 0) fixflag = 1;
      else if (strcmp(arg[iarg],"virial") == 0) {
        pairflag = 1;
        bondflag = angleflag = dihedralflag = improperflag = 1;
        kspaceflag = fixflag = 1;
      } else error->all(FLERR,"Illegal compute pressure command");
      iarg++;
    }
  }

  // error check

  if (keflag && id_temp == nullptr)
    error->all(FLERR,"Compute pressure requires temperature ID "
               "to include kinetic energy");

  vector = new double[size_vector];
  nvirial = 0;
  vptr = nullptr;
}

/* ---------------------------------------------------------------------- */

ComputePressureChunk::~ComputePressureChunk()
{
  delete [] id_temp;
  delete [] vector;
  delete [] vptr;
  delete [] pstyle;
}

/* ---------------------------------------------------------------------- */

void ComputePressureChunk::init()
{
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

  // recheck if pair style with and without suffix exists

  if (pairhybridflag) {
    pairhybrid = (Pair *) force->pair_match(pstyle,1,nsub);
    if (!pairhybrid && lmp->suffix) {
      strcat(pstyle,"/");
      strcat(pstyle,lmp->suffix);
      pairhybrid = (Pair *) force->pair_match(pstyle,1,nsub);
    }

    if (!pairhybrid)
      error->all(FLERR,"Unrecognized pair style in compute pressure command");
  }

  // detect contributions to virial
  // vptr points to all virial[6] contributions

  delete [] vptr;
  nvirial = 0;
  vptr = nullptr;

  //if (pairhybridflag && force->pair) nvirial++;
  if (pairflag && force->pair) nvirial++;
  /*
  if (atom->molecular != Atom::ATOMIC) {
    if (bondflag && force->bond) nvirial++;
    if (angleflag && force->angle) nvirial++;
    if (dihedralflag && force->dihedral) nvirial++;
    if (improperflag && force->improper) nvirial++;
  }
  if (fixflag)
    for (auto &ifix : modify->get_fix_list())
      if (ifix->thermo_virial) nvirial++;
  */
  if (nvirial) {
    vptr = new double*[nvirial];
    nvirial = 0;
    /*
    if (pairhybridflag && force->pair) {
      auto ph = dynamic_cast<PairHybrid *>( force->pair);
      ph->no_virial_fdotr_compute = 1;
      vptr[nvirial++] = pairhybrid->virial;
    }
    */
    if (pairflag && force->pair) vptr[nvirial++] = force->pair->chunk_virial;
    /*
    if (bondflag && force->bond) vptr[nvirial++] = force->bond->virial;
    if (angleflag && force->angle) vptr[nvirial++] = force->angle->virial;
    if (dihedralflag && force->dihedral)
      vptr[nvirial++] = force->dihedral->virial;
    if (improperflag && force->improper)
      vptr[nvirial++] = force->improper->virial;
    if (fixflag)
    for (auto &ifix : modify->get_fix_list())
      if (ifix->virial_global_flag && ifix->thermo_virial)
          vptr[nvirial++] = ifix->virial;
    */
  }

  // flag Kspace contribution separately, since not summed across procs

  if (kspaceflag && force->kspace) kspace_virial = force->kspace->virial;
  else kspace_virial = nullptr;
}

/* ----------------------------------------------------------------------
   compute total pressure, averaged over Pxx, Pyy, Pzz
------------------------------------------------------------------------- */

double ComputePressureChunk::compute_scalar()
{
  invoked_scalar = update->ntimestep;
  if (update->vflag_global != invoked_scalar)
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

void ComputePressureChunk::compute_vector()
{
  invoked_vector = update->ntimestep;
  if (update->vflag_global != invoked_vector)
    error->all(FLERR,"Virial was not tallied on needed timestep");

  if (force->kspace && kspace_virial && force->kspace->scalar_pressure_flag)
    error->all(FLERR,"Must use 'kspace_modify pressure/scalar no' for "
               "tensor components with kspace_style msm");

  // invoke temperature if it hasn't been already

  int i;
  double ke_tensor[9];
  double *temp_tensor;
  if (keflag) {
    if (temperature->invoked_vector != update->ntimestep)
      temperature->compute_vector();
    // TODO EVK: need to make this a 3x3 tensor
    temp_tensor = temperature->vector;
    // The kinetic energy tensor is symmetric by definition, but we still need the full 9 elements
    // so copy them and duplicate as necessary
    for(i=0; i < 6; i++)
      ke_tensor[i] = temp_tensor[i];
    ke_tensor[6] = temp_tensor[3];
    ke_tensor[7] = temp_tensor[4];
    ke_tensor[8] = temp_tensor[5];
  }

  if (dimension == 3) {
    inv_volume = 1.0 / (domain->xprd * domain->yprd * domain->zprd);
    // TODO: overhaul this to use chunks where appropriate
    //virial_compute(6,3);
    double *vchunk = force->pair->chunk_virial;
    double v[9];
    MPI_Allreduce(vchunk,v,9,MPI_DOUBLE,MPI_SUM,world);
    if (keflag) {
      for (int i = 0; i < 9; i++)
        vector[i] = (ke_tensor[i] + v[i]) * inv_volume * nktv2p;
    } else {
      for (int i = 0; i < 9; i++)
        vector[i] = v[i] * inv_volume * nktv2p;
    }
  } else {
    inv_volume = 1.0 / (domain->xprd * domain->yprd);
    double *vchunk = force->pair->chunk_virial;
    double v[9];
    MPI_Allreduce(vchunk,v,9,MPI_DOUBLE,MPI_SUM,world);

    if (keflag) {
      for (int i = 0; i < 9; i++)
        vector[i] = (ke_tensor[i] + v[i]) * inv_volume * nktv2p;
    } else {
      for (int i = 0; i < 9; i++)
        vector[i] = v[i] * inv_volume * nktv2p;
    }
  }
}

/* ---------------------------------------------------------------------- */

void ComputePressureChunk::virial_compute(int n, int ndiag)
{
  int i,j;
  double v[6],*vcomponent;

  for (i = 0; i < n; i++) v[i] = 0.0;

  // sum contributions to virial from forces and fixes

  for (j = 0; j < nvirial; j++) {
    vcomponent = vptr[j];
    for (i = 0; i < n; i++) v[i] += vcomponent[i];
  }

  // sum virial across procs

  MPI_Allreduce(v,virial,n,MPI_DOUBLE,MPI_SUM,world);

  // KSpace virial contribution is already summed across procs

  if (kspace_virial)
    for (i = 0; i < n; i++) virial[i] += kspace_virial[i];

  // LJ long-range tail correction, only if pair contributions are included

  if (force->pair && pairflag && force->pair->tail_flag)
    for (i = 0; i < ndiag; i++) virial[i] += force->pair->ptail * inv_volume;
}

/* ---------------------------------------------------------------------- */

void ComputePressureChunk::reset_extra_compute_fix(const char *id_new)
{
  delete [] id_temp;
  id_temp = utils::strdup(id_new);
}
