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

#include "fix_nvt_sllod_chunk.h"

#include "atom.h"
#include "compute.h"
#include "compute_chunk_atom.h"
#include "compute_com_chunk.h"
#include "compute_vcm_chunk.h"
#include "domain.h"
#include "error.h"
#include "fix_deform.h"
#include "group.h"
#include "math_extra.h"
#include "modify.h"
#include "memory.h"

#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixNVTSllodChunk::FixNVTSllodChunk(LAMMPS *lmp, int narg, char **arg) :
  FixNH(lmp, narg, arg)
{
  if (!tstat_flag)
    error->all(FLERR,"Temperature control must be used with fix nvt/sllod/chunk");
  if (pstat_flag)
    error->all(FLERR,"Pressure control can not be used with fix nvt/sllod/chunk");

  // default values

  if (mtchain_default_flag) mtchain = 1;

  kickflag = 0;

  int iarg = 3;

  while (iarg < narg) {
    if (strcmp(arg[iarg++], "kick")==0) {
      if (iarg >= narg) error->all(FLERR,"Invalid fix nvt/sllod/chunk command");
      if (strcmp(arg[iarg], "yes")==0) {
        kickflag = 1;
      } else if (strcmp(arg[iarg], "no")==0) {
        kickflag = 0;
      } else error->all(FLERR,"Invalid fix nvt/sllod/chunk command");
      ++iarg;
    }
  }

  // create a new compute temp style
  // id = fix-ID + temp

  id_temp = utils::strdup(std::string(id) + "_temp");
  modify->add_compute(fmt::format("{} {} temp/deform",
                                  id_temp,group->names[igroup]));
  tcomputeflag = 1;
  maxchunk = 0;
  vcm = nullptr;
  vcmall = nullptr;
  masstotal = nullptr;
  massproc = nullptr;
}

/* ---------------------------------------------------------------------- */
FixNVTSllodChunk::~FixNVTSllodChunk() {
  memory->destroy(vcm);
  memory->destroy(vcmall);
  memory->destroy(massproc);
  memory->destroy(masstotal);
}

/* ---------------------------------------------------------------------- */

void FixNVTSllodChunk::init() {
  FixNH::init();

  if (!temperature->tempbias)
    error->all(FLERR,"Temperature for fix nvt/sllod/chunk does not have a bias");

  nondeformbias = 0;
  if (strcmp(temperature->style,"temp/deform/chunk") != 0) nondeformbias = 1;

  // check fix deform remap settings

  int i;
  for (i = 0; i < modify->nfix; i++)
    if (strncmp(modify->fix[i]->style,"deform",6) == 0) {
      if ((dynamic_cast<FixDeform *>( modify->fix[i]))->remapflag != Domain::V_REMAP)
        error->all(FLERR,"Using fix nvt/sllod/chunk with inconsistent fix deform "
                   "remap option");
      break;
    }
  if (i == modify->nfix)
    error->all(FLERR,"Using fix nvt/sllod/chunk with no fix deform defined");
  
  // Chunk compute
  if(idchunk == nullptr)
    error->all(FLERR,"fix nvt/sllod/chunk does not use chunk/atom compute");
  int icompute = modify->find_compute(idchunk);
  if (icompute < 0)
    error->all(FLERR,"Chunk/atom compute does not exist for "
               "fix nvt/sllod/chunk");
  cchunk = dynamic_cast<ComputeChunkAtom *>( modify->compute[icompute]);
  if (strcmp(cchunk->style,"chunk/atom") != 0)
    error->all(FLERR,"fix nvt/sllod/chunk does not use chunk/atom compute");

  // Chunk VCM compute
  if(idchunk == nullptr)
    error->all(FLERR,"fix nvt/sllod/chunk does not use vcm/chunk compute");
  icompute = modify->find_compute(idvcm);
  if (icompute < 0)
    error->all(FLERR,"vcm/chunk compute does not exist for "
               "fix nvt/sllod/chunk");
  cvcm = dynamic_cast<ComputeVCMChunk *>( modify->compute[icompute]);
  if (strcmp(cvcm->style,"vcm/chunk") != 0)
    error->all(FLERR," does not use vcm/chunk compute");

}

void FixNVTSllodChunk::setup(int vflag) {
  FixNH::setup(vflag);

  // Apply kick if necessary
  if (kickflag) {
    // Call remove_bias first to calculate biases
    temperature->compute_scalar();
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

void FixNVTSllodChunk::nh_v_temp() {
  // remove and restore bias = streaming velocity = Hrate*lamda + Hratelo
  // thermostat thermal velocity only
  // vdelu = SLLOD correction = Hrate*Hinv*vthermal
  // for non temp/deform BIAS:
  //   calculate temperature since some computes require temp
  //   computed on current nlocal atoms to remove bias

  if (nondeformbias) temperature->compute_scalar();

  // Remove bias from all atoms at once to avoid re-calculating the COM positions
  temperature->remove_bias_all();

  // Use molecular/chunk centre-of-mass velocity when calculating SLLOD correction
  vcm_thermal_compute();
  nchunk = cchunk->setup_chunks();
  cchunk->compute_ichunk();
  int *ichunk = cchunk->ichunk;
  int index;


  double **v = atom->v;
  int *mask = atom->mask;
  int *type = atom->type;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  double massone;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  double h_two[6],vdelu[3];
  MathExtra::multiply_shape_shape(domain->h_rate,domain->h_inv,h_two);

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      index = ichunk[i]-1;
      if (index < 0) continue;
      // NOTE: This uses the thermal velocity of the chunk centre-of-mass in all cases
      vdelu[0] = h_two[0]*vcmall[index][0] + h_two[5]*vcmall[index][1] + h_two[4]*vcmall[index][2];
      vdelu[1] = h_two[1]*vcmall[index][1] + h_two[3]*vcmall[index][2];
      vdelu[2] = h_two[2]*vcmall[index][2];
      v[i][0] = vcmall[index][0]*factor_eta - dthalf*vdelu[0];
      v[i][1] = vcmall[index][1]*factor_eta - dthalf*vdelu[1];
      v[i][2] = vcmall[index][2]*factor_eta - dthalf*vdelu[2];
    }
  }
  temperature->restore_bias_all();
}

/* calculate COM thermal velocity. 
 * Pre: atom velocities should have streaming bias removed
 *      COM positions should already be computed when removing biases
 */
void FixNVTSllodChunk::vcm_thermal_compute() {
  int index;
  double massone;

  // compute chunk/atom assigns atoms to chunk IDs
  // extract ichunk index vector from compute
  // ichunk = 1 to Nchunk for included atoms, 0 for excluded atoms
  nchunk = cchunk->setup_chunks();
  cchunk->compute_ichunk();
  int *ichunk = cchunk->ichunk;

  if (nchunk > maxchunk) {
    maxchunk = nchunk;
    memory->destroy(vcm);
    memory->destroy(vcmall);
    memory->destroy(massproc);
    memory->destroy(masstotal);
    memory->create(vcm,maxchunk,3,"nvt/sllod/chunk:vcm");
    memory->create(vcmall,maxchunk,3,"nvt/sllod/chunk:vcmall");
    memory->create(massproc,maxchunk,"nvt/sllod/chunk:massproc");
    memory->create(masstotal,maxchunk,"nvt/sllod/chunk:masstotal");
  }

  // zero local per-chunk values

  for (int i = 0; i < nchunk; i++){
    vcm[i][0] = vcm[i][1] = vcm[i][2] = 0.0;
    massproc[i] = 0.0;
  }

  // compute COM and VCM for each chunk

  double **v = atom->v;
  int *mask = atom->mask;
  int *type = atom->type;

  imageint *image = atom->image;
  int xbox, ybox, zbox;
  double v_adjust[3];

  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      index = ichunk[i]-1;
      if (index < 0) continue;
      if (rmass) {
        massone = rmass[i];
      } else {
        massone = mass[type[i]];
      }
      // Adjust the velocity to reflect the thermal velocity 
      vcm[index][0] += v[i][0] * massone;
      vcm[index][1] += v[i][1] * massone;
      vcm[index][2] += v[i][2] * massone;
      massproc[index] += massone;
    }

  MPI_Allreduce(&vcm[0][0],&vcmall[0][0],3*nchunk,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(massproc,masstotal,nchunk,MPI_DOUBLE,MPI_SUM,world);

  for (int i = 0; i < nchunk; i++) {
    if (masstotal[i] > 0.0) {
      vcmall[i][0] /= masstotal[i];
      vcmall[i][1] /= masstotal[i];
      vcmall[i][2] /= masstotal[i];
    } else {
      vcmall[i][0] = vcmall[i][1] = vcmall[i][2] = 0.0;
    }
  }
}
