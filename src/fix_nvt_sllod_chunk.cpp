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


  // create a new compute temp style
  // id = fix-ID + temp

  id_temp = utils::strdup(std::string(id) + "_temp");
  modify->add_compute(fmt::format("{} {} temp/deform",
                                  id_temp,group->names[igroup]));
  tcomputeflag = 1;
}

/* ---------------------------------------------------------------------- */

void FixNVTSllodChunk::init()
{
  FixNH::init();

  if (!temperature->tempbias)
    error->all(FLERR,"Temperature for fix nvt/sllod/chunk does not have a bias");

  nondeformbias = 0;
  if (strcmp(temperature->style,"temp/deform") != 0) nondeformbias = 1;

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
  int icompute = modify->find_compute(idchunk);
  if (icompute < 0)
    error->all(FLERR,"Chunk/atom compute does not exist for "
               "fix nvt/sllod/chunk");
  cchunk = dynamic_cast<ComputeChunkAtom *>( modify->compute[icompute]);
  if (strcmp(cchunk->style,"chunk/atom") != 0)
    error->all(FLERR,"fix nvt/sllod/chunk does not use chunk/atom compute");

  // Chunk VCM compute
  icompute = modify->find_compute(idvcm);
  if (icompute < 0)
    error->all(FLERR,"vcm/chunk compute does not exist for "
               "fix nvt/sllod/chunk");
  cvcm = dynamic_cast<ComputeVCMChunk *>( modify->compute[icompute]);
  if (strcmp(cvcm->style,"vcm/chunk") != 0)
    error->all(FLERR," does not use vcm/chunk compute");

}

/* ----------------------------------------------------------------------
   perform half-step scaling of velocities
-----------------------------------------------------------------------*/

void FixNVTSllodChunk::nh_v_temp()
{
  // remove and restore bias = streaming velocity = Hrate*lamda + Hratelo
  // thermostat thermal velocity only
  // vdelu = SLLOD correction = Hrate*Hinv*vthermal
  // for non temp/deform BIAS:
  //   calculate temperature since some computes require temp
  //   computed on current nlocal atoms to remove bias

  if (nondeformbias) temperature->compute_scalar();

  // Use molecular/chunk centre-of-mass velocity when calculating SLLOD correction
  cvcm->compute_array();
  double **vcm = cvcm->array;
  int nchunk = cchunk->setup_chunks();
  cchunk->compute_ichunk();
  int *ichunk = cchunk->ichunk;
  int index;

  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  double h_two[6],vdelu[3];
  MathExtra::multiply_shape_shape(domain->h_rate,domain->h_inv,h_two);

  // TODO EVK: do we need to rescale by the molecular mass here?
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      index = ichunk[i]-1;
      if (index < 0) continue;
      vdelu[0] = h_two[0]*vcm[index][0] + h_two[5]*vcm[index][1] + h_two[4]*vcm[index][2];
      vdelu[1] = h_two[1]*vcm[index][1] + h_two[3]*vcm[index][2];
      vdelu[2] = h_two[2]*vcm[index][2];
      temperature->remove_bias(i,v[i]);
      v[i][0] = v[i][0]*factor_eta - dthalf*vdelu[0];
      v[i][1] = v[i][1]*factor_eta - dthalf*vdelu[1];
      v[i][2] = v[i][2]*factor_eta - dthalf*vdelu[2];
      temperature->restore_bias(i,v[i]);
    }
  }
}
