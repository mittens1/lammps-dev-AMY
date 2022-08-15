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
   Contributing author: Emily Kahl (Uni of QLD)
------------------------------------------------------------------------- */

#include "compute_temp_deform_chunk.h"

#include "atom.h"
#include "comm.h"
#include "compute_chunk_atom.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "fix_deform.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "update.h"

#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeTempDeformChunk::ComputeTempDeformChunk(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg),
  which(nullptr), idchunk(nullptr), id_bias(nullptr), sum(nullptr), sumall(nullptr), count(nullptr),
  countall(nullptr), massproc(nullptr), masstotal(nullptr), vcm(nullptr), vcmall(nullptr),
  com(nullptr), comall(nullptr)
{
  if (narg < 4) error->all(FLERR,"Illegal compute temp/chunk/deform command");

  scalar_flag = vector_flag = 1;
  size_vector = 6;
  extscalar = 0;
  extvector = 1;
  tempflag = 1;
  tempbias = 1;
  maxbias = 0;
  vbiasall = nullptr;
  vthermal = nullptr;

  // ID of compute chunk/atom

  idchunk = utils::strdup(arg[3]);

  ComputeTempDeformChunk::init();

  // optional per-chunk values

  nvalues = narg-4;
  which = new int[nvalues];
  nvalues = 0;

  int iarg = 4;

  // optional args

  comflag = 0;
  biasflag = 0;
  adof = domain->dimension;
  cdof = 0.0;

  /*
  while (iarg < narg) {
    if (strcmp(arg[iarg],"com") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal compute temp/chunk/deform command");
      comflag = utils::logical(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"bias") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal compute temp/chunk/deform command");
      biasflag = 1;
      id_bias = utils::strdup(arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"adof") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal compute temp/chunk/deform command");
      adof = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"cdof") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal compute temp/chunk/deform command");
      cdof = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else error->all(FLERR,"Illegal compute temp/chunk/deform command");
  }
  */

  // vector data

  vector = new double[size_vector];


  if (nvalues)  {
    array_flag = 1;
    size_array_cols = nvalues;
    size_array_rows = 0;
    size_array_rows_variable = 1;
    extarray = 0;
  }

  // chunk-based data
  nchunk = 1;
  maxchunk = 0;
  nmax = 0;
  allocate();
  comstep = -1;
}

/* ---------------------------------------------------------------------- */

ComputeTempDeformChunk::~ComputeTempDeformChunk()
{
  delete [] idchunk;
  delete [] which;
  delete [] id_bias;
  delete [] vector;
  memory->destroy(sum);
  memory->destroy(sumall);
  memory->destroy(count);
  memory->destroy(countall);
  memory->destroy(array);
  memory->destroy(massproc);
  memory->destroy(masstotal);
  memory->destroy(com);
  memory->destroy(comall);
  memory->destroy(vcm);
  memory->destroy(vcmall);
  memory->destroy(vbiasall);
  memory->destroy(vthermal);
}

/* ---------------------------------------------------------------------- */

void ComputeTempDeformChunk::init()
{
  int icompute = modify->find_compute(idchunk);
  if (icompute < 0)
    error->all(FLERR,"Chunk/atom compute does not exist for "
               "compute temp/chunk/deform");
  cchunk = dynamic_cast<ComputeChunkAtom *>( modify->compute[icompute]);
  if (strcmp(cchunk->style,"chunk/atom") != 0)
    error->all(FLERR,"compute temp/chunk/deform does not use chunk/atom compute");

  auto fixes = modify->get_fix_by_style("^deform");
  if (fixes.size() > 0) {
    if ((dynamic_cast<FixDeform *>(fixes[0]))->remapflag == Domain::X_REMAP && comm->me == 0)
      error->warning(FLERR, "Using compute temp/deform with inconsistent fix deform remap option");
  } else
    error->warning(FLERR, "Using compute temp/deform with no fix deform defined");
}

/* ---------------------------------------------------------------------- */

double ComputeTempDeformChunk::compute_scalar()
{
  int i;

  // calculate chunk assignments,
  //   since only atoms in chunks contribute to global temperature
  // compute chunk/atom assigns atoms to chunk IDs
  // extract ichunk index vector from compute
  // ichunk = 1 to Nchunk for included atoms, 0 for excluded atoms

  nchunk = cchunk->setup_chunks();
  cchunk->compute_ichunk();
  int *ichunk = cchunk->ichunk;

  if ((nchunk > maxchunk) || (atom->nlocal > nmax)) allocate();

  // calculate COM position for each chunk
  // This will be used to calculate the streaming velocity at the chunk's COM
  // TODO EVK: Need to find a sensible caching strategy - too slow to recalculate every time
  if(comstep != update->ntimestep) {
    com_compute();
  }
 
  // lamda = COM position in triclinic lamda coords
  // vstream = COM streaming velocity = Hrate*lamda + Hratelo. Will be the same for each atom in the chunk
  // vthermal = thermal velocity = v - vstream
  double lamda[3], vstream_chunk[3], vstream_atom[3];

  double *h_rate = domain->h_rate;
  double *h_ratelo = domain->h_ratelo;

  // calculate global temperature

  double **v = atom->v;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int index;
  int xbox, ybox, zbox;
  imageint *image = atom->image;

  double t = 0.0;
  int mycount = 0;
 
  for (i = 0; i < atom->nlocal; i++) {
    if (mask[i] & groupbit) {
      index = ichunk[i]-1;
      if (index < 0) continue;

      // Calculate streaming velocity at the chunk's centre of mass 
      domain->x2lamda(comall[index], lamda);
      vstream_chunk[0] = h_rate[0] * lamda[0] + h_rate[5] * lamda[1] + h_rate[4] * lamda[2] + h_ratelo[0];
      vstream_chunk[1] = h_rate[1] * lamda[1] + h_rate[3] * lamda[2] + h_ratelo[1];
      vstream_chunk[2] = h_rate[2] * lamda[2] + h_ratelo[2];

      // Now calculate the atomic streaming velocity at the unwrapped coordinates
      xbox = (image[i] & IMGMASK) - IMGMAX;
      ybox = (image[i] >> IMGBITS & IMGMASK) - IMGMAX;
      zbox = (image[i] >> IMG2BITS) - IMGMAX;
      // TODO EVK: should be xbox + 1 if xbox < 0?
      vstream_atom[0] = xbox*domain->h_rate[0] + ybox*domain->h_rate[5] + zbox*domain->h_rate[4];
      vstream_atom[1] = ybox*domain->h_rate[1] + zbox*domain->h_rate[3];
      vstream_atom[2] = zbox*domain->h_rate[2];

      // Calculate the thermal velocity of the atom in its unwrapped position. Need to 
      // add the new atomic streaming velocity, but subtract the COM streaming velocity
      vthermal[i][0] = v[i][0] + vstream_atom[0] - vstream_chunk[0];
      vthermal[i][1] = v[i][1] + vstream_atom[1] - vstream_chunk[1];
      vthermal[i][2] = v[i][2] + vstream_atom[2] - vstream_chunk[2];
    }
  }
  // Calculate the thermal velocity (total minus streaming) of all chunks
  vcm_thermal_compute();

  // Tally up the chunk COM velocities to get the kinetic temperature
  for (i = 0; i < nchunk; i++) {
    t += (vcmall[i][0]*vcmall[i][0] + vcmall[i][1]*vcmall[i][1] + vcmall[i][2]*vcmall[i][2]) * 
          masstotal[i];
  } 

  // final temperature
  //MPI_Allreduce(&t,&scalar,1,MPI_DOUBLE,MPI_SUM,world);
  //double rcount = mycount;
  //double allcount;
  //MPI_Allreduce(&rcount,&allcount,1,MPI_DOUBLE,MPI_SUM,world);
  //printf("proc %d: nchunk = %d, allcount = %d\n", comm->me, nchunk, allcount);

  dof_compute();
  if (dof < 0.0)
    error->all(FLERR,"Temperature compute degrees of freedom < 0");
  scalar = t*tfactor;
  return scalar;
}

/* ---------------------------------------------------------------------- */

void ComputeTempDeformChunk::compute_vector()
{
  int i;

  // calculate chunk assignments,
  //   since only atoms in chunks contribute to global temperature
  // compute chunk/atom assigns atoms to chunk IDs
  // extract ichunk index vector from compute
  // ichunk = 1 to Nchunk for included atoms, 0 for excluded atoms

  nchunk = cchunk->setup_chunks();
  cchunk->compute_ichunk();
  int *ichunk = cchunk->ichunk;

  if (nchunk > maxchunk) allocate();

  // calculate COM position and velocity for each chunk
  // This will be used to calculate the streaming velocity at the chunk's COM
  // TODO EVK: Need to find a sensible caching strategy - too slow to recalculate every time
  if(comstep != update->ntimestep) {
    com_compute();
  }

  // lamda = COM position in triclinic lamda coords
  // vstream = COM streaming velocity = Hrate*lamda + Hratelo. Will be the same for each atom in the chunk
  double lamda[3], vstream_chunk[3], vstream_atom[3];

  double *h_rate = domain->h_rate;
  double *h_ratelo = domain->h_ratelo;

  double **v = atom->v;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int index;

  int xbox, ybox, zbox;
  imageint *image = atom->image;

  double massone,t[6];
  for (i = 0; i < 6; i++) t[i] = 0.0;

  // calculate KE tensor, removing COM streaming velocity
  for (i = 0; i < nmax; i++) {
    if (mask[i] & groupbit) {
      index = ichunk[i]-1;
      if (index < 0) continue;

      // Calculate streaming velocity at the chunk's centre of mass 
      domain->x2lamda(comall[index], lamda);
      vstream_chunk[0] = h_rate[0] * lamda[0] + h_rate[5] * lamda[1] + h_rate[4] * lamda[2] + h_ratelo[0];
      vstream_chunk[1] = h_rate[1] * lamda[1] + h_rate[3] * lamda[2] + h_ratelo[1];
      vstream_chunk[2] = h_rate[2] * lamda[2] + h_ratelo[2];

      // Now calculate the atomic streaming velocity at the unwrapped coordinates
      xbox = (image[i] & IMGMASK) - IMGMAX;
      ybox = (image[i] >> IMGBITS & IMGMASK) - IMGMAX;
      zbox = (image[i] >> IMG2BITS) - IMGMAX;
      // TODO EVK: should be xbox + 1 if xbox < 0?
      vstream_atom[0] = xbox*domain->h_rate[0] + ybox*domain->h_rate[5] + zbox*domain->h_rate[4];
      vstream_atom[1] = ybox*domain->h_rate[1] + zbox*domain->h_rate[3];
      vstream_atom[2] = zbox*domain->h_rate[2];

      // Calculate the thermal velocity of the atom in its unwrapped position. Need to 
      // add the new atomic streaming velocity, but subtract the COM streaming velocity
      vthermal[i][0] = v[i][0] + vstream_atom[0] - vstream_chunk[0];
      vthermal[i][1] = v[i][1] + vstream_atom[1] - vstream_chunk[1];
      vthermal[i][2] = v[i][2] + vstream_atom[2] - vstream_chunk[2];
    }
  }
  // Calculate the thermal velocity (total minus streaming) of all chunks
  vcm_thermal_compute();

  // Tally up the chunk COM velocities to get the kinetic temperature
  // No need for MPI reductions, since every processor knows the chunk VCMs
  for (i = 0; i < nchunk; i++) {
      t[0] += masstotal[i] * vcmall[i][0] * vcmall[i][0];
      t[1] += masstotal[i] * vcmall[i][1] * vcmall[i][1];
      t[2] += masstotal[i] * vcmall[i][2] * vcmall[i][2];
      t[3] += masstotal[i] * vcmall[i][0] * vcmall[i][1];
      t[4] += masstotal[i] * vcmall[i][0] * vcmall[i][2];
      t[5] += masstotal[i] * vcmall[i][1] * vcmall[i][2];
  }
  // final KE
  //MPI_Allreduce(t,vector,6,MPI_DOUBLE,MPI_SUM,world);
  for (i = 0; i < 6; i++) vector[i] = t[i]*force->mvv2e;
}


/* ----------------------------------------------------------------------
   Degrees of freedom for chunk temperature
------------------------------------------------------------------------- */

void ComputeTempDeformChunk::dof_compute()
{
  nchunk = cchunk->setup_chunks();
  adjust_dof_fix();
  dof = domain->dimension * nchunk;
  // TODO EVK: This will vary on the type of constraint
  // e.g. if they're bond constraints then they're irrelevant to
  // the molecular temperature
  dof -= extra_dof + fix_dof;
  if (dof > 0)
    tfactor = force->mvv2e / (dof * force->boltz);
  else
    tfactor = 0.0;
}

/* ----------------------------------------------------------------------
   calculate COM for each chunk
------------------------------------------------------------------------- */

void ComputeTempDeformChunk::com_compute()
{
  int index;
  double massone;
  double unwrap[3];

  comstep = update->ntimestep;

  // compute chunk/atom assigns atoms to chunk IDs
  // extract ichunk index vector from compute
  // ichunk = 1 to Nchunk for included atoms, 0 for excluded atoms

  nchunk = cchunk->setup_chunks();
  cchunk->compute_ichunk();
  int *ichunk = cchunk->ichunk;

  if (nchunk > maxchunk) allocate();
  size_array_rows = nchunk;

  // zero local per-chunk values

  for (int i = 0; i < nchunk; i++){
    com[i][0] = com[i][1] = com[i][2] = 0.0;
    massproc[i] = 0.0;
  }

  // compute COM and VCM for each chunk

  double **x = atom->x;
  double **v = atom->v;
  int *mask = atom->mask;
  int *type = atom->type;
  imageint *image = atom->image;

  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      index = ichunk[i]-1;
      if (index < 0) continue;
      if (rmass) massone = rmass[i];
      else massone = mass[type[i]];
      
      // Have to compute COM in unwrapped coordinates
      domain->unmap(x[i],image[i],unwrap);
      com[index][0] += unwrap[0] * massone;
      com[index][1] += unwrap[1] * massone;
      com[index][2] += unwrap[2] * massone;
      massproc[index] += massone;
    }

  MPI_Allreduce(&com[0][0],&comall[0][0],3*nchunk,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(massproc,masstotal,nchunk,MPI_DOUBLE,MPI_SUM,world);

  for (int i = 0; i < nchunk; i++) {
    if (masstotal[i] > 0.0) {
      comall[i][0] /= masstotal[i];
      comall[i][1] /= masstotal[i];
      comall[i][2] /= masstotal[i];
    } else {
      comall[i][0] = comall[i][1] = comall[i][2] = 0.0;
    }
  }
}
/* ----------------------------------------------------------------------
   calculate thermal centre-of-mass velocity (lab-frame minus streaming) 
   for each chunk.
   PRE: com_compute() must have completed
  --------------------------------------------------------------------*/

void ComputeTempDeformChunk::vcm_thermal_compute()
{
  int index;
  double massone;
  double unwrap[3];

  // compute chunk/atom assigns atoms to chunk IDs
  // extract ichunk index vector from compute
  // ichunk = 1 to Nchunk for included atoms, 0 for excluded atoms

  nchunk = cchunk->setup_chunks();
  cchunk->compute_ichunk();
  int *ichunk = cchunk->ichunk;

  if (nchunk > maxchunk) allocate();
  size_array_rows = nchunk;

  // zero local per-chunk values

  for (int i = 0; i < nchunk; i++){
    vcm[i][0] = vcm[i][1] = vcm[i][2] = 0.0;
  }

  // compute COM and VCM for each chunk

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
      index = ichunk[i]-1;
      if (index < 0) continue;
      if (rmass) massone = rmass[i];
      else massone = mass[type[i]];
      vcm[index][0] += vthermal[i][0] * massone;
      vcm[index][1] += vthermal[i][1] * massone;
      vcm[index][2] += vthermal[i][2] * massone;
    }

  MPI_Allreduce(&vcm[0][0],&vcmall[0][0],3*nchunk,MPI_DOUBLE,MPI_SUM,world);
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

/* ----------------------------------------------------------------------
   bias methods: called by thermostats
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   remove velocity bias from atom I to leave thermal velocity
------------------------------------------------------------------------- */

void ComputeTempDeformChunk::remove_bias(int i, double *v)
{
  double lamda[3], vstream_chunk[3], vstream_atom[3];
  double *h_rate = domain->h_rate;
  double *h_ratelo = domain->h_ratelo;
  int xbox, ybox, zbox;
  imageint *image = atom->image;

  int index = cchunk->ichunk[i]-1;
  if (index < 0) return;

  if(comstep != update->ntimestep) {
    com_compute();
  }

  domain->x2lamda(comall[index], lamda);
  vstream_chunk[0] = h_rate[0] * lamda[0] + h_rate[5] * lamda[1] + h_rate[4] * lamda[2] + h_ratelo[0];
  vstream_chunk[1] = h_rate[1] * lamda[1] + h_rate[3] * lamda[2] + h_ratelo[1];
  vstream_chunk[2] = h_rate[2] * lamda[2] + h_ratelo[2];

  // Now calculate the atomic streaming velocity at the unwrapped coordinates
  xbox = (image[i] & IMGMASK) - IMGMAX;
  ybox = (image[i] >> IMGBITS & IMGMASK) - IMGMAX;
  zbox = (image[i] >> IMG2BITS) - IMGMAX;
  vstream_atom[0] = xbox*domain->h_rate[0] + ybox*domain->h_rate[5] + zbox*domain->h_rate[4];
  vstream_atom[1] = ybox*domain->h_rate[1] + zbox*domain->h_rate[3];
  vstream_atom[2] = zbox*domain->h_rate[2];

  // Calculate the thermal velocity of the atom in its unwrapped position. Need to 
  // add the new atomic streaming velocity, but subtract the COM streaming velocity
  vbias[0] = vstream_chunk[0] - vstream_atom[0];
  vbias[1] = vstream_chunk[1] - vstream_atom[1];
  vbias[2] = vstream_chunk[2] - vstream_atom[2];
  v[0] = v[0] - vbias[0];
  v[1] = v[1] - vbias[1];
  v[2] = v[2] - vbias[2];
}

/* ----------------------------------------------------------------------
   remove velocity bias from all atoms to leave thermal velocity
------------------------------------------------------------------------- */

void ComputeTempDeformChunk::remove_bias_all()
{
  double lamda[3], vstream_chunk[3], vstream_atom[3];
  double *h_rate = domain->h_rate;
  double *h_ratelo = domain->h_ratelo;
  int xbox, ybox, zbox;
  imageint *image = atom->image;

  if (atom->nmax > maxbias) {
    memory->destroy(vbiasall);
    maxbias = atom->nmax;
    memory->create(vbiasall, maxbias, 3, "temp/deform:vbiasall");
  }

  int index;
  int *ichunk = cchunk->ichunk;

  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  if(comstep != update->ntimestep) {
    com_compute();
  }

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      index = ichunk[i]-1;
      if (index < 0) continue;
      domain->x2lamda(comall[index], lamda);
      vstream_chunk[0] = h_rate[0] * lamda[0] + h_rate[5] * lamda[1] + h_rate[4] * lamda[2] + h_ratelo[0];
      vstream_chunk[1] = h_rate[1] * lamda[1] + h_rate[3] * lamda[2] + h_ratelo[1];
      vstream_chunk[2] = h_rate[2] * lamda[2] + h_ratelo[2];

      // Now calculate the atomic streaming velocity at the unwrapped coordinates
      xbox = (image[i] & IMGMASK) - IMGMAX;
      ybox = (image[i] >> IMGBITS & IMGMASK) - IMGMAX;
      zbox = (image[i] >> IMG2BITS) - IMGMAX;
      vstream_atom[0] = xbox*domain->h_rate[0] + ybox*domain->h_rate[5] + zbox*domain->h_rate[4];
      vstream_atom[1] = ybox*domain->h_rate[1] + zbox*domain->h_rate[3];
      vstream_atom[2] = zbox*domain->h_rate[2];

      // Calculate the thermal velocity of the atom in its unwrapped position. Need to 
      // add the new atomic streaming velocity, but subtract the COM streaming velocity
      vbiasall[i][0] = vstream_chunk[0] - vstream_atom[0];
      vbiasall[i][1] = vstream_chunk[1] - vstream_atom[1];
      vbiasall[i][2] = vstream_chunk[2] - vstream_atom[2];
    
      v[i][0] -= vbiasall[i][0];
      v[i][1] -= vbiasall[i][1];
      v[i][2] -= vbiasall[i][2];
    }
}

/* ----------------------------------------------------------------------
   add back in velocity bias to atom I removed by remove_bias()
   assume remove_bias() was previously called
------------------------------------------------------------------------- */

void ComputeTempDeformChunk::restore_bias(int i, double *v)
{
  double lamda[3];
  double *h_rate = domain->h_rate;
  double *h_ratelo = domain->h_ratelo;

  int index = cchunk->ichunk[i]-1;
  if (index < 0) return;

  v[0] += vbias[0];
  v[1] += vbias[1];
  v[2] += vbias[2];
}

/* ----------------------------------------------------------------------
   add back in velocity bias to all atoms removed by remove_bias_all()
   assume remove_bias_all() was previously called
------------------------------------------------------------------------- */

void ComputeTempDeformChunk::restore_bias_all()
{
  double lamda[3], vbias[3];
  double *h_rate = domain->h_rate;
  double *h_ratelo = domain->h_ratelo;

  int index;
  int *ichunk = cchunk->ichunk;

  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      index = ichunk[i]-1;
      if (index < 0) continue;
 
      v[i][0] += vbiasall[i][0];
      v[i][1] += vbiasall[i][1];
      v[i][2] += vbiasall[i][2];
    }
}

/* ----------------------------------------------------------------------
   lock methods: called by fix ave/time
   these methods insure vector/array size is locked for Nfreq epoch
     by passing lock info along to compute chunk/atom
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   increment lock counter
------------------------------------------------------------------------- */

void ComputeTempDeformChunk::lock_enable()
{
  cchunk->lockcount++;
}

/* ----------------------------------------------------------------------
   decrement lock counter in compute chunk/atom, it if still exists
------------------------------------------------------------------------- */

void ComputeTempDeformChunk::lock_disable()
{
  int icompute = modify->find_compute(idchunk);
  if (icompute >= 0) {
    cchunk = dynamic_cast<ComputeChunkAtom *>( modify->compute[icompute]);
    cchunk->lockcount--;
  }
}

/* ----------------------------------------------------------------------
   calculate and return # of chunks = length of vector/array
------------------------------------------------------------------------- */

int ComputeTempDeformChunk::lock_length()
{
  nchunk = cchunk->setup_chunks();
  return nchunk;
}

/* ----------------------------------------------------------------------
   set the lock from startstep to stopstep
------------------------------------------------------------------------- */

void ComputeTempDeformChunk::lock(Fix *fixptr, bigint startstep, bigint stopstep)
{
  cchunk->lock(fixptr,startstep,stopstep);
}

/* ----------------------------------------------------------------------
   unset the lock
------------------------------------------------------------------------- */

void ComputeTempDeformChunk::unlock(Fix *fixptr)
{
  cchunk->unlock(fixptr);
}

/* ----------------------------------------------------------------------
   free and reallocate per-chunk arrays
------------------------------------------------------------------------- */

void ComputeTempDeformChunk::allocate()
{
  memory->destroy(vthermal);
  memory->destroy(sum);
  memory->destroy(sumall);
  memory->destroy(count);
  memory->destroy(countall);
  memory->destroy(array);
  memory->destroy(vcm);
  memory->destroy(vcmall);
  memory->destroy(com);
  memory->destroy(comall);
  memory->destroy(massproc);
  memory->destroy(masstotal);
  maxchunk = nchunk;
  nmax = atom->nlocal;
  memory->create(vthermal,nmax,3,"temp/deform/chunk:vcmall");
  memory->create(sum,maxchunk,"temp/deform/chunk:sum");
  memory->create(sumall,maxchunk,"temp/deform/chunk:sumall");
  memory->create(count,maxchunk,"temp/deform/chunk:count");
  memory->create(countall,maxchunk,"temp/deform/chunk:countall");
  memory->create(array,maxchunk,nvalues,"temp/chunk:array");
  memory->create(vcm,maxchunk,3,"temp/deform/chunk:vcm");
  memory->create(vcmall,maxchunk,3,"temp/deform/chunk:vcmall");
  memory->create(com,maxchunk,3,"temp/deform/chunk:com");
  memory->create(comall,maxchunk,3,"temp/deform/chunk:comall");
  memory->create(massproc,maxchunk,"temp/deform/chunk:massproc");
  memory->create(masstotal,maxchunk,"temp/deform/chunk:masstotal");

}

/* ----------------------------------------------------------------------
   memory usage of local data
------------------------------------------------------------------------- */

double ComputeTempDeformChunk::memory_usage()
{
  // TODO EVK: this is completely wrong
  double bytes = (bigint) maxchunk * 2 * sizeof(double);
  bytes += (double) maxchunk * 2 * sizeof(int);
  bytes += (double) maxchunk * nvalues * sizeof(double);
  if (comflag || nvalues) {
    bytes += (double) maxchunk * 2 * sizeof(double);
    bytes += (double) maxchunk * 2*3 * sizeof(double);
  }
  return bytes;
}
