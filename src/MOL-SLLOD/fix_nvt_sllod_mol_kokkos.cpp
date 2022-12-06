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

#include "fix_nvt_sllod_mol_kokkos.h"

#include "atom.h"
#include "atom_kokkos.h"
#include "comm.h"
#include "compute.h"
#include "compute_temp_mol.h"
#include "domain.h"
#include "error.h"
#include "fix_deform.h"
#include "fix_deform_kokkos.h"
#include "fix_property_mol.h"
#include "group.h"
#include "math_extra.h"
#include "modify.h"
#include "memory_kokkos.h"

#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

// from FixNH:
enum{NOBIAS,BIAS};

// from FixDeform:
enum{NONE=0,FINAL,DELTA,SCALE,VEL,ERATE,TRATE,VOLUME,WIGGLE,VARIABLE};
/* ---------------------------------------------------------------------- */

template <class DeviceType>
FixNVTSllodMolKokkos<DeviceType>::FixNVTSllodMolKokkos(LAMMPS *lmp, int narg, char **arg) :
  FixNHKokkos<DeviceType>(lmp, narg, arg), molprop(nullptr), id_molprop(nullptr)
{
  atomKK = (AtomKokkos *) this->atom;
  this->kokkosable = 1;
  this->domainKK = (DomainKokkos *) this->domain;

  molpropflag = 1;

  if (!this->tstat_flag)
    this->error->all(FLERR,"Temperature control must be used with fix nvt/sllod/mol");
  if (this->pstat_flag)
    this->error->all(FLERR,"Pressure control can not be used with fix nvt/sllod/mol");

  for (int iarg = 0; iarg < narg; ++iarg) {
    if (strcmp(arg[iarg], "molprop")==0) {
      if (iarg+1 >= narg)
        this->error->all(FLERR,"Expected name of property/mol fix after 'molprop'");
      this->id_molprop = utils::strdup(arg[iarg+1]);
      this->molpropflag = 0;
      iarg += 2;
    }
  }

  // default values

  if (this->mtchain_default_flag) {
    this->mtchain = 1;

    // Fix allocation of chain thermostats so that size_vector is correct
    int ich;
    delete[] this->eta;
    delete[] this->eta_dot;
    delete[] this->eta_dotdot;
    delete[] this->eta_mass;
    this->eta = new double[this->mtchain];

    // add one extra dummy thermostat, set to zero

    this->eta_dot = new double[this->mtchain+1];
    this->eta_dot[this->mtchain] = 0.0;
    this->eta_dotdot = new double[this->mtchain];
    for (ich = 0; ich < this->mtchain; ich++) {
      this->eta[ich] = this->eta_dot[ich] = this->eta_dotdot[ich] = 0.0;
    }
    this->eta_mass = new double[this->mtchain];

    // Default mtchain in fix_nh is 3.
    this->size_vector -= 2*2*(3-this->mtchain);
  }

  // create a new fix property/mol if needed
  // id = fix-ID + _molprop
  if (molpropflag) {
    id_molprop = utils::strdup(std::string(this->id) + "_molprop");
  }

  // create a new compute temp style
  // id = fix-ID + _temp
  this->id_temp = utils::strdup(std::string(this->id) + "_temp");
  this->modify->add_compute(fmt::format("{} {} temp/mol {}",
                      this->id_temp, this->group->names[igroup], id_molprop));
  tcomputeflag = 1;
}

/* ----------------------------------------------------------------------
   Create a fix property/mol if required
---------------------------------------------------------------------- */
template <class DeviceType>
void FixNVTSllodMolKokkos<DeviceType>::post_constructor() {
  if (molpropflag)
    this->modify->add_fix(fmt::format(
          "{} {} property/mol", id_molprop, this->group->names[this->igroup]));
}

/* ---------------------------------------------------------------------- */
template <class DeviceType>
FixNVTSllodMolKokkos<DeviceType>::~FixNVTSllodMolKokkos() {
  if (molpropflag && this->modify->nfix) this->modify->delete_fix(id_molprop);
  delete [] id_molprop;
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void FixNVTSllodMolKokkos<DeviceType>::init() {
  FixNHKokkos<DeviceType>::init();

  // Check that temperature calculates a molecular temperature
  // TODO(SS): add moltemp flag to compute.h that we can check?
  if (strcmp(this->temperature->style, "temp/mol") != 0)
    this->error->all(FLERR,"fix nvt/sllod/mol requires temperature computed by "
        "compute temp/mol");

  // check fix deform remap settings

  int i;
  for (i = 0; i < this->modify->nfix; i++)
    if (strncmp(this->modify->fix[i]->style,"deform",6) == 0) {
      auto def = dynamic_cast<FixDeform *>(this->modify->fix[i]);
      if (def->remapflag != Domain::NO_REMAP)
        this->error->all(FLERR,"Using fix nvt/sllod/mol with inconsistent fix deform "
                   "remap option");
      bool elongation = false;
      for (int j = 0; j < 3; ++j) {
        if (def->set[j].style) {
          elongation = true;
          if (def->set[j].style != TRATE)
            this->error->all(FLERR,"fix nvt/sllod/mol requires the trate style for "
                "x/y/z deformation");
        }
      }
      for (int j = 3; j < 6; ++j) {
        if (def->set[j].style && def->set[j].style != ERATE) {
          if (elongation)
            this->error->all(FLERR,"fix nvt/sllod/mol requires the erate style for "
                "xy/xz/yz deformation under mixed shear/extensional flow");
          else if (comm->me == 0)
            this->error->warning(FLERR,
                "Using non-constant shear rate with fix nvt/sllod/mol");
        }
      }
      if (comm->me == 0) {
        if (def->set[5].style && def->set[5].rate != 0.0 &&
            (def->set[3].style || this->domain->yz != 0.0) &&
            (def->set[4].style != ERATE || def->set[5].style != ERATE
             || (def->set[3].style && def->set[3].style != ERATE))
            )
          this->error->warning(FLERR,"Shearing xy with a yz tilt is only handled "
              "correctly if fix deform uses the erate style for xy, xz and yz");
        if (def->end_flag)
          this->error->warning(FLERR,"SLLOD equations of motion require box deformation"
              " to occur with position updates to be strictly correct. Set the N"
              " parameter of fix deform to 0 to enable this.");
      }
      break;
    }
  if (i == this->modify->nfix)
    this->error->all(FLERR,"Using fix nvt/sllod/mol with no fix deform defined");

  // Get id of molprop
  molprop = dynamic_cast<FixPropertyMol*>(this->modify->get_fix_by_id(id_molprop));
  if (molprop == nullptr)
    this->error->all(FLERR, "Fix nvt/sllod/mol could not find a fix property/mol with id {}", id_molprop);
  // Make sure CoM can be computed
  molprop->request_com();

  // Check for exact group match since it's relied on for counting DoF by the temp compute
  if (igroup != molprop->igroup)
    this->error->all(FLERR, "Fix property/mol must be defined for the same group as fix nvt/sllod/mol");
}

/* ----------------------------------------------------------------------
   perform half-step scaling of velocities
-----------------------------------------------------------------------*/
template <class DeviceType>
void FixNVTSllodMolKokkos<DeviceType>::nh_v_temp() {
  // velocities stored as peculiar velocity (i.e. they don't include the SLLOD
  //   streaming velocity), so remove/restore bias will only be needed if some
  //   extra bias is being calculated.
  // thermostat thermal velocity only
  // vdelu = SLLOD correction = Hrate*Hinv*vthermal
  // for temperature compute with BIAS:
  //   calculate temperature since some computes require temp
  //   computed on current nlocal atoms to remove bias

  if (which == BIAS) {
    this->temperature->compute_scalar();
    this->temperature->remove_bias_all();
  }

  // Use molecular centre-of-mass velocity when calculating thermostat force
  // No need to pass ke_singles pointer since we only care about vcmall
  auto temp_mol = dynamic_cast<ComputeTempMol*>(this->temperature);
  temp_mol->vcm_compute();
  double **vcmall = temp_mol->vcmall;

  tagint *molecule = atom->molecule;
  int m;


  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  double *vcom;
  double grad_u[6], vfac[3], vcom_new[3];
  double* h_rate = this->domain->h_rate;
  double* h_inv = this->domain->h_inv;
  MathExtra::multiply_shape_shape(h_rate, h_inv, grad_u);

  double dt4 = 0.5*dthalf;
  vfac[0] = exp(-grad_u[0]*this->dt4);
  vfac[1] = exp(-grad_u[1]*this->dt4);
  vfac[2] = exp(-grad_u[2]*this->dt4);

/*  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      m = molecule[i]-1;
      if (m < 0) vcom = v[i];  // CoM velocity of single atom is just v[i]
      else vcom = vcmall[m];

      // First half step SLLOD force on CoM.
      // Don't overwrite vcom since we may need it for multiple atoms.
      vcom_new[0] = vcom[0]*vfac[0];
      vcom_new[1] = vcom[1]*vfac[1];
      vcom_new[2] = vcom[2]*vfac[2];
      vcom_new[1] -= this->dt4*grad_u[3]*vcom_new[2];
      vcom_new[0] -= this->dt4*(grad_u[5]*vcom_new[1] + grad_u[4]*vcom_new[2]);

      // Thermostat force
      vcom_new[0] *= this->factor_eta;
      vcom_new[1] *= this->factor_eta;
      vcom_new[2] *= this->factor_eta;

      // 2nd half step SLLOD force on CoM
      vcom_new[0] -= this->dt4*(grad_u[5]*vcom[1] + grad_u[4]*vcom[2]);
      vcom_new[1] -= this->dt4*grad_u[3]*vcom[2];
      vcom_new[0] *= vfac[0];
      vcom_new[1] *= vfac[1];
      vcom_new[2] *= vfac[2];

      // Update atom velocity with new CoM velocity
      v[i][0] = v[i][0] - vcom[0] + vcom_new[0];
      v[i][1] = v[i][1] - vcom[1] + vcom_new[1];
      v[i][2] = v[i][2] - vcom[2] + vcom_new[2];
    }
  }*/
  
  this->copymode = 0;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixNVTSllodMolKokkos_compute0>(0, nlocal),*this);
  this->copymode = 1;

  if (which == BIAS) this->temperature->restore_bias_all();
}


/* ----------------------------------------------------------------------
   perform full-step update of positions
-----------------------------------------------------------------------*/

template <class DeviceType>
void FixNVTSllodMolKokkos<DeviceType>::nve_x()
{
  double **x = atom->x;
  double **v = atom->v;
  int *mask = atom->mask;
  double grad_u[6], xfac[3];
  double dtv2 = this->dtv*0.5;
  int nlocal = atom->nlocal;

  double *xcom, xcom_half[3], molcom[3];
  double **&com = molprop->com;
  tagint *molecule = atom->molecule;
  tagint m;

  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  // x update by full step only for atoms in group

  double* h_rate = this->domain->h_rate;
  double* h_inv = this->domain->h_inv;
  MathExtra::multiply_shape_shape(h_rate, h_inv, grad_u);
  xfac[0] = exp(grad_u[0]*dtv2);
  xfac[1] = exp(grad_u[1]*dtv2);
  xfac[2] = exp(grad_u[2]*dtv2);

  // Calculate CoM
  molprop->com_compute();

  // First half step
  /*for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      m = molecule[i]-1;
      if (m < 0) xcom = x[i];
      else {
        // CoM stored in unwrapped coords.
        // Need to wrap to same image as x[i] streaming velocity is correct.
        molcom[0] = com[m][0];
        molcom[1] = com[m][1];
        molcom[2] = com[m][2];
        // Use inverted sign of atom's image to map CoM to correct position
        imageint ix = (2*IMGMAX - (atom->image[i] & IMGMASK)) & IMGMASK;
        imageint iy = (2*IMGMAX - (atom->image[i] >> IMGBITS & IMGMASK)) & IMGMASK;
        imageint iz = (2*IMGMAX - (atom->image[i] >> IMG2BITS)) & IMGMASK;
        this->domain->unmap(molcom, ix | (iy << IMGBITS) | (iz << IMG2BITS));
        xcom = molcom;
      }

      xcom_half[0] = xcom[0]*xfac[0];
      xcom_half[1] = xcom[1]*xfac[1];
      xcom_half[2] = xcom[2]*xfac[2];
      xcom_half[1] += dtv2*grad_u[3]*xcom_half[2];
      xcom_half[0] += dtv2*(grad_u[5]*xcom_half[1] + grad_u[4]*xcom_half[2]);

      x[i][0] = x[i][0] - xcom[0] + xcom_half[0] + this->dtv*v[i][0];
      x[i][1] = x[i][1] - xcom[1] + xcom_half[1] + this->dtv*v[i][1];
      x[i][2] = x[i][2] - xcom[2] + xcom_half[2] + this->dtv*v[i][2];
    }
  } */

  this->copymode=0;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixNVTSllodMolKokkos_compute1>(0, nlocal),*this);
  this->copymode=1;

  // Update CoM
  molprop->com_compute();

  // 2nd reversible half step
  /*for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      m = molecule[i]-1;
      if (m < 0) xcom = x[i];
      else {
        molcom[0] = com[m][0];
        molcom[1] = com[m][1];
        molcom[2] = com[m][2];
        imageint ix = (2*IMGMAX - (atom->image[i] & IMGMASK)) & IMGMASK;
        imageint iy = (2*IMGMAX - (atom->image[i] >> IMGBITS & IMGMASK)) & IMGMASK;
        imageint iz = (2*IMGMAX - (atom->image[i] >> IMG2BITS)) & IMGMASK;
        this->domain->unmap(molcom, ix | (iy << IMGBITS) | (iz << IMG2BITS));
        xcom = molcom;
      }

      xcom_half[0] = xcom[0] + dtv2*(grad_u[5]*xcom[1] + grad_u[4]*xcom[2]);
      xcom_half[1] = xcom[1] + dtv2*grad_u[3]*xcom[2];
      xcom_half[2] = xcom[2]*xfac[2];
      xcom_half[1] *= xfac[1];
      xcom_half[0] *= xfac[0];

      x[i][0] = x[i][0] - xcom[0] + xcom_half[0];
      x[i][1] = x[i][1] - xcom[1] + xcom_half[1];
      x[i][2] = x[i][2] - xcom[2] + xcom_half[2];
    }
  } */ 
  this->copymode=0;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixNVTSllodMolKokkos_compute2>(0, nlocal),*this);
  this->copymode=1;
 } 

template <class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixNVTSllodMolKokkos<DeviceType>::operator(TagFixNVTSllodMolKokkos_compute0, const int &i) const{
    if (mask[i] & groupbit) {
      m = molecule[i]-1;
      if (m < 0) vcom = v[i];  // CoM velocity of single atom is just v[i]
      else vcom = vcmall[m];

      // First half step SLLOD force on CoM.
      // Don't overwrite vcom since we may need it for multiple atoms.
      vcom_new[0] = vcom[0]*vfac[0];
      vcom_new[1] = vcom[1]*vfac[1];
      vcom_new[2] = vcom[2]*vfac[2];
      vcom_new[1] -= this->dt4*grad_u[3]*vcom_new[2];
      vcom_new[0] -= this->dt4*(grad_u[5]*vcom_new[1] + grad_u[4]*vcom_new[2]);

      // Thermostat force
      vcom_new[0] *= this->factor_eta;
      vcom_new[1] *= this->factor_eta;
      vcom_new[2] *= this->factor_eta;

      // 2nd half step SLLOD force on CoM
      vcom_new[0] -= this->dt4*(grad_u[5]*vcom[1] + grad_u[4]*vcom[2]);
      vcom_new[1] -= this->dt4*grad_u[3]*vcom[2];
      vcom_new[0] *= vfac[0];
      vcom_new[1] *= vfac[1];
      vcom_new[2] *= vfac[2];

      // Update atom velocity with new CoM velocity
      v[i][0] = v[i][0] - vcom[0] + vcom_new[0];
      v[i][1] = v[i][1] - vcom[1] + vcom_new[1];
      v[i][2] = v[i][2] - vcom[2] + vcom_new[2];
    }
  }

template <class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixNVTSllodMolKokkos<DeviceType>::operator(TagFixNVTSllodMolKokkos_compute1, const int &i) const{
    if (mask[i] & groupbit) {
      m = molecule[i]-1;
      if (m < 0) xcom = x[i];
      else {
        // CoM stored in unwrapped coords.
        // Need to wrap to same image as x[i] streaming velocity is correct.
        molcom[0] = com[m][0];
        molcom[1] = com[m][1];
        molcom[2] = com[m][2];
        // Use inverted sign of atom's image to map CoM to correct position
        imageint ix = (2*IMGMAX - (atom->image[i] & IMGMASK)) & IMGMASK;
        imageint iy = (2*IMGMAX - (atom->image[i] >> IMGBITS & IMGMASK)) & IMGMASK;
        imageint iz = (2*IMGMAX - (atom->image[i] >> IMG2BITS)) & IMGMASK;
        this->domain->unmap(molcom, ix | (iy << IMGBITS) | (iz << IMG2BITS));
        xcom = molcom;
      }

      xcom_half[0] = xcom[0]*xfac[0];
      xcom_half[1] = xcom[1]*xfac[1];
      xcom_half[2] = xcom[2]*xfac[2];
      xcom_half[1] += dtv2*grad_u[3]*xcom_half[2];
      xcom_half[0] += dtv2*(grad_u[5]*xcom_half[1] + grad_u[4]*xcom_half[2]);

      x[i][0] = x[i][0] - xcom[0] + xcom_half[0] + this->dtv*v[i][0];
      x[i][1] = x[i][1] - xcom[1] + xcom_half[1] + this->dtv*v[i][1];
      x[i][2] = x[i][2] - xcom[2] + xcom_half[2] + this->dtv*v[i][2];
    }
  }


template <class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixNVTSllodMolKokkos<DeviceType>::operator(TagFixNVTSllodMolKokkos_compute2, const int &i) const{
    if (mask[i] & groupbit) {
      m = molecule[i]-1;
      if (m < 0) xcom = x[i];
      else {
        molcom[0] = com[m][0];
        molcom[1] = com[m][1];
        molcom[2] = com[m][2];
        imageint ix = (2*IMGMAX - (atom->image[i] & IMGMASK)) & IMGMASK;
        imageint iy = (2*IMGMAX - (atom->image[i] >> IMGBITS & IMGMASK)) & IMGMASK;
        imageint iz = (2*IMGMAX - (atom->image[i] >> IMG2BITS)) & IMGMASK;
        this->domain->unmap(molcom, ix | (iy << IMGBITS) | (iz << IMG2BITS));
        xcom = molcom;
      }

      xcom_half[0] = xcom[0] + dtv2*(grad_u[5]*xcom[1] + grad_u[4]*xcom[2]);
      xcom_half[1] = xcom[1] + dtv2*grad_u[3]*xcom[2];
      xcom_half[2] = xcom[2]*xfac[2];
      xcom_half[1] *= xfac[1];
      xcom_half[0] *= xfac[0];

      x[i][0] = x[i][0] - xcom[0] + xcom_half[0];
      x[i][1] = x[i][1] - xcom[1] + xcom_half[1];
      x[i][2] = x[i][2] - xcom[2] + xcom_half[2];
    } 
  }

}
