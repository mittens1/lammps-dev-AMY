/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   www.cs.sandia.gov/~sjplimp/lammps.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(nvt/sllod/mol/kk,FixNVTSllodMolKokkos<LMPDeviceType>);
FixStyle(nvt/m-sllod/mol/kk,FixNVTSllodMolKokkos<LMPDeviceType>);
FixStyle(nvt/sllod/mol/kk/device,FixNVTSllodMolKokkos<LMPDeviceType>);
FixStyle(nvt/sllod/mol/kk/host,FixNVTSllodMolKokkos<LMPHostType>);
FixStyle(nvt/m-sllod/mol/kk/device,FixNVTSllodMolKokkos<LMPDeviceType>);
FixStyle(nvt/m-sllod/mol/kk/host,FixNVTSllodMolKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_FIX_NVT_SLLOD_MOL_KOKKOS_H
#define LMP_FIX_NVT_SLLOD_MOL_KOKKOS_H

#include "fix_nh_kokkos.h"
#include "kokkos_few.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

struct TagFixNVTSllodMolKokkos_compute0{};
struct TagFixNVTSllodMolKokkos_compute1{};
struct TagFixNVTSllodMolKokkos_compute2{};


template <class DeviceType>
class FixNVTSllodMolKokkos : public FixNHKokkos<DeviceType> {
 
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  FixNVTSllodMolKokkos(class LAMMPS *, int, char **);
  ~FixNVTSllodMolKokkos();

  void post_constructor() override;
  void init() override;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixNVTSllodMolKokkos_compute0, const int& i) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixNVTSllodMolKokkos_compute1, const int& i) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixNVTSllodMolKokkos_compute2, const int& i) const;

 protected:
  int molpropflag;    // 1 = molprop created by nvt/sllod/mol, 0 = user supplied
  char *id_molprop;   // Name of property/molecule fix
  
  

  class FixPropertyMol *molprop;
  class DomainKokkos *domainKK;
  class AtomKokkos *atomKK; 

  typename AT::t_x_array x;
  typename AT::t_x_array xcom;
  typename AT::t_x_array xcom_half;
  typename AT::t_x_array molcom;
  typename AT::t_x_array com;
  typename AT::t_x_array comm;
  typename AT::t_v_array v;
  typename AT::t_v_array vdelu;
  typename AT::t_v_array vcmall; 
  typename AT::t_v_array vcom;
  typename AT::t_v_array vcom_new;
  typename AT::t_v_array vfac;
  typename AT::t_f_array_const f;
  typename AT::t_float_1d rmass;
  typename AT::t_float_1d mass;
  typename AT::t_float_1d masstotal;
  typename AT::t_float_1d massproc;
  typename AT::t_int_1d type;
  typename AT::t_int_1d mask;
 
 private:
  void nh_v_temp() override;
  void nve_x() override;
};

}    // namespace LAMMPS_NS

#endif
#endif
