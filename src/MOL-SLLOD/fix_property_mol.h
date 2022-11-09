/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(property/mol,FixPropertyMol);
// clang-format on
#else

#ifndef LMP_FIX_PROPERTY_MOL_H
#define LMP_FIX_PROPERTY_MOL_H

#include "fix.h"

namespace LAMMPS_NS {

class FixPropertyMol : public Fix {
 public:
  FixPropertyMol(class LAMMPS *, int, char **);

  ~FixPropertyMol() override;
  int setmask() override;
  void init() override;
  void setup_pre_force(int) override;
  void setup_pre_force_respa(int, int) override;
  double memory_usage() override;
  double compute_array(int, int) override;

  struct PerMolecule {
    std::string name;     // Identifier
    void *address;        // Main address
    int datatype;         // INT or DOUBLE
    int cols;             // number of columns (0 for vectors)
  };

  // Register memory to be grown with the number of molecules
  // Allocation called in setup_pre_force(), so no guarantee of availability
  // until runtime, and registration should be done in init()
  void register_permolecule(std::string, void*, int, int);

  // Call to deallocate memory when no longer required
  void destroy_permolecule(void*);

  // Calculate nmolecule and grow permolecule vectors/arrays as needed
  void grow_permolecule(int=0);

  double *mass;       // per molecule mass
  double **com;       // per molecule center of mass in unwrapped coords
  bigint com_step;    // last step where com was updated
  bigint mass_step;   // last step where mass was updated

  tagint molmax;              // Max. molecule id
  int com_flag, mass_flag;    // flags for specific fields
  
  int dynamic_group;  // 1 = group is dynamic (nmolecule could change)
  int dynamic_mols;   // 1 = number of molecules could change during run

  bigint count_step;  // Last step where count_molecules was called
  tagint nmolecule;   // Number of molecules in the group
  void count_molecules();
  void mass_compute();
  void com_compute();

 protected:
  tagint nmax;                // length of permolecule arrays the last time they grew
  std::vector<PerMolecule> permolecule;

  double *massproc, **comproc;

  void mem_create(PerMolecule &item);
  void mem_grow(PerMolecule &item);
  void mem_destroy(PerMolecule &item);

 private:
  template<typename T> inline
  void mem_create_impl(PerMolecule &item);
  template<typename T> inline
  void mem_grow_impl(PerMolecule &item);
  template<typename T> inline
  void mem_destroy_impl(PerMolecule &item);
};

}    // namespace LAMMPS_NS

#endif
#endif
