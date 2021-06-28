# KSpace style pppm/conp/intel requires INTEL to be installed
set(CONP2_SOURCES_DIR ${LAMMPS_SOURCE_DIR}/USER-CONP2)

if(NOT PKG_USER-INTEL)
  get_target_property(LAMMPS_SOURCES lammps SOURCES)
  list(REMOVE_ITEM LAMMPS_SOURCES ${CONP2_SOURCES_DIR}/incl_pppm_intel_templates.cpp)
  set_property(TARGET lammps PROPERTY SOURCES ${LAMMPS_SOURCES})
endif()

target_include_directories(lammps PRIVATE ${CONP2_SOURCES_DIR})
