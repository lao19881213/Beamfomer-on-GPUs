# $Id: CMakeLists.txt 24465 2013-03-28 15:06:51Z mol $

lofar_package(OutputProc 1.0 DEPENDS Common ApplCommon Stream CoInterface MSLofar LofarStMan)

include(LofarFindPackage)
lofar_find_package(Casacore COMPONENTS casa ms tables REQUIRED)
lofar_find_package(DAL)

if(DAL_FOUND)
  lofar_find_package(HDF5 REQUIRED)
endif(DAL_FOUND)

lofar_find_package(Boost REQUIRED)

add_subdirectory(src)
add_subdirectory(test)
