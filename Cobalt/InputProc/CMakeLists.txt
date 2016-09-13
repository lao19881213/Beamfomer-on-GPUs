# $Id: CMakeLists.txt 25318 2013-06-13 11:48:50Z mol $

lofar_package(InputProc 1.0
  DEPENDS ApplCommon Common Stream CoInterface)

include(LofarFindPackage)
lofar_find_package(Boost REQUIRED COMPONENTS date_time)
lofar_find_package(Casacore COMPONENTS casa measures)
lofar_find_package(OpenMP REQUIRED)
lofar_find_package(MPI)
lofar_find_package(UnitTest++)

add_subdirectory(src)
add_subdirectory(test)
